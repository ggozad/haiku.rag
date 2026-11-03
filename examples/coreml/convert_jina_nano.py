"""Convert jina-embeddings-v5-omni-nano-retrieval's text tower for the `coreml` provider.

Writes a model directory the provider reads:

    <out>/model.mlpackage   one function per sequence length (L64, L128, ...)
    <out>/tokenizer.json

Every function takes `input_ids` and `attention_mask` (int32, shape [1, L]) and
returns `embedding` ([1, 768], L2-normalized, last-token pooled). The provider
runs each text on the smallest length that fits it.

Requirements (conversion only, not needed at runtime):
    torch, sentence-transformers, coremltools>=9

Usage:
    python examples/coreml/convert_jina_nano.py \
        jinaai/jina-embeddings-v5-omni-nano-retrieval ./jina-nano-coreml
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.models.utils import MultiFunctionDescriptor, save_multifunction
from sentence_transformers import SentenceTransformer

SAMPLE = "Document: blues jazz sixth chords"


class TextTower(torch.nn.Module):
    """Bidirectional Llama-style forward with constant RoPE tables.

    coremltools cannot convert transformers' dynamic mask and RoPE helpers, so
    the trace rebuilds the forward from the model's own layers.
    """

    def __init__(self, lm, length: int):
        super().__init__()
        self.lm = lm
        self.length = length
        cfg = lm.config
        self.heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        positions = torch.arange(length)[None]
        cos, sin = lm.rotary_emb(torch.zeros(1, length, cfg.hidden_size), positions)
        self.register_buffer("cos", cos[:, None].float())
        self.register_buffer("sin", sin[:, None].float())

    @staticmethod
    def _rotate(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)

    def forward(self, input_ids, attention_mask):
        batch, length = 1, self.length
        mask = attention_mask.to(torch.float32)
        additive = ((1.0 - mask) * -1e4)[:, None, None, :]
        h = self.lm.embed_tokens(input_ids)
        for layer in self.lm.layers:
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            shape = (batch, length, self.heads, self.head_dim)
            q = attn.q_proj(x).view(shape).transpose(1, 2)
            k = attn.k_proj(x).view(shape).transpose(1, 2)
            v = attn.v_proj(x).view(shape).transpose(1, 2)
            q = q * self.cos + self._rotate(q) * self.sin
            k = k * self.cos + self._rotate(k) * self.sin
            scores = q @ k.transpose(-1, -2) * self.head_dim**-0.5 + additive
            out = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2)
            h = h + attn.o_proj(out.reshape(batch, length, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.lm.norm(h)
        # Inputs are right-padded: the last real token is where the mask drops to 0.
        following = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], dim=1)
        last = mask - following
        pooled = (h * last[:, :, None]).sum(dim=1)
        return pooled / torch.sqrt((pooled * pooled).sum(dim=-1, keepdim=True) + 1e-12)


def convert(lm, tokenizer, length: int, reference: np.ndarray, out: Path) -> Path:
    tower = TextTower(lm, length).eval()
    enc = tokenizer(
        [SAMPLE], padding="max_length", max_length=length, return_tensors="pt"
    )
    ids = enc["input_ids"].to(torch.int32)
    mask = enc["attention_mask"].to(torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(tower, (ids, mask), strict=False)
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType("input_ids", shape=(1, length), dtype=np.int32),
            ct.TensorType("attention_mask", shape=(1, length), dtype=np.int32),
        ],
        outputs=[ct.TensorType("embedding", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    path = out / f"L{length}.mlpackage"
    model.save(str(path))
    got = model.predict({"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
    vector = got["embedding"][0].astype(np.float32)
    cosine = float(reference @ vector / np.linalg.norm(vector))
    print(f"L{length}: cosine against sentence-transformers {cosine:.5f}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", help="Hugging Face id or local directory")
    parser.add_argument("out", type=Path, help="Output model directory")
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    args = parser.parse_args()

    st = SentenceTransformer(
        args.model,
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"modality": "text", "torch_dtype": torch.float32},
    )
    lm = st[0].model.language_model.float().eval()
    tokenizer = st[0].tokenizer
    reference = st.encode([SAMPLE], normalize_embeddings=True)[0]

    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = [
            convert(lm, tokenizer, length, reference, Path(tmp))
            for length in sorted(args.lengths)
        ]
        descriptor = MultiFunctionDescriptor()
        for part in parts:
            descriptor.add_function(
                str(part), src_function_name="main", target_function_name=part.stem
            )
        descriptor.default_function_name = parts[0].stem
        target = args.out / "model.mlpackage"
        shutil.rmtree(target, ignore_errors=True)
        save_multifunction(descriptor, str(target))
    tokenizer.backend_tokenizer.save(str(args.out / "tokenizer.json"))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
