import contextlib
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from haiku.rag.config import AppConfig, EmbeddingModelConfig, EmbeddingsConfig
from haiku.rag.embeddings import coreml, get_embedder
from haiku.rag.embeddings.coreml import CoreMLEmbedder, compiled_path, model_functions

DIM = 4
WORDS = ["[UNK]", *(f"w{i}" for i in range(1, 40))]
INT32, FLOAT16, FLOAT32, DOUBLE = 131104, 65552, 65568, 65600
NE = "CPU_AND_NE"


def _feature(shape):
    return SimpleNamespace(
        multiArrayConstraint=lambda: SimpleNamespace(shape=lambda: shape)
    )


def _description(
    length, dim=DIM, inputs=("input_ids", "attention_mask"), output="embedding"
):
    return SimpleNamespace(
        inputDescriptionsByName=lambda: {n: _feature([1, length]) for n in inputs},
        outputDescriptionsByName=lambda: {output: _feature([1, dim])},
    )


def _write_package(model_dir: Path) -> Path:
    package = model_dir / coreml.MODEL_FILE
    package.mkdir(parents=True)
    (package / "Manifest.json").write_text("{}", encoding="utf-8")
    return package


def _write_tokenizer(model_dir: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / coreml.TOKENIZER_FILE))


def _error(message):
    return SimpleNamespace(localizedDescription=lambda: message)


class _Buffer:
    def __init__(self, data: bytearray):
        self.data = data

    def as_buffer(self, size: int) -> memoryview:
        return memoryview(self.data)[:size]


class _URL:
    def __init__(self, path):
        self._path = str(path)

    def path(self):
        return self._path


class _Config:
    function = None
    units = None

    def setComputeUnits_(self, units):
        self.units = units

    def setFunctionName_(self, name):
        self.function = name


class _Array:
    def __init__(self, data: bytearray, dtype=INT32, step=1):
        self.data, self.dtype, self.step = data, dtype, step

    def dataPointer(self):
        return _Buffer(self.data)

    def dataType(self):
        return self.dtype

    def strides(self):
        return [DIM * self.step, self.step]


def _alloc(factory):
    return SimpleNamespace(alloc=factory)


class FakeCoreML:
    """A stand-in for PyObjC's CoreML, Foundation and objc modules."""

    def __init__(self, descriptions: dict[str | None, Any]):
        self.descriptions = descriptions
        self.loaded: list[tuple[str, Any, str | None]] = []
        self.predictions: list[tuple[str | None, list[int], list[int]]] = []
        self.compiled: list[str] = []
        self.compile_error: str | None = None
        self.output_dtype = FLOAT16
        self.output_step = 1

    def _predict(self, function, features):
        ids = np.frombuffer(features["input_ids"].data, np.int32)
        mask = np.frombuffer(features["attention_mask"].data, np.int32)
        self.predictions.append((function, ids.tolist(), mask.tolist()))
        values = [int(mask.sum()), int(ids.sum()), len(ids), 1]
        dtype = {FLOAT16: np.float16, FLOAT32: np.float32, DOUBLE: np.float64}
        out = np.zeros(DIM * self.output_step, dtype[self.output_dtype])
        out[:: self.output_step] = values
        value = _Array(bytearray(out.tobytes()), self.output_dtype, self.output_step)
        feature = SimpleNamespace(multiArrayValue=lambda: value)
        return SimpleNamespace(featureValueForName_=lambda name: feature), None

    def _compile(self, tmp_path, url, _):
        if self.compile_error:
            return None, _error(self.compile_error)
        out = tmp_path / f"compiled-{len(self.compiled)}.mlmodelc"
        out.mkdir()
        self.compiled.append(url.path())
        return _URL(out), None

    def _load(self, url, config, _):
        self.loaded.append((url.path(), config.units, config.function))
        model = SimpleNamespace(
            predictionFromFeatures_error_=lambda features, _: self._predict(
                config.function, features
            )
        )
        return model, None

    def _asset(self):
        descriptions = self.descriptions
        return SimpleNamespace(
            functionNamesWithCompletionHandler_=lambda done: done(
                [n for n in descriptions if n is not None], None
            ),
            modelDescriptionWithCompletionHandler_=lambda done: done(
                descriptions[None], None
            ),
            modelDescriptionOfFunctionNamed_completionHandler_=lambda name, done: done(
                descriptions[name], None
            ),
        )

    def install(self, monkeypatch, tmp_path: Path) -> None:
        def multi_array(shape, dtype, _):
            assert dtype == INT32
            return _Array(bytearray(4 * shape[0] * shape[1])), None

        core_ml = types.ModuleType("CoreML")
        core_ml.__dict__.update(
            MLMultiArrayDataTypeInt32=INT32,
            MLMultiArrayDataTypeFloat16=FLOAT16,
            MLMultiArrayDataTypeFloat32=FLOAT32,
            MLMultiArrayDataTypeDouble=DOUBLE,
            MLComputeUnitsCPUAndNeuralEngine=NE,
            MLModel=SimpleNamespace(
                compileModelAtURL_error_=lambda url, e: self._compile(tmp_path, url, e),
                modelWithContentsOfURL_configuration_error_=self._load,
            ),
            MLModelAsset=SimpleNamespace(
                modelAssetWithURL_error_=lambda url, _: (self._asset(), None)
            ),
            MLModelConfiguration=_alloc(lambda: SimpleNamespace(init=_Config)),
            MLMultiArray=_alloc(
                lambda: SimpleNamespace(initWithShape_dataType_error_=multi_array)
            ),
            MLDictionaryFeatureProvider=_alloc(
                lambda: SimpleNamespace(
                    initWithDictionary_error_=lambda d, _: (d, None)
                )
            ),
        )
        foundation = types.ModuleType("Foundation")
        foundation.__dict__.update(NSURL=SimpleNamespace(fileURLWithPath_=_URL))
        objc = types.ModuleType("objc")
        objc.__dict__.update(autorelease_pool=contextlib.nullcontext)
        for name, module in [
            ("CoreML", core_ml),
            ("Foundation", foundation),
            ("objc", objc),
        ]:
            monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    directory = tmp_path / "model"
    directory.mkdir()
    _write_package(directory)
    _write_tokenizer(directory)
    return directory


def _install(monkeypatch, tmp_path, descriptions) -> FakeCoreML:
    fake = FakeCoreML(descriptions)
    fake.install(monkeypatch, tmp_path)
    return fake


@pytest.fixture
def fake(model_dir, tmp_path, monkeypatch):
    return _install(
        monkeypatch, tmp_path, {"L8": _description(8), "L4": _description(4)}
    )


def _unwritable(**kwargs):
    raise PermissionError("read-only")


def _config(name: str, **model) -> AppConfig:
    return AppConfig(
        embeddings=EmbeddingsConfig(
            model=EmbeddingModelConfig(
                provider="coreml", name=name, vector_dim=DIM, **model
            )
        )
    )


def test_get_embedder_builds_the_coreml_embedder(model_dir):
    embedder = get_embedder(_config(str(model_dir), query_prefix="w1 "))

    assert isinstance(embedder, CoreMLEmbedder)
    assert embedder.vector_dim == DIM
    assert embedder.supports_images is False


def test_coreml_refuses_multimodal(model_dir):
    with pytest.raises(ValueError, match="does not support multimodal"):
        get_embedder(_config(str(model_dir), multimodal=True))


def test_coreml_runs_on_macos_only(model_dir, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(RuntimeError, match="macOS only"):
        CoreMLEmbedder(str(model_dir), DIM)


@pytest.mark.parametrize("missing", [coreml.MODEL_FILE, coreml.TOKENIZER_FILE])
def test_model_directory_must_hold_model_and_tokenizer(model_dir, missing):
    target = model_dir / missing
    if target.is_dir():
        (target / "Manifest.json").unlink()
        target.rmdir()
    else:
        target.unlink()

    with pytest.raises(FileNotFoundError, match=missing):
        CoreMLEmbedder(str(model_dir), DIM)


async def test_texts_run_on_the_shortest_function_that_fits(model_dir, fake):
    embedder = CoreMLEmbedder(str(model_dir), DIM, document_prefix="w1 ")

    vectors = await embedder.embed_documents(["w2 w3", "w2 w3 w4 w5 w6"])

    assert [p[0] for p in fake.predictions] == ["L4", "L8"]
    assert fake.predictions[0][1:] == ([1, 2, 3, 0], [1, 1, 1, 0])
    assert vectors == [[3.0, 6.0, 4.0, 1.0], [6.0, 21.0, 8.0, 1.0]]
    assert all(units == NE for _, units, _ in fake.loaded)


async def test_a_text_longer_than_every_function_is_truncated(model_dir, fake):
    embedder = CoreMLEmbedder(str(model_dir), DIM)

    await embedder.embed_documents([" ".join(["w1"] * 20)])

    assert fake.predictions[0] == ("L8", [1] * 8, [1] * 8)


async def test_query_is_prefixed_and_functions_load_once(model_dir, fake):
    embedder = CoreMLEmbedder(str(model_dir), DIM, query_prefix="w9 ")

    first = await embedder.embed_query("w2")
    await embedder.embed_query("w3")

    assert first == [2.0, 11.0, 4.0, 1.0]
    assert [f for _, _, f in fake.loaded] == ["L4"]
    assert len(fake.compiled) == 1


async def test_a_single_function_model_loads_without_a_function_name(
    model_dir, tmp_path, monkeypatch
):
    fake = _install(monkeypatch, tmp_path, {None: _description(8)})
    embedder = CoreMLEmbedder(str(model_dir), DIM)

    await embedder.embed_query("w1")

    assert [f for _, _, f in fake.loaded] == [None]


@pytest.mark.parametrize("dtype,step", [(FLOAT32, 1), (DOUBLE, 1), (FLOAT16, 2)])
async def test_outputs_are_read_by_type_and_stride(model_dir, fake, dtype, step):
    fake.output_dtype, fake.output_step = dtype, step
    embedder = CoreMLEmbedder(str(model_dir), DIM)

    assert await embedder.embed_query("w1 w2") == [2.0, 3.0, 4.0, 1.0]


async def test_aclose_releases_the_model_and_the_next_call_reloads(model_dir, fake):
    embedder = CoreMLEmbedder(str(model_dir), DIM)
    await embedder.aclose()

    await embedder.embed_query("w1")
    await embedder.aclose()
    await embedder.embed_query("w1")

    assert [f for _, _, f in fake.loaded] == ["L4", "L4"]


async def test_a_temporary_compile_is_removed_on_close(model_dir, fake, monkeypatch):
    monkeypatch.setattr(coreml.tempfile, "mkdtemp", _unwritable)
    embedder = CoreMLEmbedder(str(model_dir), DIM)
    await embedder.embed_query("w1")
    compiled = Path(fake.loaded[0][0])
    assert compiled.exists()

    await embedder.aclose()

    assert not compiled.exists()
    assert not (model_dir / "model.mlmodelc").exists()


async def test_a_model_breaking_the_contract_fails_and_is_released(
    model_dir, tmp_path, monkeypatch
):
    _install(monkeypatch, tmp_path, {"L8": _description(8, dim=768)})
    monkeypatch.setattr(coreml.tempfile, "mkdtemp", _unwritable)
    embedder = CoreMLEmbedder(str(model_dir), DIM)

    with pytest.raises(ValueError, match="returns 768 dimensions"):
        await embedder.embed_query("w1")

    assert not (tmp_path / "compiled-0.mlmodelc").exists()


async def test_a_core_ml_error_is_raised_with_its_description(model_dir, fake):
    fake.compile_error = "bad package"
    embedder = CoreMLEmbedder(str(model_dir), DIM)

    with pytest.raises(RuntimeError, match="Core ML: bad package"):
        await embedder.embed_query("w1")


@pytest.mark.parametrize("module", ["CoreML", "tokenizers"])
async def test_missing_dependency_names_the_extra(model_dir, fake, monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, None)
    embedder = CoreMLEmbedder(str(model_dir), DIM)

    with pytest.raises(ImportError, match=r"haiku\.rag-slim\[coreml\]"):
        await embedder.embed_query("w1")


def _compiler(tmp_path, calls):
    def compile_model(package):
        out = tmp_path / f"built-{len(calls)}.mlmodelc"
        out.mkdir()
        (out / "model.mil").touch()
        calls.append(package)
        return str(out)

    return compile_model


def test_compiled_model_is_kept_beside_the_package_and_reused(model_dir, tmp_path):
    package = model_dir / coreml.MODEL_FILE
    calls = []

    first = compiled_path(package, 0.0, _compiler(tmp_path, calls))
    second = compiled_path(package, 0.0, _compiler(tmp_path, calls))

    assert first == second == (str(model_dir / "model.mlmodelc"), False)
    assert (model_dir / "model.mlmodelc" / "model.mil").exists()
    assert len(calls) == 1
    assert [p.name for p in model_dir.iterdir() if p.name.startswith(".")] == []


def test_a_changed_package_is_compiled_again(model_dir, tmp_path):
    package = model_dir / coreml.MODEL_FILE
    calls = []
    compiled_path(package, 0.0, _compiler(tmp_path, calls))
    target = model_dir / "model.mlmodelc"
    (target / "stale").touch()
    os.utime(target, (1.0, 1.0))

    compiled_path(package, 2.0, _compiler(tmp_path, calls))

    assert len(calls) == 2
    assert not (target / "stale").exists()


def test_model_functions_are_sorted_by_length():
    descriptions = [("L512", _description(512)), ("L64", _description(64))]

    assert model_functions(descriptions, DIM) == [("L64", 64), ("L512", 512)]


def test_an_output_without_a_declared_shape_is_accepted():
    description = _description(8)
    description.outputDescriptionsByName = lambda: {"embedding": _feature([])}

    assert model_functions([("L8", description)], DIM) == [("L8", 8)]


@pytest.mark.parametrize(
    "name,description,message",
    [
        ("L8", _description(8, inputs=("input_ids", "mask")), "function L8 takes"),
        (None, _description(8, inputs=("input_ids",)), "the model takes"),
        ("L8", _description(8, dim=768), "returns 768 dimensions"),
        ("L8", _description(8, output="pooled"), "no 'embedding' output"),
    ],
)
def test_model_functions_enforce_the_contract(name, description, message):
    with pytest.raises(ValueError, match=message):
        model_functions([(name, description)], DIM)


@pytest.mark.skipif(sys.platform != "darwin", reason="Core ML runs on macOS only")
async def test_a_real_core_ml_model_embeds_on_this_machine(tmp_path):
    ct = pytest.importorskip("coremltools")
    from coremltools.converters.mil import Builder

    mb: Any = Builder  # ops are registered at runtime
    length, vocab = 8, len(WORDS)
    table = np.arange(vocab * DIM, dtype=np.float32).reshape(vocab, DIM)
    int32 = ct.converters.mil.mil.types.int32

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, length), dtype=int32),
            mb.TensorSpec(shape=(1, length), dtype=int32),
        ],
        opset_version=ct.target.macOS15,
    )
    def program(input_ids, attention_mask):
        rows = mb.gather(x=table, indices=input_ids, axis=0)
        weights = mb.expand_dims(x=mb.cast(x=attention_mask, dtype="fp32"), axes=[-1])
        return mb.reduce_sum(x=mb.mul(x=rows, y=weights), axes=[1], name="embedding")

    model_dir = tmp_path / "tiny"
    model_dir.mkdir()
    ct.convert(
        program,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15,
    ).save(str(model_dir / coreml.MODEL_FILE))
    _write_tokenizer(model_dir)

    embedder = CoreMLEmbedder(str(model_dir), DIM)
    try:
        vectors = await embedder.embed_documents(["w1 w2", "w3"])
    finally:
        await embedder.aclose()

    assert vectors == [(table[1] + table[2]).tolist(), table[3].tolist()]
    assert (model_dir / "model.mlmodelc").is_dir()
