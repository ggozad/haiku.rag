import asyncio
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from haiku.rag.embeddings import EmbedderWrapper
from haiku.rag.utils.dependencies import raise_missing_extra

MODEL_FILE = "model.mlpackage"
TOKENIZER_FILE = "tokenizer.json"
INPUTS = {"input_ids", "attention_mask"}
OUTPUT = "embedding"


class CoreMLEmbedder(EmbedderWrapper):
    """Text embedder running a Core ML model in-process on the Neural Engine.

    `model_dir` holds `model.mlpackage` and `tokenizer.json`. The model is loaded
    on first use and released by `aclose()`.
    """

    def __init__(
        self,
        model_dir: str,
        vector_dim: int,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        super().__init__(
            embedder=None,
            vector_dim=vector_dim,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )
        if sys.platform != "darwin":
            raise RuntimeError("The coreml embedding provider runs on macOS only.")
        self._dir = Path(model_dir).expanduser()
        for name in (MODEL_FILE, TOKENIZER_FILE):
            if not (self._dir / name).exists():
                raise FileNotFoundError(
                    f"{self._dir / name} not found. A coreml model directory "
                    f"holds {MODEL_FILE} and {TOKENIZER_FILE}."
                )
        self._lock = threading.Lock()
        self._session: _Session | None = None

    async def _embed_query(self, text: str) -> list[float]:
        return (await self._embed_documents([text]))[0]

    async def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode, texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            if self._session is None:
                self._session = _Session(self._dir, self._vector_dim)
            return self._session.encode(texts)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._unload)

    def _unload(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None


class _Session:
    """A compiled model, its per-length functions, and the tokenizer."""

    def __init__(self, model_dir: Path, vector_dim: int):
        try:
            import CoreML
            import Foundation
            import objc
        except ModuleNotFoundError as exc:
            raise_missing_extra(str(exc.name), "coreml", exc)
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError as exc:
            raise_missing_extra("tokenizers", "coreml", exc)

        # PyObjC creates the framework attributes at runtime.
        self._coreml: Any = CoreML
        self._foundation: Any = Foundation
        self._objc: Any = objc
        self._vector_dim = vector_dim
        self._dtypes = {
            self._coreml.MLMultiArrayDataTypeFloat16: np.float16,
            self._coreml.MLMultiArrayDataTypeFloat32: np.float32,
            self._coreml.MLMultiArrayDataTypeDouble: np.float64,
        }
        self._models: dict[int, Any] = {}
        package = model_dir / MODEL_FILE
        self._compiled, self._temporary = compiled_path(
            package, (package / "Manifest.json").stat().st_mtime, self._compile
        )
        try:
            self._functions = model_functions(self._describe(), vector_dim)
        except BaseException:
            self.close()
            raise
        self._tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(self._functions[-1][1])

    def _url(self, path: str | Path) -> Any:
        return self._foundation.NSURL.fileURLWithPath_(str(path))

    def _compile(self, package: Path) -> str:
        url = _checked(
            self._coreml.MLModel.compileModelAtURL_error_(self._url(package), None)
        )
        return str(url.path())

    def _describe(self) -> list[tuple[str | None, Any]]:
        """Each function's name and model description."""
        asset = _checked(
            self._coreml.MLModelAsset.modelAssetWithURL_error_(
                self._url(self._compiled), None
            )
        )
        names = _completed(asset.functionNamesWithCompletionHandler_)
        if not names:
            return [(None, _completed(asset.modelDescriptionWithCompletionHandler_))]
        describe = asset.modelDescriptionOfFunctionNamed_completionHandler_
        return [(str(name), _completed(partial(describe, name))) for name in names]

    def _model(self, function: str | None, length: int) -> Any:
        if length not in self._models:
            config = self._coreml.MLModelConfiguration.alloc().init()
            config.setComputeUnits_(self._coreml.MLComputeUnitsCPUAndNeuralEngine)
            if function is not None:
                config.setFunctionName_(function)
            self._models[length] = _checked(
                self._coreml.MLModel.modelWithContentsOfURL_configuration_error_(
                    self._url(self._compiled), config, None
                )
            )
        return self._models[length]

    def _input(self, values: list[int], length: int) -> Any:
        array = _checked(
            self._coreml.MLMultiArray.alloc().initWithShape_dataType_error_(
                [1, length], self._coreml.MLMultiArrayDataTypeInt32, None
            )
        )
        data = np.frombuffer(array.dataPointer().as_buffer(length * 4), np.int32)
        data[:] = 0
        data[: len(values)] = values
        return array

    def _output(self, prediction: Any) -> list[float]:
        value = prediction.featureValueForName_(OUTPUT).multiArrayValue()
        dtype = np.dtype(self._dtypes[value.dataType()])
        step = int(value.strides()[-1])
        size = ((self._vector_dim - 1) * step + 1) * dtype.itemsize
        data = np.frombuffer(value.dataPointer().as_buffer(size), dtype)[::step]
        return data.astype(np.float32).tolist()

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        with self._objc.autorelease_pool():
            for encoding in self._tokenizer.encode_batch(texts):
                ids = encoding.ids
                function, length = next(f for f in self._functions if f[1] >= len(ids))
                features = _checked(
                    self._coreml.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
                        {
                            "input_ids": self._input(ids, length),
                            "attention_mask": self._input([1] * len(ids), length),
                        },
                        None,
                    )
                )
                prediction = _checked(
                    self._model(function, length).predictionFromFeatures_error_(
                        features, None
                    )
                )
                vectors.append(self._output(prediction))
        return vectors

    def close(self) -> None:
        self._models = {}
        if self._temporary:
            shutil.rmtree(self._compiled, ignore_errors=True)


def _checked(result: tuple[Any, Any]) -> Any:
    """The value of an Objective-C `...error:` call, raising its NSError."""
    value, error = result
    if error is not None:
        raise RuntimeError(f"Core ML: {error.localizedDescription()}")
    return value


def _completed(call: Callable[[Callable[..., None]], Any]) -> Any:
    """Run an Objective-C call taking a completion handler and wait for it."""
    done = threading.Event()
    result: list[Any] = []

    def handler(value: Any, error: Any) -> None:
        result.append((value, error))
        done.set()

    call(handler)
    done.wait()
    return _checked(result[0])


def compiled_path(
    package: Path, changed: float, compile_model: Callable[[Path], str]
) -> tuple[str, bool]:
    """Compile `package` to `model.mlmodelc` beside it, reused while newer than `changed`.

    Returns the compiled path and whether it is temporary. Core ML loads a
    compiled model at a stable path from its cache, so a read-only directory,
    which keeps the temporary compile, loads slower.
    """
    target = package.with_suffix(".mlmodelc")
    if target.exists() and target.stat().st_mtime >= changed:
        return str(target), False
    built = compile_model(package)
    try:
        staging = tempfile.mkdtemp(prefix=".haiku-rag-", dir=package.parent)
    except OSError:
        return built, True
    try:
        staged = shutil.move(built, Path(staging) / target.name)
        shutil.rmtree(target, ignore_errors=True)
        os.replace(staged, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return str(target), False


def model_functions(
    descriptions: Sequence[tuple[str | None, Any]], vector_dim: int
) -> list[tuple[str | None, int]]:
    """The model's functions and their sequence lengths, shortest first.

    Every function must take `input_ids` and `attention_mask` and return
    `embedding` of `vector_dim`.
    """
    functions = []
    for name, description in descriptions:
        label = f"function {name}" if name else "the model"
        inputs = dict(description.inputDescriptionsByName())
        if set(inputs) != INPUTS:
            raise ValueError(
                f"Core ML {label} takes {sorted(inputs)}, expected {sorted(INPUTS)}."
            )
        output = dict(description.outputDescriptionsByName()).get(OUTPUT)
        if output is None:
            raise ValueError(f"Core ML {label} has no '{OUTPUT}' output.")
        shape = list(output.multiArrayConstraint().shape())
        if shape and shape[-1] != vector_dim:
            raise ValueError(
                f"Core ML {label} returns {shape[-1]} dimensions, "
                f"embeddings.model.vector_dim is {vector_dim}."
            )
        length = inputs["input_ids"].multiArrayConstraint().shape()[-1]
        functions.append((name, int(length)))
    return sorted(functions, key=lambda f: f[1])
