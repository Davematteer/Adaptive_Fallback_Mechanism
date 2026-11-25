"""Runs a neural network file and returns its raw outputs as NumPy arrays.

- `.engine` -> TensorRT (on the Jetson). Uses the tensor-name API (`num_io_tensors`,
  `set_tensor_address`, `execute_async_v3`), available in TensorRT 8.5+ and 10.x.
  GPU buffers are torch CUDA tensors (torch is already installed with ultralytics),
  so pycuda is not needed.
- `.onnx`   -> ONNX Runtime (for testing the same pipeline on a PC without TensorRT).
"""

from pathlib import Path

import numpy as np


def load_runner(path: str | Path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    return TensorRTRunner(path) if path.suffix == ".engine" else OnnxRunner(path)


class TensorRTRunner:
    def __init__(self, path: Path):
        import tensorrt as trt
        import torch

        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not load {path.name}: built with a different TensorRT version?")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.input_name, self.input_buf, self.outputs = None, None, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            np_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            buf = torch.empty(shape, dtype=torch.from_numpy(np.empty(0, np_dtype)).dtype, device="cuda")
            self.context.set_tensor_address(name, buf.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name, self.input_buf = name, buf
            else:
                self.outputs[name] = buf
        self.input_shape = tuple(self.input_buf.shape)

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        with torch.cuda.stream(self.stream):  # copy in, run and copy out on one stream, in order
            self.input_buf.copy_(torch.from_numpy(np.ascontiguousarray(x)).to(self.input_buf.dtype), non_blocking=False)
            self.context.execute_async_v3(self.stream.cuda_stream)
            self.stream.synchronize()
            return {name: buf.cpu().numpy() for name, buf in self.outputs.items()}


class OnnxRunner:
    def __init__(self, path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name, self.input_shape = inp.name, tuple(inp.shape)
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, x: np.ndarray) -> dict[str, np.ndarray]:
        return dict(zip(self.output_names, self.session.run(None, {self.input_name: np.ascontiguousarray(x)})))
