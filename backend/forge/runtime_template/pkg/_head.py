"""CLM projection heads in pure NumPy, loaded straight from ``head.pt`` (no torch needed).

Adapted from github.com/Contrastive-LM/CLM ``src/clm/heads.py`` (Apache-2.0, see LICENSE/NOTICE):
the same ``hidden -> width -> ... -> proj`` MLP (GELU, optional LayerNorm / residual hidden
blocks). Each head maps an encoder embedding (``task.json`` -> ``encoder.dim``) to a 512-d vector and a (state, option)
pair scores ``exp(logit_scale) * cos(state_head(s), action_head(c))``.

``head.pt`` is a regular ``torch.save`` checkpoint (``state_head``, ``action_head``,
``logit_scale``, ``cfg``). ``load_checkpoint`` reads its zip container with a restricted
unpickler that only rebuilds plain tensors as NumPy arrays and refuses any other global,
so loading it is safe and torch is not a dependency.
"""
from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np

_DTYPES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
           "BFloat16Storage": None, "LongStorage": np.int64, "IntStorage": np.int32}


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, zf: zipfile.ZipFile, prefix: str, little: bool):
        super().__init__(fh)
        self.zf, self.prefix, self.little = zf, prefix, little

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch" and name in _DTYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            return dict
        raise pickle.UnpicklingError(f"head.pt: refusing global {module}.{name}")

    def persistent_load(self, pid):
        _, storage_type, key, _location, _numel = pid
        dtype = _DTYPES.get(storage_type)
        if dtype is None:
            raise pickle.UnpicklingError(f"head.pt: unsupported storage {storage_type}")
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        dt = np.dtype(dtype).newbyteorder("<" if self.little else ">")
        return np.frombuffer(raw, dtype=dt).astype(dtype)


def _rebuild_tensor(storage, offset, size, stride, *_):
    if not size:
        return np.array(storage[offset], dtype=np.float32)
    itemsize = storage.itemsize
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride)).copy()


def load_checkpoint(path: str | Path) -> dict:
    """Read a ``torch.save`` checkpoint of plain tensors/dicts into NumPy arrays."""
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl.rsplit("/", 1)[0]
        bo = f"{prefix}/byteorder"
        little = zf.read(bo).decode().strip() != "big" if bo in zf.namelist() else True
        with zf.open(pkl) as fh:
            return _Unpickler(fh, zf, prefix, little).load()


def _erf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7): exact-GELU without scipy
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


_ACT = {"gelu": lambda x: 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0, dtype=np.float32))),
        "relu": lambda x: np.maximum(x, 0),
        "silu": lambda x: x / (1.0 + np.exp(-x))}


class Head:
    """One projection head: ``inp`` -> [hidden blocks] -> ``out``."""

    def __init__(self, sd: dict, cfg: dict):
        f = lambda k: np.ascontiguousarray(sd[k], dtype=np.float32)  # noqa: E731
        self.w_in, self.b_in = f("inp.weight").T.copy(), f("inp.bias")
        n_hidden = cfg["depth"] - 2
        self.hidden = [(f(f"hidden.{i}.weight").T.copy(), f(f"hidden.{i}.bias")) for i in range(n_hidden)]
        self.norms = [(f(f"norms.{i}.weight"), f(f"norms.{i}.bias")) if cfg.get("layernorm") else None
                      for i in range(n_hidden)]
        self.w_out, self.b_out = f("out.weight").T.copy(), f("out.bias")
        self.act = _ACT[cfg.get("activation", "gelu")]
        self.residual = bool(cfg.get("residual", False))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.act(x @ self.w_in + self.b_in)
        for (w, b), nrm in zip(self.hidden, self.norms):
            h = x @ w + b
            if nrm is not None:
                mu = h.mean(-1, keepdims=True)
                var = ((h - mu) ** 2).mean(-1, keepdims=True)
                h = (h - mu) / np.sqrt(var + 1e-5) * nrm[0] + nrm[1]
            h = self.act(h)
            x = x + h if self.residual else h
        return x @ self.w_out + self.b_out


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class HeadPair:
    """State + action heads and the logit scale from one checkpoint."""

    def __init__(self, path: str | Path):
        ck = load_checkpoint(path)
        self.cfg = dict(ck["cfg"])
        self.state_head = Head(ck["state_head"], self.cfg)
        self.action_head = Head(ck["action_head"], self.cfg)
        self.scale = float(min(np.exp(float(np.asarray(ck["logit_scale"]))), 100.0))
        self.in_dim = int(self.state_head.w_in.shape[0])   # the training encoder's embedding width

    def _check(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"encoder produced {x.shape[-1]}-d embeddings but this head was trained on "
                             f"{self.in_dim}-d ones; use the training encoder (task.json -> encoder)")
        return x

    def project_states(self, x: np.ndarray) -> np.ndarray:
        return _l2(self.state_head(self._check(x)))

    def project_actions(self, x: np.ndarray) -> np.ndarray:
        return _l2(self.action_head(self._check(x)))
