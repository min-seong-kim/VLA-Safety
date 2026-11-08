"""Smoke test: JAX sees the allocated GH200s and can run bf16 matmuls on each one."""

import json
import platform
import socket

import jax
import jax.numpy as jnp

devices = jax.devices()
assert devices and devices[0].platform == "gpu", f"No GPU backend: {devices}"
out = {"host": socket.gethostname(), "arch": platform.machine(), "jax": jax.__version__, "gpus": []}
for d in devices:
    x = jax.device_put(jnp.ones((2048, 2048), dtype=jnp.bfloat16), d)
    y = (x @ x).block_until_ready()
    assert float(y[0, 0]) == 2048.0, float(y[0, 0])
    stats = d.memory_stats() or {}
    out["gpus"].append(
        {
            "id": d.id,
            "kind": d.device_kind,
            "bytes_limit_gib": round(stats.get("bytes_limit", 0) / 2**30, 1),
        }
    )
out["status"] = "PASS"
print(json.dumps(out), flush=True)
