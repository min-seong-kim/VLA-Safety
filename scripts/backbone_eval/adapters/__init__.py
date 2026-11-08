"""Model adapters. Each adapter exposes the same prompt-only readout interface (see base.PrefixReader).

    pi0, pi05              -> openpi_prefix.OpenPiPrefixReader   (JAX, openpi/.venv)
    openvla, prismatic,
    openvla_oft            -> prismatic_hf.PrismaticHFReader     (PyTorch container + envs/ctpkgs)
"""

MODEL_SPECS = {
    "pi0": {"family": "paligemma", "module": "openpi_prefix", "checkpoint": "checkpoints/pi0_base", "pi05": False},
    "pi05": {"family": "paligemma", "module": "openpi_prefix", "checkpoint": "checkpoints/pi05_base", "pi05": True},
    "paligemma": {"family": "paligemma", "module": "paligemma_hf", "checkpoint": "checkpoints/hf/paligemma-3b-pt-224", "letterbox": False},
    "paligemma_lb": {"family": "paligemma", "module": "paligemma_hf", "checkpoint": "checkpoints/hf/paligemma-3b-pt-224", "letterbox": True},
    "openvla": {"family": "prismatic", "module": "prismatic_hf", "checkpoint": "checkpoints/hf/openvla-7b", "kind": "hf"},
    "openvla_oft": {"family": "prismatic", "module": "prismatic_hf", "checkpoint": "checkpoints/hf/openvla-7b-oft-libero-spatial",
                    "kind": "hf_oft"},
    "prismatic": {"family": "prismatic", "module": "prismatic_hf", "checkpoint": "checkpoints/hf/prism-dinosiglip-224px+7b",
                  "kind": "tri_pt", "tokenizer_from": "checkpoints/hf/openvla-7b"},
    "pi0_libero": {"family": "paligemma", "module": "openpi_prefix", "checkpoint": "checkpoints/pi0_libero",
                   "pi05": False, "source": "gs://openpi-assets/checkpoints/pi0_libero"},
    "pi05_libero": {"family": "paligemma", "module": "openpi_prefix", "checkpoint": "checkpoints/pi05_libero",
                    "pi05": True, "model_config": {"action_horizon": 10, "discrete_state_input": False},
                    "source": "gs://openpi-assets/checkpoints/pi05_libero"},
    "openvla_libero_spatial": {
        "family": "prismatic", "module": "prismatic_hf", "kind": "hf",
        "checkpoint": "checkpoints/hf/openvla-7b-finetuned-libero-spatial",
        "tokenizer_from": "checkpoints/hf/openvla-7b", "code_from": "checkpoints/hf/openvla-7b",
        "source": "openvla/openvla-7b-finetuned-libero-spatial",
    },
    "openvla_oft_goal": {
        "family": "prismatic", "module": "prismatic_hf", "kind": "hf_oft",
        "checkpoint": "checkpoints/hf/openvla-7b-oft-libero-goal",
        "source": "moojink/openvla-7b-oft-finetuned-libero-goal",
    },
}


def load_reader(name: str, **kw):
    spec = MODEL_SPECS[name]
    if spec.get("blocked"):
        raise RuntimeError(f"{name} is blocked: {spec['blocked']}")
    if spec["module"] == "openpi_prefix":
        from .openpi_prefix import OpenPiPrefixReader

        return OpenPiPrefixReader(name, spec, **kw)
    if spec["module"] == "paligemma_hf":
        from .paligemma_hf import PaliGemmaHFReader

        return PaliGemmaHFReader(name, spec, **kw)
    from .prismatic_hf import PrismaticHFReader

    return PrismaticHFReader(name, spec, **kw)
