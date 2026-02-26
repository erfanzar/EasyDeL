from easydel.infra.elarge_model.builders import to_from_pretrained_kwargs
from easydel.infra.elarge_model.elarge_model import eLargeModel


def test_to_from_pretrained_kwargs_materializes_qmm_quantization_overrides():
    cfg = {
        "model": {"name_or_path": "dummy-model"},
        "quantization": {
            "model": {"dtype": "nf4", "group_size": 64},
            "apply_quantization": True,
            "use_qmm_best_config": False,
            "qmm_platform_override": "xla",
            "qmm_tpu_path_override": "packed",
        },
    }

    kwargs = to_from_pretrained_kwargs(cfg)
    config_kwargs = kwargs["config_kwargs"]

    assert config_kwargs["use_qmm_best_config"] is False
    assert config_kwargs["qmm_platform_override"] == "xla"
    assert config_kwargs["qmm_tpu_path_override"] == "packed"


def test_set_quantization_exposes_qmm_controls():
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "quantization": {}}

    elm.set_quantization(
        method="nf4",
        group_size=64,
        use_qmm_best_config=False,
        qmm_platform_override="xla",
        qmm_tpu_path_override="packed",
    )

    quant = elm.config["quantization"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert quant["use_qmm_best_config"] is False  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert quant["qmm_platform_override"] == "xla"  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert quant["qmm_tpu_path_override"] == "packed"  # pyright: ignore[reportTypedDictNotRequiredAccess]
