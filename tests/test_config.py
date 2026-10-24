from leafdx.config import ROOT, load_config, path


def test_config_loads():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert set(cfg["crops"]) == {"turmeric", "blackgram"}
    ratios = cfg["split"]["ratios"]
    assert abs(sum(ratios.values()) - 1.0) < 1e-9


def test_paths_resolve_under_repo():
    assert path("splits").is_relative_to(ROOT)
