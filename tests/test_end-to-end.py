import importlib, sys, types, pathlib

def test_pipeline_full(tiny_cfg, monkeypatch, tmp_path):
    """Run the real main() but with our in‑memory config and temp dirs."""
    import ancient_greek_nlp as agn

    # patch load_config() to return tiny_cfg directly
    monkeypatch.setattr(agn, "load_config", lambda _: tiny_cfg)
    agn.main(tiny_cfg)                 # should finish without errors

    out = pathlib.Path(tiny_cfg["out_dir"])
    assert (out / "topics.jsonl").exists()
    assert any(out.glob("figs/*.png"))
