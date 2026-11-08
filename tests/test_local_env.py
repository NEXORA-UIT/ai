import os

from scripts.local_env import load_local_env


def test_env_file_supplies_defaults_without_overriding_process(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_TEST_OVERRIDE", "process")
    monkeypatch.delenv("NEXORA_TEST_DEFAULT", raising=False)
    config = tmp_path / ".env"
    config.write_text(
        '# comment\nNEXORA_TEST_OVERRIDE=file\n NEXORA_TEST_DEFAULT = "a=b"\n',
        encoding="utf-8",
    )
    try:
        load_local_env(config)
        assert os.environ["NEXORA_TEST_OVERRIDE"] == "process"
        assert os.environ["NEXORA_TEST_DEFAULT"] == "a=b"
    finally:
        os.environ.pop("NEXORA_TEST_DEFAULT", None)
