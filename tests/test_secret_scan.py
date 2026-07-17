from pathlib import Path

from scripts.check_secrets import find_secrets


def test_secret_scan_accepts_clean_tree(tmp_path):
    (tmp_path / "example.py").write_text("TOKEN = 'not-a-real-credential'\n")
    assert find_secrets(tmp_path) == []


def test_secret_scan_detects_private_key_without_returning_value(tmp_path):
    marker = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    credential = marker + "\nprivate-material\n"
    (tmp_path / "identity").write_text(credential)

    findings = find_secrets(tmp_path)

    assert findings == [(Path("identity"), "private key")]
    assert credential not in repr(findings)


def test_secret_scan_ignores_generated_directories(tmp_path):
    generated = tmp_path / ".venv"
    generated.mkdir()
    marker = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
    (generated / "identity").write_text(marker)
    assert find_secrets(tmp_path) == []
