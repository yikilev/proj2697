"""Импорт всех прикладных модулей в чистом процессе без внешних действий."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_application_imports_without_io_or_test_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("SEARCH_MODE", "invalid-until-startup")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "absent" / "theses.sqlite3"))
    modules = ["main"]
    for package in ("api", "config", "domain", "external", "infrastructure", "scripts", "services"):
        modules.extend(
            ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
            for path in sorted((ROOT / package).rglob("*.py"))
        )
    script = r'''
import importlib
import os
import sys

sys.path.insert(0, sys.argv[1])
violations = []


def reject(action):
    violations.append(action)
    raise AssertionError(action)


class NoIntegrationOrTestImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"tests", "sentence_transformers", "torch"}:
            reject("Unexpected dependency: " + fullname)


def audit(event, args):
    if event in {
        "sqlite3.connect", "socket.connect", "socket.bind", "socket.getaddrinfo",
        "subprocess.Popen", "os.system", "os.mkdir", "os.remove", "os.rename",
    }:
        reject("External action: " + event)
    if event == "open":
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            reject("File write: " + str(path))
        if isinstance(path, str) and path.endswith((".json", ".db", ".sqlite", ".sqlite3", ".bin", ".safetensors")):
            reject("Data read during import: " + path)


sys.meta_path.insert(0, NoIntegrationOrTestImports())
sys.addaudithook(audit)
for module in sys.argv[2:]:
    importlib.import_module(module)
assert not violations, violations
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(ROOT), *modules],
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", check=False, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
