from pathlib import Path
import py_compile
def test_scripts_compile():
    root=Path(__file__).resolve().parents[1]
    for path in (root/"scripts").glob("*.py"):
        py_compile.compile(str(path),doraise=True)
