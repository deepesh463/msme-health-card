"""Convert notebooks/_src/*.py (percent format) into executed .ipynb files.

    python tools/build_notebooks.py            # all notebooks
    python tools/build_notebooks.py 01_eda     # one notebook

Cells start with "# %%" (code) or "# %% [markdown]" (markdown, lines prefixed with "# ").
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "_src"
OUT = ROOT / "notebooks"


def parse(text: str):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "md":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("md" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def build(name: str):
    src = SRC / f"{name}.py"
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(src.read_text(encoding="utf-8"))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
    nbformat.write(nb, OUT / f"{name}.ipynb")
    print("built", name)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in SRC.glob("*.py"))
    for n in names:
        build(n)
