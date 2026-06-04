# -*- coding: utf-8 -*-
"""Генерирует relax.ipynb из make_report_plots.py (формат ячеек `# %%`).

Запускать после правок графиков:
    python build_notebook.py
"""
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SRC = HERE / "make_report_plots.py"
DST = HERE / "relax.ipynb"

lines = SRC.read_text(encoding="utf-8").split("\n")

cells = []
cur_type = None          # None → ещё не дошли до первой ячейки (заголовок файла)
cur_label = ""
cur_lines = []


def flush():
    if cur_type is None:
        return
    text = "\n".join(cur_lines).strip("\n")
    if cur_type == "markdown":
        md = "\n".join(re.sub(r"^# ?", "", ln) for ln in cur_lines).strip()
        if md:
            cells.append(nbf.v4.new_markdown_cell(md))
    else:
        body = text
        if cur_label.strip():
            body = f"# {cur_label.strip()}\n{body}" if body else f"# {cur_label.strip()}"
        if body.strip():
            cells.append(nbf.v4.new_code_cell(body))


for line in lines:
    m = re.match(r"^# %%(.*)$", line)
    if m:
        flush()
        tail = m.group(1)
        cur_type = "markdown" if "[markdown]" in tail else "code"
        cur_label = "" if "[markdown]" in tail else tail
        cur_lines = []
    elif cur_type is not None:
        cur_lines.append(line)
flush()

nb = nbf.v4.new_notebook()
nb.cells = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, str(DST))
print(f"записано {DST}  ({len(cells)} ячеек)")
