"""Make one Jupyter notebook with every step of the pipeline, from the Python scripts.

The scripts are the originals and are left as they are: the notebook is a copy
of them, cut into cells, one section per script, in the order the pipeline
runs (PIPELINE below). Nothing in the code is rewritten except the two things
a notebook cannot do the way a script does (rules 4 and 5).

Rules:

1. Each script gets a "Step N" section. The docstring at the top of the script
   becomes its first text cell. Its "Examples:" block is taken out and turned
   into option examples (rule 5).
2. The imports go in one cell. Constants that follow each other go in one cell.
   Every function and every class gets its own cell.
3. The comments written above a piece of code stay with it.
4. A notebook has no docstring of its own, so `__doc__.splitlines()[0]` is
   replaced by the first line of the docstring, written out.
5. A notebook has no command line, so the options are typed in a cell,
   `OPTIONS = "..."`, with the same words as on the command line.
   `parser.parse_args()` becomes `parser.parse_args(shlex.split(OPTIONS))`,
   and the `if __name__ == "__main__":` block becomes a cell that calls
   `main()`.
6. The scripts reuse names (every one has a `main()`), so each step defines
   its own versions just before it runs. Run a step's cells from the top of
   its section, not only its `main()` cell.

A script that does not have exactly one `.parse_args()` and one
`if __name__ == "__main__":` block is left out, and the reason is printed.

A notebook that already exists is not replaced, in case it was edited by hand:
--force replaces it. The notebook is saved with no output in it, and the same
scripts always give the same notebook.

The notebook goes next to the scripts by default, because the scripts read and
write `data/...` from the folder they are started in, and a notebook is started
in the folder it is saved in.

Examples:
    python to_notebooks.py
    python to_notebooks.py --force
    python to_notebooks.py --in clean_news.py analyze_news.py --out two_steps.ipynb
"""

import argparse
import ast
import json
from pathlib import Path

# The steps of the pipeline, in the order they run: the notebook follows this order
PIPELINE = [
    "google_ad_news.py",
    "fetch_summaries.py",
    "clean_news.py",
    "fetch_leads.py",
    "analyze_news.py",
    "scan_topics.py",
]
# Steps that go to the web, and what running them again does to the files in data/
WEB_WARNINGS = {
    "google_ad_news.py": "Goes to the web and **replaces** `data/google_ad_news.csv` with a new collection: "
    "the results will not be the same as the ones in the assignment note. Skip this cell to keep the saved data.",
    "fetch_summaries.py": "Goes to the web. Only fetches the links not yet in `data/article_summaries.csv`, "
    "so a second run on the same data does nothing new.",
    "fetch_leads.py": "Goes to the web. By default it writes to `data/article_leads.csv`, the file that was made by hand "
    "with TinyFish: use `OPTIONS = '--out data/article_leads_script.csv'` to keep it.",
}
# The title written above the first cell of each kind
HEADINGS = {
    "imports": "Imports",
    "constants": "Constants",
    "function": "Functions",
    "main": "main(): the steps, in the order they run",
}
# Kinds of code where pieces that follow each other share one cell
MERGED_KINDS = {"imports", "constants"}
# What the notebook cannot run as it is (rules 4 and 5), and what replaces it
FIRST_LINE_OF_DOC = "__doc__.splitlines()[0]"
PARSE_ARGS = ".parse_args()"
PARSE_OPTIONS = ".parse_args(shlex.split(OPTIONS))"
# The kernel the notebook asks for: the Python of the environment Jupyter was started from
KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}
# The name of the notebook written by default
DEFAULT_OUT = "google_ad_news_all_steps.ipynb"


def kind_of(node):
    """Say what a top-level piece of a script is: imports, constants, function, main or run."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "imports"
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return "main" if node.name == "main" else "function"
    if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
        return "run"
    return "constants"


def cut_in_pieces(source):
    """Return the docstring of a script and its code as a list of (kind, text)."""
    tree = ast.parse(source)
    lines = source.splitlines()
    nodes = tree.body
    docstring = ast.get_docstring(tree) or ""
    previous_end = 0
    if docstring:
        previous_end = nodes[0].end_lineno
        nodes = nodes[1:]

    # Each piece starts right after the piece before it, to take its comments with it
    spans = []
    for node in nodes:
        kind = kind_of(node)
        if spans and spans[-1][0] == kind and kind in MERGED_KINDS:
            spans[-1][2] = node.end_lineno
        else:
            spans.append([kind, previous_end, node.end_lineno])
        previous_end = node.end_lineno
    pieces = [(kind, "\n".join(lines[start:end]).strip("\n")) for kind, start, end in spans]
    return docstring, pieces


def split_examples(docstring, script_name):
    """Take the "Examples:" block out of a docstring. Return the text left and the options of each example."""
    text, options, in_examples = [], [], False
    for line in docstring.splitlines():
        if line.strip() == "Examples:":
            in_examples = True
        elif in_examples and line.startswith(" "):
            options.append(line.strip().removeprefix("python").strip().removeprefix(script_name).strip())
        else:
            in_examples = False
            text.append(line)
    return "\n".join(text).strip(), options


def options_cell(examples):
    """Write the cell where the options are typed."""
    lines = [
        "import shlex  # cuts the OPTIONS text into words, the way the command line does",
        "",
        '# The options, written as on the command line. "" means every default.',
        '# "--help" lists all of them (and then stops the cell, which is normal).',
    ]
    if any(examples):
        lines.append("# Examples from the script:")
        lines.extend(f"#   OPTIONS = {example!r}" for example in examples if example)
    lines.append('OPTIONS = ""')
    return "\n".join(lines)


def cell(cell_type, text, number):
    """Return one notebook cell, in the format of a .ipynb file."""
    result = {"cell_type": cell_type, "id": f"cell-{number}", "metadata": {}, "source": text.splitlines(keepends=True)}
    if cell_type == "code":
        result["execution_count"] = None
        result["outputs"] = []
    return result


def step_cells(script, step):
    """Return the cells of one step, as a list of (cell type, text). Raises ValueError when the script cannot be used."""
    source = script.read_text(encoding="utf-8")
    docstring, pieces = cut_in_pieces(source)
    kinds = [kind for kind, _ in pieces]
    if source.count(PARSE_ARGS) != 1:
        raise ValueError(f"{source.count(PARSE_ARGS)} `{PARSE_ARGS}` in it, 1 expected")
    if kinds.count("run") != 1 or kinds[-1] != "run":
        raise ValueError('it does not end with one `if __name__ == "__main__":` block')

    text, examples = split_examples(docstring, script.name)
    first_line = docstring.splitlines()[0] if docstring else script.name
    cells = [("markdown", f"## Step {step}: `{script.name}`\n\n{text}")]
    previous_kind = ""
    for kind, code in pieces:
        if kind == "run":
            cells.append(("markdown", f"### Options of step {step}"))
            cells.append(("code", options_cell(examples)))
            run_text = f"### Run step {step}"
            if script.name in WEB_WARNINGS:
                run_text += f"\n\n> **Warning:** {WEB_WARNINGS[script.name]}"
            cells.append(("markdown", run_text))
            cells.append(("code", "main()"))
            continue
        if kind != previous_kind:
            cells.append(("markdown", f"### {HEADINGS[kind]}"))
        code = code.replace(FIRST_LINE_OF_DOC, repr(first_line)).replace(PARSE_ARGS, PARSE_OPTIONS)
        cells.append(("code", code))
        previous_kind = kind
    return cells


def intro_cell(scripts):
    """Write the first cell of the notebook: what it is, the list of steps and how to use it."""
    lines = [
        "# Google ad news: every step of the pipeline",
        "",
        "One section per script, in the order they run. The `.py` scripts are the originals; "
        "this notebook is a copy made by `to_notebooks.py`. See `README.md` for what each step does and why.",
        "",
        "| Step | Script |",
        "| --- | --- |",
    ]
    lines.extend(f"| {step} | `{script.name}` |" for step, script in enumerate(scripts, start=1))
    lines += [
        "",
        "How to use it:",
        "- Start Jupyter from this folder, so `data/...` points to the data of the assignment.",
        "- Every script has a `main()`, so run a step's cells from the top of its section, not only its last cell.",
        "- Steps that go to the web have a warning above their `main()` cell.",
    ]
    return "\n".join(lines)


def build_notebook(scripts):
    """Return the notebook made from the scripts, as a dictionary, and the scripts left out with the reason."""
    cells, used, left_out = [], [], []
    for script in scripts:
        try:
            cells += step_cells(script, len(used) + 1)
            used.append(script)
        except (OSError, SyntaxError, ValueError) as error:
            left_out.append((script, error))
    cells.insert(0, ("markdown", intro_cell(used)))
    notebook = {
        "cells": [cell(cell_type, text, number) for number, (cell_type, text) in enumerate(cells, start=1)],
        "metadata": {"kernelspec": KERNEL, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, used, left_out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="sources", nargs="*", help="scripts to put in the notebook, in this order (default: the PIPELINE list)")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"notebook to write (default: {DEFAULT_OUT})")
    parser.add_argument("--force", action="store_true", help="replace the notebook if it already exists")
    args = parser.parse_args()

    scripts = [Path(source) for source in (args.sources or PIPELINE)]
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"NOT WRITTEN: {out} already exists, use --force to replace it")
        return

    notebook, used, left_out = build_notebook(scripts)
    for script, error in left_out:
        print(f"{script}: LEFT OUT ({error})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(used)} of {len(scripts)} scripts put in the notebook, {len(notebook['cells'])} cells -> {out.resolve()}")


if __name__ == "__main__":
    main()
