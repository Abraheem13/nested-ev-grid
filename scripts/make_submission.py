#!/usr/bin/env python3
"""Builds the journal submission package from paper/:

  python scripts/make_submission.py

Copies the LaTeX sources, the class file and exactly the generated tables,
figures and number macros the paper uses into paper/submission/, removes
comment-only lines and internal "% claim:" tags, compiles the copy in that
folder (pdflatex, three passes), checks that its text equals that of
paper/main.pdf, and writes paper/submission.zip.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = PAPER / "submission"


def clean(tex: str) -> str:
    lines = []
    for line in tex.splitlines():
        if line.lstrip().startswith("%"):
            continue                                        # comment-only line: drop
        lines.append(re.sub(r"(?<!\\)%\s*claim:.*$", "%", line))
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "generated").mkdir(parents=True)
    queue, seen = ["main.tex"], set()
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        src = PAPER / name
        text = src.read_text()
        for inc in re.findall(r"\\input\{([^}]+)\}", text):
            queue.append(inc if inc.endswith(".tex") else inc + ".tex")
        for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            shutil.copy2(PAPER / fig, OUT / fig)
        (OUT / name).write_text(clean(text) if not name.startswith("generated/") else text)
    shutil.copy2(PAPER / "IEEEtran.cls", OUT / "IEEEtran.cls")
    for _ in range(3):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                           cwd=OUT, capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stdout[-3000:])
    txt = lambda p: subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True).stdout
    if txt(OUT / "main.pdf") != txt(PAPER / "main.pdf"):
        sys.exit("submission PDF text differs from paper/main.pdf")
    for f in OUT.glob("main.*"):
        if f.suffix in (".aux", ".log", ".out"):
            f.unlink()
    zpath = PAPER / "submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f.name != "main.pdf":
                z.write(f, f.relative_to(OUT))
    n = sum(1 for f in OUT.rglob("*") if f.is_file())
    print(f"submission: {OUT.relative_to(ROOT)} ({n} files), {zpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
