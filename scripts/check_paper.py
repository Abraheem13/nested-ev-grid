#!/usr/bin/env python3
"""Static checks on paper/main.tex (run by reproduce.py after the analysis).

  * every \\ref / \\eqref has a \\label and every \\label is referenced
  * every \\cite key has a \\bibitem, every \\bibitem is cited, and the
    bibliography is numbered in order of first citation
  * every generated macro used in the text is defined in generated/numbers.tex
  * every \\input / \\includegraphics target exists
  * environments are balanced
  * every qualitative statement tagged "% claim: <name>" holds on the data
    (generated/claims.json, computed by nflev.eval.analysis)
  * no digits typed into result sentences: numbers in the Results, Discussion,
    Conclusion and Abstract must come from macros, except for an allow-list of
    configuration constants (scenario names, bus numbers, years, ...)
After compilation, the log is scanned for undefined references/citations,
overfull boxes and LaTeX errors.
Exit code 0 means all checks passed.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def section(tex: str, start: str, stops: list[str]) -> str:
    i = tex.find(start)
    if i < 0:
        return ""
    j = min([k for k in (tex.find(s, i + len(start)) for s in stops) if k > 0] or [len(tex)])
    return tex[i:j]


def expand(tex: str, base: pathlib.Path = PAPER) -> str:
    """Inline \\input{...} of hand-written files (generated files are checked
    separately: they must exist, and their macros must be defined)."""
    def sub(m):
        name = m.group(1)
        if name.startswith("generated/"):
            return m.group(0)
        f = base / (name if name.endswith(".tex") else name + ".tex")
        return expand(strip_comments(f.read_text()), base) if f.exists() else m.group(0)
    return re.sub(r"\\input\{([^}]+)\}", sub, tex)


def check(tex_path: pathlib.Path = PAPER / "main.tex") -> list[str]:
    errs: list[str] = []
    raw = tex_path.read_text()
    tex = expand(strip_comments(raw))

    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    dup = {x for x in labels if labels.count(x) > 1}
    errs += [f"duplicate label {x}" for x in sorted(dup)]
    refs = set(re.findall(r"\\(?:eq|auto|c|C)?ref\{([^}]+)\}", tex))
    errs += [f"undefined reference {r}" for r in sorted(refs - set(labels))]
    errs += [f"label never referenced: {x}" for x in sorted(set(labels) - refs)
             if not x.startswith(("sec:", "alg:line"))]

    cites = set()
    for grp in re.findall(r"\\cite\{([^}]+)\}", tex):
        cites |= {c.strip() for c in grp.split(",")}
    items = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    errs += [f"duplicate bibitem {x}" for x in sorted({x for x in items if items.count(x) > 1})]
    errs += [f"citation without bibitem: {c}" for c in sorted(cites - set(items))]
    errs += [f"bibitem never cited: {b}" for b in sorted(set(items) - cites)]
    first = []                                   # IEEE: numbered in order of first citation
    for grp in re.findall(r"\\cite\{([^}]+)\}", tex.split("\\begin{thebibliography}")[0]):
        first += [k for k in (c.strip() for c in grp.split(",")) if k not in first]
    for i, (f, b) in enumerate(zip(first, items), 1):
        if f != b:
            errs.append(f"bibliography not in citation order: [{i}] is {b}, first cited is {f}")
            break

    # qualitative claims: every "% claim: name" tag must name a claim that holds
    claims_f = PAPER / "generated" / "claims.json"
    claims = json.loads(claims_f.read_text()) if claims_f.exists() else {}
    raw_all = raw + "".join((PAPER / f).read_text() for f in ("results.tex", "discussion.tex", "conclusion.tex")
                            if (PAPER / f).exists())
    for name in re.findall(r"%\s*claim:\s*([A-Za-z0-9_]+)", raw_all):
        if name not in claims:
            errs.append(f"claim '{name}' is not computed by the analysis")
        elif not claims[name]["holds"]:
            errs.append(f"claim '{name}' does NOT hold on the data: {claims[name].get('detail', '')}")

    numbers = PAPER / "generated" / "numbers.tex"
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers.read_text())) if numbers.exists() else set()
    local = set(re.findall(r"\\(?:re)?newcommand\{?\\([A-Za-z]+)", tex))
    import_cmds = set(re.findall(r"\\DeclareMathOperator\{\\([A-Za-z]+)\}", tex))
    used = set(re.findall(r"\\(G[A-Z][A-Za-z]*)\b", tex))         # generated macros carry a G prefix
    errs += [f"generated macro not defined: \\{m}" for m in sorted(used - defined - local - import_cmds)]

    for f in re.findall(r"\\input\{([^}]+)\}", tex):
        p = PAPER / (f if f.endswith(".tex") else f + ".tex")
        if not p.exists():
            errs.append(f"missing \\input file {p.relative_to(ROOT)}")
    for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        if not (PAPER / f).exists():
            errs.append(f"missing figure {f}")

    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", tex):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        elif not stack or stack.pop() != m.group(2):
            errs.append(f"unbalanced environment near \\end{{{m.group(2)}}} (offset {m.start()})")
            break
    if stack:
        errs.append(f"unclosed environments: {stack}")
    if tex.count("{") != tex.count("}"):
        errs.append(f"unbalanced braces: {tex.count('{')} '{{' vs {tex.count('}')} '}}'")

    # hand-typed numbers in result-bearing text
    allowed = re.compile(
        r"^(?:S[1-7]|[0-9]{4}|[1-9]|1[0-9]|2[0-4]|33|69|50|60|95|99\.5|0\.95|0\.955|1\.03|"
        r"10[,{}\\ ]*000|15|12|300|500|600|0\.001|0\.0005|0\.002|11|S|"
        r"120|180|240|360|100|20|0\.12|0\.20|0\.30|0\.40|0\.05|25|30|45)$")
    body = re.sub(r"\\begin\{(table\*?|figure\*?|equation|align)\}.*?\\end\{\1\}", " ", tex, flags=re.S)
    parts = {"abstract": section(body, r"\begin{abstract}", [r"\end{abstract}"]),
             "results": section(body, r"\section{Results}", [r"\section{Discussion}"]),
             "discussion": section(body, r"\section{Discussion}", [r"\section{Conclusion}"]),
             "conclusion": section(body, r"\section{Conclusion}", [r"\section*{Data"])}
    for name, txt in parts.items():
        txt = re.sub(r"\\(?:ref|eqref|cite|label|input|includegraphics|SI|si)(?:\[[^\]]*\])?\{[^}]*\}", " ", txt)
        txt = re.sub(r"\\[A-Za-z]+", " ", txt)
        txt = re.sub(r"\$[^$]*\$", lambda m: " " if "=" not in m.group(0) else m.group(0), txt)
        for num in re.findall(r"(?<![A-Za-z0-9.])[0-9]+(?:\.[0-9]+)?", txt):
            if not allowed.match(num):
                ctx = txt[max(0, txt.find(num) - 40): txt.find(num) + 40].replace("\n", " ")
                errs.append(f"hand-typed number '{num}' in {name}: ...{ctx}...")
    return errs


def check_log(log_path: pathlib.Path = PAPER / "main.log") -> list[str]:
    if not log_path.exists():
        return ["no main.log (paper not compiled)"]
    log = log_path.read_text(errors="replace")
    errs = []
    for pat, msg in [(r"^! .*", "LaTeX error"),
                     (r"LaTeX Warning: (?:Reference|Citation) .* undefined", "undefined ref/cite"),
                     (r"There were undefined references", "undefined references"),
                     (r"Overfull \\hbox \((\d+\.\d+)pt", "overfull hbox"),
                     (r"LaTeX Warning: Label .* multiply defined", "duplicate label")]:
        for m in re.finditer(pat, log, re.M):
            if msg == "overfull hbox" and float(m.group(1)) < 1.0:
                continue
            line = log[m.start(): log.find("\n", m.start())]
            errs.append(f"{msg}: {line.strip()}")
    return errs


if __name__ == "__main__":
    e = check()
    if "--log" in sys.argv:
        e += check_log()
    for x in e:
        print("FAIL", x)
    print(f"{len(e)} problem(s)")
    sys.exit(1 if e else 0)
