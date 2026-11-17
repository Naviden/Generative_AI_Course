"""Build complete reading-group handouts with Pandoc 3+ and XeLaTeX.

Usage: python scripts/build_reading_group_pdfs.py [--check]
--check needs only Python; it rejects PDFs with stale sources or changed bytes.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "reading-group"
OUTPUT = LESSONS / "pdf"
BASE = "https://github.com/Naviden/Generative_AI_Course/blob/main/reading-group/"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs():
    paths = [Path(__file__), ROOT / "scripts/pdf/lesson-header.tex"]
    paths += sorted(LESSONS.glob("[0-9][0-9]-*.md"))
    paths += sorted((LESSONS / "images").glob("*"))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def walk(node):
    if isinstance(node, list):
        return [walk(item) for item in node]
    if not isinstance(node, dict):
        return node
    if node.get("t") == "Link":
        node["c"][2][0] = urljoin(BASE, node["c"][2][0])
    if node.get("t") == "Image":
        node["c"][2][0] = str(LESSONS / node["c"][2][0])
    if node.get("t") == "Code" and re.fullmatch(r"[A-Za-z0-9_./-]+", node["c"][1]):
        # Allow long field names and file paths to wrap inside narrow tables.
        code = node["c"][1].replace("_", r"\_\allowbreak{}")
        code = code.replace("/", r"/\allowbreak{}").replace("-", r"-\allowbreak{}")
        return {"t": "RawInline", "c": ["latex", r"\texttt{" + code + "}"]}
    if node.get("t") == "Table":
        # GFM has no column widths; TeX otherwise gives prose columns their
        # unwrapped natural width, which can run past the page edge.
        columns = node["c"][2]
        widths = {2: [0.38, 0.62], 3: [0.24, 0.40, 0.36]}.get(
            len(columns), [1 / len(columns)] * len(columns))
        for column, width in zip(columns, widths):
            column[1] = {"t": "ColWidth", "c": width}
    return {key: walk(value) for key, value in node.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest_path = OUTPUT / "manifest.json"
    if args.check:
        manifest = json.loads(manifest_path.read_text())
        expected = {f"{p.stem}.pdf" for p in LESSONS.glob("[0-9][0-9]-*.md")}
        if (manifest["sources"] != inputs()
                or set(manifest["pdfs"]) != expected
                or any(not (OUTPUT / name).is_file()
                       or digest(OUTPUT / name) != checksum
                       for name, checksum in manifest["pdfs"].items())):
            raise SystemExit("Reading-group PDFs are stale. Run python scripts/build_reading_group_pdfs.py")
        print("Reading-group PDFs match all lesson sources and image assets.")
        return
    for tool in ("pandoc", "xelatex"):
        if not shutil.which(tool):
            raise SystemExit(f"Install {tool}; see CONTRIBUTING.md, Reading-group PDFs.")
    OUTPUT.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="reading-group-pdfs-") as tmp:
        staging = Path(tmp)
        for source in sorted(LESSONS.glob("[0-9][0-9]-*.md")):
            text = source.read_text()
            title, text = text.split("\n", 1)
            # Expand optional derivations for print; keep every explanation and answer.
            text = re.sub(r"^<summary>(.*?)</summary>$", r"### \1", text, flags=re.M)
            text = re.sub(r"^</?details>\s*$", "", text, flags=re.M)
            text = re.sub(r"^\[Download as PDF\].*\n", "", text, flags=re.M)
            ast = json.loads(subprocess.check_output(
                ["pandoc", "--from=gfm", "--to=json"], input=text.encode()))
            ast = walk(ast)
            # Heading 1 is the document title; promote remaining heading levels.
            for block in ast["blocks"]:
                if block["t"] == "Header":
                    block["c"][0] = max(1, block["c"][0] - 1)
            kept = []
            for block in ast["blocks"]:
                kept.append(block)
                if (block["t"] == "Para" and len(block["c"]) == 1
                        and block["c"][0]["t"] == "Strong"):
                    kept.append({"t": "RawBlock", "c": ["latex", r"\nopagebreak[4]"]})
            ast["blocks"] = kept
            target = staging / f"{source.stem}.pdf"
            subprocess.run([
                "pandoc", "--from=json", "--pdf-engine=xelatex",
                "--resource-path", str(LESSONS),
                "--include-in-header", str(ROOT / "scripts/pdf/lesson-header.tex"),
                "--metadata", f"title={title.removeprefix('# ')}",
                "--variable", "papersize=a4", "--variable", "fontsize=11pt",
                "--variable", "geometry:margin=22mm", "--variable", "colorlinks=true",
                "--variable", "urlcolor=MidnightBlue", "--variable", "linkcolor=MidnightBlue",
                "--syntax-highlighting=none", "--output", str(target),
            ], input=json.dumps(ast).encode(), check=True)
            print(f"Built {target.name}", flush=True)
        # Publish locally only after every lesson built successfully.
        for target in staging.glob("*.pdf"):
            shutil.copyfile(target, OUTPUT / target.name)
    manifest = {
        "sources": inputs(),
        "pdfs": {p.name: digest(p) for p in sorted(OUTPUT.glob("*.pdf"))},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
