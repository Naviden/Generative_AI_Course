"""Validate local links, Python syntax, notebook structure and class timing."""
import ast
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import nbformat

ROOT = Path(__file__).resolve().parents[1]
errors = []
paths = [p for p in ROOT.rglob('*') if not any(part in {'.git', '.venv', 'runs', '__pycache__', '.venv-lock'} or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts)]
for path in paths:
    if not path.is_file():
        continue
    try:
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == '.ipynb':
            nbformat.validate(nbformat.read(path, as_version=4))
        elif path.suffix == '.md':
            for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
                url = urlsplit(target.strip('<>'))
                if not url.scheme and url.path and not (path.parent / unquote(url.path)).exists():
                    errors.append(f'{path.relative_to(ROOT)}: missing link {target}')
    except Exception as exc:
        errors.append(f'{path.relative_to(ROOT)}: {exc}')

syllabus = (ROOT / 'docs/syllabus.md').read_text()
classes = re.split(r'^## Class \d', syllabus, flags=re.M)[1:]
if len(classes) != 3:
    errors.append('expected exactly three class agendas')
for i, content in enumerate(classes, 1):
    rows = re.findall(r'^\| (\d+)–(\d+) \| ([^|]+)\|', content, flags=re.M)
    end = 0
    for a, b, label in rows:
        a, b = int(a), int(b)
        if a != end or b <= a:
            errors.append(f'class {i}: gap/overlap at {a}–{b}')
        end = b
    if end != 240:
        errors.append(f'class {i}: expected 240 minutes')
if errors:
    raise SystemExit('\n'.join(errors))
print(json.dumps({'status': 'ok', 'class_minutes': [240, 240, 240],
                  'checks': ['local Markdown targets', 'Python syntax', 'notebook schema', 'continuous class timing']}))
