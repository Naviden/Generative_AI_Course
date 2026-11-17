# Maintaining the course

Preserve the three-lesson structure. Required content, labs and assessment must fit the timed syllabus; mark extra material as optional and outside contact time. Keep the core labs CPU-only, offline after installation, and independent of paid or gated model services.

Before accepting changes:

1. Check every new equation's assumptions, signs and reduction conventions. Add numerical reference/gradient checks when changing an objective.
2. Run `python -m pytest`, `python scripts/check_materials.py`, `python -m pip check` and the four classroom CLI examples in a clean environment.
3. Check the timetable still totals 3 × 240 minutes.
4. Verify new citations against primary sources. Distinguish author claims from replication, and pin versions when discussing a recent preprint.
5. Record what was actually executed in `docs/validation.md`; do not mark legacy notebook schema validation as notebook execution.
6. For dependency changes, regenerate `requirements/core.lock` from `pyproject.toml` with `uv pip compile pyproject.toml --extra dev --universal --python-version 3.11 --torch-backend cpu --emit-index-url --index https://download.pytorch.org/whl/cpu -o requirements/core.lock`. Use the declared Python range, test the classroom platform, and let CI exercise the supported Python versions. Record the resolver version.
7. Do not add a license for inherited third-party files without confirming rights. Link to primary sources rather than copying full papers.

Before each course delivery, refresh the selected research readings, rerun the locked environment, verify external links and record the review date. Package upgrades must preserve tested behavior, not merely increase version numbers. Research outputs should disclose seeds, data/model revisions, split units, compute, uncertainty and unsuccessful runs.


## Repository layout

Keep the numbered student sequence in `lessons/`. Put historical scripts,
notebooks, inherited reference documents, data and environment snapshots in
`archive/`; do not reintroduce the old topic folders at the root. Shared
supported implementations remain in `genai_course/`, with current dependencies
in `requirements/`. When moving material, update local links and test paths,
verify asset preservation, and record the old-to-new mapping in the archive.

## Reading-group PDFs

Each reading-group lesson links to a complete PDF in `reading-group/pdf/`.
After editing these lessons, their images or the PDF builder, regenerate all
three downloads with `python scripts/build_reading_group_pdfs.py`. This requires
Pandoc 3 or newer and XeLaTeX with `fvextra` (TeX Live's `texlive-latex-extra`
package on Ubuntu; MacTeX on macOS). Student installation requirements do not change.

The builder reads the lesson Markdown directly, preserves GitHub math, expands
optional derivations, and turns repository links into clickable web links.
Review the resulting PDFs for equation, table and code layout, then commit the
PDFs and `manifest.json` with the source changes. Run
`python scripts/build_reading_group_pdfs.py --check` to verify freshness without
installing any PDF tools; this check also runs in CI. The editable probability
diagram is an SVG; regenerate its PNG with `rsvg-convert -w 2400
reading-group/images/conditional-probability.svg -o
reading-group/images/conditional-probability.png` after an image edit.
