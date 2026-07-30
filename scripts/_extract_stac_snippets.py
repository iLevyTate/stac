"""Print STAC-related snippets from a source PDF.

Usage: python scripts/_extract_stac_snippets.py [path/to/paper.pdf]

The default path points at a manuscript that is not committed to this repository, so
pass the PDF explicitly (or drop it at that path) before running.
"""
import sys
from pathlib import Path

try:
    import pypdf
except ImportError as e:
    raise SystemExit(
        "pypdf is required to run this helper. Install it with `pip install pypdf`."
    ) from e

# Resolve the PDF relative to the repo root so this works regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = REPO_ROOT / 'docs' / 'Aligned Minds Efficient Machines Publication 3.pdf'
pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
if not pdf_path.is_file():
    raise SystemExit(
        f"PDF not found: {pdf_path}\n"
        "This manuscript is not part of the repository. Pass a path explicitly:\n"
        "  python scripts/_extract_stac_snippets.py path/to/paper.pdf"
    )
reader = pypdf.PdfReader(str(pdf_path))
text = '\n'.join((page.extract_text() or '') for page in reader.pages)
keywords = ['STAC','Spiking','Loihi','neuromorphic','SNN','ANN-to-SNN','conversion','energy','V1','V2']
lines = text.splitlines()
seen = set()
for i, line in enumerate(lines):
    if any(k in line for k in keywords):
        snippet = '\n'.join(lines[max(0, i-3): min(len(lines), i+4)])
        if snippet in seen:
            continue
        seen.add(snippet)
        print('\n---\n' + snippet)
