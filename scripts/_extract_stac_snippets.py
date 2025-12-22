import sys, subprocess, math, textwrap, re
from pathlib import Path

try:
    import pypdf
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pypdf', '-q'])
    import pypdf

pdf_path = Path('docs/Aligned Minds Efficient Machines Publication 3.pdf')
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
