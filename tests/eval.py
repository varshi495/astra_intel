"""Tiny retrieval eval. Write questions + the page each answer is on, then run:
    python eval.py your.pdf eval_questions.json
Reports how often the correct page shows up in the top-k retrieved chunks."""
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# pyrefly: ignore [missing-import]
from rag import DocIndex

try:
    pdf, qfile = sys.argv[1], sys.argv[2]
except IndexError:
    print("Usage: python tests/eval.py <path_to_pdf> <path_to_questions.json>")
    sys.exit(1)

index = DocIndex(open(pdf, "rb").read(), pdf)
cases = json.load(open(qfile))

hit1 = hitk = 0
for c in cases:
    pages = [h["page"] for h in index.search(c["q"])]
    ok1 = len(pages) > 0 and pages[0] == c["page"]
    okk = c["page"] in pages
    hit1 += ok1
    hitk += okk
    print(f"{'OK ' if okk else 'MISS'} p{c['page']} <- {pages}  | {c['q']}")

n = len(cases)
print(f"\nTop-1 page accuracy: {hit1}/{n} = {hit1/n:.0%}")
print(f"Top-{len(pages)} page accuracy: {hitk}/{n} = {hitk/n:.0%}")
