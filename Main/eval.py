"""Tiny retrieval eval. Write questions + the page each answer is on, then run:
    python eval.py your.pdf eval_questions.json
Reports how often the correct page shows up in the top-k retrieved chunks."""
import json
import sys

from rag import DocIndex

pdf, qfile = sys.argv[1], sys.argv[2]
index = DocIndex(open(pdf, "rb").read(), pdf)
cases = json.load(open(qfile))

hit1 = hitk = 0
for c in cases:
    pages = [h["page"] for h in index.search(c["q"])]
    ok1, okk = pages[0] == c["page"], c["page"] in pages
    hit1 += ok1
    hitk += okk
    print(f"{'OK ' if okk else 'MISS'} p{c['page']} <- {pages}  | {c['q']}")

n = len(cases)
print(f"\nTop-1 page accuracy: {hit1}/{n} = {hit1/n:.0%}")
print(f"Top-{len(pages)} page accuracy: {hitk}/{n} = {hitk/n:.0%}")
