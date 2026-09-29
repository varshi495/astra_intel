# Example Questions — ASTRA INTEL

Sample questions a candidate's system should be able to answer from the
three provided documents. Mix of single-section, multi-page and synthesis
questions.

> **Note for ASTRA:** verify each question against the final PDFs before
> release (Wikipedia content changes over time). Replace or reword any
> question whose answer the document no longer contains.

## Factual (single section)

1. What are the main categories or classes of UAVs described in the
   *Unmanned aerial vehicle* document?
2. What are the three primary mission types of electronic warfare listed in
   the *Electronic warfare* document?
3. What applications of unmanned ground vehicles for explosive ordnance
   disposal are described in the *Unmanned ground vehicle* document?

## Multi-page / retrieval

4. Which sections of the documents discuss autonomy or "sense-and-avoid"
   style capabilities? Point me to them.
5. What advantages over manned platforms are listed for long-endurance
   surveillance UAVs?
6. How does the *Electronic warfare* document distinguish jamming from
   deception?
7. Name one specific UGV program mentioned in the documents and its
   developer.

## Synthesis (across documents)

8. Compare unmanned aerial and unmanned ground vehicles as presented in the
   two documents: what shared operational roles do they have?
9. Across all three documents, what limitations or risks of autonomous
   systems are mentioned? Answer only with statements actually present.

## Honesty test (important)

10. "What percentage of UAV missions are fully autonomous, according to the
    documents?"

    - **Grounded behaviour:** the system should say the documents do not
      contain this statistic (they don't claim one).
    - **Hallucinated behaviour (bad):** inventing a number such as "about
      60% of missions are fully autonomous, per page 12."

## What a good grounded answer looks like

> "The documents state that electronic warfare has three mission types —
> electronic attack, electronic protection and electronic warfare support
> (*Electronic warfare*, section 'Mission types'). They do not provide
> statistics on autonomous mission shares."

Answer **+ source pointer** = minimum requirement #5 and #6 of the challenge.
