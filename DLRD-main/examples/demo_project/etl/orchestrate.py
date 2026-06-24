"""Demo orchestration module. Illustration only: the analyzer parses this
statically and never executes it. It imports a helper from a module that is
intentionally absent from the project to illustrate a missing reference."""

from missing_metrics import compute_kpi  # module intentionally absent from the project


def summarize(rows):
    """Return a small local summary of the given rows."""
    total = sum(r.get("amount", 0) for r in rows)
    return {"count": len(rows), "total": total}


def run(rows):
    """Summarize locally, then hand off to a metric helper from the missing module."""
    summary = summarize(rows)
    return compute_kpi(summary)
