"""Agentic fact-checking: LLM judge -> DuckDuckGo search -> LLM comparison."""

from config import WEB_MAX_RESULTS
from llm import compare_claim_with_web, judge_chunk

try:
    from ddgs import DDGS  # current package name
except ImportError:  # older installs
    from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = WEB_MAX_RESULTS) -> list:
    """Search DuckDuckGo (no API key needed). Returns [{title, snippet, url}]."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in results
        ]
    except Exception as e:
        return [{"title": "search error", "snippet": str(e), "url": ""}]


def fact_check_chunks(model: str, retrieved: list, on_status=None) -> list:
    """
    Run the agent over each retrieved chunk.

    1. LLM judges the chunk using its own knowledge.
    2. If doubtful -> search the web and let the LLM compare claim vs results.

    Returns one report per chunk:
      {chunk, judge, web, verification}
    """
    log = on_status or (lambda msg: None)
    reports = []

    for n, chunk in enumerate(retrieved, 1):
        log(f"Checking chunk {n}/{len(retrieved)} (page {chunk['page']})...")
        judge = judge_chunk(model, chunk["text"])
        report = {"chunk": chunk, "judge": judge, "web": None, "verification": None}

        is_doubtful = judge.get("verdict") == "doubtful"
        query = judge.get("search_query") or judge.get("suspect_claim")

        if is_doubtful and query:
            log(f"Doubtful claim found, searching the web: *{query}*")
            report["web"] = web_search(query)
            report["verification"] = compare_claim_with_web(
                model, judge.get("suspect_claim") or chunk["text"], report["web"]
            )

        reports.append(report)

    return reports


def skip_fact_check(retrieved: list) -> list:
    """Build reports with no checking (used when fact-check is toggled off)."""
    return [
        {"chunk": c, "judge": {"verdict": "correct"}, "web": None, "verification": None}
        for c in retrieved
    ]