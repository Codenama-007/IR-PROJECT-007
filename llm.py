"""All interaction with the local LLM (Ollama): judging, comparing and answering."""

import json

import ollama

JUDGE_SYSTEM = """You are a careful fact-checker. You are given a passage from a document.
Using only your own knowledge, decide whether the passage contains a verifiable factual
statement (dates, numbers, scientific facts, historical events, definitions, named facts)
that looks WRONG or SUSPICIOUS.
- Opinions, instructions, company-internal info, and things you cannot judge are "correct" (no doubt).
- Only answer "doubtful" if a specific factual claim conflicts with what you know.
Reply ONLY with JSON:
{"verdict": "correct" | "doubtful", "suspect_claim": "<the exact claim or empty string>", "search_query": "<short web search query to verify it or empty string>"}"""

COMPARE_SYSTEM = """You compare a claim from a document against web search results.
Decide:
- "contradicted": the web results clearly show the claim is false.
- "supported": the web results confirm the claim.
- "inconclusive": the results are unrelated or insufficient.
Be strict: only say "contradicted" if the results explicitly give a different fact.
Reply ONLY with JSON:
{"verdict": "contradicted" | "supported" | "inconclusive", "correct_information": "<what the web says is correct, or empty>", "explanation": "<one or two sentences>", "source_urls": ["<url>", ...]}"""

ANSWER_SYSTEM = """You answer questions about an uploaded document using ONLY the provided context.
Some context chunks come with a FACT-CHECK NOTE. If a note says a claim is CONTRADICTED BY THE WEB,
you must clearly tell the user that this statement in the document appears to be wrong, quote what the
document says, state what the correct information is, and mention the source URLs.
If the context does not contain the answer, say so. Be concise."""


def _chat_json(model: str, system: str, user: str) -> dict:
    """Call Ollama in JSON mode and parse safely."""
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            options={"temperature": 0},
        )
        return json.loads(resp["message"]["content"])
    except Exception as e:  # Ollama offline, model missing, bad JSON...
        return {"error": str(e)}


def judge_chunk(model: str, chunk_text: str) -> dict:
    """Step 1 of the agent: does this chunk look correct from the LLM's own knowledge?"""
    return _chat_json(model, JUDGE_SYSTEM, f"PASSAGE:\n{chunk_text}")


def compare_claim_with_web(model: str, claim: str, web_results: list) -> dict:
    """Step 2 of the agent: compare the suspect claim with web search results."""
    web_text = "\n\n".join(
        f"[{i + 1}] {r['title']}\n{r['snippet']}\nURL: {r['url']}"
        for i, r in enumerate(web_results)
    )
    return _chat_json(
        model, COMPARE_SYSTEM, f"CLAIM FROM DOCUMENT:\n{claim}\n\nWEB RESULTS:\n{web_text}"
    )


def build_context(reports: list) -> str:
    """Format retrieved chunks (plus any fact-check notes) as context for the answer prompt."""
    parts = []
    for i, r in enumerate(reports, 1):
        c = r["chunk"]
        block = f"[Chunk {i} | {c['source']} p.{c['page']}]\n{c['text']}"
        v = r.get("verification")
        if v and v.get("verdict") == "contradicted":
            block += (
                f"\nFACT-CHECK NOTE: CONTRADICTED BY THE WEB. "
                f"Suspect claim: {r['judge'].get('suspect_claim')}. "
                f"Correct info: {v.get('correct_information')}. "
                f"Explanation: {v.get('explanation')}. "
                f"Sources: {', '.join(v.get('source_urls', []))}"
            )
        parts.append(block)
    return "\n\n".join(parts)


def stream_answer(model: str, question: str, context: str):
    """Yield the final answer token by token."""
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ],
        options={"temperature": 0.2},
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]