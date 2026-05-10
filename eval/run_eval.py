"""
SQL eval for the Lake County contracts chatbot.

For each test case, routes the question, runs the generated SQL, and compares
returned source_filenames against a curated ground truth set.

Metrics per question:
  precision       = correct hits / total returned
  recall          = correct hits / total expected
  F1              = harmonic mean of precision and recall
  chunk_hit_rate  = fraction of expected filenames found in RAG-retrieved chunks
                    (hybrid cases only — checks retrieval layer independent of SQL)
"""

import json
import os
import sys
import argparse
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client

SCHEMA = """
Table: contracts (PostgreSQL)
  id, contract_instance_id, contract_id, vendor_name, vendor_name_normalized,
  contract_title, document_type (Agreement|SOW|Modification|Renewal|Award|Lease|Other),
  execution_date, effective_date, expiration_date,
  total_contract_value_usd (numeric, null for hourly contracts),
  hourly_rates (text, e.g. '$150/hr (Engineer)'), auto_renewal_flag (boolean),
  payment_terms, internal_department, source_filename
"""

ROUTER_SYSTEM = f"""You are a query router for a Lake County contracts database.

Given a user question, decide how to answer it and return JSON only — no other text.

Modes:
- "sql"    — answer can be derived entirely from schema columns (dates, values, vendors, counts, departments, flags like auto_renewal_flag, payment_terms, document_type, etc.). Prefer sql whenever a schema column can answer the question.
- "rag"    — question asks about contract content not captured in any schema column (specific clause language, obligations, liability terms, detailed conditions in the document text)
- "hybrid" — needs both: first find the right documents via SQL, then search their full text content

Return this JSON format:
{{
  "mode": "sql" | "rag" | "hybrid",
  "sql": "<SELECT query or null>",
  "reasoning": "<one sentence>"
}}

For sql and hybrid modes, write a valid PostgreSQL SELECT query against the contracts table. Make sure to include source_filename as well.
Strip trailing semicolons from SQL.
The sql field should be null for rag mode.

Schema:
{SCHEMA}
"""


def route(claude: anthropic.Anthropic, question: str) -> dict:
    import re
    resp = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=ROUTER_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = resp.content[0].text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        text = match.group(1).strip() if match else text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"mode": "rag", "sql": None, "reasoning": "Router fallback"}


def run_sql(supabase, sql: str) -> list[dict]:
    result = supabase.rpc("run_read_query", {"query_text": sql.rstrip(";")}).execute()
    return result.data or []


def run_rag(supabase, embedder: SentenceTransformer, question: str, match_count: int = 15) -> list[dict]:
    embedding = embedder.encode(question).tolist()
    result = supabase.rpc("match_chunks", {
        "query_embedding": embedding,
        "query_text": question,
        "match_count": match_count,
    }).execute()
    return result.data or []


def chunk_hit_rate(expected: set, chunks: list[dict]) -> float:
    retrieved_filenames = {c["source_filename"] for c in chunks if c.get("source_filename")}
    hits = len(expected & retrieved_filenames)
    return hits / len(expected) if expected else 1.0

load_dotenv(Path(__file__).parent.parent / ".env")


def precision_recall_f1(expected: set, returned: set) -> dict:
    if not expected and not returned:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "hits": 0}
    if not returned:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hits": 0}
    if not expected:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "hits": 0}

    hits = len(expected & returned)
    precision = hits / len(returned)
    recall = hits / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "hits": hits}


def run_case(case: dict, claude: anthropic.Anthropic, supabase, embedder: SentenceTransformer) -> dict:
    question = case["question"]
    result = {"id": case["id"], "question": question, "status": "ok"}

    routing = route(claude, question)
    result["mode"] = routing.get("mode")
    result["sql"] = routing.get("sql")

    mode = routing.get("mode")
    sql = routing.get("sql")

    if mode in ("sql", "hybrid"):
        if not sql:
            result["status"] = "error"
            result["reason"] = "no SQL generated"
            return result
        try:
            rows = run_sql(supabase, sql)
        except Exception as e:
            result["status"] = "error"
            result["reason"] = str(e)
            return result

        returned_filenames = {r["source_filename"] for r in rows if r.get("source_filename")}
        result["returned_filenames"] = sorted(returned_filenames)
        result["returned_count"] = len(rows)

        if "expected_filenames" in case:
            expected = set(case["expected_filenames"])
            result["expected_filenames"] = sorted(expected)
            result.update(precision_recall_f1(expected, returned_filenames))
        elif "expected_count" in case:
            result["expected_count"] = case["expected_count"]
            result["count_match"] = len(rows) == case["expected_count"]

    if "expected_filenames" in case:
        expected = set(case["expected_filenames"])
        if expected and mode in ("hybrid", "rag"):
            chunks = run_rag(supabase, embedder, question)
            result["chunk_hit_rate"] = chunk_hit_rate(expected, chunks)

    return result


def print_result(r: dict):
    status = r["status"]
    print(f"\n[{r['id']}] {r['question']}")

    if status == "skipped":
        print(f"  SKIPPED — {r['reason']}")
        return
    if status == "error":
        print(f"  ERROR — {r['reason']}")
        return

    print(f"  mode={r.get('mode')}")
    if "f1" in r:
        print(f"  precision={r['precision']:.2f}  recall={r['recall']:.2f}  f1={r['f1']:.2f}  hits={r['hits']}/{len(r['expected_filenames'])}")
        missing = set(r["expected_filenames"]) - set(r["returned_filenames"])
        extra   = set(r["returned_filenames"]) - set(r["expected_filenames"])
        if missing:
            print(f"  missing: {sorted(missing)}")
        if extra:
            print(f"  extra:   {sorted(extra)}")
    elif "count_match" in r:
        match = "PASS" if r["count_match"] else "FAIL"
        print(f"  count: expected={r['expected_count']} returned={r['returned_count']}  {match}")
    if "chunk_hit_rate" in r:
        print(f"  chunk_hit_rate={r['chunk_hit_rate']:.2f}  (expected filenames found in RAG top-15)")
    elif r.get("mode") in ("rag", "hybrid") and not r.get("expected_filenames"):
        print(f"  chunk_hit_rate=N/A  (no expected filenames to check)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).parent / "test_cases.json"))
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API") or os.environ.get("ANTHROPIC_API_KEY")
    claude   = anthropic.Anthropic(api_key=api_key)
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with open(args.cases) as f:
        all_cases = json.load(f)

    cases = [c for c in all_cases if c.get("type") == "retrieval"]
    print(f"Running {len(cases)} retrieval eval cases...\n")
    results = [run_case(c, claude, supabase, embedder) for c in cases]

    for r in results:
        print_result(r)

    scored = [r for r in results if "f1" in r]
    if scored:
        avg_f1 = sum(r["f1"] for r in scored) / len(scored)
        print(f"\nOverall F1:         {avg_f1:.2f}  ({len(scored)} cases scored)")

    chunk_scored = [r for r in results if "chunk_hit_rate" in r]
    if chunk_scored:
        avg_chr = sum(r["chunk_hit_rate"] for r in chunk_scored) / len(chunk_scored)
        print(f"Avg chunk hit rate: {avg_chr:.2f}  ({len(chunk_scored)} cases scored)")


if __name__ == "__main__":
    main()
