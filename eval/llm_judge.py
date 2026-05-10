"""
LLM-as-a-judge evaluation for the contracts chatbot.

Runs each judge-type test case through the full pipeline, then scores the
response on three dimensions using OpenAI gpt-5-mini as the judge:

  1. Factual correctness  (1-10) — are the stated facts accurate per ground truth?
  2. Completeness         (1-10) — does the answer cover all relevant contracts/items per ground truth?
  3. Citation quality     (1-10) — are specific document names or contract IDs cited?

Factual correctness and completeness are skipped when ground_truth is null.
Final score per question: average of scored dimensions (always out of 10).
Overall score: average across all questions.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

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

ANSWER_SYSTEM = """You are a helpful assistant summarizing Lake County government contract data.
Answer the user's question clearly and concisely using only the provided context.
Be specific — include vendor names, dollar amounts, and dates where relevant.
Format lists as markdown bullet points.
Always cite which document(s) the information comes from.
If the context doesn't contain enough information to answer, say so clearly."""

JUDGE_SYSTEM = """You are an evaluator scoring a chatbot response about government contracts.
Score the response on the requested dimension on a scale of 1 to 10 and return JSON only — no other text.
Be strict but fair."""


def route(claude_client, question: str) -> dict:
    resp = claude_client.messages.create(
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


def run_rag(supabase, embedder, question: str, filter_filenames=None) -> list[dict]:
    embedding = embedder.encode(question).tolist()
    params = {"query_embedding": embedding, "query_text": question, "match_count": 15}
    if filter_filenames:
        params["filter_filenames"] = filter_filenames
    result = supabase.rpc("match_chunks", params).execute()
    return result.data or []


def get_answer(question: str, claude_client, supabase, embedder) -> tuple[str, str, list[str]]:
    routing = route(claude_client, question)
    mode = routing.get("mode", "rag")
    sql = routing.get("sql")

    if mode == "sql":
        try:
            rows = run_sql(supabase, sql)
        except Exception as e:
            return f"Query failed: {e}", mode, []
        contexts = [json.dumps(rows, default=str)]

    elif mode == "rag":
        chunks = run_rag(supabase, embedder, question)
        contexts = [f"[{c['source_filename']}]\n{c['chunk_text']}" for c in chunks]

    else:  # hybrid
        try:
            rows = run_sql(supabase, sql)
        except Exception as e:
            return f"SQL step failed: {e}", mode, []
        filenames = [r["source_filename"] for r in rows if r.get("source_filename")]
        chunks = run_rag(supabase, embedder, question, filter_filenames=filenames or None)
        contexts = [json.dumps(rows, default=str)] + [
            f"[{c['source_filename']}]\n{c['chunk_text']}" for c in chunks
        ]

    context_str = "\n\n".join(contexts)
    resp = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\nContext:\n{context_str}"}],
    )
    return resp.content[0].text.strip(), mode, contexts


def judge_raw(openai_client, prompt: str) -> str:
    resp = openai_client.chat.completions.create(
        model="gpt-5.4",
        max_completion_tokens=256,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def parse_score(text: str) -> tuple[float, str]:
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        text = match.group(1).strip() if match else text
    try:
        data = json.loads(text)
        score = float(data.get("score", 1))
        reason = data.get("reason", "")
        score = max(1.0, min(10.0, score))
        return score, reason
    except (json.JSONDecodeError, ValueError):
        return 1.0, "parse error"


def score_factual(openai_client, question: str, answer: str, ground_truth: str) -> tuple[float, str]:
    prompt = f"""Question: {question}

Ground truth: {ground_truth}

Model answer: {answer}

Score factual correctness on a scale of 1-10: does the answer contain the key facts from the ground truth?
- 9-10: all key facts present and accurate
- 6-8: most facts present, minor gaps or imprecision
- 3-5: some facts present but notably incomplete or partially wrong
- 1-2: missing key facts or factually incorrect

Return JSON: {{"score": <1-10>, "reason": "<one sentence>"}}"""

    return parse_score(judge_raw(openai_client, prompt))


def score_citation(openai_client, question: str, answer: str) -> tuple[float, str]:
    prompt = f"""Question: {question}

Model answer: {answer}

Score citation quality on a scale of 1-10: does the answer cite specific documents, filenames, or contract IDs?
- 9-10: specific document names or contract IDs cited for all key claims
- 6-8: most claims cited, some vague or missing
- 3-5: vague references (e.g., "in the contract") without specifics
- 1-2: no citations at all
- Note: If the ground truth does not have a citation, and the model does not add any citations, that can still be a 10. Penalize for missing citations only when the ground truth includes them.

Return JSON: {{"score": <1-10>, "reason": "<one sentence>"}}"""

    return parse_score(judge_raw(openai_client, prompt))


def score_completeness(openai_client, question: str, answer: str, ground_truth: str) -> tuple[float, str]:
    prompt = f"""Question: {question}

Ground truth (expected coverage): {ground_truth}

Model answer: {answer}

Score completeness on a scale of 1-10: does the answer cover all the relevant contracts, vendors, or items that the ground truth indicates should be present?
- 9-10: all expected items are present in the answer
- 6-8: most items covered, a few missing
- 3-5: roughly half the expected items present
- 1-2: most expected items missing

Do not penalize for extra items beyond the ground truth — only penalize for missing ones.
Return JSON: {{"score": <1-10>, "reason": "<one sentence>"}}"""

    return parse_score(judge_raw(openai_client, prompt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).parent / "full_test_case.json"))
    parser.add_argument("--verbose", action="store_true", help="Print full answers")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API") or os.environ.get("ANTHROPIC_API_KEY")
    claude_client = anthropic.Anthropic(api_key=api_key)
    openai_client = OpenAI(api_key=os.environ["OPENAI_KEY"])
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with open(args.cases) as f:
        all_cases = json.load(f)

    judge_cases = [c for c in all_cases if c.get("type") == "judge" and not c.get("skip")]
    if not judge_cases:
        print("No judge test cases found.")
        return

    print(f"Running {len(judge_cases)} judge cases...\n")

    results = []
    case_averages = []

    for case in judge_cases:
        cid = case["id"]
        question = case["question"]
        expected_routing = case.get("expected_routing")
        ground_truth = case.get("ground_truth")

        print(f"[{cid}] {question}")

        try:
            answer, actual_mode, contexts = get_answer(question, claude_client, supabase, embedder)
        except Exception as e:
            print(f"  ERROR running pipeline: {e}\n")
            continue

        if args.verbose:
            print(f"  Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")

        # LLM-judged dimensions
        factual_score, factual_reason = None, "skipped — no ground truth"
        completeness_score, completeness_reason = None, "skipped — no ground truth"
        if ground_truth:
            factual_score, factual_reason = score_factual(openai_client, question, answer, ground_truth)
            completeness_score, completeness_reason = score_completeness(openai_client, question, answer, ground_truth)

        citation_score, citation_reason = score_citation(openai_client, question, answer)

        # Average across scored dimensions (skip nulls)
        scored = [citation_score]
        if factual_score is not None:
            scored.append(factual_score)
        if completeness_score is not None:
            scored.append(completeness_score)
        case_avg = sum(scored) / len(scored)
        case_averages.append(case_avg)

        if factual_score is not None:
            print(f"  factual:      {factual_score:.1f}/10  ({factual_reason})")
        else:
            print(f"  factual:      --    (no ground truth)")
        if completeness_score is not None:
            print(f"  completeness: {completeness_score:.1f}/10  ({completeness_reason})")
        else:
            print(f"  completeness: --    (no ground truth)")
        print(f"  citation:     {citation_score:.1f}/10  ({citation_reason})")
        print(f"  AVERAGE:      {case_avg:.1f}/10\n")

        results.append({
            "id": cid,
            "question": question,
            "actual_routing": actual_mode,
            "factual_score": factual_score,
            "completeness_score": completeness_score,
            "citation_score": citation_score,
            "average": round(case_avg, 2),
        })

    overall = sum(case_averages) / len(case_averages) if case_averages else 0.0
    print("=" * 60)
    print(f"OVERALL AVERAGE: {overall:.1f} / 10  ({len(case_averages)} cases)")

    out_path = Path(args.cases).parent / "judge_results.json"
    with open(out_path, "w") as f:
        json.dump({"overall_average": round(overall, 2), "cases": results}, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
