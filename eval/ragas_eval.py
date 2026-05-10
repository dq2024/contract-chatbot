"""
RAGAS evaluation for RAG-mode questions.

Metrics (no ground truth required):
  faithfulness      — is the answer grounded in the retrieved context?
  answer_relevancy  — does the answer address the question?
  context_precision — were the retrieved chunks relevant?

Uses the same routing and retrieval pipeline as the live app so scores
reflect real production behavior.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import anthropic
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

# Reuse the same routing + retrieval logic as the app
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


def route(claude_client, question: str) -> dict:
    import re
    resp = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
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


def get_answer_and_contexts(question: str, claude_client, supabase, embedder) -> tuple[str, list[str]]:
    routing = route(claude_client, question)
    mode = routing.get("mode", "rag")
    sql = routing.get("sql")

    if mode == "sql":
        try:
            rows = run_sql(supabase, sql)
        except Exception as e:
            return f"Query failed: {e}", []
        contexts = [json.dumps(rows, default=str)]

    elif mode == "rag":
        chunks = run_rag(supabase, embedder, question)
        contexts = [f"[{c['source_filename']}]\n{c['chunk_text']}" for c in chunks]

    else:  # hybrid
        try:
            rows = run_sql(supabase, sql)
        except Exception as e:
            return f"SQL step failed: {e}", []
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
    return resp.content[0].text.strip(), contexts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).parent / "test_cases.json"))
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API") or os.environ.get("ANTHROPIC_API_KEY")
    claude_client = anthropic.Anthropic(api_key=api_key)
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with open(args.cases) as f:
        all_cases = json.load(f)

    rag_cases = [c for c in all_cases if c.get("type") == "ragas"]
    if not rag_cases:
        print("No RAGAS test cases found. Add cases with \"type\": \"ragas\" to test_cases.json.")
        return

    print(f"Running {len(rag_cases)} RAGAS cases through pipeline...\n")

    samples = []
    for case in rag_cases:
        question = case["question"]
        print(f"  {case['id']}: {question}")
        try:
            answer, contexts = get_answer_and_contexts(question, claude_client, supabase, embedder)
            samples.append(SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            ))
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nScoring {len(samples)} samples with RAGAS...\n")

    openai_key = os.environ.get("OPENAI_KEY")
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, max_tokens=2048))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_key))

    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings

    dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        run_config=RunConfig(timeout=300, max_retries=3, max_workers=1),
    )

    df = results.to_pandas()
    print("\nPer-question scores:")
    print(df[["user_input", "faithfulness", "answer_relevancy"]].to_string(index=False))

    print("\nMean scores:")
    for col in ["faithfulness", "answer_relevancy"]:
        print(f"  {col}: {df[col].mean():.3f}")


if __name__ == "__main__":
    main()
