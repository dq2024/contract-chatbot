"""
Streamlit chat UI for Lake County contracts.

Routing logic:
  SQL    — structured questions answered via text-to-SQL
  RAG    — content questions answered via vector search over chunks
  HYBRID — SQL to find relevant documents, then RAG filtered to those docs
"""

import json
import os
import anthropic
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from supabase import create_client

load_dotenv(Path(__file__).parent / ".env")

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
- "sql"    — question asks about structured fields (dates, values, vendors, counts, departments)
- "rag"    — question asks about contract content (clauses, terms, obligations, details in the text)
- "hybrid" — needs both: first find the right documents via SQL, then search their content

Return this JSON format:
{{
  "mode": "sql" | "rag" | "hybrid",
  "sql": "<SELECT query or null>",
  "reasoning": "<one sentence>"
}}

For sql and hybrid modes, write a valid PostgreSQL SELECT query against the contracts table.
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


@st.cache_resource
def get_clients():
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    claude = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API") or os.environ.get("ANTHROPIC_API_KEY")
    )
    return supabase, claude


@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def route(claude: anthropic.Anthropic, question: str) -> dict:
    resp = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=ROUTER_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    text = resp.content[0].text.strip()
    # Strip markdown fences if present
    if "```" in text:
        import re
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        text = match.group(1).strip() if match else text
    return json.loads(text)


def run_sql(supabase, sql: str) -> list[dict]:
    result = supabase.rpc("run_read_query", {"query_text": sql.rstrip(";")}).execute()
    return result.data or []


def run_rag(supabase, embedder, question: str, filter_filenames: list[str] | None = None) -> list[dict]:
    embedding = embedder.encode(question).tolist()
    params = {"query_embedding": embedding, "query_text": question, "match_count": 15}
    if filter_filenames:
        params["filter_filenames"] = filter_filenames
    result = supabase.rpc("match_chunks", params).execute()
    return result.data or []


def format_answer(claude: anthropic.Anthropic, question: str, context: str) -> str:
    resp = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}],
    )
    return resp.content[0].text.strip()


def handle_question(question: str, supabase, claude, embedder) -> tuple[str, str | None, str]:
    routing = route(claude, question)
    mode = routing.get("mode", "rag")
    sql = routing.get("sql")

    if mode == "sql":
        try:
            rows = run_sql(supabase, sql)
        except Exception as e:
            return f"Query failed: {e}", sql, mode
        context = f"SQL results:\n{json.dumps(rows, default=str)}"

    elif mode == "rag":
        chunks = run_rag(supabase, embedder, question)
        context = "\n\n".join(
            f"[{c['source_filename']}]\n{c['chunk_text']}" for c in chunks
        )

    else:  # hybrid
        try:
            rows = run_sql(supabase, sql)
        except Exception as e:
            return f"SQL step failed: {e}", sql, mode
        filenames = [r["source_filename"] for r in rows if r.get("source_filename")]
        chunks = run_rag(supabase, embedder, question, filter_filenames=filenames or None)
        sql_summary = f"SQL results:\n{json.dumps(rows, default=str)}"
        rag_context = "\n\n".join(
            f"[{c['source_filename']}]\n{c['chunk_text']}" for c in chunks
        )
        context = f"{sql_summary}\n\nRelevant document excerpts:\n{rag_context}"

    answer = format_answer(claude, question, context)
    return answer, sql, mode


def main():
    st.set_page_config(page_title="Lake County Contracts", page_icon="📄", layout="wide")
    st.title("Lake County Contracts")
    st.caption("Ask questions about vendors, values, dates, departments, or contract content.")

    supabase, claude = get_clients()
    embedder = get_embedder()

    with st.sidebar:
        st.markdown("### Example questions")
        st.markdown("**Structured**")
        sql_examples = [
            "Which contracts have the highest total value?",
            "Which department has the most contracts?",
            "List all contracts expiring in 2026.",
            "Which contracts have hourly rates?",
            "Show all modifications to contract 22046.",
        ]
        for ex in sql_examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state["prefill"] = ex

        st.markdown("**Content**")
        rag_examples = [
            "Do any contracts contain indemnification clauses?",
            "What are the payment terms in the Motorola leases?",
            "What does contract 23159 say about job order contracting?",
            "Which contracts have termination for convenience clauses?",
            "What are the insurance requirements in the Tyler Technologies agreement?",
        ]
        for ex in rag_examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state["prefill"] = ex

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander(f"View SQL ({msg.get('mode', '')})"):
                    st.code(msg["sql"], language="sql")

    prefill = st.session_state.pop("prefill", None)
    question = st.chat_input("Ask about contracts...") or prefill

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sql, mode = handle_question(question, supabase, claude, embedder)
            st.markdown(answer)
            if sql:
                with st.expander(f"SQL · {mode.upper()}"):
                    st.code(sql, language="sql")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sql": sql,
            "mode": mode,
        })

        if prefill:
            st.rerun()


if __name__ == "__main__":
    main()
