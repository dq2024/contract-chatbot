import json
import os
import re

import anthropic
import pandas as pd
import plotly.express as px
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
If the context doesn't contain enough information to answer, say so clearly.
For questions asking about total contract value, you should select the highest most recent number. Do not sum across older documents for the same vendor."""


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


@st.cache_data(ttl=300)
def load_all_contracts(_supabase):
    rows = _supabase.table("contracts").select("*").execute().data
    df = pd.DataFrame(rows)
    for col in ["execution_date", "effective_date", "expiration_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["total_contract_value_usd"] = pd.to_numeric(df["total_contract_value_usd"], errors="coerce")
    df["auto_renewal_flag"] = df["auto_renewal_flag"].astype(bool)
    return df


def route(claude: anthropic.Anthropic, question: str) -> dict:
    resp = claude.messages.create(
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
        return {"mode": "rag", "sql": None, "reasoning": "Router fallback — defaulting to RAG"}


def run_sql(supabase, sql: str) -> list[dict]:
    result = supabase.rpc("run_read_query", {"query_text": sql.rstrip(";")}).execute()
    return result.data or []


def run_rag(supabase, embedder, question: str, filter_filenames: list[str] | None = None) -> list[dict]:
    embedding = embedder.encode(question).tolist()
    params = {"query_embedding": embedding, "query_text": question, "match_count": 15}
    if filter_filenames:
        # In hybrid mode, restrict vector search to only the documents identified by SQL
        params["filter_filenames"] = filter_filenames
    result = supabase.rpc("match_chunks", params).execute()
    return result.data or []


def format_answer(claude: anthropic.Anthropic, question: str, context: str, history: list[dict]) -> str:
    # Include the last 6 messages for conversational context without blowing the context window
    messages = []
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"})
    resp = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=ANSWER_SYSTEM,
        messages=messages,
    )
    return resp.content[0].text.strip()


def handle_question(question: str, supabase, claude, embedder, history: list[dict] | None = None) -> tuple[str, str | None, str]:
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

    else:  # hybrid: SQL narrows to relevant documents, RAG searches their full text
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

    answer = format_answer(claude, question, context, history or [])
    return answer, sql, mode


def fmt_usd(val):
    if pd.isna(val):
        return "—"
    return f"${val:,.0f}"


def render_dashboard(supabase):
    df = load_all_contracts(supabase)

    if df.empty:
        st.info("No contract data available.")
        return

    today = pd.Timestamp.today().normalize()
    horizon = today + pd.DateOffset(months=18)

    # Top-level metrics
    total_docs = len(df)
    unique_contracts = df["contract_id"].replace("", pd.NA).dropna().nunique()
    known_value = df["total_contract_value_usd"].sum()
    auto_count = int(df["auto_renewal_flag"].sum())
    auto_value = df.loc[df["auto_renewal_flag"], "total_contract_value_usd"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", total_docs)
    c2.metric("Unique Contracts", unique_contracts)
    c3.metric("Total Value (known)", fmt_usd(known_value))
    c4.metric("Auto-Renewing", f"{auto_count}  ·  {fmt_usd(auto_value)}")
    st.caption("Values are summed across all documents per contract family and may include amendments.")

    st.divider()

    # Upcoming expirations 
    st.subheader("Contracts Expiring in the Next 18 Months")
    expiring = df[(df["expiration_date"] >= today) & (df["expiration_date"] <= horizon)].copy()

    if expiring.empty:
        st.info("No contracts expiring in the next 18 months.")
    else:
        expiring["Month"] = expiring["expiration_date"].dt.to_period("M").astype(str)
        expiring["Status"] = expiring["auto_renewal_flag"].map(
            {True: "Auto-Renews", False: "Needs Action"}
        )
        monthly = (
            expiring.groupby(["Month", "Status"])
            .size()
            .reset_index(name="Count")
            .sort_values("Month")
        )
        fig = px.bar(
            monthly, x="Month", y="Count", color="Status", barmode="stack",
            color_discrete_map={"Auto-Renews": "#66BB6A", "Needs Action": "#EF5350"},
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # For value, use max across the contract family since renewal docs often omit it
        family_max_value = df.groupby("contract_id")["total_contract_value_usd"].max()
        expiring["family_value"] = expiring["contract_id"].map(family_max_value)

        needs_action = expiring[~expiring["auto_renewal_flag"]][
            ["vendor_name", "contract_title", "document_type", "expiration_date", "family_value", "internal_department"]
        ].sort_values("expiration_date").copy()
        needs_action["expiration_date"] = needs_action["expiration_date"].dt.date
        needs_action["family_value"] = needs_action["family_value"].apply(fmt_usd)
        needs_action.columns = ["Vendor", "Title", "Type", "Expires", "Value", "Department"]
        st.caption(f"{len(needs_action)} of {len(expiring)} expiring contracts require active renewal")
        st.dataframe(needs_action, use_container_width=True, hide_index=True)

    st.divider()

    # Vendor spend  +  Department spend
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Top Vendors by Contract Value")
        # Deduplicate to one value per contract family before summing per vendor
        per_contract = df.groupby(["contract_id", "vendor_name"])["total_contract_value_usd"].max().reset_index()
        vendor_spend = (
            per_contract.groupby("vendor_name")["total_contract_value_usd"]
            .sum()
            .dropna()
            .nlargest(15)
            .sort_values()
            .reset_index()
        )
        vendor_spend.columns = ["Vendor", "Value"]
        fig = px.bar(
            vendor_spend, x="Value", y="Vendor", orientation="h",
            labels={"Value": "Total Value (USD)", "Vendor": ""},
        )
        fig.update_layout(yaxis=dict(tickfont=dict(size=10)), margin=dict(l=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Spend by Department")
        dept_spend = (
            df[df["internal_department"].fillna("").str.strip() != ""]
            .groupby("internal_department")["total_contract_value_usd"]
            .sum()
            .reset_index()
        )
        dept_spend.columns = ["Department", "Value"]
        dept_spend = dept_spend[dept_spend["Value"] > 0].sort_values("Value")
        fig = px.bar(
            dept_spend, x="Value", y="Department", orientation="h",
            labels={"Value": "Total Value (USD)", "Department": ""},
        )
        fig.update_layout(yaxis=dict(tickfont=dict(size=10)), margin=dict(l=10))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Document type  +  Value distribution
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Document Type Breakdown")
        doc_counts = df["document_type"].value_counts().reset_index()
        doc_counts.columns = ["Type", "Count"]
        fig = px.pie(doc_counts, names="Type", values="Count", hole=0.4)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Contract Value Distribution")
        valued = df["total_contract_value_usd"].dropna()
        fig = px.histogram(
            valued, x=valued, nbins=25,
            labels={"x": "Contract Value (USD)", "count": "Number of Documents"},
        )
        fig.update_layout(bargap=0.05, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Data completeness scorecard
    st.subheader("Extraction Completeness")
    fields = {
        "expiration_date": "Expiration Date",
        "effective_date": "Effective Date",
        "execution_date": "Execution Date",
        "total_contract_value_usd": "Contract Value",
        "internal_department": "Department",
        "payment_terms": "Payment Terms",
        "hourly_rates": "Hourly Rates",
    }
    completeness_rows = []
    for col, label in fields.items():
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            missing = df[col].isna()
        else:
            missing = df[col].isna() | (df[col].fillna("").astype(str).str.strip() == "")
        missing_n = int(missing.sum())
        pct = round((1 - missing_n / len(df)) * 100, 1)
        completeness_rows.append({"Field": label, "% Complete": pct, "Missing": missing_n})

    comp_df = pd.DataFrame(completeness_rows).sort_values("% Complete")
    fig = px.bar(
        comp_df, x="% Complete", y="Field", orientation="h",
        color="% Complete",
        color_continuous_scale=["#EF5350", "#FFA726", "#66BB6A"],
        range_color=[0, 100],
        text="% Complete",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_range=[0, 115],
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Hourly Rates and Payment Terms are intentionally null for many contracts. "
        "Low completeness on date fields indicates extraction gaps worth reviewing."
    )


def main():
    st.set_page_config(page_title="Lake County Contracts", page_icon="📄", layout="wide")
    st.title("Lake County Contracts")

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

    tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])

    with tab_chat:
        st.caption("Ask questions about vendors, values, dates, departments, or contract content.")

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
                    answer, sql, mode = handle_question(question, supabase, claude, embedder, st.session_state.messages)
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

    with tab_dashboard:
        render_dashboard(supabase)


if __name__ == "__main__":
    main()
