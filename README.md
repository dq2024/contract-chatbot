# Lake County Contracts Chatbot

A generative AI pipeline that structures Lake County government contract documents into a searchable database and lets non-technical users query them in natural language.

Live app: [Streamlit deployment URL]
Database: Supabase (available for live inspection during walkthrough)

---

## Architecture

The pipeline has four stages: document curation, structured extraction, vector store, and a chat interface.

### Stage 1: Document Curation

Started with ~400 PDF contract files. `src/get_document_subset.py` groups documents by shared 5-digit contract ID prefix using regex on filenames (e.g., all files containing "22046" belong to the same contract family). Documents without a recognizable ID in the filename were excluded. In the future, using content-based ID extraction as a fallback could be a good experiment.

Sorted contract families by cumulative file size, smallest first, and selected 139 documents into `data/selected_contracts/`, outputting `data/selected_contracts.csv`. Prioritizing smaller groups first saves on API costs.

`src/extract_text.py` then uses `pymupdf4llm` to convert each selected PDF to markdown, preserving document structure (headers, tables) for the chunking stage. Outputs text files to `data/selected_contracts_text/`. This text is used for chunking the documents for the vector database described in Stage 3.

### Stage 2: Structured Extraction

`src/extract_contracts.py` sends each document to Claude Sonnet 4.6 with forced tool use to extract 12 fields. Two additional fields (`contract_instance_id`, `vendor_name_normalized`) are derived in Python from the extracted values:

| Field | Source | Description |
|---|---|---|
| contract_id | extracted | shared ID across all documents in a contract family, e.g., "22046" |
| vendor_name | extracted | vendor name as written in the document |
| contract_title | extracted | e.g., "Lake County Jail Inmate Food Service" |
| document_type | extracted | Agreement, SOW, Modification, Renewal, Award, Lease, Other |
| execution_date | extracted | date the contract was signed |
| effective_date | extracted | date the contract period begins |
| expiration_date | extracted | date the contract period ends |
| total_contract_value_usd | extracted | total dollar amount of the contract; null for open-ended unit-rate contracts |
| hourly_rates | extracted | rate schedule for T&M contracts, e.g., "$150/hr (Engineer)" |
| auto_renewal_flag | extracted | true if the contract renews automatically without action from either party |
| payment_terms | extracted | e.g., "Net 30 per IL Prompt Payment Act" |
| internal_department | extracted | Lake County department responsible for the contract, e.g., "Justice Division", "LCSO" |
| vendor_name_normalized | derived | lowercased vendor name with legal suffixes stripped, for deduplication and grouping |
| contract_instance_id | derived | stable key shared across a contract family: contract_id + first three tokens of normalized vendor name |

Extraction runs asynchronously with a concurrency limit of 2. Missing PDFs and extraction errors are logged and skipped without halting the pipeline.

A second-pass verification script (`src/verify_fields.py`) re-runs Claude on each document with more targeted prompts for the three fields most prone to error: `auto_renewal_flag`, `internal_department`, and the three date fields.

Known edge cases: some documents do not state a total contract value explicitly (left null); some have multiple hourly rates for different service tiers (stored as a single text field).

### Stage 3: Supabase Database

Two tables:

**`contracts`**: one row per document, all 14 fields plus `source_filename` as the upsert key. Exposed via a `run_read_query` RPC function that allows safe read-only SQL execution from the app.

**`contract_chunks`**: vector store for full-text retrieval. Each document is chunked and embedded using `all-MiniLM-L6-v2` (384-dim, runs locally). Exposed via a `match_chunks` RPC function that performs cosine similarity search with optional filename filtering.

Chunking strategy (`src/chunk_and_embed.py`): splits on `##` section headers first, since Lake County government contracts have consistent structure that maps naturally to semantic units. Sections under ~400 characters are merged with their neighbor. Sections over ~1,500 characters are split at sentence boundaries with ~150 character overlap. Documents with no headers fall back to fixed-size windows.

### Stage 4: Chat Interface

`app.py` is a Streamlit app with three-path retrieval and a sidebar of example questions for both structured queries and content searches:

**SQL mode**: question can be answered from schema columns (dates, values, vendor names, flags). Claude generates a SELECT query, runs it via the RPC function, and formats the result as markdown. The generated SQL is shown in a collapsible expander.

**RAG mode**: question asks about clause language or document content not captured in the schema. Embeds the question, retrieves top-15 chunks by cosine similarity, Claude answers from the chunks citing source documents.

**Hybrid mode**: needs both. SQL first identifies the relevant documents by filename, then RAG is run filtered to only those documents to retrieve the specific clause text.

The router is a separate Claude call that returns structured JSON with the mode, SQL query if applicable, and a one-sentence reasoning. If the router fails to return valid JSON, the app falls back to RAG mode. The last 6 messages of conversation history are passed to the answer call so follow-up questions work correctly.

**Error handling:** SQL execution errors are caught and surfaced to the user as a plain-text message rather than an exception. Empty results (no rows returned, no chunks retrieved) are passed as empty context; the answer system prompt instructs Claude to say explicitly when context is insufficient rather than hallucinate. Documents that fail during extraction are logged and skipped without stopping the pipeline.

---

## Evaluation

Three eval scripts in `eval/`:

**`run_eval.py`**: retrieval precision/recall/F1 against ground truth filenames for SQL and hybrid queries. Also computes chunk hit rate for RAG and hybrid cases: runs an unfiltered vector search and checks whether the expected documents appear in the top-15 results. If chunk hit rate is low while SQL F1 is high, the failure is in the embedding layer, not the query generation. Current scores: Overall F1 0.74 (6 cases scored), Avg chunk hit rate 0.79 (3 cases scored)

**`ragas_eval.py`**: RAGAS faithfulness and answer relevancy on 5 RAG test cases using GPT-4o-mini. Faithfulness measures whether the answer stays within the retrieved context; answer relevancy measures whether it addresses the question. Current scores: faithfulness 0.954, answer relevancy 0.871.

**`llm_judge.py`**: LLM-as-a-judge scoring on 9 test cases across three dimensions (1-10 scale): factual correctness against ground truth, completeness (did it cover all relevant contracts), and citation quality (did it name specific documents). Uses GPT-5.4 as judge.

---

## Known Limitations

**Embedding model.** `all-MiniLM-L6-v2` is fast and free but weak on legal language. Queries for specific clause types (e.g., "indemnification") can miss relevant documents because the embedding does not capture the semantic relationship between the question and dense legal text, even when "Consultant agrees to indemnify" would get a hit. A domain-adapted or larger model would meaningfully improve retrieval quality.

**Extraction scope.** The structured table was built from 139 documents. Fields like `auto_renewal_flag` required a second verification pass and may still have errors in ambiguous cases. Contracts with "option to renew" language are correctly classified as false; "automatically renews unless notice is given" is classified as true. Edge cases between those two remain the most common extraction error.

**No access control.** The app uses a service key with full database access. A production deployment would need row-level security, user authentication, and audit logging, especially for sensitive contract terms. Confidentiality for legal documents would also require encrypting fields at rest and restricting which users can query which contract families.

**Router brittleness.** Some questions sit on the boundary between SQL and hybrid (e.g., "How much are we paying Vendor X?" the total value is in the schema, but the payment structure may require reading the document). The router sometimes picks SQL when hybrid would give a better answer.

---

## Top Improvements

**Better embedding model.** Swap `all-MiniLM-L6-v2` for a larger or legal-domain model. This is the highest-leverage improvement. Retrieval quality directly caps answer quality, and the current model visibly struggles with clause-level queries. The chunk hit rate metric in the eval suite makes this easy to measure before and after.

**Expand document coverage and improve extraction reliability.** The current 139-document scope is a proof of concept. A production system would need robust handling of documents without an ID in the filename, better extraction of multi-rate pricing structures, and a confidence score per extracted field so low-confidence values can be flagged for human review before entering the database.

**Expand eval coverage.** The current eval suite covers a representative but limited set of queries. Adding more ground truth answers and more edge cases (ambiguous questions, cross-contract comparisons) would give a more complete and reliable signal on where the system succeeds and fails.

**Scheduled eval with alerting.** Run the eval suite on a daily/weekly cron job, store scores in a time series, and alert on drops below a threshold. This catches prompt drift when Anthropic updates Claude, and document format drift when the county starts uploading differently structured contracts.
