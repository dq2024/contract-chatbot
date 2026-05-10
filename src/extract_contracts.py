"""
Reads data/selected_contracts.csv, extracts structured fields from each PDF via Claude,
and writes to Supabase (TEST_MODE=False) or a local CSV (TEST_MODE=True).

Run from project root:
    python src/extract_contracts.py
"""

import asyncio
import base64
import csv
import os
import re
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR     = Path(__file__).parent.parent
MANIFEST_CSV = BASE_DIR / "data" / "selected_contracts.csv"
PDF_DIR      = BASE_DIR / "data" / "selected_contracts"

MODEL          = "claude-sonnet-4-6"
MAX_CONCURRENT = 2
MAX_RETRIES    = 10
BASE_BACKOFF   = 2.0

TEST_MODE       = False
TEST_LIMIT      = 3
TEST_OUTPUT_CSV = BASE_DIR / "SQL_DB.csv"

EXTRACT_TOOL = {
    "name": "extract_contract_fields",
    "description": "Extract structured fields from a contract document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "contract_id": {
                "type": "string",
                "description": "Contract number, e.g. '16069', '22046'. Empty string if not found.",
            },
            "vendor_name": {
                "type": "string",
                "description": "Full legal name of the vendor or counterparty.",
            },
            "contract_title": {
                "type": "string",
                "description": "Description of the contract scope, e.g. 'Lake County Jail Inmate Food Service'.",
            },
            "document_type": {
                "type": "string",
                "enum": ["Agreement", "SOW", "Modification", "Renewal", "Award", "Lease", "Other"],
                "description": "Type of document.",
            },
            "execution_date": {
                "type": "string",
                "description": "Date the document was signed, ISO format YYYY-MM-DD. Empty string if not found.",
            },
            "effective_date": {
                "type": "string",
                "description": "Contract period start date, ISO format YYYY-MM-DD. Empty string if not found.",
            },
            "expiration_date": {
                "type": "string",
                "description": "Contract period end date, ISO format YYYY-MM-DD. Empty string if not found.",
            },
            "total_contract_value_usd": {
                "type": "string",
                "description": "Total amount of the contract in USD as a plain number string, e.g. '1000000'. Empty string if not found.",
            },
            "hourly_rates": {
                "type": "string",
                "description": "Hourly rates if billed hourly, e.g. '$150/hr (Lead Engineer), $95/hr (Technician)'. List all rates found. Empty string if none.",
            },
            "auto_renewal_flag": {
                "type": "boolean",
                "description": "True if the contract contains an auto-renewal clause.",
            },
            "payment_terms": {
                "type": "string",
                "description": "Payment terms, e.g. 'Net 30 per IL Prompt Payment Act'. Empty string if not stated.",
            },
            "internal_department": {
                "type": "string",
                "description": "Lake County department or division, e.g. 'Justice Division', 'LCHD'. Empty string if not stated.",
            },
        },
        "required": [
            "contract_id", "vendor_name", "contract_title", "document_type",
            "execution_date", "effective_date", "expiration_date",
            "total_contract_value_usd", "hourly_rates", "auto_renewal_flag",
            "payment_terms", "internal_department",
        ],
    },
}


def normalize_vendor(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"\b(inc|llc|ltd|corp|co|pc|lp|llp|pllc)\.?$", "", n)
    n = re.sub(r"[^\w\s]", "", n)
    return re.sub(r"\s+", " ", n).strip()


def make_instance_id(contract_id: str, vendor_name: str) -> str:
    # Stable ID shared across all documents in a contract family (base, amendments, renewals)
    normalized = normalize_vendor(vendor_name)
    slug = "_".join(normalized.split()[:3])
    return f"{contract_id}-{slug}" if contract_id else slug


def parse_float(value: str) -> float | None:
    try:
        return float(re.sub(r"[,$]", "", value.strip()))
    except (ValueError, AttributeError):
        return None


def parse_date(value: str) -> str | None:
    if not value or not value.strip():
        return None
    v = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return v


def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")
    return create_client(url, key)


def upsert_row(supabase: Client, row: dict):
    supabase.table("contracts").upsert(row, on_conflict="source_filename").execute()


async def call_llm(client: anthropic.AsyncAnthropic, pdf_bytes: bytes) -> dict:
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    for attempt in range(MAX_RETRIES):
        try:
            # tool_choice forces Claude to always call the tool, so response.content[0] is always a ToolUseBlock
            response = await client.messages.create(
                model=MODEL,
                max_tokens=1024,
                tools=[EXTRACT_TOOL],
                tool_choice={"type": "tool", "name": "extract_contract_fields"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Extract the requested fields from this contract document. Return an empty string for any field not present.",
                            },
                        ],
                    }
                ],
            )
            return response.content[0].input
        except anthropic.RateLimitError:
            wait = min(BASE_BACKOFF * (2 ** attempt), 60)
            print(f"    Rate limited. Waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries.")


def build_row(fields: dict, manifest_row: dict, filename: str) -> dict:
    contract_id = fields.get("contract_id", "").strip() or manifest_row.get("contract_number", "")
    vendor_name = fields.get("vendor_name", "").strip()
    return {
        "contract_instance_id":    make_instance_id(contract_id, vendor_name),
        "contract_id":             contract_id,
        "vendor_name":             vendor_name,
        "vendor_name_normalized":  normalize_vendor(vendor_name),
        "contract_title":          fields.get("contract_title", "").strip(),
        "document_type":           fields.get("document_type", "Other"),
        "execution_date":          parse_date(fields.get("execution_date", "")),
        "effective_date":          parse_date(fields.get("effective_date", "")),
        "expiration_date":         parse_date(fields.get("expiration_date", "")),
        "total_contract_value_usd": parse_float(fields.get("total_contract_value_usd", "")),
        "hourly_rates":            fields.get("hourly_rates", "").strip(),
        "auto_renewal_flag":       int(fields.get("auto_renewal_flag", False)),
        "payment_terms":           fields.get("payment_terms", "").strip(),
        "internal_department":     fields.get("internal_department", "").strip(),
        "source_filename":         filename,
    }


async def process_document(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    supabase: Client,
    manifest_row: dict,
    lock: asyncio.Lock,
) -> bool:
    filename = manifest_row["filename"]
    pdf_path = PDF_DIR / filename

    if not pdf_path.exists():
        print(f"  MISSING: {filename}")
        return False

    pdf_bytes = pdf_path.read_bytes()

    async with semaphore:
        try:
            fields = await call_llm(client, pdf_bytes)
        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            return False

    # Lock prevents concurrent writes to Supabase from racing on the same row
    async with lock:
        upsert_row(supabase, build_row(fields, manifest_row, filename))

    return True


async def process_document_test(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    manifest_row: dict,
) -> dict | None:
    filename = manifest_row["filename"]
    pdf_path = PDF_DIR / filename

    if not pdf_path.exists():
        print(f"  MISSING: {filename}")
        return None

    pdf_bytes = pdf_path.read_bytes()

    async with semaphore:
        try:
            fields = await call_llm(client, pdf_bytes)
        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            return None

    return build_row(fields, manifest_row, filename)


async def run(manifest: list[dict], supabase: Client):
    api_key = os.environ.get("ANTHROPIC_API") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API or ANTHROPIC_API_KEY not set.")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    lock = asyncio.Lock()

    tasks = [process_document(client, semaphore, supabase, row, lock) for row in manifest]
    total = len(tasks)
    success = 0

    # as_completed lets us print progress as each task finishes rather than waiting for all
    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        ok = await coro
        if ok:
            success += 1
        print(f"  [{i:>3}/{total}] {'OK' if ok else 'FAIL'} — {success} extracted so far")

    print(f"\nDone: {success}/{total} documents extracted → Supabase")


async def run_test(manifest: list[dict]) -> list[dict]:
    api_key = os.environ.get("ANTHROPIC_API") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API or ANTHROPIC_API_KEY not set.")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [process_document_test(client, semaphore, row) for row in manifest]
    total = len(tasks)
    results = []

    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        if result is None:
            print(f"  [{i:>3}/{total}] FAIL")
        else:
            results.append(result)
            print(f"  [{i:>3}/{total}] OK — {result['source_filename']}")

    print(f"\nExtracted: {len(results)}, Failed: {total - len(results)}")
    return results


def main():
    with MANIFEST_CSV.open(encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    if TEST_MODE:
        manifest = manifest[:TEST_LIMIT]
        print(f"TEST MODE: processing {len(manifest)} documents\n")
        results = asyncio.run(run_test(manifest))
        if results:
            with TEST_OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(f"Saved {len(results)} rows → {TEST_OUTPUT_CSV}")
    else:
        print(f"Extracting {len(manifest)} documents → Supabase\n")
        supabase = get_supabase_client()
        asyncio.run(run(manifest, supabase))


if __name__ == "__main__":
    main()
