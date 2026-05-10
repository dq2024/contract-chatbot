"""
Re-runs Claude on each contract PDF to verify and correct auto_renewal_flag,
internal_department, and null date fields. Updates Supabase via source_filename.

Run from project root:
    python src/verify_fields.py [--dry-run] [--files FILENAME ...]
"""

import argparse
import asyncio
import base64
import os
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
PDF_DIR  = BASE_DIR / "data" / "selected_contracts"

MODEL          = "claude-sonnet-4-6"
MAX_CONCURRENT = 2
MAX_RETRIES    = 5
BASE_BACKOFF   = 2.0

VERIFY_TOOL = {
    "name": "verify_contract_fields",
    "description": "Re-extract specific fields from a contract document for quality assurance.",
    "input_schema": {
        "type": "object",
        "properties": {
            "auto_renewal_flag": {
                "type": "boolean",
                "description": (
                    "True if the contract renews automatically at the end of its term unless a party "
                    "sends notice to stop it. False if a party must take affirmative action to renew. "
                    "When in doubt, default to false.\n\n"
                    "Set false for: 'option to renew', 'right to renew', 'may renew', "
                    "'renewal options', 'upon written notice of exercising this option', "
                    "'right to extend'. Also false for award letters that list renewal periods "
                    "(e.g. 'two years with three one-year renewals') and for any language describing "
                    "pricing during a renewal without stating the renewal is automatic.\n\n"
                    "Set true for: 'shall automatically renew', 'will automatically renew', "
                    "'automatically extends', 'continues unless written notice of termination is given'."
                ),
            },
            "internal_department": {
                "type": "string",
                "description": (
                    "The Lake County department or division that will use the services. "
                    "Ignore procurement, purchasing, and finance — they sign most contracts and are not the answer. "
                    "Look first in the recitals, then the scope of services, then the agreement title, "
                    "then the project manager's title in the notices section. "
                    "Return empty string if not determinable."
                ),
            },
            "execution_date": {
                "type": "string",
                "description": "Date the document was signed by all parties, ISO format YYYY-MM-DD. Empty string if not found.",
            },
            "effective_date": {
                "type": "string",
                "description": "Contract period start date, ISO format YYYY-MM-DD. Empty string if not found.",
            },
            "expiration_date": {
                "type": "string",
                "description": "Contract period end date, ISO format YYYY-MM-DD. Empty string if not found.",
            },
        },
        "required": [
            "auto_renewal_flag", "internal_department",
            "execution_date", "effective_date", "expiration_date",
        ],
    },
}


async def call_llm(client: anthropic.AsyncAnthropic, pdf_bytes: bytes) -> dict:
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=512,
                tools=[VERIFY_TOOL],
                tool_choice={"type": "tool", "name": "verify_contract_fields"},
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64},
                        },
                        {
                            "type": "text",
                            "text": "Re-extract the specified fields from this contract for quality assurance. Apply the definitions exactly as described.",
                        },
                    ],
                }],
            )
            return response.content[0].input
        except anthropic.RateLimitError:
            wait = min(BASE_BACKOFF * (2 ** attempt), 60)
            print(f"    Rate limited. Waiting {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries.")


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


def build_updates(new_fields: dict, existing: dict) -> dict:
    updates = {}

    # Always re-evaluate auto_renewal_flag — first-pass extraction got this wrong most often
    new_flag = bool(new_fields.get("auto_renewal_flag", False))
    if new_flag != bool(existing.get("auto_renewal_flag")):
        updates["auto_renewal_flag"] = new_flag

    # Only overwrite department if Claude returned something non-empty and different
    new_dept = new_fields.get("internal_department", "").strip()
    if new_dept and new_dept != (existing.get("internal_department") or ""):
        updates["internal_department"] = new_dept

    # Only fill in dates that are missing — don't overwrite values already in Supabase
    for field in ("execution_date", "effective_date", "expiration_date"):
        if not existing.get(field):
            parsed = parse_date(new_fields.get(field, ""))
            if parsed:
                updates[field] = parsed

    return updates


async def verify_row(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    existing: dict,
    dry_run: bool,
    supabase: Client,
) -> None:
    filename = existing["source_filename"]
    pdf_path = PDF_DIR / filename

    if not pdf_path.exists():
        print(f"  MISSING  {filename}")
        return

    async with semaphore:
        try:
            new_fields = await call_llm(client, pdf_path.read_bytes())
        except Exception as e:
            print(f"  ERROR    {filename}: {e}")
            return

    updates = build_updates(new_fields, existing)

    if not updates:
        print(f"  OK       {filename}")
        return

    print(f"  DIFF     {filename}")
    for field, new_val in updates.items():
        print(f"           {field}: {existing.get(field)!r} -> {new_val!r}")

    if not dry_run:
        supabase.table("contracts").update(updates).eq("source_filename", filename).execute()
        print(f"           updated")


async def run(rows: list[dict], dry_run: bool, supabase: Client) -> None:
    api_key = os.environ.get("ANTHROPIC_API") or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    await asyncio.gather(*[
        verify_row(client, semaphore, row, dry_run, supabase)
        for row in rows
    ])


def main():
    parser = argparse.ArgumentParser(description="Verify and correct specific contract fields.")
    parser.add_argument("--dry-run", action="store_true", help="Print diffs without updating Supabase")
    parser.add_argument("--files", nargs="+", metavar="FILENAME", help="Only verify these filenames")
    args = parser.parse_args()

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    rows = supabase.table("contracts").select(
        "source_filename, auto_renewal_flag, internal_department, "
        "execution_date, effective_date, expiration_date"
    ).execute().data

    if args.files:
        rows = [r for r in rows if r["source_filename"] in args.files]

    label = "DRY RUN  " if args.dry_run else ""
    print(f"{label}Verifying {len(rows)} contracts...\n")
    asyncio.run(run(rows, args.dry_run, supabase))
    print("\nDone.")


if __name__ == "__main__":
    main()
