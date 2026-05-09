"""
One-shot: read test_extraction.csv and upsert all rows into Supabase contracts table.
"""

import csv
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "test_extraction.csv"


def coerce_row(row: dict) -> dict:
    def nullable(v):
        return v if v not in ("", "None", "none") else None

    return {
        "contract_instance_id":    row["contract_instance_id"],
        "contract_id":             nullable(row["contract_id"]),
        "vendor_name":             nullable(row["vendor_name"]),
        "vendor_name_normalized":  nullable(row["vendor_name_normalized"]),
        "contract_title":          nullable(row["contract_title"]),
        "document_type":           nullable(row["document_type"]),
        "execution_date":          nullable(row["execution_date"]),
        "effective_date":          nullable(row["effective_date"]),
        "expiration_date":         nullable(row["expiration_date"]),
        "total_contract_value_usd": float(row["total_contract_value_usd"]) if row["total_contract_value_usd"] not in ("", "None") else None,
        "auto_renewal_flag":       row["auto_renewal_flag"].lower() == "true",
        "payment_terms":           nullable(row["payment_terms"]),
        "internal_department":     nullable(row["internal_department"]),
        "source_filename":         nullable(row["source_filename"]),
    }


def main():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    supabase = create_client(url, key)

    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Uploading {len(rows)} rows to Supabase...")
    coerced = [coerce_row(r) for r in rows]
    result = supabase.table("contracts").upsert(coerced, on_conflict="source_filename").execute()
    print(f"Done. {len(result.data)} rows upserted.")


if __name__ == "__main__":
    main()
