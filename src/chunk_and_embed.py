"""
Chunks contract markdown text and stores embeddings in Supabase pgvector.

Chunking strategy:
  - Split on ## headers for semantic chunks
  - Accumulate short sections (< MIN_CHARS) into the next section
  - Split oversized sections (> MAX_CHARS) at sentence boundaries with overlap
  - Strip OCR image artifacts before chunking
  - Fall back to fixed-size splitting for docs with no headers

Run from project root:
    python src/chunk_and_embed.py
"""

import re
import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.parent
TEXT_DIR = BASE_DIR / "selected_contracts_text"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
MIN_CHARS  = 400   # sections smaller than this are merged with the next
MAX_CHARS  = 1500  # sections larger than this are split at sentence boundaries
OVERLAP    = 150   # character overlap when splitting large sections


def clean_text(text: str) -> str:
    text = re.sub(r'\*\*==> picture \[.*?\] intentionally omitted <==\*\*', '', text)
    text = re.sub(r'<br>\s*', '\n', text)
    text = re.sub(r'\*\*----- Start of picture text -----\*\*\s*', '', text)
    text = re.sub(r'\s*\*\*----- End of picture text -----\*\*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_at_sentences(text: str) -> list[str]:
    """Split large text into MAX_CHARS chunks at sentence boundaries with overlap."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > MAX_CHARS and current:
            chunks.append(' '.join(current))
            # carry back enough sentences for overlap
            overlap, overlap_len = [], 0
            for s in reversed(current):
                if overlap_len + len(s) > OVERLAP:
                    break
                overlap.insert(0, s)
                overlap_len += len(s)
            current = overlap + [sent]
            current_len = overlap_len + len(sent)
        else:
            current.append(sent)
            current_len += len(sent)

    if current:
        chunks.append(' '.join(current))

    return chunks or [text]


def chunk_document(text: str) -> list[dict]:
    text = clean_text(text)

    raw_sections = re.split(r'\n(?=## )', text)

    # No headers — fixed-size fallback
    if len(raw_sections) <= 1:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append({'text': text[start:start + MAX_CHARS].strip(), 'section_title': ''})
            start += MAX_CHARS - OVERLAP
        return [c for c in chunks if c['text']]

    # Parse title + body from each section
    sections = []
    for part in raw_sections:
        part = part.strip()
        if not part:
            continue
        lines = part.split('\n', 1)
        title = lines[0].lstrip('#').strip()
        body  = lines[1].strip() if len(lines) > 1 else ''
        sections.append((title, body or part))

    # Accumulate sections into chunks, merging short ones and splitting large ones
    chunks = []
    pending_text  = ''
    pending_title = ''

    for title, body in sections:
        candidate = (pending_text + '\n\n' + body).strip() if pending_text else body

        if len(candidate) <= MAX_CHARS:
            pending_text  = candidate
            pending_title = pending_title or title
        else:
            # Flush pending buffer
            if pending_text:
                for sub in (split_at_sentences(pending_text) if len(pending_text) > MAX_CHARS else [pending_text]):
                    chunks.append({'text': sub, 'section_title': pending_title})
            pending_text  = body
            pending_title = title

    # Flush remainder
    if pending_text:
        for sub in (split_at_sentences(pending_text) if len(pending_text) > MAX_CHARS else [pending_text]):
            chunks.append({'text': sub, 'section_title': pending_title})

    return [c for c in chunks if c['text'].strip()]


def main():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    supabase = create_client(url, key)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Fetching contracts from Supabase...")
    rows = supabase.table("contracts").select("source_filename, contract_instance_id").execute().data
    print(f"  {len(rows)} contracts found\n")

    total_chunks = 0
    for i, row in enumerate(rows, 1):
        pdf_filename = row["source_filename"]
        txt_path = TEXT_DIR / (Path(pdf_filename).stem + ".txt")

        if not txt_path.exists():
            print(f"  [{i:>3}/{len(rows)}] MISSING {txt_path.name}")
            continue

        text   = txt_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_document(text)

        embeddings = model.encode(
            [c['text'] for c in chunks], batch_size=BATCH_SIZE, show_progress_bar=False
        )

        records = [
            {
                "source_filename":      pdf_filename,
                "contract_instance_id": row["contract_instance_id"],
                "chunk_index":          idx,
                "section_title":        chunk['section_title'],
                "chunk_text":           chunk['text'],
                "embedding":            embedding.tolist(),
            }
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        supabase.table("contract_chunks").upsert(
            records, on_conflict="source_filename,chunk_index"
        ).execute()

        total_chunks += len(records)
        print(f"  [{i:>3}/{len(rows)}] {pdf_filename} → {len(records)} chunks")

    print(f"\nDone. {total_chunks} total chunks uploaded.")


if __name__ == "__main__":
    main()
