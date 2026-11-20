"""Generate synthetic multi-domain documents via LM Studio (qwen3.6-35b-a3b).

Writes NDJSON to --output with checkpoint/resume support.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from random import randint

import requests

DOC_CATEGORIES = [
    "invoice_receipt",
    "legal_contract",
    "technical_spec",
    "executive_memo",
    "financial_table_report",
    "academic_abstract_notes",
    "sop_policy_doc",
    "messy_meeting_notes",
]

CATEGORY_PROMPTS = {
    "invoice_receipt": (
        "Generate a realistic commercial invoice/receipt from a fictional company. "
        "Include: vendor name & address, client details, itemized line items with unit prices and quantities, "
        "subtotal, tax rate, total amount in USD, payment terms, invoice number, issue date, due date. "
        "Format as Markdown with at least one table of line items."
    ),
    "legal_contract": (
        "Generate a realistic business contract between two fictional companies. "
        "Include: parties, recitals, definitions, obligations, payment terms, termination clause, "
        "governing law, signature blocks. Use formal legal language and numbered sections."
    ),
    "technical_spec": (
        "Generate a technical specification document for a software system or hardware component. "
        "Include: overview, architecture description, API endpoints or interface specs, "
        "performance requirements, dependencies, version history table."
    ),
    "executive_memo": (
        "Generate an internal executive memo from a fictional CEO/CTO to their leadership team. "
        "Include: subject line, date, recipients, strategic context, key decisions, action items with owners and deadlines, "
        "and a brief financial or operational impact summary."
    ),
    "financial_table_report": (
        "Generate a quarterly financial performance report for a fictional company. "
        "Include: revenue breakdown by product line, operating expenses, EBITDA, net income, "
        "year-over-year comparisons in Markdown tables, and analyst commentary."
    ),
    "academic_abstract_notes": (
        "Generate academic research notes for a fictional paper in AI/ML or materials science. "
        "Include: title, abstract, introduction, methodology summary, results with a data table, "
        "discussion, limitations, references section."
    ),
    "sop_policy_doc": (
        "Generate a Standard Operating Procedure / company policy document. "
        "Include: purpose, scope, definitions, step-by-step procedures, responsibilities matrix, "
        "compliance requirements, revision history table."
    ),
    "messy_meeting_notes": (
        "Generate realistic messy meeting notes from a fictional cross-functional team standup. "
        "Include: date/time, attendees (partial names), agenda items with bullet points of discussion, "
        "action items assigned to people, open questions, and some informal commentary or side conversations."
    ),
}


def load_existing_ids(output_path):
    """Load existing doc_ids from a checkpoint file."""
    seen = set()
    p = Path(output_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    seen.add(record.get("doc_id"))
                except json.JSONDecodeError:
                    continue
    return seen


def generate_document(category):
    """Build a prompt for Qwen3.6-35B-A3B to generate one document."""
    doc_id = f"doc_{randint(0, (2**31) - 1)}"
    system_prompt = (
        "You are a professional document writer. Generate realistic, self-contained documents. "
        "Do NOT include meta-commentary. Output clean Markdown only."
    )
    user_prompt = (
        f"Generate a {category} document.\n\n"
        f"{CATEGORY_PROMPTS[category]}\n\n"
        "The document should be between 600 and 1800 words with realistic names, dates, tables, clauses, and numbers."
    )
    return doc_id, system_prompt, user_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic documents via LM Studio"
    )
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", type=str, default="data/synthetic_docs/docs.jsonl")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://192.168.56.1:1234/v1/chat/completions",
    )
    parser.add_argument("--model", type=str, default="qwen3.6-35b-a3b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    existing_ids = load_existing_ids(args.output)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    count = 0
    retries = 0
    with open(args.output, "a", encoding="utf-8") as fout:
        while count < args.count:
            cat = DOC_CATEGORIES[randint(0, len(DOC_CATEGORIES) - 1)]
            doc_id, sys_prompt, usr_prompt = generate_document(cat)

            if doc_id in existing_ids:
                continue

            body = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt},
                ],
                "temperature": 0.85,
                "max_tokens": 2400,
            }

            try:
                resp = requests.post(args.endpoint, json=body, timeout=args.timeout)
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                raw = msg.get("reasoning_content") or msg.get("content", "")
                content = raw.strip()
            except (KeyError, OSError) as exc:
                retries += 1
                if retries >= 2:
                    print(f"ERROR after 2 retries for {doc_id}: {exc}", file=sys.stderr)
                    retries = 0
                    continue
                time.sleep(2 ** min(retries, 3))
                continue

            retries = 0
            record = {
                "doc_id": doc_id,
                "category": cat,
                "content": content,
                "system_prompt": sys_prompt,
                "user_prompt": usr_prompt,
            }
            line = json.dumps(record, ensure_ascii=False)
            fout.write(line + "\n")
            fout.flush()
            count += 1
            print(f"[{count}/{args.count}] {doc_id}", flush=True)

    print(f"Done. Wrote {count} documents to {args.output}")


if __name__ == "__main__":
    main()
