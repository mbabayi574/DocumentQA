"""One-time builder for the eval corpus's Latin PDF (plan.md §9.1).

The other five corpus documents are committed as Markdown or plain text. A PDF needs a
font, and there is no Arabic-capable font on this machine — which is why the Persian PDF
in the eval corpus is a symlink to the committed ``tests/fixtures/docs/fa/ai-engineer.pdf``
rather than a file built here. That fixture already carries a real Persian text layer, and
plan.md §9.1 names it as the source for exactly this reason.

The generated PDF is committed, so the corpus is deterministic and needs no build step.
Re-run only if the text below changes:

    uv run python scripts/build_eval_corpus.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

TARGET = Path(__file__).resolve().parents[1] / "tests" / "eval" / "corpus" / "en" / "limits.pdf"

# Plain Latin prose on purpose: Base14 needs no embedded font, so the text layer extracts
# identically on any machine. Every number in here is a quota, distinct by construction
# from the runtime limits in the operator handbook.
PAGES: list[tuple[str, list[str]]] = [
    (
        "AURORA GATEWAY - SERVICE LIMITS AND QUOTAS",
        [
            "This sheet states the per-tenant quotas enforced by the Aurora Gateway. Runtime",
            "behaviour such as the listen port or the session ceiling is described in the",
            "operator handbook instead; the two documents never restate each other's numbers.",
        ],
    ),
    (
        "Throughput",
        [
            "Each tenant is entitled to a sustained throughput of 200 requests per second.",
            "Bursts are permitted up to 400 requests per second, after which the gateway",
            "throttles rather than rejects. A burst allowance lets a cold client catch up",
            "without letting one tenant consume the whole service.",
        ],
    ),
    (
        "Payloads",
        [
            "The largest request body a tenant may send is 4 MB. The largest response the",
            "gateway will produce for one request is 8 MB. Both ceilings are enforced before",
            "the request reaches the session pool, so an oversized upload costs a tenant",
            "nothing but the bytes it already sent.",
        ],
    ),
    (
        "Storage and retention",
        [
            "A tenant may store at most 5,000 documents. Documents are retained for 90 days",
            "from the moment they are ingested; after that they are purged and the quota is",
            "released. A document that is deleted by its owner is released immediately and",
            "does not wait for the retention window to close.",
        ],
    ),
    (
        "Enforcement",
        [
            "A request that exceeds its quota is answered with HTTP 429 and a Retry-After",
            "header naming the second at which the quota resets. Quotas reset at 00:00 UTC.",
            "There is no burst credit carried across a quota reset: a tenant that has used",
            "its full throughput in the final second of a minute starts the next minute with",
            "the full allowance and nothing more.",
        ],
    ),
]


def main() -> None:
    document = pymupdf.open()
    for heading, lines in PAGES:
        page = document.new_page(width=595, height=842)  # A4
        page.insert_text((64, 90), heading, fontsize=15, fontname="hebo")
        top = 130.0
        for line in lines:
            page.insert_text((64, top), line, fontsize=10.5, fontname="helv")
            top += 16
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_bytes(document.tobytes(garbage=4, deflate=True))
    document.close()
    print(f"wrote {TARGET} ({TARGET.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
