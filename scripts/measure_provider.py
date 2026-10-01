"""Measure the real embedding provider, for tuning decisions (plan.md §2.1, P9).

Run with ``RUN_LIVE=1 uv run python scripts/measure_provider.py``. Every number printed
here is quoted in ``docs/DECISIONS.md``; re-run it before trusting any of them.

The point is to replace assumptions with measurements:

* batch size and item length vs latency, so ``MAX_ITEMS_PER_BATCH`` and
  ``CHARS_PER_ITEM`` are chosen from data rather than round numbers;
* the true single-item ceiling, so ``MAX_CHARS_PER_ITEM`` has a known margin;
* ingest throughput, so the rate-limit setting is defensible;
* a retrieval spot-check on the real fixtures with the *current* pipeline, so the
  dense-retrieval premise is verified rather than inherited from an earlier phase.

Pacing is done here rather than by the client's limiter, so the reported latencies are
network time and not queue time. The provider allows 120 requests/minute (C5).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qasystem.chunking.chunker import CHARS_PER_TOKEN, Chunker
from qasystem.config import load_settings
from qasystem.domain.models import Chunk
from qasystem.domain.ports import Embedder
from qasystem.embeddings.client import EmbeddingClient
from qasystem.embeddings.rate_limit import RateLimiter
from qasystem.parsing.registry import ParserRegistry
from qasystem.storage.lexical import LexicalIndex
from qasystem.storage.sqlite_store import SqliteStore

PROBES_PER_MIN = 100  # stay under C5's 120 with room for the calls below
PROVIDER_CHAR_CEILING = 200_000  # §2.1: 200 000 chars is the provider's per-request cap
GAP = 60.0 / PROBES_PER_MIN
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "docs"


async def paced(embed: Embedder, texts: Sequence[str]) -> tuple[list[list[float]], float]:
    """One embed call, with the quota gap applied *before* it so timing is pure network."""
    await asyncio.sleep(GAP)
    started = time.perf_counter()
    vectors = await embed.embed(list(texts))
    return vectors, time.perf_counter() - started


def l2(vector: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


async def sweep_batch_size(
    embed: Embedder, item_chars: int, model_id: str
) -> list[tuple[int, float, float]]:
    """Latency and per-char cost as the item count in one request grows."""
    print(f"\n== batch size sweep ({item_chars}-char items, {model_id}) ==")
    print(f"{'items':>6} {'chars':>8} {'median s':>9} {'ms/1k chars':>12} {'chunks/s':>10}")
    rows: list[tuple[int, float, float]] = []
    for count in (1, 4, 8, 16, 32, 64, 128):
        batch = [f"item {i} " + "word " * (item_chars // 5) for i in range(count)]
        samples = [(await paced(embed, batch))[1] for _ in range(3)]
        median = statistics.median(samples)
        chars = sum(len(t) for t in batch)
        rows.append((count, median, chars))
        print(
            f"{count:>6} {chars:>8} {median:>9.3f} "
            f"{median / chars * 1000 * 1000:>12.2f} {count / median:>10.1f}"
        )
    return rows


async def sweep_item_length(embed: Embedder, items: int, model_id: str) -> list[tuple[int, float]]:
    print(f"\n== item length sweep ({items} items per request, {model_id}) ==")
    print(f"{'chars/item':>11} {'median s':>9} {'chunks/s':>10}")
    rows: list[tuple[int, float]] = []
    for item_chars in (200, 500, 1000, 2000, 4000, 8000):
        batch = [f"item {i} " + "word " * (item_chars // 5) for i in range(items)]
        median = statistics.median([(await paced(embed, batch))[1] for _ in range(3)])
        rows.append((item_chars, median))
        print(f"{item_chars:>11} {median:>9.3f} {items / median:>10.1f}")
    return rows


async def find_item_ceiling(model_id: str, dimension: int) -> tuple[int, str]:
    """Binary-search the largest single item the *provider* accepts.

    §2.1 measured 40 000 chars accepted and 45 000 rejected. The exact boundary matters
    because ``MAX_CHARS_PER_ITEM`` is the only thing between a chunker and a hard
    failure, so its margin should be known rather than assumed.

    The probe gets its own client with the per-item cap lifted: measuring the ceiling
    *through* the guard would just measure the guard (which is what the first run of this
    script did, and it reported a "ceiling" of exactly 20 000).
    """
    print(f"\n== single-item ceiling, provider limit ({model_id}) ==")
    probe = EmbeddingClient(
        load_settings(
            max_chars_per_item=PROVIDER_CHAR_CEILING,
            max_chars_per_request=PROVIDER_CHAR_CEILING,
        ),
        limiter=RateLimiter(10**6),
    )
    await probe.start()
    assert probe.model_id == model_id and probe.dimension == dimension

    async def accepted(chars: int) -> tuple[bool, str]:
        try:
            await paced(probe, ["word " * (chars // 5)])
            return True, "ok"
        except Exception as exc:
            return False, str(exc)[:100]

    low, high, last_error = 20_000, PROVIDER_CHAR_CEILING, ""
    while low < high:
        middle = (low + high + 1) // 2
        ok, detail = await accepted(middle)
        print(f"  {middle:>7} chars -> {'accepted' if ok else 'REJECTED: ' + detail}")
        if ok:
            low = middle
        else:
            high, last_error = middle - 1, detail
    print(f"  provider ceiling: {low} chars")
    print(
        f"  MAX_CHARS_PER_ITEM={load_settings().max_chars_per_item} "
        f"-> margin {low / load_settings().max_chars_per_item:.2f}x"
    )
    await probe.aclose()
    return low, last_error


async def measure_norms(embed: Embedder) -> tuple[float, float]:
    vectors, _ = await paced(embed, ["a short sentence", "متن کوتاه فارسی", "x" * 500])
    norms = [l2(v) for v in vectors]
    print(f"\n== vector norms ==\n  {[round(n, 6) for n in norms]}")
    return min(norms), max(norms)


async def measure_ingest_throughput(embed: Embedder, model_id: str, settings: object) -> None:
    """Chars per second at the production batching settings, over a real chunk.

    This is the number that decides whether a full re-ingest fits inside the quota.
    """
    print(f"\n== ingest throughput at production settings ({model_id}) ==")
    parser, chunker = ParserRegistry(), Chunker(settings)  # type: ignore[arg-type]
    document = parser.parse(
        (FIXTURES / "fa" / "justforfun_book_a4.pdf").read_bytes(), "justforfun_book_a4.pdf"
    )
    chunks = chunker.chunk(document)
    texts = [f"{document.title} > {' > '.join(c.section_path)}\n\n{c.text}" for c in chunks]
    sizes = [len(t) for t in texts]
    print(f"  corpus: {len(texts)} chunks, {sum(sizes)} chars, largest {max(sizes)} chars")
    print(
        f"  chunk sizing: CHARS_PER_TOKEN={CHARS_PER_TOKEN}"
        f" -> hard cap {chunker.hard_max_chars} chars"
    )

    started, calls, done = time.perf_counter(), 0, 0
    for start in range(0, len(texts), settings.max_items_per_batch):  # type: ignore[attr-defined]
        batch = texts[start : start + settings.max_items_per_batch]  # type: ignore[attr-defined]
        await paced(embed, batch)
        calls += 1
        done += len(batch)
    elapsed = time.perf_counter() - started
    print(
        f"  {done} chunks in {calls} requests, {elapsed:.1f}s wall\n"
        f"  {sum(sizes) / elapsed:,.0f} chars/s | {done / elapsed:.1f} chunks/s | "
        f"{calls / (elapsed / 60):.0f} req/min sustained"
    )
    print(f"  at 120 req/min the floor is {calls} x 0.5s = {calls * 0.5:.0f}s of pure quota wait")


# --- retrieval spot-check on the real fixtures, with the current pipeline -------------

# Gold is an exact phrase from the source, not a chunk id: chunk ids change whenever
# chunking is tuned, so they are never gold (plan.md §9.1). A hit is a retrieved chunk
# from the expected document whose text contains the phrase.
SPOT_CHECK: list[tuple[str, str, str]] = [
    # (question, expected document, phrase that must appear in the retrieved chunk)
    ("Who is Eleanor?", "storyen.md", "Eleanor"),
    ("What did Sarah find in the attic?", "storyen.md", "photograph"),
    ("What was in the sealed envelope?", "storyen.md", "envelope"),
    ("Why did Sarah go through her grandmother's things?", "storyen.md", "grandmother"),
    ("مرد در ایستگاه قطار چه می‌کرد؟", "storyfa.md", "ایستگاه قطار"),
    ("راننده تاکسی چه گفت؟", "storyfa.md", "راننده تاکسی"),
    ("چرا مرد به شهر کوچک برگشته بود؟", "storyfa.md", "برگشته بود"),
    ("What is the goal of this book?", "clean-code-excerpt.pdf", "goal of this book"),
    ("Who should read this book?", "clean-code-excerpt.pdf", "read this book"),
    ("What does the book say about functions?", "clean-code-excerpt.pdf", "function"),
    ("What must a knowledge system do about deleted documents?", "ai-engineer.pdf", "حذف"),
    ("متن سامانه درباره پرسش و پاسخ چه می‌گوید؟", "ai-engineer.pdf", "پرسش"),
    # Unanswerable: near-topic and off-topic, with no gold by construction.
    ("What is the capital of France?", "", ""),
    ("How do I configure the payment gateway?", "", ""),
]


CACHE = Path(__file__).resolve().parents[1] / "data" / "measure" / "fixture-vectors.json"


def rrf_k() -> int:
    return int(load_settings().rrf_k)


def _rrf(rank_d: int | None, rank_l: int | None, weight_d: float, weight_l: float) -> float:
    """Weighted reciprocal rank fusion (plan.md §7.1). Missing on one side costs only that side."""
    score = 0.0
    if rank_d is not None:
        score += weight_d / (rrf_k() + rank_d)
    if rank_l is not None:
        score += weight_l / (rrf_k() + rank_l)
    return score


def _build_corpus(cap_per_document: int | None) -> list[dict[str, object]]:
    """Chunk every fixture with the production parser and chunker."""
    parser, chunker = ParserRegistry(), Chunker(load_settings())
    rows: list[dict[str, object]] = []
    for path in sorted(FIXTURES.rglob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".pdf") or "Clean Code" in path.name:
            continue
        document = parser.parse(path.read_bytes(), path.name)
        chunks = chunker.chunk(document)
        if cap_per_document is not None:
            chunks = chunks[:cap_per_document]
        rows.extend(
            {
                "doc": path.name,
                "text": f"{document.title} > {' > '.join(c.section_path)}\n\n{c.text}",
                "section": " > ".join(c.section_path),
            }
            for c in chunks
        )
    return rows


async def _vectors_for(rows: list[dict[str, object]], embed: Embedder) -> list[list[float]]:
    """Vectors for the corpus, cached on disk.

    The fixture corpus is ~1200 chunks, which is ~40 requests and over a minute at the
    measured rate. Fusion weights have to be swept many times, so re-embedding for every
    experiment is the slowest thing this script could do. The cache is keyed by the exact
    embedded strings, so changing chunking invalidates it correctly.
    """
    texts = [str(row["text"]) for row in rows]
    key = hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16]
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            print(f"   vectors: {len(cached['vectors'])} from cache ({CACHE.name})")
            return [list(v) for v in cached["vectors"]]
    print(f"   vectors: embedding {len(texts)} chunks ({len(texts) // 32} requests)")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 32):
        vectors.extend((await paced(embed, texts[start : start + 32]))[0])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"key": key, "vectors": vectors}))
    print(f"   vectors: cached to {CACHE}")
    return vectors


async def retrieval_spot_check(embed: Embedder, model_id: str) -> dict[str, float]:
    """Dense vs lexical vs hybrid on the real fixtures, with the production retrieval code.

    §2.2a reported 12/12 for pure dense retrieval on 75 chunks from three small documents.
    That is re-measured here on the committed corpus with chunking-independent phrase gold
    (plan.md §9.1), because the gap between the two numbers is the design input P7 needs:
    if dense alone is not accurate enough to answer from, hybrid and the evidence gate
    are load-bearing rather than optional.
    """
    print(f"\n== retrieval on the committed fixtures ({model_id}) ==")
    rows = _build_corpus(None)
    vectors = await _vectors_for(rows, embed)
    print(f"   corpus: {len(rows)} chunks from {len({str(r['doc']) for r in rows})} documents")

    # The lexical side runs through the real FTS5 index, so this measures P5, not a rerun.
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteStore(Path(tmp) / "q.db")
        index = LexicalIndex(store)
        for doc in sorted({str(row["doc"]) for row in rows}):
            version = store.begin_staging(doc, doc, f"{doc}.md", "md", "c", "p", "en")
            subset = [r for r in rows if r["doc"] == doc]
            store.stage_version(
                doc,
                version,
                "c",
                " ".join(str(r["text"]) for r in subset),
                [
                    Chunk(i, 0, len(str(r["text"])), str(r["text"]), ("s",), f"h{i}", "en")
                    for i, r in enumerate(subset)
                ],
                [str(r["text"]) for r in subset],
            )
            store.publish(doc, version, "c", "p")
        results = await _compare(
            strategies=[("dense only", 1.0, 0.0), ("lexical only", 0.0, 1.0)]
            + [(f"hybrid {d}/{1 - d:.1f}", d, 1 - d) for d in (0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1)],
            rows=rows,
            vectors=vectors,
            index=index,
            embed=embed,
        )
        store.close()
    return results


def _chunk_id_map(rows: list[dict[str, object]]) -> dict[str, int]:
    """FTS chunk id -> corpus row index.

    Ordinals restart per document, so the id is rebuilt from the same
    ``(doc, within-document position)`` the staging code used.
    """
    mapping: dict[str, int] = {}
    seen: dict[str, int] = {}
    for position, row in enumerate(rows):
        doc = str(row["doc"])
        ordinal = seen.get(doc, 0)
        mapping[f"{doc}:v1:{ordinal}"] = position
        seen[doc] = ordinal + 1
    return mapping


async def _compare(
    *,
    strategies: list[tuple[str, float, float]],
    rows: list[dict[str, object]],
    vectors: list[list[float]],
    index: LexicalIndex,
    embed: Embedder,
) -> dict[str, float]:
    """Score each fusion strategy on the same gold, so the comparison is like for like.

    Faithful to plan.md §7.1: dense over-fetches ``CANDIDATES_N x OVERFETCH``, lexical
    returns ``CANDIDATES_N``, and fusion runs over the **union** of the two. Fusing
    full-corpus dense ranks instead lets one rank-1 lexical hit outrank a correct dense
    hit, which measures a different and much worse system than the one P7 will build --
    an earlier version of this script made exactly that mistake and reported hybrid as
    worse than dense, which was an artifact.
    """
    settings = load_settings()
    docs = [str(row["doc"]) for row in rows]
    by_id = _chunk_id_map(rows)
    window = settings.candidates_n * settings.overfetch
    print(
        f"\n   candidate window: dense top {window}, lexical top {settings.candidates_n},"
        f" RRF k={settings.rrf_k}"
    )
    print(f"   {'strategy':<18} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR@5':>7}")
    table: dict[str, float] = {}

    for label, weight_d, weight_l in strategies:
        ranks: list[int] = []
        for question, expected, phrase in SPOT_CHECK:
            if not expected:
                continue
            (query,), _ = await paced(embed, [question])
            dense_order = sorted(
                range(len(rows)),
                key=lambda i: -sum(a * b for a, b in zip(vectors[i], query, strict=True)),
            )[:window]
            dense_rank = {i: rank for rank, i in enumerate(dense_order, start=1)}
            lexical_rank = {
                by_id[hit.chunk_id]: rank
                for rank, hit in enumerate(
                    index.search(question, limit=settings.candidates_n), start=1
                )
                if hit.chunk_id in by_id
            }
            scores = {
                i: _rrf(dense_rank.get(i), lexical_rank.get(i), weight_d, weight_l)
                for i in set(dense_rank) | set(lexical_rank)
            }
            order = sorted(scores, key=lambda i: -scores[i])
            ranks.append(
                next(
                    (
                        position
                        for position, i in enumerate(order, start=1)
                        if docs[i] == expected
                        and phrase.casefold() in str(rows[i]["text"]).casefold()
                    ),
                    len(order) + 1,  # no gold phrase anywhere in the candidate window
                )
            )
        n = len(ranks)
        table[label] = sum(r == 1 for r in ranks) / n
        print(
            f"   {label:<18} {table[label]:>6.2f} {sum(r <= 3 for r in ranks) / n:>6.2f} "
            f"{sum(r <= 5 for r in ranks) / n:>6.2f} "
            f"{sum(1 / r for r in ranks if r <= 5) / n:>7.3f}"
        )
    best = max(table, key=lambda label: table[label])
    print(f"\n   best by R@1 here: {best} ({table[best]:.2f})")
    print(f"   caveat: {len(ranks)} answerable questions, so one question is worth 0.08 R@1.")
    return table


async def main() -> int:
    if not load_settings().embedding_api_key:
        print("EMBEDDING_API_KEY is not set", file=sys.stderr)
        return 1
    # A permissive limiter: pacing is handled above so latencies exclude queue time.
    async with EmbeddingClient(load_settings(), limiter=RateLimiter(10**6)) as client:
        print(f"model={client.model_id} dimension={client.dimension}")
        print(f"available={list(client.available_models)}")
        low, high = await measure_norms(client)
        print(f"  L2-normalized: min={low:.6f} max={high:.6f} (ingest skips normalizing)")
        await sweep_batch_size(client, 500, client.model_id)
        await sweep_item_length(client, 16, client.model_id)
        await find_item_ceiling(client.model_id, client.dimension)
        await measure_ingest_throughput(client, client.model_id, load_settings())
        await retrieval_spot_check(client, client.model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
