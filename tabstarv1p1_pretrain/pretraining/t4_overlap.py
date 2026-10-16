"""Content-based table matching, used to drop T4 near-duplicates and T4 copies of evaluation tables.

Each column with enough distinct values gets a bottom-k sketch: the k smallest keyed hashes of its
distinct values. Two sketches estimate the Jaccard similarity of the underlying value sets, so a
copy that keeps a subset of rows still scores in proportion to what it kept, and renaming columns
does not hide it. A table matches a reference table when their lowercased column-name sets are
identical, or when enough of its columns match columns of that one reference table.
"""
from collections import Counter
from hashlib import blake2b, sha256
import json
from math import ceil
from typing import Dict, Iterable, List, Optional, Tuple

from pandas import DataFrame

SKETCH_SIZE = 64
# Changing the key invalidates every stored sketch.
SKETCH_KEY = b"t4-overlap-v1"
# Low-cardinality columns (Yes/No, small category sets) hash alike across unrelated tables.
MIN_DISTINCT_VALUES = 25
# Below the ~0.5 a half-kept column scores, so row subsets still match.
COLUMN_MATCH_THRESHOLD = 0.30
MIN_MATCHING_COLUMNS = 2
# Bounds memory when streaming large reference tables; only the distinct-value floor reads it.
DISTINCT_CAP = 100_000

Sketch = List[int]


def _hash(value: str) -> int:
    digest = blake2b(value.encode("utf-8"), digest_size=8, key=SKETCH_KEY).digest()
    return int.from_bytes(digest, "big")


def schema_signature(columns: Iterable) -> str:
    names = sorted({str(c).strip().lower() for c in columns})
    return sha256(json.dumps(names).encode("utf-8")).hexdigest()


def _is_contiguous_int_range(n_distinct: int, lo: int, hi: int) -> bool:
    span = hi - lo + 1
    return span > 0 and n_distinct / span >= 0.9


def is_generic_integer_column(values: Iterable) -> bool:
    """Row indices, sequential ids and dense year columns match across unrelated tables."""
    integers = set()
    for value in values:
        try:
            integers.add(int(str(value).strip()))
        except (TypeError, ValueError):
            return False
    if len(integers) < 2:
        return False
    return _is_contiguous_int_range(len(integers), lo=min(integers), hi=max(integers))


def column_sketch(values: Iterable, size: int = SKETCH_SIZE) -> Optional[Sketch]:
    """Bottom-k sketch of a column's distinct values, or None if the column is uninformative."""
    distinct = {str(v) for v in values}
    if len(distinct) < MIN_DISTINCT_VALUES or is_generic_integer_column(distinct):
        return None
    return sorted(_hash(v) for v in distinct)[:size]


def frame_signatures(frame: DataFrame) -> Tuple[str, List[Sketch]]:
    sketches = []
    for col in frame.columns:
        try:
            values = frame[col].dropna().astype(str).unique()
        except Exception:
            continue
        sketch = column_sketch(values)
        if sketch:
            sketches.append(sketch)
    return schema_signature(frame.columns), sketches


class _StreamingColumn:
    """Tracks one column's bottom-k sketch and integer range without holding its values."""

    def __init__(self):
        self.sketch: Sketch = []
        self.in_sketch = set()
        self.distinct = set()
        self.all_int = True
        self.lo: Optional[int] = None
        self.hi: Optional[int] = None

    def update(self, values: Iterable[str]):
        for value in values:
            if len(self.distinct) < DISTINCT_CAP:
                self.distinct.add(value)
            self._update_range(value)
            digest = _hash(value)
            if digest in self.in_sketch:
                continue
            if len(self.sketch) < SKETCH_SIZE:
                self._insert(digest)
            elif digest < self.sketch[-1]:
                self.in_sketch.discard(self.sketch.pop())
                self._insert(digest)

    def _insert(self, digest: int):
        self.sketch.append(digest)
        self.sketch.sort()
        self.in_sketch.add(digest)

    def _update_range(self, value: str):
        if not self.all_int:
            return
        try:
            number = int(value.strip())
        except (TypeError, ValueError):
            self.all_int = False
            return
        self.lo = number if self.lo is None else min(self.lo, number)
        self.hi = number if self.hi is None else max(self.hi, number)

    def finalize(self) -> Optional[Sketch]:
        n_distinct = len(self.distinct)
        if n_distinct < MIN_DISTINCT_VALUES:
            return None
        if self.all_int and self.lo is not None and _is_contiguous_int_range(n_distinct, self.lo, self.hi):
            return None
        return self.sketch


def parquet_signatures(path: str) -> Tuple[str, List[Sketch]]:
    """Same signatures as `frame_signatures`, streamed so multi-GB reference tables fit in memory."""
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(path)
    names = list(reader.schema.names)
    columns = {name: _StreamingColumn() for name in names}
    for batch in reader.iter_batches(batch_size=65536):
        frame = batch.to_pandas()
        for name in names:
            try:
                values = frame[name].dropna().astype(str).unique()
            except Exception:
                continue
            columns[name].update(values)
    sketches = [s for s in (c.finalize() for c in columns.values()) if s]
    return schema_signature(names), sketches


def sketch_jaccard(left: Sketch, right: Sketch) -> float:
    """Estimated Jaccard over the k smallest hashes of the union, valid when set sizes differ."""
    if not left or not right:
        return 0.0
    width = min(len(left), len(right))
    union = sorted(set(left) | set(right))[:width]
    shared = set(left) & set(right)
    return sum(1 for value in union if value in shared) / len(union)


class OverlapIndex:
    """Finds which indexed table a candidate table is a copy of."""

    def __init__(self, threshold: float = COLUMN_MATCH_THRESHOLD, min_columns: int = MIN_MATCHING_COLUMNS):
        self.threshold = threshold
        self.min_columns = min_columns
        self._schemas: Dict[str, str] = {}
        self._sketches: List[Tuple[str, Sketch]] = []
        # Similar sets share small hashes, so candidates come from an inverted index, not every pair.
        self._by_hash: Dict[int, set] = {}

    @property
    def size(self) -> int:
        return len(self._sketches)

    def add_table(self, table_id: str, schema: Optional[str], sketches: Iterable[Sketch]):
        if schema:
            self._schemas[schema] = table_id
        for sketch in sketches:
            if not sketch:
                continue
            position = len(self._sketches)
            self._sketches.append((table_id, list(sketch)))
            for value in sketch:
                self._by_hash.setdefault(value, set()).add(position)

    def match_columns(self, sketches: Iterable[Sketch]) -> Dict[str, int]:
        """How many of the candidate's columns match each indexed table."""
        hits: Dict[str, int] = {}
        for sketch in sketches:
            if not sketch:
                continue
            shared_counts: Counter = Counter()
            for value in sketch:
                shared_counts.update(self._by_hash.get(value, ()))
            best: Dict[str, float] = {}
            for position, shared in shared_counts.items():
                table_id, other = self._sketches[position]
                min_shared = ceil(self.threshold * min(len(sketch), len(other)))
                if shared < min_shared:
                    continue
                best[table_id] = max(best.get(table_id, 0.0), sketch_jaccard(sketch, other))
            # A candidate column vouches for a table at most once.
            for table_id, score in best.items():
                if score >= self.threshold:
                    hits[table_id] = hits.get(table_id, 0) + 1
        return hits

    def blocking_table(self, schema: Optional[str], sketches: List[Sketch]) -> Optional[Tuple[str, str]]:
        """The indexed table this candidate copies and the evidence, or None."""
        if schema and schema in self._schemas:
            return self._schemas[schema], "schema"
        hits = self.match_columns(sketches)
        if not hits:
            return None
        table_id, count = max(hits.items(), key=lambda item: (item[1], item[0]))
        if count >= self.min_columns:
            return table_id, f"content:{count}-columns"
        return None


def load_fingerprint_index(path: str) -> OverlapIndex:
    """Builds an index from a fingerprint JSON written by `do_t4_eval_fingerprints.py`."""
    with open(path) as f:
        payload = json.load(f)
    index = OverlapIndex()
    for table in payload["tables"]:
        index.add_table(table["table_id"], table["schema"], table["sketches"])
    return index
