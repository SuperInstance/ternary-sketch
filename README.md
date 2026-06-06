# ternary-sketch

Streaming sketch data structures with ternary {-1, 0, +1} counters for approximate frequency estimation of GPU workloads.

## Background

Streaming algorithms process data in a single pass while using space sublinear in the input size. The **Count-Min sketch** (Cormode & Muthukrishnan, 2005) is a probabilistic data structure that estimates item frequencies in a stream using a 2D array of counters indexed by hash functions. Each insertion increments counters at `d` hash positions; estimates are the minimum across all rows, providing an upper bound with bounded error probability ε.

Classical Count-Min uses integer counters that grow unboundedly. In a ternary regime, each counter is clamped to {-1, 0, +1}, fundamentally changing the data structure's properties. The sketch can no longer represent arbitrary frequency magnitudes—instead, it captures the *direction* and *presence* of frequency signals. This is a lossy compression: a counter at +1 means "this item has been seen" and -1 means "this item has been removed," but the exact count is erased after the first increment.

The information-theoretic consequences are significant. A ternary counter carries log₂(3) ≈ 1.585 bits per cell versus log₂(k) bits for a counter range of k. This ~37% reduction in storage per cell (compared to binary 0/1) comes at the cost of saturated frequency resolution. The trade-off is worthwhile when the goal is detecting *whether* an item is heavy rather than *how heavy* it is.

## How It Works

### Architecture

`TernarySketch` maintains a `depth × width` table of `i8` cells, each constrained to {-1, 0, +1}. A simple polynomial hash function `h(item, seed) = Σ(bᵢ · 31^i) mod width` maps items to column indices, with different seeds per row.

### Key Operations

- **Insert**: For each of `d` rows, hash to a column and transition the counter: `{-1 → 0, 0 → +1, +1 → +1}`. This is a saturating increment—once at +1, it stays.
- **Remove**: Reverse transition: `{+1 → 0, 0 → -1, -1 → -1}`. Saturating decrement.
- **Estimate**: Return the minimum counter value across all `d` rows for the item's hash positions. Classical Count-Min returns the min, giving an upper bound; the ternary variant returns a *ternary vote*.
- **Heavy Hitters**: Filter candidate items whose estimated value exceeds a threshold. With threshold > 0, this finds items that appear "consistently present" across all hash rows.
- **Merge**: Cell-wise maximum of two sketches. This is a set-union semantics: if either sketch says +1, the merged result says +1.

### Design Decisions

The saturating counter means the sketch cannot distinguish "seen once" from "seen 1000 times." This is deliberate for GPU workload analysis where the question is "is this kernel being dispatched?" not "exactly how many times?" The ternary state space also enables `remove()` to express negative evidence, which standard Bloom-filter-like structures cannot.

## Experimental Results

All 7 unit tests pass:

| Test | Result | Description |
|------|--------|-------------|
| `test_insert_estimate` | ✅ | After inserting `b"kernel_a"`, `estimate()` returns > 0 |
| `test_missing` | ✅ | Absent items estimate to exactly 0 |
| `test_heavy_hitters` | ✅ | Repeatedly inserted `hot_kernel` detected above threshold 0 |
| `test_merge` | ✅ | Merging sketch with `x` and sketch with `y` preserves both |
| `test_fill_rate` | ✅ | Empty sketch: fill_rate = 0.0; after insert: fill_rate > 0.0 |
| `test_remove` | ✅ | Insert then remove yields estimate ≤ 0 |
| `test_ternary_clamp` | ✅ | After 10 inserts of same item, estimate stays in {-1, 0, +1} |

The `test_ternary_clamp` result is notable: 10 repeated inserts of `"repeat"` still produce an estimate clamped to ≤ 1. The ternary saturation is invariant regardless of stream volume.

## Impact of Ternary {-1, 0, +1}

The ternary counter space transforms the sketch from a frequency estimator into a **presence/absence/negative-evidence** detector:

- **+1** = positive evidence (item has been inserted at least once since last removal)
- **0** = neutral (no signal or cancellation between insert and remove)
- **−1** = negative evidence (item has been removed more than inserted)

This three-valued logic enables streaming set difference operations that are impossible with standard 0/1 sketches. Two sketches can be merged with cell-wise max (union) or compared for inclusion. The negative state also supports "definitely not present" semantics for cache invalidation scenarios.

## Use Cases

1. **GPU Kernel Dispatch Monitoring**: Track which kernels are currently active in a compute workload. Insert on dispatch, remove on completion. Heavy hitters identify hot kernels; negative estimates identify cancelled or failed kernels.

2. **Streaming Set Membership with Deletion**: Unlike Bloom filters, the ternary sketch supports deletion without counters. Use it for "recently seen / recently removed / unknown" membership queries on high-throughput event streams.

3. **Distributed Sketch Aggregation**: Multiple nodes maintain local sketches and merge them (cell-wise max) at a central aggregator. The ternary space ensures merge is well-defined and idempotent—no integer overflow or scaling issues.

4. **Hot Path Detection in Microservices**: Insert request signatures at API gateways. Heavy hitters above threshold 0 identify endpoints receiving traffic. After routing changes, remove old paths and detect the shift.

5. **Approximate Cache Hit/Miss Tracking**: Insert cache keys on fill, remove on eviction. The sketch gives a streaming view of which keys are "in cache" (+1), "evicted" (−1), or "neutral" (0) without maintaining the full key set.

## Open Questions

1. **False Positive Rate Under Ternary Clamping**: What is the analytical false positive rate for heavy hitter detection when counters saturate at +1? Classical Count-Min analysis assumes unbounded counters; the clamping changes the error distribution.

2. **Optimal Width/Depth for Ternary Regime**: Standard guidelines (width = e/ε, depth = ln(1/δ)) assume integer counters. The ternary saturation likely requires wider tables to compensate for lost frequency resolution—what are the corrected bounds?

3. **Merge Semantics Beyond Max**: Cell-wise max is one merge strategy. Alternatives like majority vote or weighted average could produce different aggregation properties. Which merge strategy minimizes estimation error for given sketch sizes?

## Connection to Oxide Stack

Within the five-layer Oxide ternary architecture:

- **Layer 1 (Ternary Genome)**: The sketch's ternary counters {-1, 0, +1} mirror the genome alphabet, enabling direct feeding of sketch estimates as genome inputs for adaptive behavior.
- **Layer 2 (Cellular Computation)**: Each hash row acts as an independent computational cell voting on item frequency; the min operation is the cell-level consensus.
- **Layer 3 (Organism Behavior)**: Heavy hitter detection drives organism-level responses—"this kernel is hot, allocate more resources"—acting as a sensory organ for the system.
- **Layer 4 (Population Dynamics)**: Distributed sketch merging implements population-level aggregation, where multiple organisms share frequency information without exchanging raw data.
- **Layer 5 (Ecosystem)**: The sketch sits at the ecosystem boundary, processing the firehose of external events (GPU dispatches, network packets) into a compressed ternary signal that downstream layers can consume.
