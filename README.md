# Ternary Sketch — Count-Min Sketch with Ternary Counters

**Ternary Sketch** is a probabilistic data structure for approximate frequency estimation using a Count-Min sketch with ternary counters {-1, 0, +1}. It provides insert, remove, estimate, heavy-hitter detection, and merge operations — all in sublinear space, suitable for GPU workload analysis where exact counting is too expensive.

## Why It Matters

Tracking the frequency of every item in a high-throughput stream is impossible when the universe is large. Count-Min sketches solve this by hashing items into a small table, trading exact counts for bounded error guarantees. The ternary variant uses counters limited to {-1, 0, +1}, making each counter fit in 2 bits — 4× denser than byte counters and 16× denser than int32. This is particularly valuable for GPU workload analysis: tracking kernel launch frequencies, memory access patterns, and cache hit rates in real-time, all in a few kilobytes of state.

## How It Works

### Structure

The sketch is a `depth × width` table of ternary counters. Each item is hashed with `depth` independent hash functions to find one cell per row.

### Insert (increment)

For each row d, hash the item to column `w = hash(item, seed_d) mod width`, then saturate the counter upward:

```
-1 → 0
 0 → +1
+1 → +1  (saturate)
```

### Remove (decrement)

Same as insert but saturating downward:

```
+1 → 0
 0 → -1
-1 → -1  (saturate)
```

### Estimate

Return the minimum counter value across all rows:

```
estimate(item) = min(sketch[d][hash(item, d)] for d in 0..depth)
```

The min reduces over-estimation error (the core Count-Min guarantee). With probability ≥ 1-δ, the estimate err·s by at most εN, where N is total insertions and width = ⌈e/ε⌉, depth = ⌈ln(1/δ)⌉. Space: O(depth × width × 2 bits).

### Heavy Hitters

Given a set of candidate items, return those with estimate > threshold. O(c × depth) for c candidates.

### Merge

Two sketches merge by taking the max per cell. This is a valid CRDT merge — commutative, idempotent, associative. Enables distributed sketch aggregation across GPU nodes.

## Quick Start

```rust
use ternary_sketch::TernarySketch;

let mut sketch = TernarySketch::new(64, 4); // width=64, depth=4

sketch.insert(b"kernel_launch");
sketch.insert(b"kernel_launch");
sketch.insert(b"memory_copy");

let freq = sketch.estimate(b"kernel_launch");
println!("Estimated frequency: {}", freq);

// Heavy hitters
let candidates = vec![b"kernel_launch".to_vec(), b"rare_event".to_vec()];
let heavy = sketch.heavy_hitters(&candidates, 0);
```

```bash
cargo add ternary-sketch
```

## API

| Type / Function | Description |
|---|---|
| `TernarySketch` | `new(width, depth)`, `insert()`, `remove()`, `estimate()`, `heavy_hitters()`, `merge()` |
| `fill_rate()` | Fraction of non-zero cells |
| `total_updates()` | Count of insertions |

## Architecture Notes

The sketch enables approximate fleet monitoring in **SuperInstance**. GPU nodes maintain local sketches of workload patterns and merge them via the CRDT property. The γ + η = C conservation manifests in the error-space trade-off: wider sketches (more γ = more information) reduce estimation error (less η = less uncertainty), for a fixed total size C. See [Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

- Cormode, Graham & Muthukrishnan, S. "An Improved Data Stream Summary," *ESA*, 2004 — Count-Min sketch.
| Mitzenmacher, Michael & Upfal, Eli. *Probability and Computing*, Cambridge UP, 2017.
| Agarwal, Pranjal et al. "Sketching for Big Data," *Found. Trends ML*, 2020.

## License

Apache-2.0
