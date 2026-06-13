# ternary-sketch

Ternary Count-Min sketch for **approximate GPU workload analysis**. Each cell stores a ternary counter ∈ {-1, 0, +1} instead of a full integer, enabling ultra-low-memory streaming frequency estimation with bounded error guarantees.

## Why It Matters

Standard Count-Min sketches use `log(W)`-bit counters per cell, which over-provisions memory for workloads where only the *sign* of frequency matters (is this kernel hot, cold, or neutral?). Ternary counters reduce per-cell storage to 2 bits while preserving:

- **Heavy-hitter detection** — identify items with frequency > threshold
- **Direction tracking** — +1 means "increasing," -1 means "decreasing," 0 means "stable"
- **Mergeability** — combine sketches from multiple GPUs via cell-wise max

The ternary constraint clamps counters to {-1, 0, +1}, so repeated increments saturate at +1. This trades absolute frequency for directional information, which is sufficient for adaptive scheduling decisions.

## How It Works

### Data Structure

A `TernarySketch` is a 2D table of `depth × width` ternary cells, initialized to 0:

```
table[d][w] ∈ {-1, 0, +1}
```

### Hashing

Each row *d* uses a different seed to hash items to columns:

```
h_d(item) = (Σ byteᵢ · 31^(len-i) + d · 997) mod width
```

This is a polynomial rolling hash with row-specific salt — not cryptographically secure, but fast and sufficiently uniform for sketch purposes.

**Complexity:** O(L) per hash, where L = item byte length.

### Insert (Increment with Clamp)

```
for d in 0..depth:
    w = h_d(item)
    table[d][w] = match table[d][w]:
        -1 → 0
         0 → +1
        +1 → +1  (saturated)
```

### Remove (Decrement with Clamp)

```
for d in 0..depth:
    w = h_d(item)
    table[d][w] = match table[d][w]:
        +1 → 0
         0 → -1
        -1 → -1  (saturated)
```

### Frequency Estimate

Take the **minimum** across all rows (standard Count-Min estimator):

```
estimate(item) = min_d  table[d][h_d(item)]
```

The min reduces over-estimation from hash collisions: if two items collide in one row, they likely don't collide in another.

**Complexity:** O(D) per estimate, where D = depth.

### Error Bounds

For a Count-Min sketch with width *W* and depth *D*:

- **False positive rate:** `P(overestimate > 0) ≤ (1/e)^D ≈ 0.368^D`
- With D = 4: `P ≈ 1.8%` per query
- The ternary clamp introduces **directional accuracy**: the estimate correctly identifies sign (positive/negative/zero) with high probability, though the magnitude is clamped.

### Merge (Cell-wise Max)

```
merged[d][w] = max(sketch_A[d][w], sketch_B[d][w])
```

This is the standard Count-Min merge for the union of two streams. Preserves the worst-case (highest frequency) estimate.

**Complexity:** O(D × W).

## Quick Start

```rust
use ternary_sketch::TernarySketch;

let mut sk = TernarySketch::new(width: 64, depth: 4);

sk.insert(b"matmul_kernel");
sk.insert(b"matmul_kernel");
sk.insert(b"conv_kernel");

assert!(sk.estimate(b"matmul_kernel") > 0);  // hot
assert!(sk.estimate(b"unknown") == 0);        // not seen

let hh = sk.heavy_hitters(&[b"matmul_kernel".to_vec(), b"unknown".to_vec()], threshold: 0);
assert!(hh.contains(&0));  // matmul_kernel is a heavy hitter
```

## API

| Method | Returns | Description |
|--------|---------|-------------|
| `new(width, depth)` | `Self` | Initialize empty sketch |
| `insert(item)` | `()` | Increment (clamp at +1) |
| `remove(item)` | `()` | Decrement (clamp at -1) |
| `estimate(item)` | `i8` | Min across rows |
| `heavy_hitters(candidates, threshold)` | `Vec<usize>` | Indices above threshold |
| `merge(other)` | `()` | Cell-wise max merge |
| `fill_rate()` | `f64` | Fraction of non-zero cells |
| `total_updates()` | `u64` | Lifetime insert count |

## Architecture Notes

The **γ + η = C** invariant: *generation* (γ) is the insert/remove stream modifying cell values, *entropy* (η) is the information loss from ternary clamping (we can't recover exact frequencies from {-1, 0, +1}), and *conservation* (C) is the error bound guarantee — the estimate never underestimates the true clamped frequency (Count-Min property). The tradeoff between γ and η is direct: more insertions (γ↑) increase collisions and thus entropy (η↑), while the conservation law (C) maintains the `min-row` lower bound on estimate accuracy.

## References

- **Count-Min Sketch:** Cormode, G. & Muthukrishnan, S. "An Improved Data Stream Summary" (2005)
- **Ternary frequency tracking:** For other applications of ternary counting, see Alemdar et al. "Ternary Weight Networks" (2017)
- **Streaming algorithms:** Muthukrishnan, S. "Data Streams: Algorithms and Applications" (2005)
- **Mergeable summaries:** Agarwal, S. et al. "Mergeable Summaries" (2013)

## License

MIT
