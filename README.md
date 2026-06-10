# ternary-sketch

Count-Min sketch with ternary counters for approximate GPU workload analysis.

## Why This Exists

When you're monitoring thousands of GPU kernel types across a fleet, you can't track exact counts for everything. Sketch data structures give you approximate frequency estimation in bounded space. But binary counters in a Count-Min sketch can only count up. Ternary counters (+1, 0, -1) let you **both insert and remove** items — tracking frequency deltas, not just totals. This means you can detect which kernels are heating up (net positive), stable (net zero), or cooling down (net negative).

## Architecture

### Core Types

- **`TernarySketch`** — A 2D array of ternary counters (`i8`) with configurable `width` and `depth`. Multiple hash functions map items to different rows.
- Each cell stores a ternary value: positive (overrepresented), zero (baseline), negative (underrepresented).

### Key Algorithms

- **insert**: Hash item to `depth` positions, increment each cell (clamped to ±1).
- **remove**: Same but decrement — useful for sliding windows.
- **estimate**: Return the minimum across all hash positions.
- **heavy_hitters**: Among candidates, return those exceeding a ternary threshold.
- **merge**: Combine two sketches pointwise (max).

## Usage

```rust
use ternary_sketch::TernarySketch;

let mut sketch = TernarySketch::new(1024, 5); // 1024 wide, 5 hash functions

sketch.insert(b"kernel::matmul_4096");
sketch.insert(b"kernel::matmul_4096");
sketch.insert(b"kernel::layernorm");

let freq = sketch.estimate(b"kernel::matmul_4096");
assert!(freq > 0); // overrepresented

// Remove to simulate sliding window
sketch.remove(b"kernel::matmul_4096");

// Heavy hitter detection
let hitters = sketch.heavy_hitters(
    &[b"kernel::matmul_4096".to_vec(), b"kernel::layernorm".to_vec()],
    0, // threshold: anything above 0
);
```

## API Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `new(width, depth)` | `TernarySketch` | Create sketch with given dimensions |
| `insert(item)` | `()` | Increment item's counters |
| `remove(item)` | `()` | Decrement item's counters |
| `estimate(item)` | `i8` | Approximate frequency (min across hashes) |
| `heavy_hitters(candidates, threshold)` | `Vec<usize>` | Indices of items above threshold |
| `merge(other)` | `()` | Merge another sketch into this one |
| `fill_rate()` | `f64` | Fraction of non-zero cells |
| `total_updates()` | `u64` | Total insert + remove operations |

## The Deeper Idea

Ternary sketches are the **signal processing** of workload monitoring. Traditional Count-Min is a low-pass filter (it only accumulates). Ternary Count-Min is a band-pass filter — it shows you what's changing, not just what's large. Items that have been inserted many times but also removed many times show as zero, which is the correct answer: "this used to be hot, now it's not." This makes ternary sketches ideal for adaptive scheduling where you care about trends, not totals.

## Related Crates

- **ternary-bloom-filter** — membership testing with ternary weighted bits
- **ternary-search-index** — ternary-weighted document search
- **ternary-accumulator** — ternary gradient accumulation
