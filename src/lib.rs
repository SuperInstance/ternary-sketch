//! # ternary-sketch
//!
//! Ternary sketch for approximate GPU workload analysis.
//! Count-Min sketch with ternary counters.

fn hash(item: &[u8], seed: u64) -> usize {
    let mut h = seed;
    for &b in item { h = h.wrapping_mul(31).wrapping_add(b as u64); }
    h as usize
}

#[derive(Debug, Clone)]
pub struct TernarySketch {
    width: usize,
    depth: usize,
    table: Vec<Vec<i8>>, // depth x width, each cell is {-1, 0, +1}
    total_updates: u64,
}

impl TernarySketch {
    pub fn new(width: usize, depth: usize) -> Self {
        Self { width, depth, table: vec![vec![0; width]; depth], total_updates: 0 }
    }

    /// Insert item: increment if possible within ternary range.
    pub fn insert(&mut self, item: &[u8]) {
        self.total_updates += 1;
        for d in 0..self.depth {
            let w = hash(item, d as u64 * 997) % self.width;
            self.table[d][w] = match self.table[d][w] { -1 => 0, 0 => 1, _ => 1 };
        }
    }

    /// Remove item: decrement.
    pub fn remove(&mut self, item: &[u8]) {
        for d in 0..self.depth {
            let w = hash(item, d as u64 * 997) % self.width;
            self.table[d][w] = match self.table[d][w] { 1 => 0, 0 => -1, _ => -1 };
        }
    }

    /// Estimate frequency: minimum across all depth rows.
    pub fn estimate(&self, item: &[u8]) -> i8 {
        (0..self.depth).map(|d| {
            let w = hash(item, d as u64 * 997) % self.width;
            self.table[d][w]
        }).min().unwrap_or(0)
    }

    /// Heavy hitters: items with estimate > threshold.
    pub fn heavy_hitters(&self, candidates: &[Vec<u8>], threshold: i8) -> Vec<usize> {
        candidates.iter().enumerate().filter(|(_, c)| self.estimate(c) > threshold).map(|(i, _)| i).collect()
    }

    /// Merge two sketches: take max per cell.
    pub fn merge(&mut self, other: &TernarySketch) {
        for d in 0..self.depth.min(other.depth) {
            for w in 0..self.width.min(other.width) {
                self.table[d][w] = self.table[d][w].max(other.table[d][w]);
            }
        }
    }

    pub fn fill_rate(&self) -> f64 {
        let total = self.depth * self.width;
        let filled = self.table.iter().flat_map(|r| r.iter()).filter(|&&v| v != 0).count();
        filled as f64 / total as f64
    }

    pub fn total_updates(&self) -> u64 { self.total_updates }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_estimate() {
        let mut sk = TernarySketch::new(64, 4);
        sk.insert(b"kernel_a");
        assert!(sk.estimate(b"kernel_a") > 0);
    }

    #[test]
    fn test_missing() {
        let sk = TernarySketch::new(64, 4);
        assert_eq!(sk.estimate(b"absent"), 0);
    }

    #[test]
    fn test_heavy_hitters() {
        let mut sk = TernarySketch::new(64, 4);
        sk.insert(b"hot_kernel");
        sk.insert(b"hot_kernel");
        let hh = sk.heavy_hitters(&[b"hot_kernel".to_vec(), b"cold".to_vec()], 0);
        assert!(hh.contains(&0));
    }

    #[test]
    fn test_merge() {
        let mut sk1 = TernarySketch::new(32, 3);
        let mut sk2 = TernarySketch::new(32, 3);
        sk1.insert(b"x");
        sk2.insert(b"y");
        sk1.merge(&sk2);
        assert!(sk1.estimate(b"x") > 0);
    }

    #[test]
    fn test_fill_rate() {
        let mut sk = TernarySketch::new(16, 2);
        assert_eq!(sk.fill_rate(), 0.0);
        sk.insert(b"a");
        assert!(sk.fill_rate() > 0.0);
    }

    #[test]
    fn test_remove() {
        let mut sk = TernarySketch::new(64, 4);
        sk.insert(b"temp");
        sk.remove(b"temp");
        assert!(sk.estimate(b"temp") <= 0);
    }

    #[test]
    fn test_ternary_clamp() {
        let mut sk = TernarySketch::new(16, 2);
        for _ in 0..10 { sk.insert(b"repeat"); }
        let est = sk.estimate(b"repeat");
        assert!(est <= 1 && est >= -1); // always ternary
    }
}
