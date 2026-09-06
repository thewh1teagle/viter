//! A binary max-heap that reproduces libstdc++'s `std::priority_queue`
//! element ordering.
//!
//! Kaldi's clustering and tree-splitting code pushes equal-keyed elements into
//! `std::priority_queue` and the pop order among them decides which merge or
//! split happens first — so parity requires matching the heap's sift
//! behaviour, not just its ordering guarantee. `push` is libstdc++'s
//! `__push_heap`; `pop` moves the last element to the root and sifts it down,
//! always descending into the larger child (the right child on ties), as
//! `__adjust_heap` does.
//!
//! Kaldi's min-heaps (`std::greater<...>`) are expressed here by wrapping keys
//! in `std::cmp::Reverse`.


/// Max-heap over `T: Ord`, replicating libstdc++ `push_heap`/`pop_heap` so that
/// the pop order among equal-comparing elements matches Kaldi exactly.
/// (Kaldi's min-heaps are expressed by wrapping keys in `std::cmp::Reverse`.)
pub(crate) struct Heap<T> {
    data: Vec<T>,
}

impl<T: Ord> Heap<T> {
    pub(crate) fn new() -> Self {
        Self { data: Vec::new() }
    }
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub(crate) fn peek(&self) -> Option<&T> {
        self.data.first()
    }
    pub(crate) fn clear(&mut self) {
        self.data.clear();
    }

    /// libstdc++ `__push_heap`: sift the new last element up while it is
    /// strictly greater than its parent.
    pub(crate) fn push(&mut self, value: T) {
        self.data.push(value);
        let mut hole = self.data.len() - 1;
        while hole > 0 {
            let parent = (hole - 1) / 2;
            if self.data[parent] < self.data[hole] {
                self.data.swap(parent, hole);
                hole = parent;
            } else {
                break;
            }
        }
    }

    /// libstdc++ `pop_heap`: move the last element into the root, then
    /// `__adjust_heap` — sift down always taking the larger child (right child
    /// on ties, matching `__is_heap`'s `if (comp(children, children-1))`),
    /// then a final push_heap of the moved value from the leaf position.
    pub(crate) fn pop(&mut self) -> Option<T> {
        let len = self.data.len();
        if len == 0 {
            return None;
        }
        if len == 1 {
            return self.data.pop();
        }
        let result = self.data.swap_remove(0);
        // `swap_remove` moved the last element to index 0; sift it down,
        // always descending into the larger child (the right child on ties,
        // matching libstdc++'s `__adjust_heap`).
        let n = self.data.len();
        let mut hole = 0usize;
        loop {
            let left = 2 * hole + 1;
            if left >= n {
                break;
            }
            let right = left + 1;
            let child = if right < n && !(self.data[right] < self.data[left]) {
                right
            } else {
                left
            };
            if self.data[hole] < self.data[child] {
                self.data.swap(hole, child);
                hole = child;
            } else {
                break;
            }
        }
        Some(result)
    }
}

/// A total order on f64 for heap keys. All values here are finite distances or
/// objective-function improvements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OrdF64(pub f64);
impl Eq for OrdF64 {}
impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pops_in_descending_order() {
        let mut h: Heap<i32> = Heap::new();
        for v in [3, 1, 4, 1, 5, 9, 2, 6] {
            h.push(v);
        }
        let mut out = Vec::new();
        while let Some(v) = h.pop() {
            out.push(v);
        }
        assert_eq!(out, vec![9, 6, 5, 4, 3, 2, 1, 1]);
    }

    #[test]
    fn reverse_gives_a_min_heap() {
        let mut h: Heap<std::cmp::Reverse<i32>> = Heap::new();
        for v in [3, 1, 4, 1, 5] {
            h.push(std::cmp::Reverse(v));
        }
        let mut out = Vec::new();
        while let Some(std::cmp::Reverse(v)) = h.pop() {
            out.push(v);
        }
        assert_eq!(out, vec![1, 1, 3, 4, 5]);
    }

    #[test]
    fn peek_and_clear() {
        let mut h: Heap<(OrdF64, usize)> = Heap::new();
        h.push((OrdF64(1.5), 0));
        h.push((OrdF64(2.5), 1));
        assert_eq!(h.peek().unwrap().1, 1);
        assert_eq!(h.len(), 2);
        h.clear();
        assert!(h.is_empty());
        assert!(h.pop().is_none());
    }

    #[test]
    fn ord_f64_orders_by_value() {
        assert!(OrdF64(1.0) < OrdF64(2.0));
        assert_eq!(OrdF64(1.0), OrdF64(1.0));
    }
}
