import operator
from typing import Callable

class SegmentTree:
    def __init__(self, capacity: int, operation: Callable, neutral_element: float):
        """Initialize Segment Tree.
        
        Args:
            capacity: Number of elements that can be stored in the tree
            operation: Function to use for combining elements (e.g., min or sum)
            neutral_element: Neutral element for the operation
        """
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        
        # Tree height and size
        self.tree_height = (capacity - 1).bit_length()
        self.tree_size = 2 ** self.tree_height
        self.tree = [self.neutral_element for _ in range(2 * self.tree_size)]
        
    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Helper function for range-based operations."""
        if start == node_start and end == node_end:
            return self.tree[node]
        
        mid = (node_start + node_end) // 2
        
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        if mid + 1 <= start:
            return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            
        left = self._operate_helper(start, mid, 2 * node, node_start, mid)
        right = self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
        return self.operation(left, right)

    def __setitem__(self, idx: int, val: float) -> None:
        """Set value in tree."""
        idx += self.tree_size
        self.tree[idx] = val
        
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(
                self.tree[2 * idx],
                self.tree[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get value from tree."""
        assert 0 <= idx < self.capacity
        return self.tree[self.tree_size + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start: int = 0, end: int = None) -> float:
        """Returns sum over [start, end)."""
        if end is None:
            end = self.capacity
        return self._operate_helper(start, end - 1, 1, 0, self.tree_size - 1)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Find the highest index `i` in the tree with sum[0, i] <= prefixsum."""
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        
        while idx < self.tree_size:
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1
                
        return idx - self.tree_size

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start: int = 0, end: int = None) -> float:
        """Returns min over [start, end)."""
        if end is None:
            end = self.capacity
        return self._operate_helper(start, end - 1, 1, 0, self.tree_size - 1) 