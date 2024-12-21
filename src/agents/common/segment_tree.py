"""
Segment Tree data structures for efficient sum and min operations.
"""

import operator
from typing import Callable

class SegmentTree:
    def __init__(self, capacity: int, operation: Callable, neutral_element: float) -> None:
        """
        Initialize Segment Tree.
        
        Args:
            capacity: Number of elements that can be stored in the tree
            operation: Operation to perform on the tree (e.g., min or sum)
            neutral_element: Neutral element for the operation
        """
        self.capacity = capacity
        self.operation = operation
        self.neutral_element = neutral_element
        
        # Tree height and size
        self.tree_height = 1
        while 2**self.tree_height < capacity:
            self.tree_height += 1
            
        self.tree_size = 2**self.tree_height
        self.tree = [self.neutral_element for _ in range(2 * self.tree_size)]
        
    def _reduce(self, start: int, end: int, node: int = 1, node_start: int = 0, node_end: int = None) -> float:
        """Reduce elements in range [start, end) using the operation."""
        if node_end is None:
            node_end = self.tree_size
            
        if start == node_start and end == node_end:
            return self.tree[node]
            
        mid = (node_start + node_end) // 2
        
        if end <= mid:
            return self._reduce(start, end, 2 * node, node_start, mid)
        elif start >= mid:
            return self._reduce(start, end, 2 * node + 1, mid, node_end)
        else:
            return self.operation(
                self._reduce(start, mid, 2 * node, node_start, mid),
                self._reduce(mid, end, 2 * node + 1, mid, node_end)
            )
    
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
        return self.tree[self.tree_size + idx]
    
    def reduce(self, start: int = 0, end: int = None) -> float:
        """Reduce elements in range [start, end) using the operation."""
        if end is None:
            end = self.capacity
        if end < 0:
            end += self.tree_size
            
        return self._reduce(start, end)

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        """Initialize Sum Segment Tree."""
        super().__init__(capacity=capacity, operation=operator.add, neutral_element=0.0)
        
    def sum(self, start: int = 0, end: int = None) -> float:
        """Return sum of elements in range [start, end)."""
        return self.reduce(start, end)
        
    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """
        Find the highest index `i` in the tree such that
        sum(tree[0] + tree[1] + ... + tree[i - 1]) <= prefixsum
        """
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
    def __init__(self, capacity: int) -> None:
        """Initialize Min Segment Tree."""
        super().__init__(capacity=capacity, operation=min, neutral_element=float('inf'))
        
    def min(self, start: int = 0, end: int = None) -> float:
        """Return min of elements in range [start, end)."""
        return self.reduce(start, end) 