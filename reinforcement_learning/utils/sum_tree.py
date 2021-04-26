# Based on:
# https://github.com/rlcode/per/blob/master/SumTree.py
# https://adventuresinmachinelearning.com/sumtree-introduction-python/
# https://en.wikipedia.org/wiki/Binary_tree
from typing import Any
import numpy as np


class SumTree:
    """
    Binary Sum Tree implementation that allows to sample from an array based on priorities assigned to each object

    Parameters
    ----------
    capacity: int
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx: int, delta_node_value: float) -> None:
        """
        Update all nodes upwards until we reached the top-parent node. It is important to work with the difference
        rather the new node value to update without knowledge of the other leaves
        Performed recursively until top-parent node (index == 0)

        Parameters
        ----------
        idx: int
        delta_node_value: float
            Difference in node-value compared to previous entry

        Returns
        -------
        None
        """

        parent_index = (idx - 1) // 2

        self.tree[parent_index] += delta_node_value

        if parent_index > 0:
            self._propagate(parent_index, delta_node_value)

    def _retrieve_leaf(self, idx: int, sampled_value: float) -> int:
        """
        Recursive method to traverse to a leave using the following algorithm:

        Decision 1:
        Have we reached a leaf, i.e. 2 * idx + 1 >= n -> index value is returned

        Decision 2:
        If there are still two children left, we follow the sum-tree traverse decision:
        - if the sampled value is smaller than the left-hand node value we traverse the left-hand path and leave
            the sampled value as it is.
        - if the sampled value is larger than the left-hand node value we traverse the right-hand path and
            subtract the value of the left-hand node from the sampled value

        Parameters
        ----------
        idx: int
            index of the current node
        sampled_value: float
            sampled value

        Returns
        -------
        int
        """

        # Binary Tree structure: children are stored as
        # 2 * parent + 1(left)/2(right)
        left = 2 * idx + 1
        right = 2 * idx + 2

        # Decision 1
        if left >= len(self.tree):
            return idx

        # Decision 2
        left_node = self.tree[left]
        if sampled_value <= left_node:
            return self._retrieve_leaf(left, sampled_value)
        else:
            return self._retrieve_leaf(right, sampled_value - left_node)

    def total(self) -> float:
        """
        Return the node value of the top-parent node which is equal to the sum of all leaf nodes. This serves as the
        range from which a sampled value needs to be drawn

        Returns
        -------
        float
            node value of top-parent node <=> sum of all leaf nodes
        """
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        """
        Store (priority, data) sample and update the parent nodes accordingly

        Parameters
        ----------
        priority: float
        data: Any

        Returns
        -------
        None
        """
        print('added item with priority', priority)
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, priority)

        # push write index by one
        self.write = (self.write + 1) % self.capacity

        # track number of (open) entries
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """
        Update upstream tree nodes

        Parameters
        ----------
        idx: int
        priority: float

        Returns
        -------
        None
        """
        delta_priority = priority - self.tree[idx]

        # Update leaf
        self.tree[idx] = priority

        # Recursive updates for parents until top-node
        self._propagate(idx, delta_priority)

    def get(self, sampled_value: float) -> Any:
        """
        Get the leaf-index/leaf-priority/data value tuple for a sampled value

        Parameters
        ----------
        sampled_value: float
            Sampled value of the continuous range that is the sum of all priorities in the tree <=> top-parent node

        Returns
        -------
        Tuple[int, float, Any]
            leaf_index: int
            leaf_priority: float
            data: Any
        """

        # Recursively retrieve the matching leaf-node
        leaf_index = self._retrieve_leaf(0, sampled_value)

        # Match leaf index to data index
        data_index = leaf_index - self.capacity + 1

        return self.data[data_index]

    def __len__(self):
        return self.n_entries


if __name__ == '__main__':
    # Test the sum tree
    tree = SumTree(10)
    objects = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    priorities = list(range(1, 11))
    for o, p in zip(objects, priorities):
        tree.add(priority=p, data=o)

    n = 100000
    test_array = np.zeros(shape=n, dtype=object)
    sampling_range = tree.total()
    for i in range(n):
        s = np.random.random() * sampling_range

        test_array[i] = tree.get(sampled_value=s)

    hist = np.histogram(test_array, bins=objects)
    print(hist)
