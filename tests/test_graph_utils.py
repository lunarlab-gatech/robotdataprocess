import itertools
import numpy as np
import os
from robotdataprocess.utils.graph_utils import kruskal_connection_weights, kruskal_spanning_forest
import unittest


@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestKruskalSpanningForest(unittest.TestCase):
    """ kruskal_spanning_forest: edges taken by (weight, edge); an edge is kept only if its nodes aren't already
    connected through edges kept before it, so the result is a spanning forest of the input. """

    def _connected_groups(self, edges) -> set:
        """ The sets of nodes connected by edges (brute force, independent of the union-find under test). """
        groups: list = []
        for a, b in edges:
            touching = [group for group in groups if a in group or b in group]
            merged = set().union({a, b}, *touching)
            groups = [group for group in groups if group not in touching] + [merged]
        return {frozenset(group) for group in groups}

    def test_cycle_closing_edge_dropped(self):
        kept = kruskal_spanning_forest({('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 2, ('c', 'd'): 3})
        self.assertEqual(kept, {('a', 'b'), ('b', 'c'), ('c', 'd')})

    def test_ties_broken_by_edge(self):
        kept = kruskal_spanning_forest({('b', 'c'): 1, ('a', 'c'): 1, ('a', 'b'): 1})  # inserted in reverse order
        self.assertEqual(kept, {('a', 'b'), ('a', 'c')})

    def test_weight_before_edge(self):
        kept = kruskal_spanning_forest({('a', 'b'): 2, ('b', 'c'): 2, ('a', 'c'): 1})
        self.assertEqual(kept, {('a', 'c'), ('a', 'b')})

    def test_separate_groups_joined_later(self):
        kept = kruskal_spanning_forest({('a', 'b'): 1, ('c', 'd'): 1, ('b', 'c'): 2, ('a', 'd'): 3})
        self.assertEqual(kept, {('a', 'b'), ('c', 'd'), ('b', 'c')})

    def test_no_edges_and_one_edge(self):
        self.assertEqual(kruskal_spanning_forest({}), set())
        self.assertEqual(kruskal_spanning_forest({('a', 'b'): 4}), {('a', 'b')})

    def test_random_inputs_give_spanning_forest(self):
        rng = np.random.RandomState(0)
        nodes = list('abcdefg')
        all_edges = list(itertools.combinations(nodes, 2))
        for trial in range(50):
            with self.subTest(trial=trial):
                chosen = [all_edges[k] for k in rng.choice(len(all_edges), size=rng.randint(1, len(all_edges)), replace=False)]
                edge_weights = {edge: int(rng.randint(1, 4)) for edge in chosen}
                kept = kruskal_spanning_forest(edge_weights)
                self.assertTrue(kept <= set(edge_weights))
                self.assertEqual(self._connected_groups(kept), self._connected_groups(edge_weights))
                num_nodes = len({node for edge in edge_weights for node in edge})
                self.assertEqual(len(kept), num_nodes - len(self._connected_groups(edge_weights)))  # a forest: no cycles



@unittest.skipIf(os.getenv("SKIP_PURE_PYTHON_TESTS") == "True", "Skipping pure python tests")
class TestKruskalConnectionWeights(unittest.TestCase):
    """ kruskal_connection_weights: each connected node pair (sorted) mapped to the weight of the edge that first
    connected their groups; never-connected pairs absent. """

    def _brute_force(self, edge_weights: dict) -> dict:
        """ Smallest weight w such that edges of weight <= w connect each pair (independent of the code under test). """
        nodes = sorted({node for edge in edge_weights for node in edge})
        expected = {}
        for weight in sorted(set(edge_weights.values())):
            neighbours = {node: set() for node in nodes}
            for (a, b), edge_weight in edge_weights.items():
                if edge_weight <= weight:
                    neighbours[a].add(b)
                    neighbours[b].add(a)
            for start in nodes:
                reached, frontier = {start}, [start]
                while frontier:
                    for nxt in neighbours[frontier.pop()]:
                        if nxt not in reached:
                            reached.add(nxt)
                            frontier.append(nxt)
                for other in reached:
                    if start < other:
                        expected.setdefault((start, other), weight)
        return expected

    def test_groups_joined_later(self):
        weights = kruskal_connection_weights({('a', 'b'): 1, ('c', 'd'): 1, ('b', 'c'): 2})
        self.assertEqual(weights, {('a', 'b'): 1, ('c', 'd'): 1, ('a', 'c'): 2, ('a', 'd'): 2, ('b', 'c'): 2, ('b', 'd'): 2})

    def test_cycle_closing_edge_does_not_restamp(self):
        weights = kruskal_connection_weights({('a', 'b'): 1, ('b', 'c'): 2, ('a', 'c'): 3})
        self.assertEqual(weights, {('a', 'b'): 1, ('a', 'c'): 2, ('b', 'c'): 2})

    def test_unconnected_pairs_absent_and_empty_input(self):
        weights = kruskal_connection_weights({('a', 'b'): 1, ('c', 'd'): 2})
        self.assertEqual(weights, {('a', 'b'): 1, ('c', 'd'): 2})
        self.assertEqual(kruskal_connection_weights({}), {})

    def test_random_inputs_match_brute_force(self):
        rng = np.random.RandomState(1)
        all_edges = list(itertools.combinations('abcdefg', 2))
        for trial in range(50):
            with self.subTest(trial=trial):
                chosen = [all_edges[k] for k in rng.choice(len(all_edges), size=rng.randint(1, len(all_edges)), replace=False)]
                edge_weights = {edge: int(rng.randint(1, 5)) for edge in chosen}
                self.assertEqual(kruskal_connection_weights(edge_weights), self._brute_force(edge_weights))


if __name__ == '__main__':
    unittest.main()
