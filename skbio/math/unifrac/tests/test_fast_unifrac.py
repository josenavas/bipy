#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# -----------------------------------------------------------------------------

from unittest import TestCase, main
import numpy as np
from numpy.testing import assert_almost_equal
from itertools import izip

from skbio.core.distance import DistanceMatrix, DissimilarityMatrix
from skbio.core.tree import TreeNode
from skbio.math.unifrac.fast_unifrac import (FastUniFrac,
                                              UnweightedFastUniFrac,
                                              WeightedFastUniFrac,
                                              CorrectedWeightedFastUniFrac,
                                              GFastUniFrac,
                                              UnnormalizedGFastUnifrac,
                                              UnnormalizedUnweightedFastUniFrac
                                              )


class BaseFastUniFracTests(TestCase):
    """"""
    def setUp(self):
        """Define test variables"""
        # Test case 1
        self.abund_mtx = np.array([[1, 1, 0, 0, 0],
                                   [0, 1, 1, 3, 0],
                                   [2, 0, 0, 0, 1]])
        self.tree_str = "((a:1,b:2):4,(c:3,(d:1,e:1):2):3);"
        self.tree = TreeNode.from_newick(self.tree_str)
        self.sample_ids = ['A', 'B', 'C']
        self.taxon_ids = ['a', 'b', 'c', 'd', 'e']
        # Test case 2
        self.l19_data = np.array([[7, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [4, 2, 0, 0, 0, 1, 0, 0, 0],
                                  [2, 4, 0, 0, 0, 1, 0, 0, 0],
                                  [1, 7, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 8, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 7, 1, 0, 0, 0, 0, 0, 0],
                                  [0, 4, 2, 0, 0, 0, 2, 0, 0],
                                  [0, 2, 4, 0, 0, 0, 1, 0, 0],
                                  [0, 1, 7, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 8, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 7, 1, 0, 0, 0, 0, 0],
                                  [0, 0, 4, 2, 0, 0, 0, 3, 0],
                                  [0, 0, 2, 4, 0, 0, 0, 1, 0],
                                  [0, 0, 1, 7, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 8, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 7, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 4, 2, 0, 0, 0, 4],
                                  [0, 0, 0, 2, 4, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 7, 0, 0, 0, 0]])
        self.l19_tree_str = ("((((tax7:0.1,tax3:0.2):.98,tax8:.3, tax4:.3):.4,"
                             "((tax1:0.3, tax6:.09):0.43,tax2:0.4):0.5):.2,"
                             "(tax9:0.3, endbigtaxon:.08));")
        self.l19_tree = TreeNode.from_newick(self.l19_tree_str)
        self.l19_sample_ids = ['sam1', 'sam2', 'sam3', 'sam4', 'sam5', 'sam6',
                               'sam7', 'sam8', 'sam9', 'sam_middle', 'sam11',
                               'sam12', 'sam13', 'sam14', 'sam15', 'sam16',
                               'sam17', 'sam18', 'sam19']
        self.l19_taxon_ids = ['tax1', 'tax2', 'tax3', 'tax4', 'endbigtaxon',
                              'tax6', 'tax7', 'tax8', 'tax9']


class FastUniFracTests(BaseFastUniFracTests):
    """"""
    def test_init(self):
        """Correctly initializes the FastUnifrac object"""
        # Test case 1
        obs = FastUniFrac(self.tree, self.abund_mtx, self.sample_ids,
                          self.taxon_ids)
        # Expected envs attribute
        exp_envs = {'a': {'A': 1, 'C': 2},
                    'b': {'A': 1, 'B': 1},
                    'c': {'B': 1},
                    'd': {'B': 3},
                    'e': {'C': 1}}
        self.assertEqual(obs._envs, exp_envs)
        # Expected tree attribute
        exp_tree = "((a:1.0,b:2.0):4.0,(c:3.0,(d:1.0,e:1.0):2.0):3.0);"
        self.assertEqual(str(obs._tree), exp_tree)
        # Expected node index and nodes attributes
        exp_node_index, exp_nodes = self.tree.index_tree()
        self.assertEqual(obs._nodes, exp_nodes)
        self.assertEqual(obs._node_index.keys(), exp_node_index.keys())
        for k in obs._node_index:
            self.assertEqual(str(obs._node_index[k]), str(exp_node_index[k]))
        # Expected unique envs attribute
        exp_unique_envs = ['A', 'B', 'C']
        self.assertEqual(obs._unique_envs, exp_unique_envs)
        # Expected env to index attribute
        exp_env_to_index = {'A': 0, 'B': 1, 'C': 2}
        self.assertEqual(obs._env_to_index, exp_env_to_index)
        # Expected count array attribute
        exp_count_array = np.array([[1, 0, 2],
                                    [1, 1, 0],
                                    [0, 3, 0],
                                    [0, 0, 1],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        assert_almost_equal(obs._count_array, exp_count_array)
        # Expected env names attribute
        exp_env_names = ['A', 'B', 'C']
        self.assertEqual(obs._env_names, exp_env_names)
        # Expected branch lengths attribute
        exp_branch_lengths = np.array([1., 2., 1., 1., 3., 2., 4., 3., 0.])
        assert_almost_equal(obs._branch_lengths, exp_branch_lengths)
        # Expected bound indices attribute
        exp_bound_indices = [
            (np.array([0., 0., 0.]), np.array([[0., 3., 0.], [0., 0., 1.]])),
            (np.array([0., 0., 0.]), np.array([[1., 0., 2.], [1., 1., 0.]])),
            (np.array([0., 0., 0.]), np.array([[0., 1., 0.], [0., 0., 0.]])),
            (np.array([0., 0., 0.]), np.array([[0., 0., 0.], [0., 0., 0.]]))]
        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])

        # Test case 2
        obs = FastUniFrac(self.l19_tree, self.l19_data, self.l19_sample_ids,
                          self.l19_taxon_ids)
        # Expected envs attribute
        exp_envs = {'endbigtaxon': {'sam17': 2, 'sam16': 1,
                                    'sam19': 7, 'sam18': 4},
                    'tax9': {'sam17': 4, 'sam18': 1},
                    'tax8': {'sam13': 1, 'sam12': 3},
                    'tax1': {'sam3': 2, 'sam2': 4, 'sam1': 7, 'sam4': 1},
                    'tax3': {'sam14': 1, 'sam13': 2, 'sam_middle': 8,
                             'sam11': 7, 'sam12': 4, 'sam9': 7,
                             'sam8': 4, 'sam7': 2, 'sam6': 1},
                    'tax2': {'sam9': 1, 'sam8': 2, 'sam3': 4,
                             'sam2': 2, 'sam1': 1, 'sam7': 4,
                             'sam6': 7, 'sam5': 8, 'sam4': 7},
                    'tax4': {'sam19': 1, 'sam18': 2, 'sam17': 4,
                             'sam16': 7, 'sam15': 8, 'sam14': 7,
                             'sam13': 4, 'sam12': 2, 'sam11': 1},
                    'tax7': {'sam8': 1, 'sam7': 2},
                    'tax6': {'sam3': 1, 'sam2': 1}}
        self.assertEqual(obs._envs, exp_envs)
        # Expected tree attribute
        exp_tree = ("((((tax7:0.1,tax3:0.2):0.98,tax8:0.3,tax4:0.3):0.4,"
                    "((tax1:0.3,tax6:0.09):0.43,tax2:0.4):0.5):0.2,"
                    "(tax9:0.3,endbigtaxon:0.08));")
        self.assertEqual(str(obs._tree), exp_tree)
        # Expected node index and nodes attributes
        exp_node_index, exp_nodes = self.l19_tree.index_tree()
        self.assertEqual(obs._nodes, exp_nodes)
        self.assertEqual(obs._node_index.keys(), exp_node_index.keys())
        for k in obs._node_index:
            self.assertEqual(str(obs._node_index[k]), str(exp_node_index[k]))
        # Expected unique envs attribute
        exp_unique_envs = ['sam1', 'sam11', 'sam12', 'sam13', 'sam14', 'sam15',
                           'sam16', 'sam17', 'sam18', 'sam19', 'sam2', 'sam3',
                           'sam4', 'sam5', 'sam6', 'sam7', 'sam8', 'sam9',
                           'sam_middle']
        self.assertEqual(obs._unique_envs, exp_unique_envs)
        # Expected env to index attribute
        exp_env_to_index = {'sam19': 9, 'sam18': 8, 'sam17': 7, 'sam16': 6,
                            'sam15': 5, 'sam14': 4, 'sam13': 3, 'sam12': 2,
                            'sam11': 1, 'sam_middle': 18, 'sam9': 17,
                            'sam8': 16, 'sam3': 11, 'sam2': 10, 'sam1': 0,
                            'sam7': 15, 'sam6': 14, 'sam5': 13, 'sam4': 12}
        self.assertEqual(obs._env_to_index, exp_env_to_index)
        # Expected count array attribute
        exp_count_array = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0],
             [0, 7, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 7, 8],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 2, 4, 7, 8, 7, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 7, 8, 7, 4, 2, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 2, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        assert_almost_equal(obs._count_array, exp_count_array)
        # Expected env names attribute
        exp_env_names = ['sam1', 'sam11', 'sam12', 'sam13', 'sam14', 'sam15',
                         'sam16', 'sam17', 'sam18', 'sam19', 'sam2', 'sam3',
                         'sam4', 'sam5', 'sam6', 'sam7', 'sam8', 'sam9',
                         'sam_middle']
        self.assertEqual(obs._env_names, exp_env_names)
        # Expected branch lengths attribute
        exp_branch_lengths = np.array([0.1, 0.2, 0.98, 0.3, 0.3, 0.3, 0.09,
                                       0.43, 0.4, 0.4, 0.5, 0.3, 0.08, 0.2,
                                       0., 0.])
        assert_almost_equal(obs._branch_lengths, exp_branch_lengths)
        # Expected bound indices attribute
        exp_bound_indices = [
            (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 2, 1, 0, 0],
                       [0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 2, 4, 7, 8]])),
            (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        4, 2, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 3, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 2, 4, 7, 8, 7, 4, 2, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        2, 4, 7, 8, 7, 4, 2, 1, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 4, 1, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 2, 4, 7,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]]))
            ]
        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])

    def test_make_envs(self):
        """FastUniFrac._make_envs already tested in test_init."""
        pass

    def test_make_subtree(self):
        """FastUniFrac._make_subtree already tested in test_init."""
        pass

    def test_index_envs(self):
        """FastUniFrac._index_envs already tested in test_init."""
        pass

    def test_bind_to_array(self):
        """FastUniFrac._bind_to_array already tested in test_init."""
        pass

    def test_traverse_reduce(self):
        """FastUniFrac._traverse_reduce already tested in test_bool_descendants
        and test_sum_descendants"""
        pass

    def test_bool_descendants(self):
        """Correctly updates the bound indices attribute"""
        # Test case 1
        obs = FastUniFrac(self.tree, self.abund_mtx, self.sample_ids,
                          self.taxon_ids)
        obs._bool_descendants()
        exp_bound_indices = [
            (np.array([0, 1, 1]), np.array([[0, 3, 0], [0, 0, 1]])),
            (np.array([1, 1, 1]), np.array([[1, 0, 2], [1, 1, 0]])),
            (np.array([0, 1, 1]), np.array([[0, 1, 0], [0, 1, 1]])),
            (np.array([1, 1, 1]), np.array([[1, 1, 1], [0, 1, 1]]))]
        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])
        # Test case 2
        obs = FastUniFrac(self.l19_tree, self.l19_data, self.l19_sample_ids,
                          self.l19_taxon_ids)
        obs._bool_descendants()
        exp_bound_indices = [
            (np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 1, 1, 1, 1, 1]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 2, 1, 0, 0],
                       [0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 2, 4, 7, 8]])),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 0, 0, 0, 0, 0, 0]),
             np.array([[7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        4, 2, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       0, 0, 0, 0, 1, 1, 1, 1, 1]),
             np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [0, 0, 3, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 2, 4, 7, 8, 7, 4, 2, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 0]),
             np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        2, 4, 7, 8, 7, 4, 2, 1, 0]])),
            (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]),
             np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1, 1, 1, 1, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 4, 1, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 2, 4, 7,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]),
             np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]]))]
        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])

    def test_sum_descendants(self):
        """Correctly updates the bound indices attribute"""
        # Test case 1
        obs = FastUniFrac(self.tree, self.abund_mtx, self.sample_ids,
                          self.taxon_ids)
        obs._sum_descendants()
        exp_bound_indices = [
            (np.array([0, 3, 1]), np.array([[0, 3, 0], [0, 0, 1]])),
            (np.array([2, 1, 2]), np.array([[1, 0, 2], [1, 1, 0]])),
            (np.array([0, 4, 1]), np.array([[0, 1, 0], [0, 3, 1]])),
            (np.array([2, 5, 3]), np.array([[2, 1, 2], [0, 4, 1]]))]
        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])
        # Test case 2
        obs = FastUniFrac(self.l19_tree, self.l19_data, self.l19_sample_ids,
                          self.l19_taxon_ids)
        obs._sum_descendants()
        exp_bound_indices = [
            (np.array([0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 1, 4, 5, 7, 8]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 2, 1, 0, 0],
                       [0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 2, 4, 7, 8]])),
            (np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       5, 3, 1, 0, 0, 0, 0, 0, 0]),
             np.array([[7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        4, 2, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([0, 8, 9, 7, 8, 8, 7, 4, 2, 1,
                       0, 0, 0, 0, 1, 4, 5, 7, 8]),
             np.array([[0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 4, 5, 7, 8],
                       [0, 0, 3, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 2, 4, 7, 8, 7, 4, 2, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       7, 7, 8, 8, 7, 4, 2, 1, 0]),
             np.array([[7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        5, 3, 1, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        2, 4, 7, 8, 7, 4, 2, 1, 0]])),
            (np.array([8, 8, 9, 7, 8, 8, 7, 4, 2, 1,
                       7, 7, 8, 8, 8, 8, 7, 8, 8]),
             np.array([[0, 8, 9, 7, 8, 8, 7, 4, 2, 1,
                        0, 0, 0, 0, 1, 4, 5, 7, 8],
                       [8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        7, 7, 8, 8, 7, 4, 2, 1, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 1, 6, 5, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 4, 1, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 2, 4, 7,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([8, 8, 9, 7, 8, 8, 8, 10, 7, 8,
                       7, 7, 8, 8, 8, 8, 7, 8, 8]),
             np.array([[8, 8, 9, 7, 8, 8, 7, 4, 2, 1,
                        7, 7, 8, 8, 8, 8, 7, 8, 8],
                       [0, 0, 0, 0, 0, 0, 1, 6, 5, 7,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]]))]
        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])

    def test_symetric_matrix(self):
        """Raises an error due to missing metric"""
        obs = FastUniFrac(self.tree, self.abund_mtx, self.sample_ids,
                          self.taxon_ids)
        with self.assertRaises(NotImplementedError):
            obs._symmetric_matrix()

    def test_asymmetric_matrix(self):
        """Raises an error due to missing metric"""
        obs = FastUniFrac(self.tree, self.abund_mtx, self.sample_ids,
                          self.taxon_ids)
        with self.assertRaises(NotImplementedError):
            obs._asymmetric_matrix()

    def test_metric(self):
        """This is a base class should raise a NotImplementedError"""
        obs = FastUniFrac(self.tree, self.abund_mtx, self.sample_ids,
                          self.taxon_ids)
        with self.assertRaises(NotImplementedError):
            obs._metric(0, 0)

    def test_matrix(self):
        """This is a base class should raise a NotImplementedError"""
        obs = FastUniFrac(self.tree, self.abund_mtx, self.sample_ids,
                          self.taxon_ids)
        with self.assertRaises(NotImplementedError):
            obs.matrix()


class UnweightedFastUniFracTests(BaseFastUniFracTests):
    """"""
    def test_init(self):
        """"""
        # Test case 1
        obs = UnweightedFastUniFrac(self.tree, self.abund_mtx,
                                    self.sample_ids, self.taxon_ids)
        exp_bound_indices = [
            (np.array([0, 1, 1]), np.array([[0, 3, 0], [0, 0, 1]])),
            (np.array([1, 1, 1]), np.array([[1, 0, 2], [1, 1, 0]])),
            (np.array([0, 1, 1]), np.array([[0, 1, 0], [0, 1, 1]])),
            (np.array([1, 1, 1]), np.array([[1, 1, 1], [0, 1, 1]]))]
        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])
        # Test case 2
        obs = UnweightedFastUniFrac(self.l19_tree, self.l19_data,
                                    self.l19_sample_ids, self.l19_taxon_ids)
        exp_bound_indices = [
            (np.array([0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 1, 1, 1, 1, 1]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 2, 1, 0, 0],
                       [0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 2, 4, 7, 8]])),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 0, 0, 0, 0, 0, 0]),
             np.array([[7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        4, 2, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       0, 0, 0, 0, 1, 1, 1, 1, 1]),
             np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [0, 0, 3, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 2, 4, 7, 8, 7, 4, 2, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       1, 1, 1, 1, 1, 1, 1, 1, 0]),
             np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        2, 4, 7, 8, 7, 4, 2, 1, 0]])),
            (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]),
             np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 0, 0, 1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 1, 1, 1, 1, 1, 1, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 4, 1, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 2, 4, 7,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1]),
             np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]]))]

        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])

    def test_matrix(self):
        """"""
        # Test case 1
        unif = UnweightedFastUniFrac(self.tree, self.abund_mtx,
                                     self.sample_ids, self.taxon_ids)
        obs = unif.matrix()
        # The object has the correct type
        self.assertEqual(type(obs), DistanceMatrix)
        # Expected matrix ids
        exp_ids = tuple(sorted(self.sample_ids))
        self.assertEqual(obs.ids, exp_ids)
        # Expected dtype
        self.assertEqual(obs.dtype, np.float64)
        # Expected matrix data
        exp_data = np.array([[0., 0.625, 0.6153846153],
                             [0.625, 0., 0.4705882352],
                             [0.6153846153, 0.4705882352, 0.]])
        assert_almost_equal(obs.data, exp_data)
        # Expected shape
        self.assertEqual(obs.shape, exp_data.shape)
        # Expected size
        self.assertEqual(obs.size, exp_data.size)

        # Test case 2
        unif = UnweightedFastUniFrac(self.l19_tree, self.l19_data,
                                     self.l19_sample_ids, self.l19_taxon_ids)
        obs = unif.matrix()
        # The object has the correct type
        self.assertEqual(type(obs), DistanceMatrix)
        # Expected matrix ids
        exp_ids = tuple(sorted(self.l19_sample_ids))
        self.assertEqual(obs.ids, exp_ids)
        # Expected dtype
        self.assertEqual(obs.dtype, np.float64)
        # Expected matrix data
        exp_data = np.array([
            [0., 0.94609164, 0.95012469, 0.95012469, 0.94609164, 0.92094862,
             0.92337165, 0.93127148, 0.93127148, 0.92337165, 0.046875,
             0.046875, 0., 0.3989071, 0.67741935, 0.68660969, 0.68660969,
             0.67741935, 0.94134897],
            [0.94609164, 0., 0.12605042, 0.12605042, 0., 0.56730769,
             0.58333333, 0.63414634, 0.63414634, 0.58333333, 0.94736842,
             0.94736842, 0.94609164, 0.93288591, 0.40268456, 0.42207792,
             0.42207792, 0.40268456, 0.14423077],
            [0.95012469, 0.12605042, 0., 0., 0.12605042, 0.62184874,
             0.63414634, 0.67391304, 0.67391304, 0.63414634, 0.95121951,
             0.95121951, 0.95012469, 0.93902439, 0.45731707, 0.47337278,
             0.47337278, 0.45731707, 0.25210084],
            [0.95012469, 0.12605042, 0., 0., 0.12605042, 0.62184874,
             0.63414634, 0.67391304, 0.67391304, 0.63414634, 0.95121951,
             0.95121951, 0.95012469, 0.93902439, 0.45731707, 0.47337278,
             0.47337278, 0.45731707, 0.25210084],
            [0.94609164, 0., 0.12605042, 0.12605042, 0., 0.56730769,
             0.58333333, 0.63414634, 0.63414634, 0.58333333, 0.94736842,
             0.94736842, 0.94609164, 0.93288591, 0.40268456, 0.42207792,
             0.42207792, 0.40268456, 0.14423077],
            [0.92094862, 0.56730769, 0.62184874, 0.62184874, 0.56730769, 0.,
             0.08163265, 0.296875, 0.296875, 0.08163265, 0.92366412,
             0.92366412, 0.92094862, 0.88888889, 0.79865772, 0.80519481,
             0.80519481, 0.79865772, 0.71153846],
            [0.92337165, 0.58333333, 0.63414634, 0.63414634, 0.58333333,
             0.08163265, 0., 0.234375, 0.234375, 0., 0.92592593, 0.92592593,
             0.92337165, 0.89361702, 0.80392157, 0.81012658, 0.81012658,
             0.80392157, 0.72222222],
            [0.93127148, 0.63414634, 0.67391304, 0.67391304, 0.63414634,
             0.296875, 0.234375, 0., 0., 0.234375, 0.93333333, 0.93333333,
             0.93127148, 0.90825688, 0.82142857, 0.8265896, 0.8265896,
             0.82142857, 0.75609756],
            [0.93127148, 0.63414634, 0.67391304, 0.67391304, 0.63414634,
             0.296875, 0.234375, 0., 0., 0.234375, 0.93333333, 0.93333333,
             0.93127148, 0.90825688, 0.82142857, 0.8265896, 0.8265896,
             0.82142857, 0.75609756],
            [0.92337165, 0.58333333, 0.63414634, 0.63414634, 0.58333333,
             0.08163265, 0., 0.234375, 0.234375, 0., 0.92592593, 0.92592593,
             0.92337165, 0.89361702, 0.80392157, 0.81012658, 0.81012658,
             0.80392157, 0.72222222],
            [0.046875, 0.94736842, 0.95121951, 0.95121951, 0.94736842,
             0.92366412, 0.92592593, 0.93333333, 0.93333333, 0.92592593, 0.,
             0., 0.046875, 0.42708333, 0.68571429, 0.69444444, 0.69444444,
             0.68571429, 0.94285714],
            [0.046875, 0.94736842, 0.95121951, 0.95121951, 0.94736842,
             0.92366412, 0.92592593, 0.93333333, 0.93333333, 0.92592593, 0.,
             0., 0.046875, 0.42708333, 0.68571429, 0.69444444, 0.69444444,
             0.68571429, 0.94285714],
            [0., 0.94609164, 0.95012469, 0.95012469, 0.94609164, 0.92094862,
             0.92337165, 0.93127148, 0.93127148, 0.92337165, 0.046875,
             0.046875, 0., 0.3989071, 0.67741935, 0.68660969, 0.68660969,
             0.67741935, 0.94134897],
            [0.3989071, 0.93288591, 0.93902439, 0.93902439, 0.93288591,
             0.88888889, 0.89361702, 0.90825688, 0.90825688, 0.89361702,
             0.42708333, 0.42708333, 0.3989071, 0., 0.58955224, 0.60431655,
             0.60431655, 0.58955224, 0.92537313],
            [0.67741935, 0.40268456, 0.45731707, 0.45731707, 0.40268456,
             0.79865772, 0.80392157, 0.82142857, 0.82142857, 0.80392157,
             0.68571429, 0.68571429, 0.67741935, 0.58955224, 0., 0.03597122,
             0.03597122, 0., 0.3358209],
            [0.68660969, 0.42207792, 0.47337278, 0.47337278, 0.42207792,
             0.80519481, 0.81012658, 0.8265896, 0.8265896, 0.81012658,
             0.69444444, 0.69444444, 0.68660969, 0.60431655, 0.03597122, 0.,
             0., 0.03597122, 0.35971223],
            [0.68660969, 0.42207792, 0.47337278, 0.47337278, 0.42207792,
             0.80519481, 0.81012658, 0.8265896, 0.8265896, 0.81012658,
             0.69444444, 0.69444444, 0.68660969, 0.60431655, 0.03597122, 0.,
             0., 0.03597122, 0.35971223],
            [0.67741935, 0.40268456, 0.45731707, 0.45731707, 0.40268456,
             0.79865772, 0.80392157, 0.82142857, 0.82142857, 0.80392157,
             0.68571429, 0.68571429, 0.67741935, 0.58955224, 0., 0.03597122,
             0.03597122, 0., 0.3358209],
            [0.94134897, 0.14423077, 0.25210084, 0.25210084, 0.14423077,
             0.71153846, 0.72222222, 0.75609756, 0.75609756, 0.72222222,
             0.94285714, 0.94285714, 0.94134897, 0.92537313, 0.3358209,
             0.35971223, 0.35971223, 0.3358209, 0.]])
        assert_almost_equal(obs.data, exp_data)
        # Expected shape
        self.assertEqual(obs.shape, exp_data.shape)
        # Expected size
        self.assertEqual(obs.size, exp_data.size)

    def test_metric(self):
        """UnweightedFastUniFrac._metric already tested on test_matrix"""
        pass


class UnnormalizedUnweightedFastUniFracTests(BaseFastUniFracTests):
    """"""
    def test_matrix(self):
        """"""
        # Test case 1
        unn_unif = UnnormalizedUnweightedFastUniFrac(self.tree, self.abund_mtx,
                                                     self.sample_ids,
                                                     self.taxon_ids)
        obs = unn_unif.matrix()
        # The object has the correct type
        self.assertEqual(type(obs), DistanceMatrix)
        # Expected matrix ids
        exp_ids = tuple(sorted(self.sample_ids))
        self.assertEqual(obs.ids, exp_ids)
        # Expected dtype
        self.assertEqual(obs.dtype, np.float64)
        # Expected matrix data
        exp_data = np.array([[0., 0.5882352941, 0.4705882352],
                             [0.5882352941, 0., 0.4705882352],
                             [0.4705882352, 0.4705882352, 0.]])
        assert_almost_equal(obs.data, exp_data)
        # Expected shape
        self.assertEqual(obs.shape, exp_data.shape)
        # Expected size
        self.assertEqual(obs.size, exp_data.size)

        # Test case 2
        unn_unif = UnnormalizedUnweightedFastUniFrac(self.l19_tree,
                                                     self.l19_data,
                                                     self.l19_sample_ids,
                                                     self.l19_taxon_ids)
        obs = unn_unif.matrix()
        # The object has the correct type
        self.assertEqual(type(obs), DistanceMatrix)
        # Expected matrix ids
        exp_ids = tuple(sorted(self.l19_sample_ids))
        self.assertEqual(obs.ids, exp_ids)
        # Expected dtype
        self.assertEqual(obs.dtype, np.float64)
        # Expected matrix data
        exp_data = np.array([
            [0., 0.76637555, 0.83187773, 0.83187773, 0.76637555, 0.50873362,
             0.52620087, 0.59170306, 0.59170306, 0.52620087, 0.01965066,
             0.01965066, 0., 0.15938865, 0.50436681, 0.52620087, 0.52620087,
             0.50436681, 0.70087336],
            [0.76637555, 0., 0.06550218, 0.06550218, 0., 0.25764192,
             0.27510917, 0.34061135, 0.34061135, 0.27510917, 0.7860262,
             0.7860262, 0.76637555, 0.6069869, 0.26200873, 0.28384279,
             0.28384279, 0.26200873, 0.06550218],
            [0.83187773, 0.06550218, 0., 0., 0.06550218, 0.3231441, 0.34061135,
             0.40611354, 0.40611354, 0.34061135, 0.85152838, 0.85152838,
             0.83187773, 0.67248908, 0.32751092, 0.34934498, 0.34934498,
             0.32751092, 0.13100437],
            [0.83187773, 0.06550218, 0., 0., 0.06550218, 0.3231441, 0.34061135,
             0.40611354, 0.40611354, 0.34061135, 0.85152838, 0.85152838,
             0.83187773, 0.67248908, 0.32751092, 0.34934498, 0.34934498,
             0.32751092, 0.13100437],
            [0.76637555, 0., 0.06550218, 0.06550218, 0., 0.25764192,
             0.27510917, 0.34061135, 0.34061135, 0.27510917, 0.7860262,
             0.7860262, 0.76637555, 0.6069869, 0.26200873, 0.28384279,
             0.28384279, 0.26200873, 0.06550218],
            [0.50873362, 0.25764192, 0.3231441, 0.3231441, 0.25764192, 0.,
             0.01746725, 0.08296943, 0.08296943, 0.01746725, 0.52838428,
             0.52838428, 0.50873362, 0.34934498, 0.51965066, 0.54148472,
             0.54148472, 0.51965066, 0.3231441],
            [0.52620087, 0.27510917, 0.34061135, 0.34061135, 0.27510917,
             0.01746725, 0., 0.06550218, 0.06550218, 0., 0.54585153,
             0.54585153, 0.52620087, 0.36681223, 0.5371179, 0.55895197,
             0.55895197, 0.5371179, 0.34061135],
            [0.59170306, 0.34061135, 0.40611354, 0.40611354, 0.34061135,
             0.08296943, 0.06550218, 0., 0., 0.06550218, 0.61135371,
             0.61135371, 0.59170306, 0.43231441, 0.60262009, 0.62445415,
             0.62445415, 0.60262009, 0.40611354],
            [0.59170306, 0.34061135, 0.40611354, 0.40611354, 0.34061135,
             0.08296943, 0.06550218, 0., 0., 0.06550218, 0.61135371,
             0.61135371, 0.59170306, 0.43231441, 0.60262009, 0.62445415,
             0.62445415, 0.60262009, 0.40611354],
            [0.52620087, 0.27510917, 0.34061135, 0.34061135, 0.27510917,
             0.01746725, 0., 0.06550218, 0.06550218, 0., 0.54585153,
             0.54585153, 0.52620087, 0.36681223, 0.5371179, 0.55895197,
             0.55895197, 0.5371179, 0.34061135],
            [0.01965066, 0.7860262, 0.85152838, 0.85152838, 0.7860262,
             0.52838428, 0.54585153, 0.61135371, 0.61135371, 0.54585153, 0.,
             0., 0.01965066, 0.1790393, 0.52401747, 0.54585153, 0.54585153,
             0.52401747, 0.72052402],
            [0.01965066, 0.7860262, 0.85152838, 0.85152838, 0.7860262,
             0.52838428, 0.54585153, 0.61135371, 0.61135371, 0.54585153, 0.,
             0., 0.01965066, 0.1790393, 0.52401747, 0.54585153, 0.54585153,
             0.52401747, 0.72052402],
            [0., 0.76637555, 0.83187773, 0.83187773, 0.76637555, 0.50873362,
             0.52620087, 0.59170306, 0.59170306, 0.52620087, 0.01965066,
             0.01965066, 0., 0.15938865, 0.50436681, 0.52620087, 0.52620087,
             0.50436681, 0.70087336],
            [0.15938865, 0.6069869, 0.67248908, 0.67248908, 0.6069869,
             0.34934498, 0.36681223, 0.43231441, 0.43231441, 0.36681223,
             0.1790393, 0.1790393, 0.15938865, 0., 0.34497817, 0.36681223,
             0.36681223, 0.34497817, 0.54148472],
            [0.50436681, 0.26200873, 0.32751092, 0.32751092, 0.26200873,
             0.51965066, 0.5371179, 0.60262009, 0.60262009, 0.5371179,
             0.52401747, 0.52401747, 0.50436681, 0.34497817, 0., 0.02183406,
             0.02183406, 0., 0.19650655],
            [0.52620087, 0.28384279, 0.34934498, 0.34934498, 0.28384279,
             0.54148472, 0.55895197, 0.62445415, 0.62445415, 0.55895197,
             0.54585153, 0.54585153, 0.52620087, 0.36681223, 0.02183406, 0.,
             0., 0.02183406, 0.21834061],
            [0.52620087, 0.28384279, 0.34934498, 0.34934498, 0.28384279,
             0.54148472, 0.55895197, 0.62445415, 0.62445415, 0.55895197,
             0.54585153, 0.54585153, 0.52620087, 0.36681223, 0.02183406, 0.,
             0., 0.02183406, 0.21834061],
            [0.50436681, 0.26200873, 0.32751092, 0.32751092, 0.26200873,
             0.51965066, 0.5371179, 0.60262009, 0.60262009, 0.5371179,
             0.52401747, 0.52401747, 0.50436681, 0.34497817, 0., 0.02183406,
             0.02183406, 0., 0.19650655],
            [0.70087336, 0.06550218, 0.13100437, 0.13100437, 0.06550218,
             0.3231441, 0.34061135, 0.40611354, 0.40611354, 0.34061135,
             0.72052402, 0.72052402, 0.70087336, 0.54148472, 0.19650655,
             0.21834061, 0.21834061, 0.19650655, 0.]])
        assert_almost_equal(obs.data, exp_data)
        # Expected shape
        self.assertEqual(obs.shape, exp_data.shape)
        # Expected size
        self.assertEqual(obs.size, exp_data.size)

    def test_metric(self):
        """UnnormalizedUnweightedFastUniFrac._metric already tested on
        test_matrix
        """
        pass


class WeightedFastUniFracTests(BaseFastUniFracTests):
    """"""
    def test_init(self):
        """"""
        obs = WeightedFastUniFrac(self.l19_tree, self.l19_data,
                                  self.l19_sample_ids, self.l19_taxon_ids)
        # Expected tip indices attribute
        exp_tip_indices = [0, 1, 3, 4, 5, 6, 8, 11, 12]
        self.assertEqual(obs._tip_indices, exp_tip_indices)
        # Expected bound indices attribute
        exp_bound_indices = [
            (np.array([0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 1, 4, 5, 7, 8]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 2, 1, 0, 0],
                       [0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 2, 4, 7, 8]])),
            (np.array([7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       5, 3, 1, 0, 0, 0, 0, 0, 0]),
             np.array([[7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        4, 2, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([0, 8, 9, 7, 8, 8, 7, 4, 2, 1,
                       0, 0, 0, 0, 1, 4, 5, 7, 8]),
             np.array([[0, 7, 4, 2, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 4, 5, 7, 8],
                       [0, 0, 3, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 1, 2, 4, 7, 8, 7, 4, 2, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       7, 7, 8, 8, 7, 4, 2, 1, 0]),
             np.array([[7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        5, 3, 1, 0, 0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        2, 4, 7, 8, 7, 4, 2, 1, 0]])),
            (np.array([8, 8, 9, 7, 8, 8, 7, 4, 2, 1,
                       7, 7, 8, 8, 8, 8, 7, 8, 8]),
             np.array([[0, 8, 9, 7, 8, 8, 7, 4, 2, 1,
                        0, 0, 0, 0, 1, 4, 5, 7, 8],
                       [8, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        7, 7, 8, 8, 7, 4, 2, 1, 0]])),
            (np.array([0, 0, 0, 0, 0, 0, 1, 6, 5, 7,
                       0, 0, 0, 0, 0, 0, 0, 0, 0]),
             np.array([[0, 0, 0, 0, 0, 0, 0, 4, 1, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 2, 4, 7,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]])),
            (np.array([8, 8, 9, 7, 8, 8, 8, 10, 7, 8,
                       7, 7, 8, 8, 8, 8, 7, 8, 8]),
             np.array([[8, 8, 9, 7, 8, 8, 7, 4, 2, 1,
                        7, 7, 8, 8, 8, 8, 7, 8, 8],
                       [0, 0, 0, 0, 0, 0, 1, 6, 5, 7,
                        0, 0, 0, 0, 0, 0, 0, 0, 0]]))]
        self.assertEqual(len(obs._bound_indices), len(exp_bound_indices))
        for o, e in izip(obs._bound_indices, exp_bound_indices):
            assert_almost_equal(o[0], e[0])
            assert_almost_equal(o[1], e[1])
        # Expected tip ds attribute
        exp_tip_ds = np.array([[1.68], [1.78], [0.], [0.9], [0.9], [1.43],
                               [1.22], [0.], [1.1], [0.], [0.], [0.3], [0.08],
                               [0.], [0.], [0.]])
        assert_almost_equal(obs._tip_ds, exp_tip_ds)
        # Expected bindings attribute
        exp_bindings = [[np.array([0.]), np.array([0.])],
                        [np.array([0.]), np.array([0.])],
                        [np.array([0.]), np.array([0.])],
                        [np.array([1.68]), np.array([0.])],
                        [np.array([1.78]), np.array([0.])],
                        [np.array([0.9]), np.array([0.])],
                        [np.array([0.9]), np.array([0.])],
                        [np.array([0.]), np.array([0.])],
                        [np.array([0.]), np.array([0.])],
                        [np.array([1.43]), np.array([0.])],
                        [np.array([1.22]), np.array([0.])],
                        [np.array([1.1]), np.array([0.])],
                        [np.array([0.]), np.array([0.])],
                        [np.array([0.3]), np.array([0.])],
                        [np.array([0.08]), np.array([0.])]]
        assert_almost_equal(obs._bindings, exp_bindings)

    def test_bind_to_parent_array(self):
        """WeightedFastUniFrac._bind_to_parent_array already tested on
        test_init"""
        pass

    def test_tip_distances(self):
        """WeightedFastUniFrac._tip_distances already tested on test_init"""
        pass

    def test_matrix(self):
        """"""
        wei_unif = WeightedFastUniFrac(self.l19_tree, self.l19_data,
                                       self.l19_sample_ids, self.l19_taxon_ids)
        obs = wei_unif.matrix()
        # The object has the correct type
        self.assertEqual(type(obs), DistanceMatrix)
        # Expected matrix ids
        exp_ids = tuple(sorted(self.l19_sample_ids))
        self.assertEqual(obs.ids, exp_ids)
        # Expected dtype
        self.assertEqual(obs.dtype, np.float64)
        # Expected matrix data
        exp_data = np.array([
            [0., 2.65875, 2.27986111, 2.14017857, 1.99875, 1.88875, 1.83625,
             1.72475, 1.62017857, 1.52125, 0.23732143, 0.56017857, 0.8475,
             0.98875, 1.19875, 1.80375, 2.17446429, 2.45875, 2.76875],
            [2.65875, 0., 0.63722222, 0.87214286, 1.11, 1.295, 1.3425, 1.611,
             1.59785714, 1.6275, 2.57571429, 2.48142857, 2.41125, 2.37, 2.06,
             1.205, 0.64142857, 0.2, 0.185],
            [2.27986111, 0.63722222, 0., 0.34920635, 0.67277778, 0.85777778,
             0.90527778, 1.17377778, 1.16063492, 1.24861111, 2.1968254,
             2.10253968, 2.03236111, 1.99111111, 1.68111111, 0.935, 0.84222222,
             0.83722222, 0.82222222],
            [2.14017857, 0.87214286, 0.34920635, 0., 0.32357143, 0.50857143,
             0.55607143, 0.92742857, 0.98285714, 1.10892857, 2.05714286,
             1.96285714, 1.89267857, 1.85142857, 1.54142857, 1.10642857,
             1.07714286, 1.07214286, 1.05714286],
            [1.99875, 1.11, 0.67277778, 0.32357143, 0., 0.185, 0.2325, 0.786,
             0.84142857, 0.9675, 1.91571429, 1.82142857, 1.75125, 1.71, 1.4,
             1.33, 1.315, 1.31, 1.295],
            [1.88875, 1.295, 0.85777778, 0.50857143, 0.185, 0., 0.1225, 0.676,
             0.73142857, 0.8575, 1.80571429, 1.71142857, 1.64125, 1.6, 1.585,
             1.515, 1.5, 1.495, 1.48],
            [1.83625, 1.3425, 0.90527778, 0.55607143, 0.2325, 0.1225, 0.,
             0.5535, 0.60892857, 0.735, 1.75321429, 1.65892857, 1.58875,
             1.5475, 1.5325, 1.4625, 1.4475, 1.4425, 1.5275],
            [1.72475, 1.611, 1.17377778, 0.92742857, 0.786, 0.676, 0.5535, 0.,
             0.20971429, 0.4215, 1.64171429, 1.54742857, 1.47725, 1.436, 1.421,
             1.431, 1.58742857, 1.711, 1.796],
            [1.62017857, 1.59785714, 1.16063492, 0.98285714, 0.84142857,
             0.73142857, 0.60892857, 0.20971429, 0., 0.21178571, 1.53714286,
             1.44285714, 1.37267857, 1.33142857, 1.31642857, 1.41785714,
             1.57428571, 1.69785714, 1.78285714],
            [1.52125, 1.6275, 1.24861111, 1.10892857, 0.9675, 0.8575, 0.735,
             0.4215, 0.21178571, 0., 1.43821429, 1.34392857, 1.27375, 1.2325,
             1.2175, 1.4475, 1.60392857, 1.7275, 1.8125],
            [0.23732143, 2.57571429, 2.1968254, 2.05714286, 1.91571429,
             1.80571429, 1.75321429, 1.64171429, 1.53714286, 1.43821429, 0.,
             0.32285714, 0.63589286, 0.77714286, 0.98714286, 1.59214286,
             1.96285714, 2.37571429, 2.68571429],
            [0.56017857, 2.48142857, 2.10253968, 1.96285714, 1.82142857,
             1.71142857, 1.65892857, 1.54742857, 1.44285714, 1.34392857,
             0.32285714, 0., 0.31303571, 0.45428571, 0.66428571, 1.32642857,
             1.86857143, 2.28142857, 2.59142857],
            [0.8475, 2.41125, 2.03236111, 1.89267857, 1.75125, 1.64125,
             1.58875, 1.47725, 1.37267857, 1.27375, 0.63589286, 0.31303571, 0.,
             0.14125, 0.35125, 1.25625, 1.79839286, 2.21125, 2.52125],
            [0.98875, 2.37, 1.99111111, 1.85142857, 1.71, 1.6, 1.5475, 1.436,
             1.33142857, 1.2325, 0.77714286, 0.45428571, 0.14125, 0., 0.31,
             1.215, 1.75714286, 2.17, 2.48],
            [1.19875, 2.06, 1.68111111, 1.54142857, 1.4, 1.585, 1.5325, 1.421,
             1.31642857, 1.2175, 0.98714286, 0.66428571, 0.35125, 0.31, 0.,
             0.905, 1.44714286, 1.86, 2.17],
            [1.80375, 1.205, 0.935, 1.10642857, 1.33, 1.515, 1.4625, 1.431,
             1.41785714, 1.4475, 1.59214286, 1.32642857, 1.25625, 1.215, 0.905,
             0., 0.56357143, 1.005, 1.315],
            [2.17446429, 0.64142857, 0.84222222, 1.07714286, 1.315, 1.5,
             1.4475, 1.58742857, 1.57428571, 1.60392857, 1.96285714,
             1.86857143, 1.79839286, 1.75714286, 1.44714286, 0.56357143, 0.,
             0.44142857, 0.75142857],
            [2.45875, 0.2, 0.83722222, 1.07214286, 1.31, 1.495, 1.4425, 1.711,
             1.69785714, 1.7275, 2.37571429, 2.28142857, 2.21125, 2.17, 1.86,
             1.005, 0.44142857, 0., 0.31],
            [2.76875, 0.185, 0.82222222, 1.05714286, 1.295, 1.48, 1.5275,
             1.796, 1.78285714, 1.8125, 2.68571429, 2.59142857, 2.52125, 2.48,
             2.17, 1.315, 0.75142857, 0.31, 0.]])
        assert_almost_equal(obs.data, exp_data)
        # Expected shape
        self.assertEqual(obs.shape, exp_data.shape)
        # Expected size
        self.assertEqual(obs.size, exp_data.size)

    def test_metric(self):
        """WeightedFastUniFrac._metric already tested on test_matrix"""
        pass


class CorrectedWeightedFastUniFracTests(BaseFastUniFracTests):
    """"""
    def test_matrix(self):
        """"""
        corr_wei_unif = CorrectedWeightedFastUniFrac(self.l19_tree,
                                                     self.l19_data,
                                                     self.l19_sample_ids,
                                                     self.l19_taxon_ids)
        obs = corr_wei_unif.matrix()
        # The object has the correct type
        self.assertEqual(type(obs), DistanceMatrix)
        # Expected matrix ids
        exp_ids = tuple(sorted(self.l19_sample_ids))
        self.assertEqual(obs.ids, exp_ids)
        # Expected dtype
        self.assertEqual(obs.dtype, np.float64)
        # Expected matrix data
        exp_data = np.array([
            [0., 0.86922763, 0.85073853, 0.84253076, 0.83324648, 0.82523211,
             0.83990852, 0.9151081, 0.93410893, 0.9681782, 0.08807741,
             0.2154385, 0.33498024, 0.3972878, 0.46576008, 0.64333482,
             0.734572, 0.79732469, 0.87376726],
            [0.86922763, 0., 0.215197, 0.30911392, 0.4141791, 0.50389105,
             0.54407295, 0.74376731, 0.79270021, 0.87854251, 0.86557849,
             0.86117997, 0.85771454, 0.85559567, 0.72154116, 0.39059968,
             0.19788453, 0.05943536, 0.05362319],
            [0.85073853, 0.215197, 0., 0.14296855, 0.29237084, 0.39148073,
             0.4334353, 0.65680179, 0.7090768, 0.84731385, 0.84596577,
             0.84016237, 0.83555073, 0.83271375, 0.67893202, 0.34551427,
             0.29422203, 0.28037209, 0.26772793],
            [0.84253076, 0.30911392, 0.14296855, 0., 0.14970258, 0.24791086,
             0.28532161, 0.56295525, 0.65648855, 0.8313253, 0.8372093,
             0.83071342, 0.82553158, 0.82233503, 0.65973708, 0.43111606,
             0.39559286, 0.37666248, 0.36062378],
            [0.83324648, 0.4141791, 0.29237084, 0.14970258, 0., 0.09685864,
             0.12863071, 0.52191235, 0.62065332, 0.81132075, 0.82726712,
             0.81993569, 0.81406159, 0.81042654, 0.63781321, 0.54845361,
             0.50940786, 0.48428835, 0.46415771],
            [0.82523211, 0.50389105, 0.39148073, 0.24791086, 0.09685864, 0.,
             0.07216495, 0.48424069, 0.58715596, 0.79214781, 0.81865285,
             0.8105548, 0.80404164, 0.8, 0.76019185, 0.65442765, 0.60693642,
             0.5761079, 0.55223881],
            [0.83990852, 0.54407295, 0.4334353, 0.28532161, 0.12863071,
             0.07216495, 0., 0.42790877, 0.53264605, 0.75, 0.83358805,
             0.82577778, 0.81947131, 0.81554677, 0.77301387, 0.66101695,
             0.61103573, 0.57873621, 0.59262852],
            [0.9151081, 0.74376731, 0.65680179, 0.56295525, 0.52191235,
             0.48424069, 0.42790877, 0., 0.24915139, 0.62122329, 0.91119569,
             0.90629183, 0.90227516, 0.89974937, 0.84533016, 0.74882261,
             0.76782753, 0.78092195, 0.78910369],
            [0.93410893, 0.79270021, 0.7090768, 0.65648855, 0.62065332,
             0.58715596, 0.53264605, 0.24915139, 0., 0.40094659, 0.93079585,
             0.9266055, 0.92314159, 0.92094862, 0.86000933, 0.80527383,
             0.82116244, 0.8319916, 0.83870968],
            [0.9681782, 0.87854251, 0.84731385, 0.8313253, 0.81132075,
             0.79214781, 0.75, 0.62122329, 0.40094659, 0., 0.96640269,
             0.96413016, 0.96222852, 0.96101365, 0.89031079, 0.90610329,
             0.9144777, 0.92010652, 0.92356688],
            [0.08807741, 0.86557849, 0.84596577, 0.8372093, 0.82726712,
             0.81865285, 0.83358805, 0.91119569, 0.93079585, 0.96640269, 0.,
             0.12826334, 0.2598701, 0.32304038, 0.39632922, 0.58519296,
             0.68222443, 0.79171626, 0.87037037],
            [0.2154385, 0.86117997, 0.84016237, 0.83071342, 0.81993569,
             0.8105548, 0.82577778, 0.90629183, 0.9266055, 0.96413016,
             0.12826334, 0., 0.13305503, 0.19653894, 0.27719821, 0.50503128,
             0.67145791, 0.78495945, 0.86628462],
            [0.33498024, 0.85771454, 0.83555073, 0.82553158, 0.81406159,
             0.80404164, 0.81947131, 0.90227516, 0.92314159, 0.96222852,
             0.2598701, 0.13305503, 0., 0.06302287, 0.15099409, 0.49144254,
             0.66295833, 0.77963861, 0.86307231],
            [0.3972878, 0.85559567, 0.83271375, 0.82233503, 0.81042654, 0.8,
             0.81554677, 0.89974937, 0.92094862, 0.96101365, 0.32304038,
             0.19653894, 0.06302287, 0., 0.1356674, 0.48310139, 0.65775401,
             0.7763864, 0.86111111],
            [0.46576008, 0.72154116, 0.67893202, 0.65973708, 0.63781321,
             0.76019185, 0.77301387, 0.84533016, 0.86000933, 0.89031079,
             0.39632922, 0.27719821, 0.15099409, 0.1356674, 0., 0.34807692,
             0.52500648, 0.64583333, 0.73187184],
            [0.64333482, 0.39059968, 0.34551427, 0.43111606, 0.54845361,
             0.65442765, 0.66101695, 0.74882261, 0.80527383, 0.90610329,
             0.58519296, 0.50503128, 0.49144254, 0.48310139, 0.34807692, 0.,
             0.18871083, 0.32315113, 0.41158059],
            [0.734572, 0.19788453, 0.29422203, 0.39559286, 0.50940786,
             0.60693642, 0.61103573, 0.76782753, 0.82116244, 0.9144777,
             0.68222443, 0.67145791, 0.66295833, 0.65775401, 0.52500648,
             0.18871083, 0., 0.13514105, 0.22421142],
            [0.79732469, 0.05943536, 0.28037209, 0.37666248, 0.48428835,
             0.5761079, 0.57873621, 0.78092195, 0.8319916, 0.92010652,
             0.79171626, 0.78495945, 0.77963861, 0.7763864, 0.64583333,
             0.32315113, 0.13514105, 0., 0.08920863],
            [0.87376726, 0.05362319, 0.26772793, 0.36062378, 0.46415771,
             0.55223881, 0.59262852, 0.78910369, 0.83870968, 0.92356688,
             0.87037037, 0.86628462, 0.86307231, 0.86111111, 0.73187184,
             0.41158059, 0.22421142, 0.08920863, 0.]])
        assert_almost_equal(obs.data, exp_data)
        # Expected shape
        self.assertEqual(obs.shape, exp_data.shape)
        # Expected size
        self.assertEqual(obs.size, exp_data.size)

    def test_metric(self):
        """CorrectedWeightedFastUniFrac._metric already tested on test_matrix
        """
        pass


class GFastUniFracTests(BaseFastUniFracTests):
    """"""

    def test_matrix(self):
        """"""
        g_unig = GFastUniFrac(self.l19_tree, self.l19_data,
                              self.l19_sample_ids, self.l19_taxon_ids)
        obs = g_unig.matrix()
        # The object has the correct type
        self.assertEqual(type(obs), DissimilarityMatrix)
        # Expected matrix ids
        exp_ids = tuple(sorted(self.l19_sample_ids))
        self.assertEqual(obs.ids, exp_ids)
        # Expected dtype
        self.assertEqual(obs.dtype, np.float64)
        # Expected matrix data
        exp_data = np.array([
            [0., 0.4393531, 0.40648379, 0.40648379, 0.4393531, 0.64426877,
             0.62452107, 0.56013746, 0.56013746, 0.62452107, 0., 0., 0.,
             0.3989071, 0.21407625, 0.20797721, 0.20797721, 0.21407625,
             0.47800587],
            [0.50673854, 0., 0., 0., 0., 0.56730769, 0.5462963, 0.4796748,
             0.4796748, 0.5462963, 0.49473684, 0.49473684, 0.50673854,
             0.63087248, 0.10067114, 0.0974026, 0.0974026, 0.10067114,
             0.14423077],
            [0.5436409, 0.12605042, 0., 0., 0.12605042, 0.62184874, 0.60162602,
             0.53623188, 0.53623188, 0.60162602, 0.53170732, 0.53170732,
             0.5436409, 0.66463415, 0.18292683, 0.17751479, 0.17751479,
             0.18292683, 0.25210084],
            [0.5436409, 0.12605042, 0., 0., 0.12605042, 0.62184874, 0.60162602,
             0.53623188, 0.53623188, 0.60162602, 0.53170732, 0.53170732,
             0.5436409, 0.66463415, 0.18292683, 0.17751479, 0.17751479,
             0.18292683, 0.25210084],
            [0.50673854, 0., 0., 0., 0., 0.56730769, 0.5462963, 0.4796748,
             0.4796748, 0.5462963, 0.49473684, 0.49473684, 0.50673854,
             0.63087248, 0.10067114, 0.0974026, 0.0974026, 0.10067114,
             0.14423077],
            [0.27667984, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.26717557,
             0.26717557, 0.27667984, 0.38888889, 0.10067114, 0.0974026,
             0.0974026, 0.10067114, 0.14423077],
            [0.29885057, 0.03703704, 0.03252033, 0.03252033, 0.03703704,
             0.08163265, 0., 0., 0., 0., 0.28888889, 0.28888889, 0.29885057,
             0.41489362, 0.12418301, 0.12025316, 0.12025316, 0.12418301,
             0.17592593],
            [0.37113402, 0.15447154, 0.13768116, 0.13768116, 0.15447154,
             0.296875, 0.234375, 0., 0., 0.234375, 0.36, 0.36, 0.37113402,
             0.49541284, 0.20238095, 0.19653179, 0.19653179, 0.20238095,
             0.27642276],
            [0.37113402, 0.15447154, 0.13768116, 0.13768116, 0.15447154,
             0.296875, 0.234375, 0., 0., 0.234375, 0.36, 0.36, 0.37113402,
             0.49541284, 0.20238095, 0.19653179, 0.19653179, 0.20238095,
             0.27642276],
            [0.29885057, 0.03703704, 0.03252033, 0.03252033, 0.03703704,
             0.08163265, 0., 0., 0., 0., 0.28888889, 0.28888889, 0.29885057,
             0.41489362, 0.12418301, 0.12025316, 0.12025316, 0.12418301,
             0.17592593],
            [0.046875, 0.45263158, 0.4195122, 0.4195122, 0.45263158,
             0.65648855, 0.63703704, 0.57333333, 0.57333333, 0.63703704, 0.,
             0., 0.046875, 0.42708333, 0.23428571, 0.22777778, 0.22777778,
             0.23428571, 0.49142857],
            [0.046875, 0.45263158, 0.4195122, 0.4195122, 0.45263158,
             0.65648855, 0.63703704, 0.57333333, 0.57333333, 0.63703704, 0.,
             0., 0.046875, 0.42708333, 0.23428571, 0.22777778, 0.22777778,
             0.23428571, 0.49142857],
            [0., 0.4393531, 0.40648379, 0.40648379, 0.4393531, 0.64426877,
             0.62452107, 0.56013746, 0.56013746, 0.62452107, 0., 0., 0.,
             0.3989071, 0.21407625, 0.20797721, 0.20797721, 0.21407625,
             0.47800587],
            [0., 0.30201342, 0.27439024, 0.27439024, 0.30201342, 0.5,
             0.4787234, 0.41284404, 0.41284404, 0.4787234, 0., 0., 0., 0., 0.,
             0., 0., 0., 0.3358209],
            [0.46334311, 0.30201342, 0.27439024, 0.27439024, 0.30201342,
             0.69798658, 0.67973856, 0.61904762, 0.61904762, 0.67973856,
             0.45142857, 0.45142857, 0.46334311, 0.58955224, 0., 0., 0., 0.,
             0.3358209],
            [0.47863248, 0.32467532, 0.29585799, 0.29585799, 0.32467532,
             0.70779221, 0.68987342, 0.6300578, 0.6300578, 0.68987342,
             0.46666667, 0.46666667, 0.47863248, 0.60431655, 0.03597122, 0.,
             0., 0.03597122, 0.35971223],
            [0.47863248, 0.32467532, 0.29585799, 0.29585799, 0.32467532,
             0.70779221, 0.68987342, 0.6300578, 0.6300578, 0.68987342,
             0.46666667, 0.46666667, 0.47863248, 0.60431655, 0.03597122, 0.,
             0., 0.03597122, 0.35971223],
            [0.46334311, 0.30201342, 0.27439024, 0.27439024, 0.30201342,
             0.69798658, 0.67973856, 0.61904762, 0.61904762, 0.67973856,
             0.45142857, 0.45142857, 0.46334311, 0.58955224, 0., 0., 0., 0.,
             0.3358209],
            [0.46334311, 0., 0., 0., 0., 0.56730769, 0.5462963, 0.4796748,
             0.4796748, 0.5462963, 0.45142857, 0.45142857, 0.46334311,
             0.58955224, 0., 0., 0., 0., 0.]])
        assert_almost_equal(obs.data, exp_data)
        # Expected shape
        self.assertEqual(obs.shape, exp_data.shape)
        # Expected size
        self.assertEqual(obs.size, exp_data.size)

    def test_metric(self):
        """GFastUniFrac._metric already tested on test_matrix"""
        pass


class UnnormalizedGFastUnifracTests(BaseFastUniFracTests):
    """"""

    def test_matrix(self):
        """"""
        un_g_unig = UnnormalizedGFastUnifrac(self.l19_tree, self.l19_data,
                                             self.l19_sample_ids,
                                             self.l19_taxon_ids)
        obs = un_g_unig.matrix()
        # The object has the correct type
        self.assertEqual(type(obs), DissimilarityMatrix)
        # Expected matrix ids
        exp_ids = tuple(sorted(self.l19_sample_ids))
        self.assertEqual(obs.ids, exp_ids)
        # Expected dtype
        self.assertEqual(obs.dtype, np.float64)
        # Expected matrix data
        exp_data = np.array([
            [0., 0.3558952, 0.3558952, 0.3558952, 0.3558952, 0.3558952,
             0.3558952, 0.3558952, 0.3558952, 0.3558952, 0., 0., 0.,
             0.15938865, 0.15938865, 0.15938865, 0.15938865, 0.15938865,
             0.3558952],
            [0.41048035, 0., 0., 0., 0., 0.25764192, 0.25764192, 0.25764192,
             0.25764192, 0.25764192, 0.41048035, 0.41048035, 0.41048035,
             0.41048035, 0.06550218, 0.06550218, 0.06550218, 0.06550218,
             0.06550218],
            [0.47598253, 0.06550218, 0., 0., 0.06550218, 0.3231441, 0.3231441,
             0.3231441, 0.3231441, 0.3231441, 0.47598253, 0.47598253,
             0.47598253, 0.47598253, 0.13100437, 0.13100437, 0.13100437,
             0.13100437, 0.13100437],
            [0.47598253, 0.06550218, 0., 0., 0.06550218, 0.3231441, 0.3231441,
             0.3231441, 0.3231441, 0.3231441, 0.47598253, 0.47598253,
             0.47598253, 0.47598253, 0.13100437, 0.13100437, 0.13100437,
             0.13100437, 0.13100437],
            [0.41048035, 0., 0., 0., 0., 0.25764192, 0.25764192, 0.25764192,
             0.25764192, 0.25764192, 0.41048035, 0.41048035, 0.41048035,
             0.41048035, 0.06550218, 0.06550218, 0.06550218, 0.06550218,
             0.06550218],
            [0.15283843, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.15283843,
             0.15283843, 0.15283843, 0.15283843, 0.06550218, 0.06550218,
             0.06550218, 0.06550218, 0.06550218],
            [0.17030568, 0.01746725, 0.01746725, 0.01746725, 0.01746725,
             0.01746725, 0., 0., 0., 0., 0.17030568, 0.17030568, 0.17030568,
             0.17030568, 0.08296943, 0.08296943, 0.08296943, 0.08296943,
             0.08296943],
            [0.23580786, 0.08296943, 0.08296943, 0.08296943, 0.08296943,
             0.08296943, 0.06550218, 0., 0., 0.06550218, 0.23580786,
             0.23580786, 0.23580786, 0.23580786, 0.14847162, 0.14847162,
             0.14847162, 0.14847162, 0.14847162],
            [0.23580786, 0.08296943, 0.08296943, 0.08296943, 0.08296943,
             0.08296943, 0.06550218, 0., 0., 0.06550218, 0.23580786,
             0.23580786, 0.23580786, 0.23580786, 0.14847162, 0.14847162,
             0.14847162, 0.14847162, 0.14847162],
            [0.17030568, 0.01746725, 0.01746725, 0.01746725, 0.01746725,
             0.01746725, 0., 0., 0., 0., 0.17030568, 0.17030568, 0.17030568,
             0.17030568, 0.08296943, 0.08296943, 0.08296943, 0.08296943,
             0.08296943],
            [0.01965066, 0.37554585, 0.37554585, 0.37554585, 0.37554585,
             0.37554585, 0.37554585, 0.37554585, 0.37554585, 0.37554585, 0.,
             0., 0.01965066, 0.1790393, 0.1790393, 0.1790393, 0.1790393,
             0.1790393, 0.37554585],
            [0.01965066, 0.37554585, 0.37554585, 0.37554585, 0.37554585,
             0.37554585, 0.37554585, 0.37554585, 0.37554585, 0.37554585, 0.,
             0., 0.01965066, 0.1790393, 0.1790393, 0.1790393, 0.1790393,
             0.1790393, 0.37554585],
            [0., 0.3558952, 0.3558952, 0.3558952, 0.3558952, 0.3558952,
             0.3558952, 0.3558952, 0.3558952, 0.3558952, 0., 0., 0.,
             0.15938865, 0.15938865, 0.15938865, 0.15938865, 0.15938865,
             0.3558952],
            [0., 0.19650655, 0.19650655, 0.19650655, 0.19650655, 0.19650655,
             0.19650655, 0.19650655, 0.19650655, 0.19650655, 0., 0., 0., 0.,
             0., 0., 0., 0., 0.19650655],
            [0.34497817, 0.19650655, 0.19650655, 0.19650655, 0.19650655,
             0.45414847, 0.45414847, 0.45414847, 0.45414847, 0.45414847,
             0.34497817, 0.34497817, 0.34497817, 0.34497817, 0., 0., 0., 0.,
             0.19650655],
            [0.36681223, 0.21834061, 0.21834061, 0.21834061, 0.21834061,
             0.47598253, 0.47598253, 0.47598253, 0.47598253, 0.47598253,
             0.36681223, 0.36681223, 0.36681223, 0.36681223, 0.02183406, 0.,
             0., 0.02183406, 0.21834061],
            [0.36681223, 0.21834061, 0.21834061, 0.21834061, 0.21834061,
             0.47598253, 0.47598253, 0.47598253, 0.47598253, 0.47598253,
             0.36681223, 0.36681223, 0.36681223, 0.36681223, 0.02183406, 0.,
             0., 0.02183406, 0.21834061],
            [0.34497817, 0.19650655, 0.19650655, 0.19650655, 0.19650655,
             0.45414847, 0.45414847, 0.45414847, 0.45414847, 0.45414847,
             0.34497817, 0.34497817, 0.34497817, 0.34497817, 0., 0., 0., 0.,
             0.19650655],
            [0.34497817, 0., 0., 0., 0., 0.25764192, 0.25764192, 0.25764192,
             0.25764192, 0.25764192, 0.34497817, 0.34497817, 0.34497817,
             0.34497817, 0., 0., 0., 0., 0.]])
        assert_almost_equal(obs.data, exp_data)
        # Expected shape
        self.assertEqual(obs.shape, exp_data.shape)
        # Expected size
        self.assertEqual(obs.size, exp_data.size)

    def test_metric(self):
        """UnnormalizedGFastUnifrac._metric already tested on test_matrix"""
        pass

if __name__ == '__main__':
    main()
