#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2013--, bipy development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

"""Fast implementation of UniFrac for use with very large datasets"""
from __future__ import division

import numpy as np

from skbio.core.distance import SymmetricDistanceMatrix, DistanceMatrix
from skbio.maths.unifrac.metrics import (unweighted_unifrac,
                                         unnormalized_unweighted_unifrac,
                                         weighted_unifrac,
                                         corrected_weighted_unifrac,
                                         G, unnormalized_G)


class FastUniFrac(object):
    """
    Parameters
    ----------
    t : UniFracTree
        Phylogenetic tree relating the sequences
    abund_mtx : 2d array
        samples (rows) by seqs (columns) numpy 2d array
    sample_ids : list of strings
        list with the sample ids. Its length should be the number of rows
        on abund_mtx
    taxon_ids : list of strings
        list with the taxon ids. Its length should be the number of columns
        on abund_mtx
    make_subtree : bool
        If true, prune the tree to only include the nodes that appear on envs

    Attributes
    ----------
    """
    def __init__(self, t, abund_mtx, sample_ids, taxon_ids, make_subtree=True):
        """Sets up all the internal variables"""
        self._make_envs_dict(abund_mtx, sample_ids, taxon_ids)

        self._tree = t.deepcopy()
        if make_subtree:
            self._make_subtree()

        # get good nodes, defined as those that are in envs
        self._envs = dict([(i.name, self._envs[i.name])
                           for i in self._tree.tips()
                           if i.name in self._envs])
        if not self._envs:
            raise ValueError("No valid samples/environments found. Check "
                             "whether tree tips match otus/taxa present in "
                             "samples/environments")

        # index tree
        self._node_index, self._nodes = self._tree.index_tree()

        # index envs
        self._index_envs()
        self._env_names = sorted(self._unique_envs)

        # Get an array of branch lengths, in self._node_index order
        self._branch_lengths = np.zeros(len(self._node_index), float)
        for i, node in self._node_index.items():
            try:
                if node.length is not None:
                    self._branch_lengths[i] = node.length
            except AttributeError:
                pass

        self._bind_to_array()

    def _make_envs_dict(self, abund_mtx, sample_ids, taxon_ids):
        """Makes an envs dict from an abundance matrix

        Parameters
        ----------
        abund_mtx : 2d array
            samples (rows) by seqs (columns) numpy 2d array
        sample_ids : list of strings
            list with the sample ids. Its length should be the number of rows
            on abund_mtx
        taxon_ids : list of strings
            list with the taxon ids. Its length should be the number of columns
            on abund_mtx
        """
        num_samples, num_seqs = abund_mtx.shape
        if (num_samples, num_seqs) != (len(sample_ids), len(taxon_ids)):
            raise ValueError("Shape of abundance matrix %s doesn't match # "
                             "of samples and # of taxa (%s and %s)" %
                             (abund_mtx.shape, num_samples, num_seqs))
        self._envs = {}
        sample_ids = np.asarray(sample_ids)
        for i, taxon in enumerate(abund_mtx.T):
            nonzeros = taxon.nonzero()
            self._envs[taxon_ids[i]] = dict(zip(sample_ids[nonzeros],
                                                taxon[nonzeros]))

    def _make_subtree(self):
        """Prune the tree to only include the tips seen on envs"""
        wanted = set(self._envs.keys())

        def delete_test(node):
            if node.is_tip() and node.name not in wanted:
                return True
            return False

        self._tree.remove_deleted(delete_test)
        self._tree.prune()

    def _index_envs(self):
        """Creates an array of taxon x env with counts of the taxon in each env
        """
        # Extract all unique envs from envs dict
        self._unique_envs = set()
        for v in self._envs.values():
            self._unique_envs.update(v.keys())
        self._unique_envs = sorted(self._unique_envs)

        self._env_to_index = dict([(e, i) for i, e
                                  in enumerate(self._unique_envs)])
        # figure out taxon label to index map
        self._node_to_index = {}
        for i, node in self._node_index.items():
            if node.name is not None:
                self._node_to_index[node.name] = i
        # walk over env_counts, adding correct slots in array
        num_nodes = len(self._node_index)
        num_envs = len(self._unique_envs)
        self._count_array = np.zeros((num_nodes, num_envs), int)
        for name in self._envs:
            curr_row_index = self._node_to_index[name]
            for env, count in self._envs[name].items():
                self._count_array[curr_row_index,
                                  self._env_to_index[env]] = count

    def _bind_to_array(self):
        """Binds the _nodes to the _count_array, creating a new list

        Takes the list (node, first_child, last_child) _nodes and creates the
        list _bound_indices of (node_row, child_rows) such that node_row points
        to the row of a that corresponds to the current node, and child_rows
        points tot the row of rows of a that correspond tot the direct children
        of the current node.

        Order is assumed to be traversal order, i.e. for the typical case of
        postorder traversal iterating over the items in the result and
        consolidating each time should give the same result as postorder
        traversal of the original tree. Should also be able to modify the
        preorder traversal.
        """
        self._bound_indices = [(self._count_array[node],
                                self._count_array[start:end+1])
                               for node, start, end in self._nodes]

    def _traverse_reduce(self, f):
        """Applies _bound_indices[i] = f(_bound_indices[j:k]) over list of
        [(_bound_indices[i], _bound_indices[j:k])]

        If list is in traversal order, has same effect as consolidating the
        function over the tree, only much faster

        Note that f(a[j:k]) must return and object that can be broadcast to
        the same shape as a[i], e.g. summing a 2D array to get a vector.
        """
        for i, s in self._bound_indices:
            i[:] = f(s, 0)

    def _bool_descendants(self):
        """For each internal node, sets col to True if any descendant is True
        """
        self._traverse_reduce(np.logical_or.reduce)

    def _sum_descendants(self):
        """For each internal node, sets col to sum of values in descendants"""
        self._traverse_reduce(np.sum)

    def _symmetric_matrix(self):
        """Returns a SymmetricDistanceMatrix with the UniFrac distances"""
        num_cols = self._count_array.shape[-1]
        cols = [self._count_array[:, i] for i in range(num_cols)]
        result = np.zeros((num_cols, num_cols), float)
        # Since unifrac is symmetric, only calc half matrix and transpose
        for i in range(1, num_cols):
            first_col = cols[i]
            row_result = []
            for j in range(i):
                second_col = cols[j]
                row_result.append(self._metric(first_col, second_col))
            result[i, :j+1] = row_result
        # can't use += because shared memory between a and transpose(a)
        result = result + np.transpose(result)
        return SymmetricDistanceMatrix(result, self._env_names)

    def _asymmetric_matrix(self):
        """Returns a DistanceMatrix with the UniFrac distances"""
        num_cols = self._count_array.shape[-1]
        cols = [self._count_array[:, i] for i in range(num_cols)]
        result = np.zeros((num_cols, num_cols), float)
        for i in range(num_cols):
            first_col = cols[i]
            result[i] = [self._metric(first_col, cols[j])
                         for j in range(num_cols)]
        return DistanceMatrix(result, self._env_names)

    def _metric(self, i, j):
        """Should be implemented by the subclasses"""
        raise NotImplementedError("_metric cannot be called on the base class")

    def matrix(self):
        """Should be implemented by the subclasses"""
        raise NotImplementedError("matrix cannot be called on the base class")


class UnweightedFastUniFrac(FastUniFrac):
    """docstring for UnweightedFastUniFrac"""
    def __init__(self, t, abund_mtx, sample_ids, taxon_ids, **kwargs):
        super(UnweightedFastUniFrac, self).__init__(t, abund_mtx,
                                                    sample_ids, taxon_ids,
                                                    **kwargs)
        self._bool_descendants()

    def _metric(self, i, j):
        """Calculates unifrac(i,j) from branch_lengths and cols i and j of the
        abundance matrix

        This is the original, unerighted UniFrac metric

        Slicing the abundance matrix (m[:, i]) returns a vector in the right
        format, note that it should be a row vector, not a columns vector.

        Parameters
        ----------
        i : list
            Should be a slice of states of the abundance matrix, same length
            as # nodes in tree
        j : list
            Should be a slice of states of the abundance matrix, same length
            as # nodes in tree
        """
        return unweighted_unifrac(self._branch_lengths, i, j)

    def matrix(self):
        return self._symmetric_matrix()


class UnnormalizedUnweightedFastUniFrac(UnweightedFastUniFrac):
    """"""
    def _metric(self, i, j):
        """UniFrac, but omits normalization for frac of tree covered"""
        return unnormalized_unweighted_unifrac(self._branch_lengths, i, j)


class GFastUniFrac(UnweightedFastUniFrac):
    """"""
    def _metric(self, i, j):
        """Calculates G(i,j) from branch_lengths and cols i,j of abund_mtx"""
        return G(self._branch_lengths, i, j)

    def matrix(self):
        return self._asymmetric_matrix()


class UnnormalizedGFastUnifrac(GFastUniFrac):
    """docstring for GFastUniFracFullTree"""

    def _metric(self, i, j):
        """"""
        return unnormalized_G(self._branch_lengths, i, j)


class WeightedFastUniFrac(FastUniFrac):
    """"""
    def __init__(self, t, abund_mtx, sample_ids, taxon_ids, **kwargs):
        super(WeightedFastUniFrac, self).__init__(t, abund_mtx, sample_ids,
                                                  taxon_ids, **kwargs)
        self._tip_indices = [n._leaf_index for n in self._tree.tips()]
        self._sum_descendants()
        self._tip_ds = self._branch_lengths.copy()[:, np.newaxis]
        self._bind_to_parent_array()
        self._tip_distances()

    def _bind_to_parent_array(self):
        """Binds tree to tip_ds

        Creates _bindings, a list of (node_row, parent_row) such that node_row
        points to the row of that corresponds to the current row, and
        parent_row points to the row of the parent

        Order will be preorder traversal, i.e. for propagating attributes from
        the root to the tip.
        """
        self._bindings = []
        for n in self._tree.traverse(self_before=True, self_after=False):
            if n is not self._tree:
                self._bindings.append([self._tip_ds[n._leaf_index],
                                       self._tip_ds[n.parent._leaf_index]])

    def _tip_distances(self):
        """Sets each tip to its distance from the root"""
        for i, s in self._bindings:
            i += s
        mask = np.zeros(len(self._tip_ds))
        np.put(mask, self._tip_indices, 1)
        self._tip_ds *= mask[:, np.newaxis]

    def _metric(self, i, j, i_sum, j_sum):
        """Calculates weighted unifrac(i, j) from branch_lengths and cols i,j
        of abund_mtx.

        It performs branch length correction if self._bl_correct = True
        """
        return weighted_unifrac(self._branch_lengths, i, j, i_sum, j_sum)

    def matrix(self):
        """Calculates weighted_unifrac(i, j) for all i, j in abund_mtx"""
        num_cols = self._count_array.shape[-1]
        cols = [self._count_array[:, i] for i in range(num_cols)]
        sums = [np.take(self._count_array[:, i], self._tip_indices).sum()
                for i in range(num_cols)]
        result = np.zeros((num_cols, num_cols), float)
        for i in range(1, num_cols):
            i_sum = sums[i]
            first_col = cols[i]
            row_result = []
            for j in range(i):
                second_col = cols[j]
                j_sum = sums[j]
                curr = self._metric(first_col, second_col, i_sum, j_sum)
                row_result.append(curr)
            result[i, :j+1] = row_result
        result = result + np.transpose(result)
        return SymmetricDistanceMatrix(result, self._env_names)


class CorrectedWeightedFastUniFrac(WeightedFastUniFrac):
    """"""
    def _metric(self, i, j, i_sum, j_sum):
        """Calculates weighted unifrac(i, j) from branch_lengths and cols i,j
        of abund_mtx.

        It performs branch length correction if self._bl_correct = True
        """
        return corrected_weighted_unifrac(self._branch_lengths, i, j,
                                          i_sum, j_sum, self._tip_ds)
