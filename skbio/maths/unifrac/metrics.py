#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2013--, bipy development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np


def unweighted_unifrac(branch_lengths, i, j):
    """Calculates unifrac(i,j) from branch lengths and cols i and j of the
    abundance matrix

    This is the original, unerighted UniFrac metric

    Slicing the abundance matrix (m[:, i]) returns a vector in the right
    format, note that it should be a row vector, not a columns vector.

    Parameters
    ----------
    branch_lengths : numpy array
        Row vector with the same length as the # of ndoes in tree
    i : list
        Should be a slice of states of the abundance matrix, same length
        as # nodes in tree
    j : list
        Should be a slice of states of the abundance matrix, same length
        as # nodes in tree
    """
    return 1 - ((branch_lengths * np.logical_and(i, j)).sum() /
                (branch_lengths * np.logical_or(i, j)).sum())


def unweighted_unifrac_full_tree(branch_lengths, i, j):
    """Calculates unifrac(i,j) from branch_lengths and cols i and j of the
    abundance matrix, but omits normalization for fraction of tree covered

    Parameters
    ----------
    branch_lengths : numpy array
        Row vector with the same length as the # of ndoes in tree
    i : list
        Should be a slice of states of the abundance matrix, same length
        as # nodes in tree
    j : list
        Should be a slice of states of the abundance matrix, same length
        as # nodes in tree
    """
    return ((branch_lengths * np.logical_xor(i, j)).sum() /
            branch_lengths.sum())


def weighted_unifrac(branch_lengths, i, j, i_sum, j_sum, tip_distances=None):
    """Calculates weighted unifrac(i, j) from branch lengths and cols i and j
    of the abundance matrix.

    Parameters
    ----------
    branch_lengths : numpy array
        Row vector with the same length as the # of ndoes in tree
    i : list
        Should be a slice of states of the abundance matrix, same length
        as # nodes in tree
    j : list
        Should be a slice of states of the abundance matrix, same length
        as # nodes in tree
    i_sum : float
        
    j_sum : float
    """
    return (branch_lengths * np.abs((i / np.float(i_sum)) -
            (j / np.float(j_sum)))).sum()


def normalized_weighted_unifrac(branch_lengths, i, j, i_sum, j_sum,
                                tip_distances):
    """"""
    result = weighted_unifrac(branch_lengths, i, j, i_sum, j_sum)
    corr = tip_distances.ravel() * ((i/np.float(i_sum)) + (j/np.float(j_sum)))
    return result / corr.sum()


def G(branch_lengths, i, j):
    """"""
    return ((branch_lengths * np.logical_and(i, np.logical_not(j))).sum() /
            (branch_lengths * np.logical_or(i, j)).sum())


def G_full_tree(branch_lengths, i, j):
    """"""
    return ((branch_lengths * np.logical_and(i, logical_not(j))).sum() /
            branch_lengths.sum())
