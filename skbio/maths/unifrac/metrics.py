#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Copyright (c) 2013--, bipy development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# -----------------------------------------------------------------------------

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
        Row vector with length equal to the number of nodes in the tree which
        contains the branch lengths
    i : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree
    j : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree

    Returns
    -------
    float
        The unweighted unifrac distance
    """
    return 1 - ((branch_lengths * np.logical_and(i, j)).sum() /
                (branch_lengths * np.logical_or(i, j)).sum())


def unnormalized_unweighted_unifrac(branch_lengths, i, j):
    """Calculates unifrac(i,j) from branch_lengths and cols i and j of the
    abundance matrix, but omits normalization for fraction of tree covered

    Parameters
    ----------
    branch_lengths : numpy array
        Row vector with length equal to the number of nodes in the tree which
        contains the branch lengths
    i : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree
    j : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree

    Returns
    -------
    float
        The unnormalized unweighted unifrac distance
    """
    return ((branch_lengths * np.logical_xor(i, j)).sum() /
            branch_lengths.sum())


def weighted_unifrac(branch_lengths, i, j, i_sum, j_sum, tip_distances=None):
    """Calculates weighted unifrac(i, j) from branch lengths and cols i and j
    of the abundance matrix.

    Parameters
    ----------
    branch_lengths : numpy array
        Row vector with length equal to the number of nodes in the tree which
        contains the branch lengths
    i : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree
    j : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree
    i_sum : float
        Sum of the tip distances of the i slice
    j_sum : float
        Sum of the tip distances of the j slice
    tip_distances :
        Unused

    Returns
    -------
    float
        The weighted unifrac distance
    """
    return (branch_lengths * np.abs((i / np.float(i_sum)) -
            (j / np.float(j_sum)))).sum()


def corrected_weighted_unifrac(branch_lengths, i, j, i_sum, j_sum,
                               tip_distances):
    """Calculates weighted unifrac(i, j) form branch lengths and cols i and j
    of the abundance matrix, applying branch length corrections.

    Parameters
    ----------
    branch_lengths : numpy array
        Row vector with length equal to the number of nodes in the tree which
        contains the branch lengths
    i : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree
    j : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree
    i_sum : float
        Sum of the tip distances of the i slice
    j_sum : float
        Sum of the tip distances of the j slice
    tip_distances : numpy array
        Vector with length equal to the number of nodes in the tree which
        contains the branch length only for the tips (all other lengths must
        be 0)

    Returns
    -------
    float
        The corrected weighted unifrac distance
    """
    result = weighted_unifrac(branch_lengths, i, j, i_sum, j_sum)
    corr = tip_distances.ravel() * ((i/np.float(i_sum)) + (j/np.float(j_sum)))
    return result / corr.sum()


def G(branch_lengths, i, j):
    """Calculates G(i, j) from branch lengths and cols i and j of the abundance
    matrix.

    Calculates fraction gain in branch length in i with respect to i+j, i.e.
    normalized for the parts of the tree that i and j cover

    Parameters
    ----------
    branch_lengths : numpy array
        Row vector with length equal to the number of nodes in the tree which
        contains the branch lengths
    i : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree
    j : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree

    Returns
    -------
    float
        The G distance
    """
    return ((branch_lengths * np.logical_and(i, np.logical_not(j))).sum() /
            (branch_lengths * np.logical_or(i, j)).sum())


def unnormalized_G(branch_lengths, i, j):
    """Calculates G(i, j) from branch lengths and cols i and j of the abundance
    matrix.

    Calculates the fraction gain in branch length of i with respect to j,
    divided by all the branch length in the tree

    Parameters
    ----------
    branch_lengths : numpy array
        Row vector with length equal to the number of nodes in the tree which
        contains the branch lengths
    i : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree
    j : numpy array
        Slice of states of the abundance matrix, same length as # nodes in tree

    Returns
    -------
    float
        The unnormalized G distance
    """
    return ((branch_lengths * np.logical_and(i, np.logical_not(j))).sum() /
            branch_lengths.sum())
