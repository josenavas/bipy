#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2013--, bipy development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

"""Fast implementation of UniFrac for use with very large datasets"""

import numpy as np
from cogent.core.tree import PhyloNode


class UniFracTree(PhyloNode):
    """"""
    def __nonzero__(self):
        """Returns True if self.Children"""
        return bool(self.Children)

    def index_tree(self):
        """Indexes nodes in-place

        Algorithm is as follows:
        for each no in post-order traversal over tree:
            if the node has children:
                set an index on each child
                for each child with children:
                    add the child and its start and end tips to the result
        """
        # A dict because adding out of order
        id_index = {}
        child_index = []
        curr_index = 0
        for n in self.traverse(self_before=False, self_after=True):
            for c in n.Children:
                c._leaf_index = curr_index
                id_index[curr_index] = c
                curr_index += 1
                if c:
                    # c has children itself, so need to add to result
                    child_index.append((c._leaf_index,
                                        c.Children[0]._leaf_index,
                                        c.Children[-1]._leaf_index))
        # Handle root, which should be self
        self._leaf_index = curr_index
        id_index[curr_index] = self
        # Only want to add to the child_index if self has children
        if self.Children:
            child_index.append((self._leaf_index, self.Children[0]._leaf_index,
                                self.Children[-1]._leaf_index))
        return id_index, child_index


class FastUniFrac(object):
    """
    Parameters
    ----------
    t : UniFracTree
        Phylogenetic tree relating the sequences
    envs : dict of dicts
        dict of {sequence:{env:count}} showing environmental abundance
    make_subtree : bool
        If true, prune the tree to only include the nodes that appear on envs

    Attributes
    ----------
    """
    def __init__(self, t, abund_mtx, sample_ids, taxon_ids, make_subtree=True):
        """Set ups all the internal"""
        num_samples, num_seqs = abund_mtx.shape
        if (num_samples, num_seqs) != (len(sample_ids), len(taxon_ids)):
            raise ValueError("Shape of matrix %s doesn't match # samples and "
                             "# taxa (%s and %s)" % (abund_mtx.shape,
                                                     len(sample_ids),
                                                     len(taxon_ids)))
        envs = {}
        sample_ids
        self._tree = t.copy()
        if make_subtree:
            wanted = set(envs.keys())

            def delete_test(node):
                if node.istip() and node.Name not in wanted:
                    return True
                return False

            self._tree.removeDeleted(delete_test)
            self._tree.prune()

        # get good nodes, defined as those that are in envs
        self._envs = dict([(i.Name, envs[i.Name]) for i in self._tree.tips()
                          if i.Name in envs])
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
                if node.Length is not None:
                    self._branch_lengths = node.Length
            except AttributeError:
                pass

        self._index_envs(self._envs, self._node_index)

    def _index_envs(self):
        """"""
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
            if node.Name is not None:
                self._node_to_index[node.Name] = i
        # walk over env_counts, adding correct slots in array
        num_nodes = len(self._node_index)
        num_envs = len(self._unique_envs)
        self._count_array = np.zeros((num_nodes, num_envs), int)
        for name in self._envs:
            curr_row_index = self._node_to_index[name]
            for env, count in self._envs[name].items():
                self._count_array[curr_row_index,
                                  self._env_to_index[env]] = count
