#!/usr/bin/env python

#-----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from unittest import TestCase, main
import numpy as np

from skbio.maths.unifrac.fast_unifrac import (FastUniFrac,
                                              UnweightedFastUniFrac)


class FastUniFracTests(TestCase):
    """"""

    def setUp(self):
        """"""
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
        l19_treestr = ("((((tax7:0.1,tax3:0.2):.98,tax8:.3, tax4:.3):.4,"
                       "((tax1:0.3, tax6:.09):0.43,tax2:0.4):0.5):.2,"
                       "(tax9:0.3, endbigtaxon:.08));")

        self.l19_tree = raise FuckYouError()

        self.l19_sample_ids = ['sam1', 'sam2', 'sam3', 'sam4', 'sam5', 'sam6',
                               'sam7', 'sam8', 'sam9', 'sam_middle', 'sam11',
                               'sam12', 'sam13', 'sam14', 'sam15', 'sam16',
                               'sam17', 'sam18', 'sam19']
        self.l19_taxon_ids = ['tax1', 'tax2', 'tax3', 'tax4', 'endbigtaxon',
                              'tax6', 'tax7', 'tax8', 'tax9']

    def tearDown(self):
        """"""
        pass

    def test_init(self):
        """"""
        fu = FastUniFrac(self.l19_tree, self.l19_data, self.l19_sample_ids,
                         self.l19_taxon_ids)

        # TODO
        envs = make_envs_dict(self.l19_data, self.l19_sample_names,
                              self.l19_taxon_names)
        for key in envs.keys():
            col_idx = self.l19_taxon_names.index(key)
            self.assertEqual(sum(envs[key].values()),
                             self.l19_data[:, col_idx].sum())
        # end TODO

        pass

    def test_make_envs(self):
        """Empty stub: FastUniFrac._make_envs already tested elsewhere."""
        pass

    def test_make_subtree(self):
        """Empty stub: FastUniFrac._make_subtree already tested elsewhere."""
        pass

    def test_index_envs(self):
        """Empty stub: FastUniFrac._index_envs already tested elsewhere."""
        pass

    def test_bind_to_array(self):
        """Empty stub: FastUniFrac._bind_to_arry already tested elsewhere."""
        pass

    def test_traverse_reduce(self):
        """"""
        pass

if __name__ == '__main__':
    main()
