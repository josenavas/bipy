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
from numpy.testing import assert_almost_equal

from skbio.maths.unifrac.metrics import (unweighted_unifrac,
                                         unnormalized_unweighted_unifrac,
                                         weighted_unifrac,
                                         corrected_weighted_unifrac,
                                         G, unnormalized_G)


class MetricsTests(TestCase):
    """Tests for skbio.maths.unifrac.metrics"""

    def setUp(self):
        """Define testing variables"""
        self.branch_lengths = np.array([1., 2., 1., 1., 3., 2., 4., 3., 0.])
        self.count_array = np.array([[1, 0, 2], [1, 1, 0], [0, 3, 0],
                                    [0, 0, 1], [0, 1, 0], [0, 0, 0],
                                    [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.tip_distances = np.array([[5.], [6.], [6.],
                                      [6.], [6.], [0.],
                                      [0.], [0.], [0.]])

    def test_unweighted_unifrac(self):
        """Correctly computes the unweighted unifrac distance"""
        obs = unweighted_unifrac(self.branch_lengths, self.count_array[:, 0],
                                 self.count_array[:, 1])
        assert_almost_equal(obs, 0.7142857142)

        obs = unweighted_unifrac(self.branch_lengths, self.count_array[:, 0],
                                 self.count_array[:, 2])
        assert_almost_equal(obs, 0.75)

        obs = unweighted_unifrac(self.branch_lengths, self.count_array[:, 1],
                                 self.count_array[:, 2])
        assert_almost_equal(obs, 1.0)

    def test_unnormalized_unweighted_unifrac(self):
        """Correctly computes the unnormalized unweighted unifrac distance"""
        obs = unnormalized_unweighted_unifrac(self.branch_lengths,
                                              self.count_array[:, 0],
                                              self.count_array[:, 1])
        assert_almost_equal(obs, 0.2941176470)

        obs = unnormalized_unweighted_unifrac(self.branch_lengths,
                                              self.count_array[:, 0],
                                              self.count_array[:, 2])
        assert_almost_equal(obs, 0.1764705882)

        obs = unnormalized_unweighted_unifrac(self.branch_lengths,
                                              self.count_array[:, 1],
                                              self.count_array[:, 2])
        assert_almost_equal(obs, 0.4705882352)

    def test_weighted_unifrac(self):
        """Correctly computes the weighted unifrac distance"""
        obs = weighted_unifrac(self.branch_lengths, self.count_array[:, 0],
                               self.count_array[:, 1], 2.0, 5.0)
        assert_almost_equal(obs, 2.3)

        obs = weighted_unifrac(self.branch_lengths, self.count_array[:, 0],
                               self.count_array[:, 2], 2.0, 3.0)
        assert_almost_equal(obs, 1.499999999)

        obs = weighted_unifrac(self.branch_lengths, self.count_array[:, 1],
                               self.count_array[:, 2], 5.0, 3.0)
        assert_almost_equal(obs, 2.599999999)

    def test_corrected_weighted_unifrac(self):
        """"""
        obs = corrected_weighted_unifrac(self.branch_lengths,
                                         self.count_array[:, 0],
                                         self.count_array[:, 1],
                                         2.0, 5.0, self.tip_distances)
        assert_almost_equal(obs, 0.2)

        obs = corrected_weighted_unifrac(self.branch_lengths,
                                         self.count_array[:, 0],
                                         self.count_array[:, 2],
                                         2.0, 3.0, self.tip_distances)
        assert_almost_equal(obs, 0.1384615384)

        obs = corrected_weighted_unifrac(self.branch_lengths,
                                         self.count_array[:, 1],
                                         self.count_array[:, 2],
                                         5.0, 3.0, self.tip_distances)
        assert_almost_equal(obs, 0.2294117647)

    def test_G(self):
        """"""
        obs = G(self.branch_lengths, self.count_array[:, 0],
                self.count_array[:, 0])
        assert_almost_equal(obs, 0.0)

        obs = G(self.branch_lengths, self.count_array[:, 0],
                self.count_array[:, 1])
        assert_almost_equal(obs, 0.1428571428)

        obs = G(self.branch_lengths, self.count_array[:, 1],
                self.count_array[:, 0])
        assert_almost_equal(obs, 0.5714285714)

    def test_unnormalized_G(self):
        """"""
        obs = unnormalized_G(self.branch_lengths, self.count_array[:, 0],
                             self.count_array[:, 0])
        assert_almost_equal(obs, 0.0)

        obs = unnormalized_G(self.branch_lengths, self.count_array[:, 0],
                             self.count_array[:, 1])
        assert_almost_equal(obs, 0.0588235294)

        obs = unnormalized_G(self.branch_lengths, self.count_array[:, 1],
                             self.count_array[:, 0])
        assert_almost_equal(obs, 0.2352941176)

if __name__ == '__main__':
    main()
