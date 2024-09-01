import unittest

import numpy as np

from denstream import micro_cluster


class TestMicroCluster(unittest.TestCase):
    def setUp(self):
        self.TOL = 1e-6

    def test_fading_cf1(self):
        """
        This test is designed to check that the fading calculations and the cf1 calculations works as expected.
        """

        t = 3
        lambd = 0.5

        # Feature-arrays
        x1 = np.array([1, 2]).reshape((1, 2))
        x2 = np.array([4, 5]).reshape((1, 2))
        x3 = np.array([0, 0]).reshape((1, 2))

        # Time values
        t1 = 1.0
        t2 = 2.0
        t3 = 3.0

        mc = micro_cluster.MicroCluster(0, lambd)
        mc.append(t1, x1)
        mc.append(t2, x2)
        mc.append(t3, x3)

        # Check that the fading calculation works
        estimated_fading = mc._calculate_fading(t)
        expected_fading = np.array([0.5, 1 / np.sqrt(2), 1]).reshape((3, 1))
        self.assertTrue(np.linalg.norm(estimated_fading - expected_fading) < self.TOL)

        # Check that the CF1 calculation works
        x = mc.features_array
        estimated_cf1 = mc.cf1_func(x, estimated_fading)
        expected_cf1 = np.array([0.5 + 4 / np.sqrt(2), 1 + 5 / np.sqrt(2)]).reshape((1, 2))
        self.assertTrue(np.linalg.norm(estimated_cf1 - expected_cf1) < self.TOL)

    def test_radius(self):
        """
        This test is designed to check that the radius calculations works as expected for the given micro-cluster.
        """

        t = 3
        lambd = 0.5

        # Feature-arrays
        x1 = np.array([1, 2], dtype=np.float32).reshape((1, 2))
        x2 = np.array([4, 5], dtype=np.float32).reshape((1, 2))
        x3 = np.array([0, 0], dtype=np.float32).reshape((1, 2))

        # Time values
        t1 = 1.0
        t2 = 2.0
        t3 = 3.0

        mc = micro_cluster.MicroCluster(0, lambd)
        mc.append(t1, x1)
        mc.append(t2, x2)
        mc.append(t3, x3)

        # (cf1 / w) squared
        expected_c1 = (21.75 + 14 / np.sqrt(2)) / (11 / 4 + 3 / np.sqrt(2))

        # abs(cf2 / w)
        expected_c2 = (0.5 + 16 / np.sqrt(2) + 2 + 25 / np.sqrt(2)) / (1.5 + 1 / np.sqrt(2))

        expected_radius = np.sqrt(expected_c2 - expected_c1)
        estimated_radius, _, _ = mc.calculate_radius(t)

        self.assertTrue(np.abs(estimated_radius - expected_radius) < self.TOL)

    def test_adding_updating(self):
        """
        This test is designed to test that adding of points works as expected.
        It also checks that the update_parameters works as expected.
        """

        t0 = 0.5
        lambd = 0.5
        mc = micro_cluster.MicroCluster(t0, lambd)

        # Feature-arrays
        x1 = np.array([1, 2]).reshape((1, 2))
        x2 = np.array([4, 5]).reshape((1, 2))
        x3 = np.array([0, 0]).reshape((1, 2))

        # Time values
        t1 = 1.0
        t2 = 2.0
        t3 = 3.0

        # Check that the array is initially empty
        self.assertEqual(len(mc.features_array), 0)
        self.assertEqual(len(mc.time_array), 0)

        # Adding one point
        mc.append(t1, x1)
        self.assertEqual(len(mc.features_array), 1)
        self.assertEqual(len(mc.time_array), 1)

        # Asserting that the append comes in the correct order
        mc.append(t2, x2)
        expected_time_array = np.array([t1, t2]).reshape((2, 1))
        self.assertTrue(np.linalg.norm(mc.time_array - expected_time_array) < self.TOL)

        # Checking so that the update works
        mc.append(t3, x3)
        t = 3
        mc.update_parameters(time=t)

        expected_weight = 1.5 + 1 / np.sqrt(2)
        expected_center = np.array([0.5 + 4 / np.sqrt(2), 1 + 5 / np.sqrt(2)]).reshape((1, 2))
        expected_center = expected_center / expected_weight

        self.assertTrue(np.abs(mc.weight - expected_weight) < self.TOL)
        self.assertTrue(np.linalg.norm(mc.center - expected_center) < self.TOL)

    def test_update_parameters_same(self):
        """
        This test nis designed to test that adding update_parameters works the same when "time" is given and not.
        """

        lambd = 0.5
        time = 1

        x1 = np.array([-4.1, -3.9]).reshape((1, 2))
        x2 = np.array([-4.2, -4.1]).reshape((1, 2))

        c1 = micro_cluster.MicroCluster(1.0, lambd)
        c1.append(1.0, x1)
        c1.append(1.0, x2)
        c1.update_parameters(time=time)

        c2 = micro_cluster.MicroCluster(1.0, lambd)
        c2.append(1.0, x1)
        c2.append(1.0, x2)

        radius, weight, cf1 = c2.calculate_radius(time)
        c2.update_parameters(cf1_score=cf1, weight=weight)

        self.assertTrue(np.abs(c1.weight - c2.weight) < self.TOL)
        self.assertTrue(np.linalg.norm(c1.center - c2.center) < self.TOL)


if __name__ == "__main__":
    unittest.main()
