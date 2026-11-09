import math
import unittest

import rf_off_analysis as rf


class BallisticGeometryTests(unittest.TestCase):
    def test_radial_outward(self):
        self.assertAlmostEqual(rf.first_outward_circle_crossing(0, 0, 2, 0, 8), 4.0)

    def test_tangential_launch_matches_sagitta_formula(self):
        r0, speed, aperture = 1.0, 2.0, 8.0
        expected = math.sqrt(aperture**2 - r0**2) / speed
        self.assertAlmostEqual(rf.first_outward_circle_crossing(r0, 0, 0, speed, aperture), expected)

    def test_zero_velocity_never_crosses(self):
        self.assertTrue(math.isinf(rf.first_outward_circle_crossing(3, 4, 0, 0, 8)))

    def test_inward_radial_launch_exits_opposite_side(self):
        self.assertAlmostEqual(rf.first_outward_circle_crossing(3, 0, -2, 0, 8), 5.5)

    def test_boundary_inward_launch_uses_later_outward_crossing(self):
        self.assertAlmostEqual(rf.first_outward_circle_crossing(8, 0, -2, 0, 8), 8.0)

    def test_crossing_at_horizon_counts_as_escape(self):
        self.assertEqual(rf.retention_at_horizon(0.003, 0.003), "escaped")
        self.assertEqual(rf.retention_at_horizon(0.0030001, 0.003), "retained")

    def test_reference_energy_cutoffs(self):
        expected_meV = {50000: 1.813747, 100000: 3.627494, 150000: 5.441242, 200000: 7.254989}
        for mz, expected in expected_meV.items():
            mass = mz * rf.AMU_KG
            e = 0.5 * mass * (rf.APERTURE_M**2 - 0.001**2) / (0.003**2) / rf.EV_J * 1000
            self.assertAlmostEqual(e, expected, places=5)

    def test_v2_input_audit(self):
        rows, audit, starts = rf.audit_inputs()
        self.assertEqual(len(rows), 2320)
        self.assertEqual(len(starts), 1160)
        self.assertEqual(audit["duplicate_valid_checkpoint_rows"], 228)
        self.assertEqual(audit["conflicting_duplicate_job_ids"], 0)
        self.assertEqual(audit["unresolved_rows_in_v2_csv"], 0)
        self.assertLess(audit["max_relative_energy_conversion_error"], 1e-12)


if __name__ == "__main__":
    unittest.main()
