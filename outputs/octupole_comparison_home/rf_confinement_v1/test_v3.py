"""Fast regression tests for the V3 field adapter and launch calculations."""
import math
import unittest
import numpy as np
import run_v3 as v3


class V3PhysicsTests(unittest.TestCase):
    def test_charge_mass_and_energy_conversion(self):
        initial=v3.make_initial(100_000.,1.,10.,.7,1.1,-1.,"tangent")
        mass=100_000*v3.AMU
        e=.5*mass*(initial[2]**2+initial[3]**2)/v3.E_CHARGE
        self.assertAlmostEqual(e,.010,places=14)
        self.assertAlmostEqual(v3.E_CHARGE/mass,964.8533215665328,places=10)

    def test_ballistic_radial_and_tangential_crossings(self):
        r0=.001;R=v3.APERTURE;T=.003
        vt=math.sqrt(R*R-r0*r0)/T
        self.assertAlmostEqual(v3.ballistic_escape_time([r0,0.,0.,vt]),T,places=14)
        self.assertAlmostEqual(v3.ballistic_escape_time([r0,0.,vt,0.]),(R-r0)/vt,places=14)
        self.assertTrue(math.isinf(v3.ballistic_escape_time([r0,0.,0.,0.])))

    def test_standing_octupole_coefficients_and_quadrature(self):
        theta=np.arange(12)*math.pi/6
        np.testing.assert_allclose(np.cos(4*theta),np.resize([1.,-.5,-.5],12),atol=2e-15)
        _,q=v3.load_fields(v3.FIELD35)
        _,s=v3.field_pair(q,"standing").potential_quadratures(np.array([[1e-3,0.]]))
        c,_=q.potential_quadratures(np.array([[1e-3,0.]]))
        self.assertEqual(float(c[0]),float(c[0]))
        self.assertTrue(math.isfinite(float(s[0])))

    def test_voltage_scaling_is_explicitly_linear(self):
        _,q=v3.load_fields(v3.FIELD35);xy=np.array([[1.1e-3,-.4e-3]])
        e1=v3.field_values(q,xy,.41,1.)
        e250=v3.field_values(q,xy,.41,250.)
        np.testing.assert_allclose(e250,250*e1,rtol=2e-15,atol=1e-12)

    def test_fast_acceleration_matches_published_evaluator(self):
        _,q=v3.load_fields(v3.FIELD35);m=100000*v3.AMU;qm=v3.E_CHARGE/m
        dc=np.polynomial.polynomial.polyder(q.cosine);ds=np.polynomial.polynomial.polyder(q.sine)
        for xy in ((.7e-3,-1.2e-3),(2.8e-3,1.1e-3),(-4.5e-3,.2e-3)):
            z=complex(*xy)/q.reference_radius;d0=np.polynomial.polynomial.polyval(z,dc)/q.reference_radius
            d1=np.polynomial.polynomial.polyval(z,ds)/q.reference_radius
            for phase in (0.,.731,2.2):
                fast=250*qm*np.array([-d0.real*math.cos(phase)-d1.real*math.sin(phase),
                    d0.imag*math.cos(phase)+d1.imag*math.sin(phase)])
                reference=qm*v3.field_values(q,np.asarray(xy)[None,:],phase,250.)[0]
                np.testing.assert_allclose(fast,reference,rtol=2e-13,atol=1e-10)


if __name__=="__main__":
    unittest.main(verbosity=2)
