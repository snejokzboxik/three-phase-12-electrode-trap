from types import SimpleNamespace
import numpy as np
import pytest
from skfem import Basis,ElementTriP2,MeshTri

from three_phase_trap.config import ParticleConfig
from three_phase_trap import dynamics
from three_phase_trap.fem import FEMBasisSolutions
from three_phase_trap.mesh import TrapMesh
from three_phase_trap.interpolation import FEMInterpolator
from three_phase_trap.validation_benchmarks import ideal_field,matched_full_state,recover_slow_state,centered_period_mean
from three_phase_trap.validation_field_audit import element_jacobians,fit_harmonic_quadratures


def test_failed_integrator_is_not_reported_as_completed(monkeypatch):
    monkeypatch.setattr(dynamics,"solve_ivp",lambda *a,**kw:SimpleNamespace(success=False,message="forced failure"))
    with pytest.raises(RuntimeError,match="forced failure"):
        dynamics.integrate_trajectory(None,ParticleConfig(),1.,1.,periods=1.,field_function=lambda p,t:np.zeros(2))


def test_micromotion_transform_recovers_nonzero_slow_velocity():
    model=ideal_field(4,1.2)
    slow=np.array([.8,.3,.07,-.04])
    full=matched_full_state(model,slow[:2],slow[2:],1.,1.,20.,.7)
    restored=recover_slow_state(model,full,1.,1.,20.,.7)
    assert np.allclose(restored,slow,rtol=0,atol=1e-12)


def test_period_mean_has_centered_timestamps_and_removes_rf_oscillation():
    t=np.arange(257)/64
    values=np.column_stack((2*t+np.cos(2*np.pi*t),-.5*t+np.sin(2*np.pi*t)))
    centres,mean=centered_period_mean(values,t,64)
    assert np.allclose(mean,np.column_stack((2*centres,-.5*centres)),rtol=0,atol=1e-13)


def test_p2_element_hessian_includes_mixed_derivative():
    m=MeshTri(np.array([[.2,1.4,-.3],[-.4,.1,1.2]]),np.array([[0],[1],[2]]),sort_t=False)
    b=Basis(m,ElementTriP2())
    x,y=b.doflocs
    phi=3*x*x+2*x*y-y*y+.3*x
    trap=TrapMesh(m,{}, {},np.array([],dtype=int))
    fields={n:np.zeros((1,2)) for n in ("A","B","C","DC")}
    interp=FEMInterpolator(FEMBasisSolutions(trap,b,{n:phi for n in fields},fields))
    actual=element_jacobians(interp,[0],"A")[0]
    assert np.allclose(actual,[[-6,-2],[-2,2]],rtol=0,atol=1e-13)


def test_harmonic_fit_predicts_heldout_gradient_and_hessian():
    original=ideal_field(4,.7)
    original.cosine=np.pad(original.cosine,(0,4))
    original.sine=np.pad(original.sine,(0,4))
    original.cosine[8]=.04+.07j
    original.sine[8]=-.03+.02j
    a=np.arange(97)*2*np.pi/97
    points=np.concatenate([r*np.column_stack((np.cos(a),np.sin(a))) for r in (.3,.6,.9)])
    pc,ps=original.potential_quadratures(points)
    fitted,_=fit_harmonic_quadratures(points,pc,ps,8,1.)
    heldout=np.array([[.43,.12],[-.2,.71],[.02,-.38]])
    for actual,expected in zip(fitted.field_quadratures(heldout),original.field_quadratures(heldout)):
        assert np.allclose(actual,expected,rtol=0,atol=1e-13)
    for actual,expected in zip(fitted.jacobian_quadratures(heldout),original.jacobian_quadratures(heldout)):
        assert np.allclose(actual,expected,rtol=0,atol=1e-12)
