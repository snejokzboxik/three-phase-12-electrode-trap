"""Common-point diagnostics of raw P2 and harmonic RF fields.

The raw force below is only the within-element gradient of the piecewise
pseudopotential. Interface terms are deliberately not treated as a global
smooth averaged force. No modifications of the original FEM are made.
"""
from __future__ import annotations

import numpy as np

from .validation_field_audit import (rf_quadratures, rf_potential_quadratures,
                                     element_jacobians)


def relative_rms(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
        raise ValueError("Expected matching finite arrays")
    denominator = np.sum(reference**2)
    return float(np.sqrt(np.sum((actual-reference)**2)/denominator)) if denominator else None


def field_quantities(ec, es, jc, js, charge, mass, omega):
    """J[i,j]=dE_i/dx_j. Returns joules and newtons, no RF phase sampling."""
    psi = charge**2*np.sum(ec*ec+es*es, axis=-1)/(4*mass*omega**2)
    force = -charge**2/(2*mass*omega**2)*(np.einsum("...ij,...i->...j", jc, ec)+np.einsum("...ij,...i->...j", js, es))
    return psi, force


def sample_raw(interpolator, points, amplitude, charge, mass, omega):
    ec, es = rf_quadratures(interpolator, points, amplitude)
    pc, ps = rf_potential_quadratures(interpolator, points, amplitude)
    _, ids = interpolator._query_triangles(points)
    ja, jb, jc0 = (element_jacobians(interpolator, ids, p) for p in ("A", "B", "C"))
    jc = np.sqrt(3)*amplitude*(jc0-jb)/2
    js = amplitude*(ja-jb/2-jc0/2)
    psi, force = field_quantities(ec, es, jc, js, charge, mass, omega)
    return {"phi": np.stack((pc, ps), axis=-1), "field": np.stack((ec, es), axis=-1),
            "jacobian": np.stack((jc, js), axis=-1), "psi": psi, "force_elementwise": force}


def sample_smooth(quadratures, points, charge, mass, omega):
    ec, es = quadratures.field_quadratures(points)
    pc, ps = quadratures.potential_quadratures(points)
    jc, js = quadratures.jacobian_quadratures(points)
    psi, force = field_quantities(ec, es, jc, js, charge, mass, omega)
    return {"phi": np.stack((pc, ps), axis=-1), "field": np.stack((ec, es), axis=-1),
            "jacobian": np.stack((jc, js), axis=-1), "psi": psi, "force": force}


def sample_jumps(interpolator, amplitude, radius=.008):
    """Both one-sided fields at the same facet midpoint; no position nudges.

    All internal facets inside 1.5 mm; every thirteenth remaining facet out
    to radius. Quantiles on these nonidentical meshes are diagnostics only.
    """
    mesh = interpolator.solutions.trap_mesh.mesh
    facets = np.flatnonzero(np.all(mesh.f2t >= 0, axis=0))
    midpoint = mesh.p[:, mesh.facets[:, facets]].mean(axis=1).T
    rr = np.linalg.norm(midpoint, axis=1)
    selected = np.r_[facets[rr <= .0015], facets[(rr > .0015) & (rr <= radius)][::13]]
    midpoint = mesh.p[:, mesh.facets[:, selected]].mean(axis=1).T
    rr = np.linalg.norm(midpoint, axis=1)
    values = []
    for side in (0, 1):
        a, b, c = (interpolator._field_values(midpoint, mesh.f2t[side, selected], p) for p in ("A", "B", "C"))
        values.append(np.stack((np.sqrt(3)*amplitude*(c-b)/2, amplitude*(a-b/2-c/2)), axis=-1))
    jump = values[1]-values[0]
    mean = (values[0]+values[1])/2
    normals = mesh.p[:, mesh.facets[1, selected]].T-mesh.p[:, mesh.facets[0, selected]].T
    normals = np.column_stack((-normals[:, 1], normals[:, 0]))
    normals /= np.linalg.norm(normals, axis=-1)[:, None]
    normal_jump = np.einsum("ni,nij->nj", normals, jump)
    tangent_jump = np.einsum("ni,nij->nj", np.column_stack((-normals[:, 1], normals[:, 0])), jump)
    return {"points_m": midpoint, "radius_m": rr,
            "jump_rf_rms": np.sqrt(np.sum(jump**2, axis=(1, 2))/2),
            "mean_field_rf_rms": np.sqrt(np.sum(mean**2, axis=(1, 2))/2),
            "normal_jump_rf_rms": np.sqrt(np.sum(normal_jump**2, axis=-1)/2),
            "tangential_jump_rf_rms": np.sqrt(np.sum(tangent_jump**2, axis=-1)/2)}


def sample_origin_limits(interpolator, amplitude):
    """One-sided gradient limits at the central mesh vertex, without a finder.

    Smooth physical RF quadratures must have E(0)=0 by the 90-degree
    equivariance of this geometry/drive. A continuous P2 potential need not
    have a unique gradient at the vertex. Retain every incident element.
    """
    tri = interpolator.solutions.triangles
    rr = np.linalg.norm(interpolator.solutions.points, axis=-1)
    ids = np.flatnonzero(np.any(rr[tri] < 1e-12, axis=-1))
    if not len(ids):
        raise ValueError("Expected the central vertex of the symmetric Gmsh mesh")
    xy = np.zeros((len(ids), 2))
    a, b, c = (interpolator._field_values(xy, ids, p) for p in ("A", "B", "C"))
    fields = np.stack((np.sqrt(3)*amplitude*(c-b)/2, amplitude*(a-b/2-c/2)), axis=-1)
    return {"triangle_ids": ids, "one_sided_field": fields,
            "one_sided_rf_rms": np.sqrt(np.sum(fields**2, axis=(1, 2))/2)}
