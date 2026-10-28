"""Direct P2/P4 RF finite elements, with an exact cyclic sector reduction.

H = Phi_A + exp(-2pi i/3) Phi_B + exp(2pi i/3) Phi_C, so that
Phi(t)=U0 Im(H exp(i chi)). At a 30-degree rotation H gains rho=exp(-2pi i/3).
The complex constraint is assembled in the variational problem, not fitted
to a harmonic expansion. All spatial content of the FE space is retained.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skfem import Basis, ElementTriP2, ElementTriP4, MeshTri, asm
import matplotlib.tri as mtri

from .fem import laplace_form
from .interpolation import InterpolationDomainError

RHO = complex(-.5, -math.sqrt(3)/2)
SECTOR_ANGLE = math.pi/6


def exponents(degree):
    return [(i, total-i) for total in range(degree+1) for i in range(total, -1, -1)]


def vandermonde(xy, degree):
    xy = np.asarray(xy)
    return np.stack([xy[..., 0]**i*xy[..., 1]**j for i, j in exponents(degree)], axis=-1)


def sector_mesh(path):
    import meshio
    data = meshio.read(path)
    triangles = data.get_cells_type("triangle")
    used = np.unique(triangles)
    remap = np.full(len(data.points), -1, int)
    remap[used] = np.arange(len(used))
    # Higher-order H1 edge DOFs require a consistent local edge orientation.
    mesh = MeshTri(data.points[used, :2].T, remap[triangles].T, sort_t=True)
    lines = remap[data.get_cells_type("line")]
    tags = data.get_cell_data("gmsh:physical", "line")
    lookup = {tuple(v): i for i, v in enumerate(np.sort(mesh.facets.T, axis=1))}
    groups = {}
    for name, tag in (("outer", 1), ("electrode", 10)):
        groups[name] = np.array([lookup[tuple(v)] for v in np.sort(lines[tags == tag], axis=1)], int)
    boundary = mesh.boundary_facets()
    for name, angle in (("minus", -math.pi/12), ("plus", math.pi/12)):
        normal = np.array([-math.sin(angle), math.cos(angle)])
        distance = np.einsum("i,ijk->jk", normal, mesh.p[:, mesh.facets[:, boundary]])
        groups[name] = boundary[np.all(abs(distance) < 1e-12, axis=0)]
    union = np.concatenate(list(groups.values()))
    if len(union) != len(np.unique(union)) or set(union) != set(boundary):
        raise ValueError("Sector boundary tags/seams do not cover the boundary exactly")
    return mesh, groups


def cyclic_reduction(basis, groups):
    minus = basis.get_dofs(facets=groups["minus"]).all()
    plus = basis.get_dofs(facets=groups["plus"]).all()
    angle = SECTOR_ANGLE
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    distance, partner = cKDTree((rot@basis.doflocs[:, minus]).T).query(basis.doflocs[:, plus].T)
    if distance.max() > 1e-12 or len(set(partner)) != len(minus) or len(plus) != len(minus):
        raise ValueError("Nonmatching periodic FE degrees of freedom")
    origin = np.flatnonzero(np.linalg.norm(basis.doflocs, axis=0) < 1e-12)
    if len(origin) != 1:
        raise ValueError("Expected one central vertex DOF")
    parent, weight = np.arange(basis.N), np.ones(basis.N, complex)
    active = plus != origin[0]
    parent[plus[active]], weight[plus[active]] = minus[partner[active]], RHO
    representatives, inverse = np.unique(parent, return_inverse=True)
    transform = csr_matrix((weight, (np.arange(basis.N), inverse)), shape=(basis.N, len(representatives)))
    outer = basis.get_dofs(facets=groups["outer"]).all()
    electrode = basis.get_dofs(facets=groups["electrode"]).all()
    fixed = np.unique(inverse[np.r_[outer, electrode, origin]])
    values = np.zeros(len(representatives), complex)
    values[inverse[electrode]] = 1.
    return transform, fixed, values, {"max_seam_DOF_mismatch_m": float(distance.max()),
                                      "matched_seam_dofs": len(plus), "central_DOF": int(origin[0])}


def canonical_nodal_values(basis, values):
    """Check local H1 orientation explicitly; never assume facet DOF ordering."""
    v = basis.mesh.p[:, basis.mesh.t]
    expected = v[:, [0], :]+np.einsum("ijk,lj->ilk", v[:, 1:, :]-v[:, [0], :], basis.elem.doflocs)
    coordinates = basis.doflocs[:, basis.element_dofs]
    # Match actual DOFs to canonical reference points per cell; also detects
    # unsupported oriented bases before any polynomial is evaluated.
    distance = np.linalg.norm(expected[:, :, None, :]-coordinates[:, None, :, :], axis=0)
    assignment = np.argmin(distance, axis=1)
    if np.max(np.min(distance, axis=1)) > 1e-11 or np.any(np.diff(np.sort(assignment, axis=0), axis=0) == 0):
        raise ValueError("Cannot identify higher-order local DOF coordinates")
    if np.any(assignment != np.arange(assignment.shape[0])[:, None]):
        raise ValueError("Native basis edge orientation is inconsistent with nodal coordinates")
    dofs = np.take_along_axis(basis.element_dofs, assignment, axis=0)
    return values[dofs].T


@dataclass
class SectorSolution:
    points: np.ndarray
    triangles: np.ndarray
    coefficients: np.ndarray
    degree: int
    diagnostics: dict

    def save(self, path):
        np.savez_compressed(path, points=self.points, triangles=self.triangles,
                            coefficients=self.coefficients, degree=self.degree)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls(a["points"], a["triangles"], a["coefficients"], int(a["degree"]), {})


def solve_sector(path, degree=4):
    mesh, groups = sector_mesh(path)
    element = ElementTriP4() if degree == 4 else ElementTriP2() if degree == 2 else None
    if element is None:
        raise ValueError("Supported direct elements: P2 and P4")
    basis = Basis(mesh, element, intorder=2*degree)
    transform, fixed, values, diagnostics = cyclic_reduction(basis, groups)
    stiffness = asm(laplace_form, basis)
    reduced = (transform.conj().T@stiffness@transform).tocsc()
    interior = np.setdiff1d(np.arange(len(values)), fixed)
    rhs = -(reduced@values)[interior]
    matrix = reduced[interior][:, interior].tocsc()
    lu = splu(matrix)
    values[interior] = lu.solve(rhs)
    residual = (reduced@values)[interior]
    scale = float(np.asarray(abs(reduced).sum(axis=1)).max())
    backward_error = float(np.max(abs(residual))/scale)
    if backward_error > 1e-10:
        raise ValueError("Direct cyclic FEM residual too large")
    solution = transform@values
    nodal = canonical_nodal_values(basis, solution)
    vander = vandermonde(element.doflocs, degree)
    coefficients = np.linalg.solve(vander, nodal.T).T
    nodal_error = float(np.max(abs(coefficients@vander.T-nodal)))
    if nodal_error > 1e-10:
        raise ValueError("Native FE polynomial does not reproduce nodal values")
    diagnostics.update({"sector_dofs": int(basis.N), "reduced_dofs": len(values), "free_dofs": len(interior),
                        "scaled_discrete_residual_max": backward_error, "nodal_polynomial_error_V": nodal_error,
                        "degree": degree, "sector_triangles": mesh.nelements,
                        "RF_encoding": "Phi/U0=Im(H*exp(i*chi)), H(R30 x)=rho H(x), rho=exp(-2pi i/3)",
                        "all_FE_content_retained": True, "harmonic_fit_used": False})
    return SectorSolution(mesh.p.T, mesh.t.T, coefficients, degree, diagnostics)


def permutation_transform(permutation, degree):
    nodes = np.array([(i/degree, j/degree) for i, j in exponents(degree)])
    vertices = np.array([[0., 0.], [1., 0.], [0., 1.]])[list(permutation)]
    old = vertices[0]+nodes[:, [0]]*(vertices[1]-vertices[0])+nodes[:, [1]]*(vertices[2]-vertices[0])
    return np.linalg.solve(vandermonde(nodes, degree), vandermonde(old, degree))


def global_polynomials(sector, full_points, full_triangles):
    vertices = full_points[full_triangles]
    centroids = vertices.mean(axis=1)
    sector_vertices = sector.points[sector.triangles]
    tree = cKDTree(sector_vertices.mean(axis=1))
    phi = np.arctan2(centroids[:, 1], centroids[:, 0])
    rotation_ids = np.floor((phi+math.pi/12)/SECTOR_ANGLE).astype(int) % 12
    coefficients = np.empty((len(vertices), len(exponents(sector.degree))), complex)
    largest = 0.
    for k in range(12):
        selected = np.flatnonzero(rotation_ids == k)
        angle = k*SECTOR_ANGLE
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        local_vertices = vertices[selected]@rotation
        distance, ids = tree.query(local_vertices.mean(axis=1))
        largest = max(largest, float(distance.max()))
        if distance.max() > 1e-11:
            raise ValueError("Full mesh is not the twelve-fold copy of the saved sector")
        distances = np.linalg.norm(local_vertices[:, :, None]-sector_vertices[ids, None, :], axis=-1)
        permutation = np.argmin(distances, axis=2)
        if np.max(np.min(distances, axis=2)) > 1e-11:
            raise ValueError("Sector/full vertex mapping differs")
        for perm in itertools.permutations(range(3)):
            mask = np.all(permutation == perm, axis=1)
            if mask.any():
                transform = permutation_transform(perm, sector.degree)
                coefficients[selected[mask]] = RHO**k*(sector.coefficients[ids[mask]]@transform.T)
    return coefficients, largest


class DirectFEMRF:
    """Native elementwise gradient, no interelement gradient recovery/fitting."""
    def __init__(self, sector, trap_mesh, amplitude=250.):
        self.points, self.triangles = trap_mesh.points, trap_mesh.triangles
        self.mesh = trap_mesh.mesh
        self.degree = sector.degree
        coefficients, mismatch = global_polynomials(sector, self.points, self.triangles)
        self.potential_coefficients = coefficients*amplitude
        vertices = self.points[self.triangles]
        self.origin = vertices[:, 0]
        # D maps local reference coordinates to physical coordinates.
        self.inverse = np.linalg.inv(np.swapaxes(vertices[:, 1:]-vertices[:, [0]], 1, 2))
        self.gradient = np.concatenate((-self.inverse.sum(axis=1)[:, None], self.inverse), axis=1)
        self._triangulation = mtri.Triangulation(self.points[:, 0], self.points[:, 1], self.triangles)
        self._finder = self._triangulation.get_trifinder()
        self.neighbors = np.empty((len(self.triangles), 3), int)
        # scikit-fem t2f local edges: (v0,v1),(v1,v2),(v0,v2).
        for face, local_edge in enumerate((1, 2, 0)):
            f = self.mesh.t2f[local_edge]
            first, second = self.mesh.f2t[:, f]
            self.neighbors[:, face] = np.where(first == np.arange(len(first)), second, first)
        powers = exponents(sector.degree)
        lower = exponents(3)
        index = {power: i for i, power in enumerate(lower)}
        du, dv = np.zeros((len(vertices), 10), complex), np.zeros((len(vertices), 10), complex)
        for source, (i, j) in enumerate(powers):
            if i:
                du[:, index[(i-1, j)]] += i*self.potential_coefficients[:, source]
            if j:
                dv[:, index[(i, j-1)]] += j*self.potential_coefficients[:, source]
        self.gx = self.inverse[:, 0, 0, None]*du+self.inverse[:, 1, 0, None]*dv
        self.gy = self.inverse[:, 0, 1, None]*du+self.inverse[:, 1, 1, None]*dv
        self.last_id = None
        self.mapping_error_m = mismatch

    def find(self, x, y):
        i = self.last_id
        if i is not None:
            for _ in range(16):
                dx, dy = x-self.origin[i, 0], y-self.origin[i, 1]
                a = self.inverse[i]
                u, v = a[0, 0]*dx+a[0, 1]*dy, a[1, 0]*dx+a[1, 1]*dy
                l0 = 1-u-v
                if min(l0, u, v) >= -1e-12:
                    self.last_id = i
                    return i, u, v
                face = 0 if l0 <= u and l0 <= v else 1 if u <= v else 2
                i = int(self.neighbors[i, face])
                if i < 0:
                    break
        i = int(self._finder(x, y))
        if i < 0:
            raise InterpolationDomainError("Direct FEM query outside vacuum")
        self.last_id = i
        return self.local_coordinates(i, x, y)

    def local_coordinates(self, i, x, y):
        dx, dy = x-self.origin[i, 0], y-self.origin[i, 1]
        a = self.inverse[i]
        return i, a[0, 0]*dx+a[0, 1]*dy, a[1, 0]*dx+a[1, 1]*dy

    @staticmethod
    def cubic(c, u, v):
        # Powers: 1,u,v,u²,uv,v²,u³,u²v,uv²,v³.
        return c[0]+u*(c[1]+u*(c[3]+u*c[6])+v*(c[4]+u*c[7]+v*c[8]))+v*(c[2]+v*(c[5]+v*c[9]))

    def local_quadratures(self, i, u, v):
        x, y = self.cubic(self.gx[i], u, v), self.cubic(self.gy[i], u, v)
        return -x.imag, -y.imag, -x.real, -y.real

    def quadratures(self, x, y):
        return self.local_quadratures(*self.find(x, y))

    def sample(self, points):
        xy = np.asarray(points)
        ids = np.asarray(self._finder(xy[:, 0], xy[:, 1]), int)
        if np.any(ids < 0):
            raise InterpolationDomainError("Direct FEM sample outside vacuum")
        uv = np.einsum("nij,nj->ni", self.inverse[ids], xy-self.origin[ids])
        phi = np.sum(self.potential_coefficients[ids]*vandermonde(uv, self.degree), axis=1)
        dx = np.sum(self.gx[ids]*vandermonde(uv, 3), axis=1)
        dy = np.sum(self.gy[ids]*vandermonde(uv, 3), axis=1)
        return phi, np.stack((-dx.imag, -dy.imag), axis=-1), np.stack((-dx.real, -dy.real), axis=-1)

    def jacobian(self, x, y, angle, cell=None):
        i, u, v = self.find(x, y) if cell is None else self.local_coordinates(cell, x, y)
        derivatives = []
        for c in (self.gx[i], self.gy[i]):
            du = c[1]+2*u*c[3]+v*c[4]+3*u*u*c[6]+2*u*v*c[7]+v*v*c[8]
            dv = c[2]+u*c[4]+2*v*c[5]+u*u*c[7]+2*u*v*c[8]+3*v*v*c[9]
            derivatives.append(np.array([du, dv])@self.inverse[i])
        hessian = np.array(derivatives)
        return -hessian.imag*math.cos(angle)-hessian.real*math.sin(angle)
