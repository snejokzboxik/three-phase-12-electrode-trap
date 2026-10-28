"""Harmonic RF potential reconstruction shared by benchmarks and model B.

Coefficients describe potential at U0=1 V unless an explicit amplitude has
already been applied. No angular symmetry or mode selection is imposed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

@dataclass
class HarmonicQuadratures:
    """Phi = Re(Pc(z)) cos(Omega*t) + Re(Ps(z)) sin(Omega*t).

    z=(x+iy)/reference_radius; polynomial coefficients are in volts.
    The fit is to potential, so the reconstructed field is curl-free.
    """
    cosine: np.ndarray
    sine: np.ndarray
    reference_radius: float

    def potential_quadratures(self, points):
        xy = np.asarray(points)
        z = (xy[...,0]+1j*xy[...,1])/self.reference_radius
        return tuple(np.polynomial.polynomial.polyval(z,c).real for c in (self.cosine,self.sine))

    def field_quadratures(self, points):
        xy = np.asarray(points)
        z = (xy[...,0]+1j*xy[...,1])/self.reference_radius
        values = []
        for c in (self.cosine,self.sine):
            d = np.polynomial.polynomial.polyval(z,np.polynomial.polynomial.polyder(c))/self.reference_radius
            values.append(np.stack((-d.real,d.imag),axis=-1))
        return tuple(values)

    def jacobian_quadratures(self, points):
        xy = np.asarray(points)
        z = (xy[...,0]+1j*xy[...,1])/self.reference_radius
        values=[]
        for c in (self.cosine,self.sine):
            d = np.polynomial.polynomial.polyval(z,np.polynomial.polynomial.polyder(c,2))/self.reference_radius**2
            values.append(np.stack((np.stack((-d.real,d.imag),axis=-1),
                                    np.stack((d.imag,d.real),axis=-1)),axis=-2))
        return tuple(values)

    def field(self, points, angle):
        ec,es = self.field_quadratures(points)
        return ec*np.cos(angle)+es*np.sin(angle)

    def pseudo(self, points, charge, mass, omega):
        ec,es = self.field_quadratures(points)
        return charge**2*(np.sum(ec*ec,axis=-1)+np.sum(es*es,axis=-1))/(4*mass*omega**2)

    def force(self, points, charge, mass, omega):
        ec,es = self.field_quadratures(points)
        jc,js = self.jacobian_quadratures(points)
        return -charge**2/(2*mass*omega**2)*(np.einsum("...ij,...i->...j",jc,ec)+np.einsum("...ij,...i->...j",js,es))


def fit_harmonic_quadratures(points,cosine,sine,order,reference_radius):
    z=(points[:,0]+1j*points[:,1])/reference_radius
    columns=[np.ones(len(z))]
    for n in range(1,order+1):
        columns.extend(((z**n).real,(z**n).imag))
    design=np.column_stack(columns)
    coefficients,_,rank,singular=np.linalg.lstsq(design,np.column_stack((cosine,sine)),rcond=None)
    if rank!=design.shape[1]:
        raise ValueError("Rank deficient harmonic audit fit")
    polynomials=[]
    for j in range(2):
        c=np.zeros(order+1,dtype=complex)
        c[0]=coefficients[0,j]
        c[1:]=coefficients[1::2,j]-1j*coefficients[2::2,j]
        polynomials.append(c)
    return HarmonicQuadratures(*polynomials,reference_radius),float(singular[0]/singular[-1])


class ReconstructionDomainError(ValueError):
    """A query is outside the independently audited reconstruction disk."""


@dataclass
class RFReconstruction:
    """Unit-voltage potential quadratures with an explicit spatial domain.

    The domain is a computational restriction, not a certificate of FEM accuracy.
    DC is deliberately absent. RF phase is the voltage phase, not pattern angle.
    """
    unit_quadratures: HarmonicQuadratures
    valid_radius: float
    metadata: dict

    def __post_init__(self):
        q = self.unit_quadratures
        if not np.isfinite(self.valid_radius) or self.valid_radius <= 0:
            raise ValueError("valid_radius must be positive and finite")
        if not np.isfinite(q.reference_radius) or q.reference_radius <= 0:
            raise ValueError("reference_radius must be positive and finite")
        for coefficients in (q.cosine, q.sine):
            if np.ndim(coefficients) != 1 or len(coefficients) < 2 or not np.all(np.isfinite(coefficients)):
                raise ValueError("Expected finite one-dimensional polynomial coefficients")

    def check_points(self, points):
        xy = np.asarray(points, dtype=float)
        if xy.ndim < 1 or xy.shape[-1] != 2 or not np.all(np.isfinite(xy)):
            raise ValueError("Expected finite (..., 2) coordinates in metres")
        if np.any(np.linalg.norm(xy, axis=-1) > self.valid_radius * (1 + 1e-12)):
            raise ReconstructionDomainError("Query outside reconstruction disk")
        return xy

    def scaled(self, amplitude):
        if not np.isfinite(amplitude):
            raise ValueError("RF amplitude must be finite")
        q = self.unit_quadratures
        return HarmonicQuadratures(amplitude*q.cosine, amplitude*q.sine, q.reference_radius)

    def field(self, points, time, amplitude, frequency, phase=0.):
        """Full RF adapter for the later A_smooth control experiment."""
        if not np.isfinite(frequency) or frequency <= 0 or not np.isfinite(phase):
            raise ValueError("Expected positive frequency and finite phase")
        return self.scaled(amplitude).field(self.check_points(points), 2*np.pi*frequency*time+phase)

    def save(self, path: Path):
        q = self.unit_quadratures
        value = {"format_version": 1, "normalization": "potential quadratures at U0=1 V",
                 "voltage_law": "A=sin(angle), B=sin(angle-2pi/3), C=sin(angle+2pi/3)",
                 "reference_radius_m": q.reference_radius, "valid_radius_m": self.valid_radius,
                 "cosine_real": q.cosine.real.tolist(), "cosine_imag": q.cosine.imag.tolist(),
                 "sine_real": q.sine.real.tolist(), "sine_imag": q.sine.imag.tolist(),
                 "metadata": self.metadata}
        Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, expected_fingerprint=None):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value["format_version"] != 1 or value["normalization"] != "potential quadratures at U0=1 V":
            raise ValueError("Unsupported RF reconstruction format or normalization")
        if expected_fingerprint is not None and value["metadata"].get("fem_fingerprint") != expected_fingerprint:
            raise ValueError("RF reconstruction FEM fingerprint mismatch")
        for name in ("cosine", "sine"):
            if len(value[name+"_real"]) != len(value[name+"_imag"]):
                raise ValueError("Inconsistent polynomial coefficient lengths")
        q = HarmonicQuadratures(np.array(value["cosine_real"])+1j*np.array(value["cosine_imag"]),
                                np.array(value["sine_real"])+1j*np.array(value["sine_imag"]),
                                value["reference_radius_m"])
        return cls(q, value["valid_radius_m"], value["metadata"])

