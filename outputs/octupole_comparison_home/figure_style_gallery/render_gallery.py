"""Render four visual styles from the immutable, published V4 data.

No RF integrations, field solves, interpolation, or resampling of scientific
outcomes are performed here. Matplotlib is unavailable in the local runtime;
the repository's tested Pillow + ReportLab renderer supplies raster and vector
exports. The source V4 renderer supplies the audited figure data selection.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V4 = ROOT / "outputs/octupole_comparison_home/confinement_maps_v4"
SOURCE_RENDERER = ROOT / "outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py"
PNG_DPI = 600

STYLES = {
    "A": {
        "name": "APS / Physical Review",
        "font_regular": "C:/Windows/Fonts/times.ttf",
        "font_bold": "C:/Windows/Fonts/timesbd.ttf",
        "family": "Times New Roman, serif",
        "colors": {"ROT": "#145A8D", "STAND": "#B33C2E", "ADVANTAGE": "#246B57",
                   "STAND_ONLY": "#8A4D79", "BOTH": "#59636D", "BOTH_ESCAPE": "#BE9A38",
                   "INK": "#181A1B", "MUTED": "#555D63", "GRID": "#D6D9DB", "PALE": "#F7F7F5"},
        "line_width_scale": 1.0,
    },
    "B": {
        "name": "Nature-style contemporary",
        "font_regular": "C:/Windows/Fonts/arial.ttf",
        "font_bold": "C:/Windows/Fonts/arialbd.ttf",
        "family": "Arial, Helvetica, sans-serif",
        "colors": {"ROT": "#007F86", "STAND": "#D04A35", "ADVANTAGE": "#287B5A",
                   "STAND_ONLY": "#8D5D91", "BOTH": "#4C5B64", "BOTH_ESCAPE": "#D09B32",
                   "INK": "#1D2A32", "MUTED": "#53616A", "GRID": "#E1E6E8", "PALE": "#F4F7F8"},
        "line_width_scale": 1.12,
    },
    "C": {
        "name": "Classical monochrome",
        "font_regular": "C:/Windows/Fonts/cambria.ttc",
        "font_bold": "C:/Windows/Fonts/cambriab.ttf",
        "family": "Cambria, serif",
        "colors": {"ROT": "#111111", "STAND": "#585858", "ADVANTAGE": "#222222",
                   "STAND_ONLY": "#666666", "BOTH": "#858585", "BOTH_ESCAPE": "#B0B0B0",
                   "INK": "#111111", "MUTED": "#4C4C4C", "GRID": "#DDDDDD", "PALE": "#FAFAFA"},
        "line_width_scale": 1.18,
    },
    "D": {
        "name": "Modern editorial scientific",
        "font_regular": "C:/Windows/Fonts/georgia.ttf",
        "font_bold": "C:/Windows/Fonts/georgiab.ttf",
        "family": "Georgia, serif",
        "colors": {"ROT": "#536F7C", "STAND": "#B66A59", "ADVANTAGE": "#557C69",
                   "STAND_ONLY": "#82748E", "BOTH": "#58646A", "BOTH_ESCAPE": "#AD9150",
                   "INK": "#29343A", "MUTED": "#647078", "GRID": "#E8E6E0", "PALE": "#F7F6F2"},
        "line_width_scale": 1.25,
    },
}


def load_renderer():
    spec = importlib.util.spec_from_file_location("v4_publication_renderer", SOURCE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import V4 renderer: {SOURCE_RENDERER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def register_style_fonts(style):
    # The renderer asks for these stable ReportLab names; each style is rendered
    # in its own process so those aliases can point to the selected typeface.
    regular = Path(style["font_regular"])
    bold = Path(style["font_bold"])
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError(f"Style font is missing: {regular} / {bold}")
    if "PubArial" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("PubArial", str(regular), subfontIndex=0 if regular.suffix.lower() == ".ttc" else 0))
        pdfmetrics.registerFont(TTFont("PubArial-Bold", str(bold)))
        pdfmetrics.registerFontFamily("PubArial", normal="PubArial", bold="PubArial-Bold",
                                      italic="PubArial", boldItalic="PubArial-Bold")
    return regular, bold


def configure_style(mod, key, style):
    colors = style["colors"]
    for name, color in colors.items():
        setattr(mod, name, color)
    mod.MODEL_COLOR = {"rotating": colors["ROT"], "standing": colors["STAND"]}
    mod.CLASS_COLOR = {
        "rotating_only_retained": colors["ADVANTAGE"],
        "standing_only_retained": colors["STAND_ONLY"],
        "both_retained": colors["BOTH"],
        "both_escaped": colors["BOTH_ESCAPE"],
    }
    out = HERE / f"style_{key.lower()}"
    out.mkdir(parents=True, exist_ok=True)
    mod.FIGURES = out
    mod.register_fonts = lambda: register_style_fonts(style)
    old_canvas = mod.FigureCanvas

    class StyledCanvas(old_canvas):
        def __init__(self, name, width_mm, height_mm, layout):
            super().__init__(name, width_mm, height_mm, layout)
            self.regular, self.bold = register_style_fonts(style)
            self.svg[2] = f'<style>text{{font-family:{style["family"]};fill:{colors["INK"]}}}</style>'
            self.svg[1] = f'<rect width="100%" height="100%" fill="white"/>'
            self.line_scale = style["line_width_scale"]

        def line(self, x1, y1, x2, y2, color=colors["INK"], lw=0.22, dash=None):
            return super().line(x1, y1, x2, y2, color, lw*self.line_scale, dash)

        def text(self, x, y, text, *args, **kwargs):
            # Use portable, legible subscript notation in embedded fonts and SVG.
            text = str(text).replace("E⊥", "E_perp").replace("E₀", "E0").replace("r₀", "r0").replace("φRF", "φ_RF")
            return super().text(x, y, text, *args, **kwargs)

        def polyline(self, points, color=colors["INK"], lw=0.22, dash=None):
            # Monochrome variant distinguishes the two models by stroke pattern.
            if key == "C" and dash is None and color == colors["STAND"]:
                dash = (1.0, 0.65)
            return super().polyline(points, color, lw*self.line_scale, dash)

    mod.FigureCanvas = StyledCanvas
    return out


def shell_data(pairs):
    energies = sorted({float(p["energy_eV"]) for p in pairs})
    out = []
    for energy in energies:
        group = [p for p in pairs if abs(float(p["energy_eV"])-energy) < 1e-11]
        n = len(group)
        kr = sum(p["rotating_status"] == "retained" for p in group)
        ks = sum(p["standing_status"] == "retained" for p in group)
        out.append({"energy":energy,"n":n,"rotating":kr/n,"standing":ks/n,"delta":(kr-ks)/n,
                    "balanced":energy in (1.,2.,3.)})
    return out


def _ax(mod, c, bounds, xlim, ylim, xticks, yticks, xfmt, yfmt):
    x,y,w,h=bounds
    return mod.Axes(c,x,y,w,h,xlim,ylim,xticks,yticks,xfmt=xfmt,yfmt=yfmt)


def _legend(mod,c,x,y,shape,color,label,fill=True,size=.85,text_size=6.0):
    mod.legend_item(c,x,y,shape,color,label,fill=fill,size=size,text_size=text_size)


def energy_figure(mod, pairs, key, layout):
    data=shell_data(pairs); base=mod.BASE_ENERGIES
    if key == "B":
        if layout == "double":
            w,h=178,126; c=mod.FigureCanvas("Fig1_energy_finite_time_confinement",w,h,layout)
            c.text(w/2,7,"Finite-time retention across the calculated energy shells",size=10.8,weight="bold")
            c.text(w/2,14,"100 kTh ion · 100 kHz · 250 V peak per model · 1000 RF periods · 8 mm aperture",size=6.7,color=mod.MUTED)
            a=_ax(mod,c,(22,25,88,56),(.37,4.13),(0,1.08),[.5,1,2,3,4],[0,.25,.5,.75,1],lambda x:f"{x:g}",lambda y:f"{y:g}")
            b=_ax(mod,c,(128,25,38,56),(.37,4.13),(-1.05,1.08),[.5,2,4],[-1,-.5,0,.5,1],lambda x:f"{x:g}",lambda y:f"{y:g}")
            mod.panel_title(c,15,20,"A","Model retention",7.6)
            mod.panel_title(c,121,20,"B","Paired Δ",7.6)
            for d in data:
                filled=d["balanced"]
                a.marker(d["energy"],d["rotating"],"circle",mod.ROT,size=1.05,fill=filled)
                a.marker(d["energy"],d["standing"],"triangle",mod.STAND,size=1.05,fill=filled)
                b.marker(d["energy"],d["delta"],"circle",mod.ADVANTAGE if d["delta"]>0 else mod.STAND_ONLY if d["delta"]<0 else mod.BOTH,size=1.05,fill=filled)
            a.ylabel("Retained fraction",x=16,size=6.8); a.xlabel("Initial transverse energy E_perp (eV)",y=85,size=6.6)
            b.ylabel("Rotating − standing",x=120,size=6.1); b.xlabel("E_perp (eV)",y=85,size=6.4)
            _legend(mod,c,28,96,"circle",mod.ROT,"Rotating",True,.8,5.8)
            _legend(mod,c,52,96,"triangle",mod.STAND,"Standing",True,.8,5.8)
            _legend(mod,c,80,96,"circle",mod.INK,"Filled: balanced",True,.8,5.8)
            _legend(mod,c,121,96,"circle",mod.INK,"Open: adaptive",False,.8,5.8)
            c.text(18,108,"n matched",size=5.8,color=mod.MUTED,ha="left")
            span=132/len(data)
            for i,d in enumerate(data):
                xx=34+(i+.5)*span
                c.text(xx,107,f"{d['energy']:g}",size=5.1,color=mod.MUTED)
                c.text(xx,113,str(d["n"]),size=5.7,weight="bold")
            c.text(w/2,122,"Points show computed shells only; filled markers denote the balanced 1, 2, 3 eV grid.",size=5.8,color=mod.MUTED)
        else:
            w,h=89,222; c=mod.FigureCanvas("Fig1_energy_finite_time_confinement",w,h,layout)
            c.text(w/2,7,"Finite-time retention by energy",size=9.3,weight="bold")
            c.text(w/2,13,"100 kTh · 100 kHz · 250 V · 1000 RF periods",size=5.9,color=mod.MUTED)
            a=_ax(mod,c,(20,23,58,52),(.37,4.13),(0,1.08),[.5,1,2,3,4],[0,.5,1],lambda x:f"{x:g}",lambda y:f"{y:g}")
            b=_ax(mod,c,(20,105,58,43),(.37,4.13),(-1.05,1.08),[.5,1,2,3,4],[-1,0,1],lambda x:f"{x:g}",lambda y:f"{y:g}")
            mod.panel_title(c,15,18,"A","Rotating and standing",7.1)
            mod.panel_title(c,15,100,"B","Paired difference",7.1)
            for d in data:
                a.marker(d["energy"],d["rotating"],"circle",mod.ROT,size=.95,fill=d["balanced"])
                a.marker(d["energy"],d["standing"],"triangle",mod.STAND,size=.95,fill=d["balanced"])
                b.marker(d["energy"],d["delta"],"circle",mod.ADVANTAGE if d["delta"]>0 else mod.STAND_ONLY if d["delta"]<0 else mod.BOTH,size=.95,fill=d["balanced"])
            a.ylabel("Retained fraction",x=15,size=6.0); a.xlabel("E_perp (eV)",y=82,size=6.1)
            b.ylabel("Rotating − standing",x=15,size=5.7); b.xlabel("Initial energy E_perp (eV)",y=158,size=6.0)
            _legend(mod,c,13,173,"circle",mod.ROT,"Rotating",True,.75,5.3)
            _legend(mod,c,44,173,"triangle",mod.STAND,"Standing",True,.75,5.3)
            _legend(mod,c,13,181,"circle",mod.INK,"Filled: balanced",True,.7,5.0)
            _legend(mod,c,47,181,"circle",mod.INK,"Open: adaptive",False,.7,5.0)
            c.text(7,193,"E_perp / n matched",size=5.0,color=mod.MUTED,ha="left")
            for i,d in enumerate(data):
                row=0 if i<6 else 1; col=i if i<6 else i-6
                xx=10+col*12.0; yy=202+row*8
                c.text(xx,yy,f"{d['energy']:g}: {d['n']}",size=4.9,color=mod.INK,ha="left")
            c.text(w/2,218,"Discrete calculated shells; descriptive grid fractions.",size=5.0,color=mod.MUTED)
        return c.save()

    if key == "C":
        # A monochrome paired-dot/lollipop plot uses line segments to display
        # the rotating-standing difference without duplicating three panels.
        w,h=(178,128) if layout=="double" else (89,126)
        c=mod.FigureCanvas("Fig1_energy_finite_time_confinement",w,h,layout)
        c.text(w/2,7,"Matched retention by initial energy",size=10.3 if layout=="double" else 9.0,weight="bold")
        c.text(w/2,13,"Black circle: rotating · gray square: standing · open: adaptive shell",size=6.4 if layout=="double" else 5.4,color=mod.MUTED)
        bounds=(19,23,151,57) if layout=="double" else (18,21,63,58)
        a=_ax(mod,c,bounds,(.37,4.13),(0,1.08),[.5,1,2,3,4],[0,.25,.5,.75,1],lambda x:f"{x:g}",lambda y:f"{y:g}")
        mod.panel_title(c,bounds[0]-5,bounds[1]-5,"A","Paired outcomes; connectors join identical launches",7.5 if layout=="double" else 6.5)
        for d in data:
            e=d["energy"]; filled=d["balanced"]
            if abs(d["rotating"]-d["standing"])>1e-12:
                a.line(e,d["rotating"],e,d["standing"],mod.MUTED,.42)
            a.marker(e,d["rotating"],"circle",mod.ROT,size=1.18 if layout=="double" else 1.0,fill=filled)
            a.marker(e,d["standing"],"square",mod.STAND,size=1.18 if layout=="double" else 1.0,fill=filled)
        a.ylabel("Retained fraction",x=bounds[0]-6,size=6.8 if layout=="double" else 5.8)
        a.xlabel("Initial transverse energy E_perp (eV)",y=bounds[1]+bounds[3]+8,size=6.5 if layout=="double" else 5.8)
        yleg=91 if layout=="double" else 91
        _legend(mod,c,23 if layout=="double" else 12,yleg,"circle",mod.ROT,"Rotating",True,.8,5.8)
        _legend(mod,c,54 if layout=="double" else 40,yleg,"square",mod.STAND,"Standing",True,.8,5.8)
        _legend(mod,c,87 if layout=="double" else 12,yleg+7,"circle",mod.INK,"Filled: balanced",True,.75,5.4)
        _legend(mod,c,131 if layout=="double" else 43,yleg+7,"circle",mod.INK,"Open: adaptive",False,.75,5.4)
        if layout=="double":
            xs=[35+(i+.5)*132/len(data) for i in range(len(data))]
            c.text(19,105,"E (eV)",size=5.2,color=mod.MUTED,ha="left")
            c.text(19,112,"n matched",size=5.2,color=mod.MUTED,ha="left")
            for xx,d in zip(xs,data):
                c.text(xx,105,f"{d['energy']:g}",size=4.9,color=mod.INK)
                c.text(xx,112,str(d["n"]),size=5.3,color=mod.INK,weight="bold")
            c.text(w/2,122,"Sample counts by shell; retention outcomes remain deterministic grid descriptors.",size=5.4,color=mod.MUTED)
        else:
            c.text(w/2,108,"All markers are computed shells; connector length is the paired retention difference.",size=4.8,color=mod.MUTED)
            c.text(w/2,118,"Balanced launches: 1, 2, 3 eV · adaptive shells remain separate by fill.",size=4.8,color=mod.MUTED)
        return c.save()

    # Editorial style: a paired horizontal slope plot places energy on the
    # vertical coordinate and retention fraction on the horizontal coordinate.
    w,h=(178,145) if layout=="double" else (89,206)
    c=mod.FigureCanvas("Fig1_energy_finite_time_confinement",w,h,layout)
    c.text(w/2,7,"Retention envelope across computed energy shells",size=10.5 if layout=="double" else 9.2,weight="bold")
    c.text(w/2,14,"A paired coordinate view of the deterministic 1000-period launches",size=6.5 if layout=="double" else 5.8,color=mod.MUTED)
    bounds=(22,24,112,83) if layout=="double" else (20,27,59,126)
    a=_ax(mod,c,bounds,(0,1.08),(.37,4.13),[0,.25,.5,.75,1],[.5,1,2,3,4],lambda x:f"{x:g}",lambda y:f"{y:g}")
    mod.panel_title(c,bounds[0]-7,bounds[1]-5,"A","Connected markers are paired outcomes",7.5 if layout=="double" else 6.5)
    for d in data:
        e=d["energy"]; filled=d["balanced"]
        if abs(d["rotating"]-d["standing"])>1e-12:
            a.line(d["rotating"],e,d["standing"],e,mod.GRID,.5)
        a.marker(d["rotating"],e,"circle",mod.ROT,size=1.25 if layout=="double" else 1.0,fill=filled)
        a.marker(d["standing"],e,"square",mod.STAND,size=1.2 if layout=="double" else .95,fill=filled)
        c.text(bounds[0]+bounds[2]+6,a.Y(e),f"n={d['n']}",size=5.2 if layout=="double" else 4.7,color=mod.MUTED,ha="left")
    a.xlabel("Retained fraction",y=bounds[1]+bounds[3]+8,size=6.6 if layout=="double" else 5.8)
    a.ylabel("Initial transverse energy E_perp (eV)",x=bounds[0]-8,size=6.7 if layout=="double" else 5.7)
    if layout=="double":
        _legend(mod,c,27,126,"circle",mod.ROT,"Rotating",True,.85,6.0)
        _legend(mod,c,58,126,"square",mod.STAND,"Standing",True,.85,6.0)
        _legend(mod,c,92,126,"circle",mod.INK,"Filled: balanced",True,.8,5.8)
        _legend(mod,c,135,126,"circle",mod.INK,"Open: adaptive",False,.8,5.8)
        c.text(w/2,140,"Energy coordinates retain their numerical spacing; no shell is interpolated.",size=5.7,color=mod.MUTED)
    else:
        _legend(mod,c,13,169,"circle",mod.ROT,"Rotating",True,.8,5.3)
        _legend(mod,c,45,169,"square",mod.STAND,"Standing",True,.8,5.3)
        _legend(mod,c,13,179,"circle",mod.INK,"Filled: balanced",True,.75,5.0)
        _legend(mod,c,47,179,"circle",mod.INK,"Open: adaptive",False,.75,5.0)
        c.text(w/2,198,"Per-shell sample counts are printed beside each matched pair.",size=4.9,color=mod.MUTED)
    return c.save()


def trajectory_figure(mod, main_rows, key, layout):
    traces=mod.get_case_traces(main_rows)
    colors=mod.MODEL_COLOR
    if key == "A":
        return mod.figure3(main_rows,layout)
    if key == "B":
        w,h=(178,126) if layout=="double" else (89,220)
        c=mod.FigureCanvas("Fig3_matched_trajectories_and_radius",w,h,layout)
        c.text(w/2,7,"One matched launch separates the two trajectories",size=10.4 if layout=="double" else 8.9,weight="bold")
        c.text(w/2,14,"1 eV · r0=1 mm · azimuth 45° · RF phase 0° · + tangent · 250 V each",size=6.3 if layout=="double" else 5.5,color=mod.MUTED)
        if layout=="double":
            boxes=[(16,26,48,46),(73,26,48,46),(130,26,37,46)]
            titles=["Rotating x–y","Standing x–y","Radius, 0–1000 RF"]
            timebox=(130,88,37,20)
            c.text(w/2,22,"Aperture radius 8 mm is the circular boundary",size=5.7,color=mod.MUTED)
        else:
            boxes=[(7,26,34,48),(48,26,34,48),(17,106,62,43)]
            titles=["Rotating x–y","Standing x–y","Radius, 0–1000 RF"]
            timebox=(17,166,62,26)
        for i,((x,y,pw,ph),title,model) in enumerate(zip(boxes[:2],titles[:2],("rotating","standing"))):
            a=_ax(mod,c,(x+3,y+4,pw-5,ph-8),(-8.6,8.6),(-8.6,8.6),[-8,0,8],[-8,0,8],lambda v:f"{v:g}",lambda v:f"{v:g}")
            mod.panel_title(c,x,y-2,"AB"[i],title,6.4 if layout=="double" else 5.6)
            cx=a.X(0); cy=a.Y(0); rx=a.w*8/17.2; ry=a.h*8/17.2
            ring=[(cx+rx*math.cos(i*2*math.pi/240),cy-ry*math.sin(i*2*math.pi/240)) for i in range(241)]
            c.polyline(ring,mod.INK,.3,dash=(1.1,.7))
            _,rows=traces[model]
            pts=[(float(q["x_mm"]),float(q["y_mm"])) for q in rows]
            a.path(pts,colors[model],.22 if layout=="double" else .18,dash=(1.2,.8) if key=="C" and model=="standing" else None)
            a.xlabel("x (mm)",y=y+ph+3,size=5.2 if layout=="double" else 4.8)
            a.ylabel("y (mm)",x=x,size=5.2 if layout=="double" else 4.8)
        x,y,pw,ph=boxes[2]
        a=_ax(mod,c,(x+5,y+4,pw-7,ph-8),(0,1000),(0,8.6),[0,500,1000],[0,4,8],lambda v:f"{int(v)}",lambda v:f"{v:g}")
        mod.panel_title(c,x,y-2,"C",titles[2],6.4 if layout=="double" else 5.6)
        for model in ("rotating","standing"):
            _,rows=traces[model]
            pts=[(float(q["rf_period"]),float(q["r_mm"])) for q in rows]
            a.path(pts,colors[model],.24 if layout=="double" else .2,dash=(1.1,.7) if key=="C" and model=="standing" else None)
        a.line(0,8,1000,8,mod.INK,.28,dash=(1.2,.8))
        a.xlabel("Time (RF periods)",y=y+ph+3,size=5.2 if layout=="double" else 4.8)
        a.ylabel("r (mm)",x=x+1,size=5.1 if layout=="double" else 4.7)
        if layout=="double":
            ix,iy,iw,ih=timebox
        else:
            ix,iy,iw,ih=timebox
        c.text(ix+iw/2,iy-2,"Early-time detail: 0–25 RF periods",size=5.1 if layout=="double" else 4.8,color=mod.MUTED)
        ia=_ax(mod,c,(ix+6,iy+2,iw-8,ih-5),(0,25),(0,8.6),[0,10,20],[0,4,8],lambda v:f"{int(v)}",lambda v:f"{v:g}")
        for model in ("rotating","standing"):
            _,rows=traces[model]
            pts=[(float(q["rf_period"]),float(q["r_mm"])) for q in rows if float(q["rf_period"])<=25]
            ia.path(pts,colors[model],.2,dash=(1.0,.7) if key=="C" and model=="standing" else None)
        ia.line(0,8,25,8,mod.INK,.2,dash=(1.0,.7))
        if layout=="double":
            _legend(mod,c,25,114,"circle",colors["rotating"],"Rotating · censored at 1000 RF",True,.85,5.8)
            _legend(mod,c,100,114,"triangle",colors["standing"],"Standing · escape at 21.31 RF",True,.85,5.8)
            c.text(w/2,123,"Each x–y panel uses the same scale; dashed circles mark the 8 mm aperture.",size=5.5,color=mod.MUTED)
        else:
            _legend(mod,c,14,202,"circle",colors["rotating"],"Rotating · right-censored at 1000 RF",True,.75,5.2)
            _legend(mod,c,14,210,"triangle",colors["standing"],"Standing · escape at 21.31 RF",True,.75,5.2)
            c.text(w/2,218,"Right-censored survivor is shown through its recorded horizon.",size=4.9,color=mod.MUTED)
        return c.save()

    # C and D share the validated trace pair but use different geometry and
    # time-window compositions from the A/B layouts.
    if key == "C":
        w,h=(178,112) if layout=="double" else (89,208)
        c=mod.FigureCanvas("Fig3_matched_trajectories_and_radius",w,h,layout)
        c.text(w/2,7,"Orbital paths and first passage",size=10.2 if layout=="double" else 8.9,weight="bold")
        c.text(w/2,14,"Matched 1 eV launch · r0=1 mm · azimuth 45° · phase 0° · 250 V",size=6.3 if layout=="double" else 5.5,color=mod.MUTED)
        if layout=="double":
            xy=(17,25,62,62); full=(96,27,70,42); zoom=(96,78,70,19)
        else:
            xy=(17,24,62,62); full=(17,106,66,38); zoom=(17,164,66,20)
        x,y,pw,ph=xy
        a=_ax(mod,c,(x+4,y+4,pw-8,ph-8),(-8.7,8.7),(-8.7,8.7),[-8,0,8],[-8,0,8],lambda v:f"{v:g}",lambda v:f"{v:g}")
        mod.panel_title(c,x,y-3,"A","Matched x–y geometry; aperture is dashed",7.0 if layout=="double" else 6.2)
        ring=[(a.X(0)+a.w*8/17.4*math.cos(i*2*math.pi/240),a.Y(0)-a.h*8/17.4*math.sin(i*2*math.pi/240)) for i in range(241)]
        c.polyline(ring,mod.INK,.3,dash=(1.1,.7))
        for model in ("rotating","standing"):
            _,rows=traces[model]; a.path([(float(q["x_mm"]),float(q["y_mm"])) for q in rows],colors[model],.22,dash=(1.1,.7) if model=="standing" else None)
        a.xlabel("x (mm)",y=y+ph+1,size=5.7); a.ylabel("y (mm)",x=x+1,size=5.7)
        for label,box,xlim in (("B",full,(0,1000)),("C",zoom,(0,25))):
            xx,yy,ww,hh=box
            aa=_ax(mod,c,(xx+6,yy+4,ww-8,hh-8),xlim,(0,8.6),[xlim[0],xlim[1]],[0,4,8],lambda v:f"{int(v)}",lambda v:f"{v:g}")
            mod.panel_title(c,xx,yy-3,label,"Radius history",6.6 if layout=="double" else 5.9)
            for model in ("rotating","standing"):
                _,rows=traces[model]
                pts=[(float(q["rf_period"]),float(q["r_mm"])) for q in rows if xlim[0]<=float(q["rf_period"])<=xlim[1]]
                aa.path(pts,colors[model],.24,dash=(1.2,.7) if model=="standing" else None)
            aa.line(xlim[0],8,xlim[1],8,mod.INK,.25,dash=(1,.7))
            aa.xlabel("RF periods",y=yy+hh+1,size=5.1); aa.ylabel("r (mm)",x=xx+1,size=5.1)
        legend_y=99 if layout=="double" else 192
        _legend(mod,c,23 if layout=="double" else 11,legend_y,"circle",colors["rotating"],"Rotating · survived to 1000 RF",True,.8,5.7)
        _legend(mod,c,91 if layout=="double" else 11,legend_y+6,"square",colors["standing"],"Standing · first passage 21.31 RF",True,.8,5.7)
        c.text(w/2,h-3,"Retained status is right-censored at 1000 RF; the standing trajectory terminates at r=8 mm.",size=5.5 if layout=="double" else 4.8,color=mod.MUTED)
        return c.save()

    # Editorial timeline is deliberately split into independently labelled
    # windows instead of compressing early escape against a long survivor.
    w,h=(178,126) if layout=="double" else (89,220)
    c=mod.FigureCanvas("Fig3_matched_trajectories_and_radius",w,h,layout)
    c.text(w/2,7,"Matched orbit and two time scales",size=10.5 if layout=="double" else 9.0,weight="bold")
    c.text(w/2,14,"1 eV · 1 mm start radius · 45° azimuth · 0° RF phase · positive tangent",size=6.3 if layout=="double" else 5.6,color=mod.MUTED)
    if layout=="double":
        xy=(14,25,62,68); tshort=(91,28,75,34); tlong=(91,78,75,29)
    else:
        xy=(16,23,58,58); tshort=(17,97,66,35); tlong=(17,153,66,35)
    x,y,pw,ph=xy
    a=_ax(mod,c,(x+4,y+4,pw-8,ph-8),(-8.7,8.7),(-8.7,8.7),[-8,0,8],[-8,0,8],lambda v:f"{v:g}",lambda v:f"{v:g}")
    mod.panel_title(c,x,y-3,"A","Rotating and standing x–y trajectories",6.8 if layout=="double" else 6.0)
    ring=[(a.X(0)+a.w*8/17.4*math.cos(i*2*math.pi/240),a.Y(0)-a.h*8/17.4*math.sin(i*2*math.pi/240)) for i in range(241)]
    c.polyline(ring,mod.INK,.32,dash=(1.1,.75))
    for model in ("rotating","standing"):
        _,rows=traces[model]; a.path([(float(q["x_mm"]),float(q["y_mm"])) for q in rows],colors[model],.22)
    a.xlabel("x (mm)",y=y+ph+1,size=5.6); a.ylabel("y (mm)",x=x+1,size=5.6)
    for label,box,xlim in (("B",tshort,(0,25)),("C",tlong,(0,1000))):
        xx,yy,ww,hh=box
        aa=_ax(mod,c,(xx+7,yy+4,ww-10,hh-9),xlim,(0,8.6),[xlim[0],xlim[1]],[0,4,8],lambda v:f"{int(v)}",lambda v:f"{v:g}")
        mod.panel_title(c,xx,yy-3,label,"Early passage" if label=="B" else "Full observation",6.8 if layout=="double" else 5.8)
        for model in ("rotating","standing"):
            _,rows=traces[model]
            pts=[(float(q["rf_period"]),float(q["r_mm"])) for q in rows if xlim[0]<=float(q["rf_period"])<=xlim[1]]
            aa.path(pts,colors[model],.23)
        aa.line(xlim[0],8,xlim[1],8,mod.INK,.28,dash=(1.1,.7))
        aa.xlabel("RF periods",y=yy+hh+1,size=5.2); aa.ylabel("r (mm)",x=xx+1,size=5.2)
    if layout=="double":
        _legend(mod,c,26,111,"circle",colors["rotating"],"Rotating · right-censored",True,.85,5.8)
        _legend(mod,c,88,111,"square",colors["standing"],"Standing · escape at 21.31 RF",True,.85,5.8)
        c.text(w/2,123,"Aperture marked at 8 mm in the x–y orbit and radial time traces.",size=5.5,color=mod.MUTED)
    else:
        _legend(mod,c,13,202,"circle",colors["rotating"],"Rotating · retained to 1000 RF",True,.75,5.2)
        _legend(mod,c,13,210,"square",colors["standing"],"Standing · escaped at 21.31 RF",True,.75,5.2)
        c.text(w/2,218,"The survivor remains right-censored through the horizon.",size=4.9,color=mod.MUTED)
    return c.save()


def make_previews(key, style_name, out):
    energy = Image.open(out / "Fig1_energy_finite_time_confinement_double.png").convert("RGB")
    traj = Image.open(out / "Fig3_matched_trajectories_and_radius_double.png").convert("RGB")
    width = 1800
    def fit(img, width):
        h = round(img.height*width/img.width)
        return img.resize((width, h), Image.Resampling.LANCZOS)
    energy = fit(energy, width)
    traj = fit(traj, width)
    header_h, gutter = 110, 38
    canvas = Image.new("RGB", (width, header_h+energy.height+gutter+traj.height+40), "white")
    draw = ImageDraw.Draw(canvas)
    fpath = STYLES[key]["font_bold"]
    font = ImageFont.truetype(fpath, 48, index=0)
    small = ImageFont.truetype(STYLES[key]["font_regular"], 26, index=0)
    draw.text((55, 23), f"STYLE {key}  |  {style_name}", fill="#202830", font=font)
    draw.text((58, 77), "Energy-dependent retention  ·  matched rotating / standing trajectories", fill="#57636B", font=small)
    canvas.paste(energy, (0, header_h))
    canvas.paste(traj, (0, header_h+energy.height+gutter))
    dest = HERE / f"STYLE_{key}_PREVIEW.png"
    canvas.save(dest, dpi=(220,220), optimize=True)
    return dest


def main():
    if len(sys.argv) == 1:
        for key in STYLES:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--style", key], check=True)
        make_overview()
        return
    if len(sys.argv) != 3 or sys.argv[1] != "--style" or sys.argv[2] not in STYLES:
        raise SystemExit("Usage: python render_gallery.py [--style A|B|C|D]")
    selected = sys.argv[2]
    mod = load_renderer()
    # Scientific content is selected and validated by the same data audit used
    # for the V4 publication figures.
    main_rows, pairs, _ = mod.load_and_validate()
    checksums = {}
    for name in ("equal_peak_main_grid.csv", "matched_pairs_equal_peak.csv"):
        path = V4/name
        checksums[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    manifest_path = HERE/"gallery_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"source": "V4 equal-peak matched data only", "source_checksums": checksums,
                    "styles": {}, "matplotlib": "unavailable; Pillow + ReportLab vector renderer used",
                    "integration_performed": False, "interpolation_or_smoothing": False,
                    "right_censoring": "retained trajectories remain right-censored at 1000 RF; escape times use first passage",
            "content": "Identical V4 energy-shell outcomes and validated matched trajectory pair; chart composition varies by style",
                    "population_claims": "none; descriptive fractions over the designed deterministic grid"}
    if manifest["source_checksums"] != checksums:
        raise ValueError("V4 source checksums changed while rendering style variants.")
    manifest["content"] = "Identical V4 energy-shell outcomes and validated matched trajectory pair; chart composition varies by style"
    (HERE/"gallery_config.json").write_text(json.dumps({
        "source_protocol":"V4 equal_peak_voltage, h35 N48 field",
        "source_checksums":checksums,
        "ion":{"mz_Th":100000,"charge_state":1},
        "rf":{"frequency_Hz":100000,"rotating_voltage_V":250,"standing_voltage_V":250,"horizon_RF":1000},
        "aperture_radius_mm":8,
        "energy_shells_eV":[d["energy"] for d in shell_data(pairs)],
        "matched_counts_by_energy":{"%.6g"%d["energy"]:d["n"] for d in shell_data(pairs)},
        "balanced_energies_eV":[1,2,3],
        "adaptive_energy_samples":"all other shells; no outcomes interpolated",
        "retention":"survival through 1000 RF periods; right-censored",
        "representative_trace":{"energy_eV":1,"r0_mm":1,"azimuth_deg":45,"phase_deg":0,"direction":"positive tangential",
                                "rotating":"retained through 1000 RF","standing":"first passage at 21.309 RF (rounded 21.31)"},
        "rendering":{"formats":["SVG","PDF","PNG"],"png_dpi":PNG_DPI,"layouts":{"single_column_mm":89,"double_column_mm":178},
                     "backend":"Pillow and ReportLab; Matplotlib unavailable"},
        "styles":{k:{"name":v["name"],"family":v["family"],"colors":v["colors"],"line_width_scale":v["line_width_scale"]} for k,v in STYLES.items()},
        "scientific_claims":"descriptive deterministic samples; no population probability or permanent-stability inference"
    },indent=2)+"\n",encoding="utf-8")
    style = STYLES[selected]
    out = configure_style(mod, selected, style)
    if selected == "A":
        energy_outputs = [mod.figure1(pairs, "single"), mod.figure1(pairs, "double")]
    else:
        energy_outputs = [energy_figure(mod,pairs,selected,"single"), energy_figure(mod,pairs,selected,"double")]
    trajectory_outputs = [trajectory_figure(mod,main_rows,selected,"single"),
                          trajectory_figure(mod,main_rows,selected,"double")]
    outputs = energy_outputs + trajectory_outputs
    manifest["styles"][selected] = {"name": style["name"], "outputs": outputs,
                                    "preview": f"STYLE_{selected}_PREVIEW.png"}
    make_previews(selected, style["name"], out)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Rendered style {selected}: {len(outputs)} figures")


def make_overview():
    paths = [HERE/f"STYLE_{key}_PREVIEW.png" for key in STYLES]
    thumb_w = 650
    thumbs=[]
    for p in paths:
        im=Image.open(p).convert("RGB")
        h=round(im.height*thumb_w/im.width)
        thumbs.append(im.resize((thumb_w,h),Image.Resampling.LANCZOS))
    gap=24; label_h=56; cols=2; rows=2
    cell_h=max(x.height for x in thumbs)+label_h
    sheet=Image.new("RGB",(gap+cols*(thumb_w+gap),gap+rows*(cell_h+gap)),"#ECEFF1")
    draw=ImageDraw.Draw(sheet)
    label_font=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",26)
    for i,(key,im) in enumerate(zip(STYLES,thumbs)):
        row,col=divmod(i,cols); x=gap+col*(thumb_w+gap); y=gap+row*(cell_h+gap)
        draw.rounded_rectangle((x-2,y-2,x+thumb_w+2,y+cell_h+2),radius=4,fill="white",outline="#D4DADF",width=2)
        draw.text((x+8,y+5),f"{key} · {STYLES[key]['name']}",font=label_font,fill="#1E2930")
        sheet.paste(im,(x,y+label_h))
    sheet.save(HERE/"ALL_STYLES_COMPARISON.png",dpi=(180,180),optimize=True)


if __name__ == "__main__":
    main()
