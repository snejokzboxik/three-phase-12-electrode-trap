"""Render publication-sized V4 confinement figures from the published V4 data.

No trajectory or field calculation is performed here.  The renderer writes
vector SVG/PDF and 600-dpi PNG in both single- and double-column layouts.
It uses Pillow and ReportLab because Matplotlib is not installed in the
available Python environment and package-network access is disabled.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import portrait
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas


HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "confinement_maps_v4"
FIGURES = HERE / "figures"
PNG_DPI = 600
ROT = "#0072B2"
STAND = "#D55E00"
ADVANTAGE = "#009E73"
STAND_ONLY = "#AA4499"
BOTH = "#58616A"
BOTH_ESCAPE = "#D69F00"
INK = "#202830"
MUTED = "#5D6872"
GRID = "#D9E0E5"
PALE = "#F4F6F8"
MODEL_COLOR = {"rotating": ROT, "standing": STAND}
CLASS_COLOR = {
    "rotating_only_retained": ADVANTAGE,
    "standing_only_retained": STAND_ONLY,
    "both_retained": BOTH,
    "both_escaped": BOTH_ESCAPE,
}
BASE_ENERGIES = {1.0, 2.0, 3.0}
ENERGY_SHELLS = [0.5, 0.625, 0.75, 0.875, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def file_path_from_v4(value: str) -> Path:
    p = Path(value)
    if p.exists():
        return p
    # Published V4 files include the original Windows absolute path.  Resolve
    # their repository-relative suffix so the study remains portable.
    marker = "outputs\\octupole_comparison_home\\confinement_maps_v4\\"
    normalized = value.replace("/", "\\")
    idx = normalized.lower().find(marker.lower())
    if idx >= 0:
        rel = normalized[idx:].replace("\\", "/")
        return V4.parents[2] / rel
    raise FileNotFoundError(value)


def launch_key(row: dict[str, str]) -> tuple:
    return tuple(float(row[k]) for k in (
        "r0_mm", "energy_eV", "azimuth_rad", "phase_rad", "direction"
    ))


def pair_key(row: dict[str, str]) -> tuple:
    return (row["normalization"], row["field"], *launch_key(row), row["velocity_kind"])


def load_and_validate():
    main = read_csv(V4 / "equal_peak_main_grid.csv")
    pairs = read_csv(V4 / "matched_pairs_equal_peak.csv")
    by_launch: dict[tuple, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in main:
        if row["normalization"] != "equal_peak_voltage" or row["field"] != "h35":
            continue
        by_launch[launch_key(row)][row["model"]] = row
    if len(by_launch) != 522 or len(pairs) != 522:
        raise ValueError(f"Expected 522 matched V4 starts; got {len(by_launch)} and {len(pairs)}")
    if any(set(models) != {"rotating", "standing"} for models in by_launch.values()):
        raise ValueError("The equal-peak dataset contains an unmatched initial state.")
    for launch, models in by_launch.items():
        a, b = models["rotating"], models["standing"]
        if launch_key(a) != launch_key(b):
            raise ValueError("Model pair does not share the same physical initial state.")
        for r in (a, b):
            if int(r["horizon_RF"]) != 1000:
                raise ValueError("Unexpected observation horizon.")
            if r["status"] == "retained":
                if r["escape_time_RF"] or float(r["right_censored_at_RF"]) != 1000:
                    raise ValueError("Retained case is not recorded as right-censored.")
            elif r["status"] == "escaped":
                if not r["escape_time_RF"] or r["right_censored_at_RF"]:
                    raise ValueError("Escape/censoring fields are inconsistent.")
            else:
                raise ValueError(f"Unknown V4 trajectory status: {r['status']}")
    for e in BASE_ENERGIES:
        if sum(abs(launch[1] - e) < 1e-12 for launch in by_launch) != 64:
            raise ValueError(f"Balanced V4 grid is incomplete at {e:g} eV.")
    if {round(float(x), 3) for x in sorted({r["r0_mm"] for r in main})} != {1.0, 2.0}:
        raise ValueError("Unexpected launch radii in V4 source data.")
    return main, pairs, by_launch


def register_fonts():
    regular = Path("C:/Windows/Fonts/cambria.ttc")
    bold = Path("C:/Windows/Fonts/cambriab.ttf")
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("Arial TTF files are required for the publication renderer.")
    if "PubArial" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("PubArial", str(regular), subfontIndex=0))
        pdfmetrics.registerFont(TTFont("PubArial-Bold", str(bold)))
        pdfmetrics.registerFontFamily("PubArial", normal="PubArial", bold="PubArial-Bold",
                                      italic="PubArial", boldItalic="PubArial-Bold")
    return regular, bold


class FigureCanvas:
    """Small multi-backend vector canvas in millimetres and point-sized text."""

    def __init__(self, name: str, width_mm: float, height_mm: float, layout: str):
        self.name, self.w, self.h, self.layout = name, width_mm, height_mm, layout
        self.px_per_mm = PNG_DPI / 25.4
        self.px = Image.new("RGB", (round(self.w*self.px_per_mm), round(self.h*self.px_per_mm)), "white")
        self.draw = ImageDraw.Draw(self.px)
        self.regular, self.bold = register_fonts()
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w:g}mm" height="{self.h:g}mm" viewBox="0 0 {self.w:g} {self.h:g}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:Cambria,serif;fill:#202830}</style>',
        ]
        self.pdf_path = FIGURES / f"{name}_{layout}.pdf"
        self.pdf = pdfcanvas.Canvas(str(self.pdf_path), pagesize=(self.w*mm, self.h*mm), pageCompression=1)
        self.pdf.setTitle(name.replace("_", " "))
        self.pdf.setAuthor("V4 confinement study")

    def _xy_px(self, x: float, y: float) -> tuple[int, int]:
        return round(x*self.px_per_mm), round(y*self.px_per_mm)

    def rect(self, x: float, y: float, w: float, h: float, color=PALE, stroke=None, lw=0.2):
        x0, y0 = self._xy_px(x, y); x1, y1 = self._xy_px(x+w, y+h)
        self.draw.rectangle((x0, y0, x1, y1), fill=color, outline=stroke, width=max(1, round(lw*self.px_per_mm)))
        self.svg.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{color}" stroke="{stroke or "none"}" stroke-width="{lw:g}"/>')
        self.pdf.setFillColor(HexColor(color)); self.pdf.setStrokeColor(HexColor(stroke or color)); self.pdf.setLineWidth(lw*mm)
        self.pdf.rect(x*mm, (self.h-y-h)*mm, w*mm, h*mm, stroke=1 if stroke else 0, fill=1)

    def line(self, x1, y1, x2, y2, color=INK, lw=0.22, dash=None):
        a, b = self._xy_px(x1, y1), self._xy_px(x2, y2)
        if not dash:
            self.draw.line((a, b), fill=color, width=max(1, round(lw*self.px_per_mm)))
        else:
            dx, dy = x2-x1, y2-y1; length = math.hypot(dx, dy)
            if length:
                dash_mm, gap_mm = dash
                t = 0.0
                while t < length:
                    end = min(length, t+dash_mm)
                    p0 = self._xy_px(x1+dx*t/length, y1+dy*t/length)
                    p1 = self._xy_px(x1+dx*end/length, y1+dy*end/length)
                    self.draw.line((p0, p1), fill=color, width=max(1, round(lw*self.px_per_mm)))
                    t = end+gap_mm
        dash_svg = f' stroke-dasharray="{dash[0]:g},{dash[1]:g}"' if dash else ""
        self.svg.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="{lw:g}"{dash_svg}/>')
        self.pdf.setStrokeColor(HexColor(color)); self.pdf.setLineWidth(lw*mm)
        self.pdf.setDash(*(dash[0]*mm, dash[1]*mm) if dash else ())
        self.pdf.line(x1*mm, (self.h-y1)*mm, x2*mm, (self.h-y2)*mm)
        self.pdf.setDash()

    def polyline(self, points, color=INK, lw=0.22, dash=None):
        if len(points) < 2:
            return
        scaled = [self._xy_px(x, y) for x, y in points]
        self.draw.line(scaled, fill=color, width=max(1, round(lw*self.px_per_mm)), joint="curve")
        ds = f' stroke-dasharray="{dash[0]:g},{dash[1]:g}"' if dash else ""
        pts = " ".join(f"{x:g},{y:g}" for x, y in points)
        self.svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{lw:g}"{ds}/>')
        self.pdf.setStrokeColor(HexColor(color)); self.pdf.setLineWidth(lw*mm)
        self.pdf.setDash(*(dash[0]*mm, dash[1]*mm) if dash else ())
        p = self.pdf.beginPath(); p.moveTo(points[0][0]*mm, (self.h-points[0][1])*mm)
        for x, y in points[1:]: p.lineTo(x*mm, (self.h-y)*mm)
        self.pdf.drawPath(p); self.pdf.setDash()

    def marker(self, x, y, shape="circle", color=INK, size=1.0, fill=True, lw=0.28):
        px, py = self._xy_px(x, y); radius = max(2, round(size*self.px_per_mm))
        fill_color = color if fill else "#FFFFFF"
        if shape == "circle":
            self.draw.ellipse((px-radius, py-radius, px+radius, py+radius), fill=fill_color, outline=color,
                              width=max(1, round(lw*self.px_per_mm)))
            self.svg.append(f'<circle cx="{x:g}" cy="{y:g}" r="{size:g}" fill="{fill_color}" stroke="{color}" stroke-width="{lw:g}"/>')
            self.pdf.setFillColor(HexColor(fill_color)); self.pdf.setStrokeColor(HexColor(color)); self.pdf.setLineWidth(lw*mm)
            self.pdf.circle(x*mm, (self.h-y)*mm, size*mm, stroke=1, fill=1)
        else:
            if shape == "square":
                pts = [(x-size,y-size),(x+size,y-size),(x+size,y+size),(x-size,y+size)]
            elif shape == "triangle":
                pts = [(x,y-size*1.2),(x-size*1.05,y+size),(x+size*1.05,y+size)]
            else:
                raise ValueError(shape)
            coords = [self._xy_px(a,b) for a,b in pts]
            self.draw.polygon(coords, fill=fill_color)
            self.draw.line(coords+[coords[0]], fill=color, width=max(1, round(lw*self.px_per_mm)))
            s = " ".join(f"{a:g},{b:g}" for a,b in pts)
            self.svg.append(f'<polygon points="{s}" fill="{fill_color}" stroke="{color}" stroke-width="{lw:g}"/>')
            path = self.pdf.beginPath(); path.moveTo(pts[0][0]*mm,(self.h-pts[0][1])*mm)
            for a,b in pts[1:]: path.lineTo(a*mm,(self.h-b)*mm)
            path.close(); self.pdf.setFillColor(HexColor(fill_color)); self.pdf.setStrokeColor(HexColor(color)); self.pdf.setLineWidth(lw*mm)
            self.pdf.drawPath(path,stroke=1,fill=1)

    def text(self, x, y, text, size=8.0, color=INK, ha="center", weight="normal", rotation=0):
        text = str(text); lines = text.split("\n"); weight_bold = weight == "bold"
        font = ImageFont.truetype(str(self.bold if weight_bold else self.regular), max(1, round(size*PNG_DPI/72)), index=0)
        line_height_mm = size*1.2*25.4/72
        # Pillow text is rendered in a rotated transparent layer when required.
        widths = [self.draw.textbbox((0,0), s, font=font)[2] for s in lines]
        hpx = max(round(line_height_mm*self.px_per_mm*len(lines)), round(size*PNG_DPI/72))
        wpx = max(widths+[2])+8
        layer = Image.new("RGBA", (wpx, hpx), (255,255,255,0)); ld = ImageDraw.Draw(layer)
        for i, s in enumerate(lines):
            yy = hpx/2 + (i-(len(lines)-1)/2)*line_height_mm*self.px_per_mm
            anchor = {"left":"lm", "center":"mm", "right":"rm"}[ha]
            xx = 2 if ha == "left" else (wpx/2 if ha == "center" else wpx-2)
            ld.text((xx, yy), s, font=font, fill=color, anchor=anchor)
        if rotation:
            layer = layer.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
        px, py = self._xy_px(x,y)
        paste_x = px if ha == "left" else (px-layer.width if ha == "right" else px-layer.width/2)
        self.px.paste(layer, (round(paste_x), round(py-layer.height/2)), layer)
        anchor = {"left":"start", "center":"middle", "right":"end"}[ha]
        transform = f' transform="rotate({-rotation:g} {x:g} {y:g})"' if rotation else ""
        self.svg.append(f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" dominant-baseline="middle" font-size="{size:g}pt" font-weight="{weight}" fill="{color}"{transform}>')
        for i, s in enumerate(lines):
            dy = 0 if i == 0 else f'{line_height_mm:g}'
            self.svg.append(f'<tspan x="{x:g}" dy="{dy}">{escape(s)}</tspan>')
        self.svg.append('</text>')
        # PDF baseline is offset below the visual centre by roughly 0.35 em.
        self.pdf.saveState(); self.pdf.setFillColor(HexColor(color)); self.pdf.setFont("PubArial-Bold" if weight_bold else "PubArial", size)
        if rotation:
            self.pdf.translate(x*mm,(self.h-y)*mm); self.pdf.rotate(rotation)
            for i,s in enumerate(lines):
                by = ((len(lines)-1)/2-i)*line_height_mm*mm - size*0.35
                self._pdf_text(0, by, s, ha)
        else:
            for i,s in enumerate(lines):
                baseline_top = y + (i-(len(lines)-1)/2)*line_height_mm + size*0.35*25.4/72
                self._pdf_text(x*mm,(self.h-baseline_top)*mm,s,ha)
        self.pdf.restoreState()

    def _pdf_text(self, x, y, s, ha):
        if ha == "center": self.pdf.drawCentredString(x,y,s)
        elif ha == "right": self.pdf.drawRightString(x,y,s)
        else: self.pdf.drawString(x,y,s)

    def save(self):
        self.svg.append("</svg>")
        svg_path = FIGURES / f"{self.name}_{self.layout}.svg"
        svg_path.write_text("\n".join(self.svg)+"\n",encoding="utf-8")
        png_path = FIGURES / f"{self.name}_{self.layout}.png"
        self.px.save(png_path, dpi=(PNG_DPI,PNG_DPI), optimize=True)
        self.pdf.showPage(); self.pdf.save()
        return {"svg":svg_path.name,"pdf":self.pdf_path.name,"png":png_path.name,
                "width_mm":self.w,"height_mm":self.h,"png_dpi":PNG_DPI}


class Axes:
    def __init__(self, c: FigureCanvas, x, y, w, h, xr, yr, xticks, yticks,
                 xfmt=lambda v:f"{v:g}", yfmt=lambda v:f"{v:g}", ylog=False):
        self.c,self.x,self.y,self.w,self.h=c,x,y,w,h
        self.xmin,self.xmax=xr; self.ymin,self.ymax=yr; self.xticks=xticks; self.yticks=yticks
        self.xfmt,self.yfmt,self.ylog=xfmt,yfmt,ylog
        self._draw_frame()

    def _yt(self, v):
        if self.ylog:
            return self.y+self.h-(math.log10(v)-math.log10(self.ymin))/(math.log10(self.ymax)-math.log10(self.ymin))*self.h
        return self.y+self.h-(v-self.ymin)/(self.ymax-self.ymin)*self.h
    def X(self,v): return self.x+(v-self.xmin)/(self.xmax-self.xmin)*self.w
    def Y(self,v): return self._yt(v)
    def _draw_frame(self):
        for v in self.xticks:
            xx=self.X(v); self.c.line(xx,self.y,xx,self.y+self.h,GRID,0.16)
            self.c.line(xx,self.y+self.h,xx,self.y+self.h+0.8,INK,0.24)
            self.c.text(xx,self.y+self.h+2.6,self.xfmt(v),size=6.2,color=MUTED)
        for v in self.yticks:
            yy=self.Y(v); self.c.line(self.x,yy,self.x+self.w,yy,GRID,0.16)
            self.c.line(self.x-0.8,yy,self.x,yy,INK,0.24)
            self.c.text(self.x-1.5,yy,self.yfmt(v),size=6.2,color=MUTED,ha="right")
        self.c.line(self.x,self.y,self.x,self.y+self.h,INK,0.28)
        self.c.line(self.x,self.y+self.h,self.x+self.w,self.y+self.h,INK,0.28)

    def xlabel(self, s, y=None, size=8.2): self.c.text(self.x+self.w/2,y if y is not None else self.y+self.h+8.5,s,size=size)
    def ylabel(self, s, x=None, size=8.0, rotation=90): self.c.text(x if x is not None else self.x-10,self.y+self.h/2,s,size=size,rotation=rotation)
    def marker(self,x,y,*args,**kwargs):
        if self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax:
            self.c.marker(self.X(x),self.Y(y),*args,**kwargs)
    def line(self,x1,y1,x2,y2,*args,**kwargs): self.c.line(self.X(x1),self.Y(y1),self.X(x2),self.Y(y2),*args,**kwargs)
    def path(self,points,*args,**kwargs): self.c.polyline([(self.X(x),self.Y(y)) for x,y in points],*args,**kwargs)


def panel_title(c, x, y, tag, title, size=8.4):
    c.text(x,y,f"{tag}  {title}",size=size,weight="bold",ha="left")


def legend_item(c,x,y,shape,color,label,fill=True,size=1.05,text_size=7.0):
    c.marker(x,y,shape,color,size,fill)
    c.text(x+2.4,y,label,size=text_size,ha="left",color=INK)


def fraction_table(c, x, y, pairs_by_energy, width):
    labels=["0.5","0.625","0.75","0.875","1","1.5","2","2.5","3","3.5","4"]
    nvals=[str(len(pairs_by_energy.get(e,[]))) for e in ENERGY_SHELLS]
    c.text(x,y,"E₀ (eV)",size=6.1,ha="left",color=MUTED)
    c.text(x+15,y,"n matched",size=6.1,ha="left",color=MUTED)
    colw=(width-31)/len(labels)
    for i,(lab,n) in enumerate(zip(labels,nvals)):
        cx=x+31+(i+.5)*colw
        c.text(cx,y-1,lab,size=5.8,color=MUTED)
        c.text(cx,y+3,n,size=6.1,color=INK,weight="bold")


def figure1(pairs, layout):
    if layout=="double":
        w,h=178,112; left, gap, top, pw, ph=15,7,23,46,48
        xs=[left+i*(pw+gap) for i in range(3)]
        c=FigureCanvas("Fig1_energy_finite_time_confinement",w,h,layout)
        c.text(w/2,7,"Finite-time confinement versus initial transverse energy",size=11.2,weight="bold")
        c.text(w/2,14,"250 V peak per model · 100 kHz · 1000 RF periods (10 ms) · aperture 8 mm",size=7.2,color=MUTED)
        panel_y=top
        footer=104
    else:
        w,h=89,236; left, gap, top, pw, ph=18,0,21,65,42
        xs=[left,left,left]; c=FigureCanvas("Fig1_energy_finite_time_confinement",w,h,layout)
        c.text(w/2,7,"Finite-time confinement vs. energy",size=10.0,weight="bold")
        c.text(w/2,13,"250 V · 100 kHz · 1000 RF periods (10 ms)",size=6.6,color=MUTED)
        panel_y=top
        footer=224
    energy_counts={e:[p for p in pairs if abs(float(p["energy_eV"])-e)<1e-10] for e in ENERGY_SHELLS}
    specs=[("(a)","Rotating", "rotating"),("(b)","Standing", "standing"),("(c)","Paired difference", "difference")]
    for i,(tag,title,model) in enumerate(specs):
        y=panel_y if layout=="double" else panel_y+i*64
        ax=Axes(c,xs[i]+6,y,pw-8,ph,(0.37,4.13),(0,1.08 if model!="difference" else 1.04),[.5,1,2,3,4],[0,.25,.5,.75,1],xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v:.2g}")
        panel_title(c,xs[i],y-5,tag,title,7.6 if layout=="double" else 8.0)
        if model=="difference":
            ax.ylabel("Δ retained fraction",x=xs[i]+1,size=7.1)
        else:
            ax.ylabel("Retained fraction",x=xs[i]+1,size=7.1)
        for j,e in enumerate(ENERGY_SHELLS):
            group=energy_counts[e]; n=len(group)
            if not n: continue
            if model=="difference":
                krot=sum(p["rotating_status"]=="retained" for p in group)
                kst=sum(p["standing_status"]=="retained" for p in group)
                val=(krot-kst)/n
                ax.marker(e,val,"circle" if e in BASE_ENERGIES else "square",ADVANTAGE if val>0 else (STAND_ONLY if val<0 else BOTH),size=1.15,fill=True)
            else:
                k=sum(p[f"{model}_status"]=="retained" for p in group)
                ax.marker(e,k/n,"circle" if e in BASE_ENERGIES else "square",MODEL_COLOR[model],size=1.15,fill=True)
        ax.xlabel("Initial energy E⊥ (eV)",y=y+ph+8.8,size=7.1)
        # Every energy shell is present on the axis; the aligned footer table
        # reports its matched launch count without crowding the marks.
    legy=86 if layout=="double" else 212
    start=(w-98)/2
    legend_item(c,start,legy,"circle",INK,"Balanced launch: 1, 2, 3 eV",size=.9,text_size=6.4)
    legend_item(c,start+60,legy,"square",INK,"Adaptive energy samples",size=.9,text_size=6.4)
    if layout=="double":
        c.text(w/2,93,"Markers are computed shells; no interpolation or smoothing. Fractions and counts describe the designed grid, not a population.",size=6.35,color=MUTED)
        fraction_table(c,15,101,energy_counts,w-30)
    else:
        c.text(w/2,219,"Markers are computed shells only; no interpolation. Designed-grid descriptor, not a population probability.",size=6.0,color=MUTED)
        fraction_table(c,6,228,energy_counts,w-12)
    return c.save()


def balanced_pairs(pairs):
    return [p for p in pairs if float(p["energy_eV"]) in BASE_ENERGIES]


def figure2(pairs, layout):
    data=balanced_pairs(pairs)
    if layout=="double":
        w,h=178,148; l=15; top=22; pw=47; ph=34; gx=9; gy=23; cols=3
        c=FigureCanvas("Fig2_balanced_phase_azimuth_pairs",w,h,layout)
        c.text(w/2,7,"Paired retention outcome on the balanced RF-phase × azimuth grid",size=10.7,weight="bold")
        c.text(w/2,14,"Equal peak voltage: 250 V each · fixed E⊥ and r₀ within each small multiple",size=7,color=MUTED)
        legend_y=135
    else:
        w,h=89,224; l=14; top=21; pw=29; ph=36; gx=11; gy=23; cols=2
        c=FigureCanvas("Fig2_balanced_phase_azimuth_pairs",w,h,layout)
        c.text(w/2,7,"Balanced phase–azimuth launches",size=9.5,weight="bold")
        c.text(w/2,13,"Equal peak voltage, 250 V each",size=6.6,color=MUTED)
        legend_y=215
    facets=[]
    for e in (1.,2.,3.):
        for r0 in (1.,2.): facets.append((e,r0))
    for idx,(e,r0) in enumerate(facets):
        row,col=divmod(idx,cols); x=l+col*(pw+gx); y=top+row*(ph+gy)
        group=[p for p in data if abs(float(p["energy_eV"])-e)<1e-12 and abs(float(p["r0_mm"])-r0)<1e-12]
        ax=Axes(c,x+7,y,pw-8,ph,( -8,278),(-8,98),[0,90,180,270],[0,30,60,90],xfmt=lambda v:f"{int(v)}",yfmt=lambda v:f"{int(v)}")
        n=len(group); c.text(x+pw/2,y-5,f"({chr(97+idx)}) {e:g} eV · r₀={r0:g} mm · n={n}",size=6.4 if layout=="double" else 5.8)
        for p in group:
            phdeg=math.degrees(float(p["phase_rad"])); azdeg=math.degrees(float(p["azimuth_rad"]))
            azdeg += 1.4 if float(p["direction"])>0 else -1.4
            ax.marker(phdeg,azdeg,"circle" if float(p["direction"])>0 else "triangle",
                      CLASS_COLOR[p["difference_class"]],size=.88 if layout=="double" else .78,fill=True)
        ax.xlabel("RF phase φ (deg)",y=y+ph+8.4,size=6.3 if layout=="double" else 5.8)
        ax.ylabel("Azimuth θ (deg)",x=x+1.3,size=6.1 if layout=="double" else 5.6)
    if layout=="double":
        # A compact explicit legend separates outcome color from launch-direction shape.
        x0=8; widths=[43,48,52,41]
        for (key,label),xx in zip([
            ("both_retained","Both retained"),("rotating_only_retained","Rotating only"),
            ("standing_only_retained","Standing only"),("both_escaped","Both escaped")],
            [x0,x0+42,x0+90,x0+140]):
            legend_item(c,xx,legend_y,"circle",CLASS_COLOR[key],label,size=.83,text_size=6.1)
        legend_item(c,59,legend_y+5,"circle",INK,"+ tangent",size=.7,text_size=5.8)
        legend_item(c,94,legend_y+5,"triangle",INK,"− tangent",size=.7,text_size=5.8)
        c.text(w/2,144,"Only balanced 1, 2, 3-eV launches are shown; direction markers are offset ±1.4° for legibility.",size=6.1,color=MUTED)
    else:
        for j,(key,label) in enumerate([
            ("both_retained","Both retained"),("rotating_only_retained","Rotating only"),
            ("standing_only_retained","Standing only"),("both_escaped","Both escaped")]):
            col=j%2; row=j//2; xx=11+col*44; yy=legend_y-7+row*5
            legend_item(c,xx,yy,"circle",CLASS_COLOR[key],label,size=.72,text_size=5.7)
        legend_item(c,14,legend_y+3,"circle",INK,"+ tangent",size=.68,text_size=5.4)
        legend_item(c,43,legend_y+3,"triangle",INK,"− tangent",size=.68,text_size=5.4)
        c.text(w/2,222,"Balanced grid only; triangles/circles are offset ±1.4° by tangential sign. Adaptive samples are in Fig. 1.",size=5.25,color=MUTED)
    return c.save()


def get_case_traces(main):
    match={
        "r0_mm":1.,"energy_eV":1.,"azimuth_rad":math.pi/4,
        "phase_rad":0.,"direction":1.,
    }
    traces={}
    for model in ("rotating","standing"):
        r=next(row for row in main if row["model"]==model and
               all(abs(float(row[k])-v)<1e-11 for k,v in match.items()) and
               row["normalization"]=="equal_peak_voltage" and row["field"]=="h35")
        with file_path_from_v4(r["sample_csv"]).open(encoding="utf-8-sig",newline="") as fcsv:
            trace=list(csv.DictReader(fcsv))
        if not trace or trace[-1]["rf_period"]=="": raise ValueError("Selected V4 trajectory trace is incomplete.")
        traces[model]=(r,trace)
    return traces


def figure3(main, layout):
    traces=get_case_traces(main)
    if layout=="double":
        w,h=178,132; c=FigureCanvas("Fig3_matched_trajectories_and_radius",w,h,layout)
        c.text(w/2,7,"Matched launch: rotating retained, standing escaped",size=10.8,weight="bold")
        c.text(w/2,14,"m/z = 100 kTh · E⊥ = 1 eV · r₀ = 1 mm · θ = 45° · φRF = 0° · + tangential · 250 V each",size=6.65,color=MUTED)
        xy=(16,24,62,62); rt=(98,24,68,45); labels_y=115
        inset=(98,84,68,22)
    else:
        w,h=89,236; c=FigureCanvas("Fig3_matched_trajectories_and_radius",w,h,layout)
        c.text(w/2,6,"Matched trajectory and radial motion",size=9.6,weight="bold")
        c.text(w/2,12,"E⊥=1 eV · r₀=1 mm · θ=45° · φRF=0° · + tangent · 250 V",size=6.2,color=MUTED)
        xy=(18,22,62,62); rt=(17,112,66,47); labels_y=215
        inset=(17,181,66,20)
    x,y,pw,ph=xy
    axxy=Axes(c,x,y,pw,ph,(-8.7,8.7),(-8.7,8.7),[-8,-4,0,4,8],[-8,-4,0,4,8],xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v:g}")
    panel_title(c,x,y-6,"(a)","x–y trajectory (mm)",size=7.8)
    # Aperture circle is drawn as a vector ring at the physical 8-mm radius.
    cx=axxy.X(0); cy=axxy.Y(0); radx=pw*8/(8.7-(-8.7)); rady=ph*8/(8.7-(-8.7))
    # 240 vertices are visually smooth and denote only the exact aperture geometry.
    ring=[(cx+radx*math.cos(2*math.pi*i/240),cy-rady*math.sin(2*math.pi*i/240)) for i in range(241)]
    c.polyline(ring,INK,0.3,dash=(1.2,.8))
    c.text(cx+radx*.70,cy-rady*.78,"8 mm",size=6.2,color=INK,weight="bold")
    for model in ("rotating","standing"):
        _,rows=traces[model]; pts=[(float(q["x_mm"]),float(q["y_mm"])) for q in rows]
        axxy.path(pts,MODEL_COLOR[model],0.21 if layout=="double" else 0.18)
    axxy.xlabel("x (mm)",y=y+ph+8.0,size=7.3)
    axxy.ylabel("y (mm)",x=x-5.5,size=7.2)
    x,y,pw,ph=rt
    axr=Axes(c,x,y,pw,ph,(0,1000),(0,8.5),[0,200,400,600,800,1000],[0,2,4,6,8],xfmt=lambda v:f"{int(v)}",yfmt=lambda v:f"{v:g}")
    panel_title(c,x,y-6,"(b)","Radius versus RF period",size=7.8)
    for model in ("rotating","standing"):
        _,rows=traces[model]
        pts=[(float(q["rf_period"]),float(q["r_mm"])) for q in rows if float(q["rf_period"])<=1000]
        axr.path(pts,MODEL_COLOR[model],0.24 if layout=="double" else 0.2)
    axr.line(0,8,1000,8,INK,0.28,dash=(1.2,.8))
    escape_rf=float(traces["standing"][0]["escape_time_RF"])
    axr.line(escape_rf,0,escape_rf,8.2,STAND,0.22,dash=(.8,.7))
    axr.xlabel("Time (RF periods)",y=y+ph+8.2,size=7.3)
    axr.ylabel("Radius r (mm)",x=x-6,size=7.2)
    if inset:
        ix,iy,iw,ih=inset
        c.rect(ix,iy,iw,ih,"#FFFFFF",stroke=INK,lw=.22)
        if layout=="double":
            inset_axes=(ix+8,iy+4,iw-10,ih-8)
        else:
            inset_axes=(ix+8,iy+4,iw-10,ih-7)
        inax=Axes(c,*inset_axes,(0,25),(0,8.5),[0,10,20],[0,4,8],xfmt=lambda v:f"{int(v)}",yfmt=lambda v:f"{v:g}")
        for model in ("rotating","standing"):
            _,rows=traces[model]
            pts=[(float(q["rf_period"]),float(q["r_mm"])) for q in rows if float(q["rf_period"])<=25]
            inax.path(pts,MODEL_COLOR[model],.18)
        inax.line(0,8,25,8,INK,.22,dash=(1,.6))
        c.text(ix+iw/2,iy-1,"Early-time zoom · 0–25 RF",size=5.2,color=MUTED)
    legend_y=labels_y
    legend_item(c,42,legend_y,"circle",ROT,"Rotating · retained to 1000 RF",size=.95,text_size=6.5)
    legend_item(c,102,legend_y,"circle",STAND,"Standing · escaped at 21.31 RF",size=.95,text_size=6.5)
    if layout=="double":
        c.text(w/2,127,"Dashed circle/line: 8-mm aperture. Saved traces use 8 samples per RF; the short panel expands the escape interval.",size=5.9,color=MUTED)
    else:
        c.text(w/2,230,"Dashed ring: 8-mm aperture. Standing crossed at 21.31 RF; rotating is right-censored at 1000 RF.",size=5.6,color=MUTED)
    return c.save()


def balanced_launch(p): return float(p["energy_eV"]) in BASE_ENERGIES


def stable_jitter(p, rank, n):
    # Deterministic display-only offsets to separate launches sharing an energy.
    # It never alters values in the source files or creates a connecting boundary.
    spread=min(.075, .34*min(abs(float(p["energy_eV"])-e) for e in ENERGY_SHELLS if abs(float(p["energy_eV"])-e)>1e-9))
    return ((rank+0.5)/max(1,n)-0.5)*2*spread


def figure4(pairs, layout):
    data=[p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["field"]=="h35"]
    by_e=defaultdict(list)
    for p in data: by_e[float(p["energy_eV"])].append(p)
    if layout=="double":
        w,h=178,128; c=FigureCanvas("Fig4_first_passage_and_orbital_differences",w,h,layout)
        c.text(w/2,7,"First-passage times and paired orbital-motion differences",size=10.8,weight="bold")
        c.text(w/2,14,"Equal peak voltage: 250 V each · matched deterministic launch grid · 1000-RF observation horizon",size=6.7,color=MUTED)
        timebox=(19,25,73,62); dybox=(107,25,59,29); drbox=(107,67,59,29); footer=107
    else:
        w,h=89,239; c=FigureCanvas("Fig4_first_passage_and_orbital_differences",w,h,layout)
        c.text(w/2,7,"First passage and orbital differences",size=9.5,weight="bold")
        c.text(w/2,13,"250 V each · matched starts · 1000 RF horizon",size=6.3,color=MUTED)
        timebox=(17,25,66,74); dybox=(17,122,66,30); drbox=(17,171,66,30); footer=219
    x,y,pw,ph=timebox
    at=Axes(c,x,y,pw,ph,(.35,4.15),(.1,1500),[.5,1,2,3,4],[.1,1,10,100,1000],xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v:g}",ylog=True)
    panel_title(c,x,y-6,"(a)","First-passage time (RF periods)",size=7.8)
    at.line(.35,1000,4.15,1000,MUTED,.28,dash=(1.2,.8))
    for e in ENERGY_SHELLS:
        group=sorted(by_e[e],key=lambda p:(float(p["azimuth_rad"]),float(p["phase_rad"]),float(p["r0_mm"]),float(p["direction"])))
        n=len(group)
        for rank,p in enumerate(group):
            jitter=stable_jitter(p,rank,n)
            for model,offset in (("rotating",-.011),("standing",.011)):
                status=p[f"{model}_status"]; color=MODEL_COLOR[model]
                xval=e+jitter+offset
                if status=="escaped":
                    t=float(p[f"{model}_escape_time_RF"])
                    at.marker(xval,t,"circle" if balanced_launch(p) else "square",color,size=.45,fill=True,lw=.16)
                else:
                    # Right-censored cases are placed above the threshold with
                    # an upward arrow, never at t=1000 as a fictitious event.
                    xpos=at.X(xval); y1=at.Y(1000); y2=at.Y(1370)
                    c.line(xpos,y1,xpos,y2,color,.15)
                    c.line(xpos,y2,xpos-.38,y2+.55,color,.15); c.line(xpos,y2,xpos+.38,y2+.55,color,.15)
                    at.marker(xval,1370,"circle" if balanced_launch(p) else "square",color,size=.45,fill=False,lw=.16)
    at.xlabel("Initial energy E⊥ (eV)",y=y+ph+8.3,size=7.2)
    at.ylabel("First outward crossing tₑ / T_RF",x=x-6.5,size=6.9)
    # Orbital diagnostics are differences for exactly matched starts.
    def orbital_panel(box, tag, title, key, yrange, yticks):
        xx,yy,ww,hh=box
        ao=Axes(c,xx+5,yy,ww-7,hh, (.35,4.15),yrange,[.5,1,2,3,4],yticks,xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v:g}")
        panel_title(c,xx,yy-6,tag,title,size=7.3)
        ao.line(.35,0,4.15,0,MUTED,.23,dash=(1,.65))
        for e in ENERGY_SHELLS:
            group=sorted(by_e[e],key=lambda p:(float(p["azimuth_rad"]),float(p["phase_rad"]),float(p["r0_mm"]),float(p["direction"])))
            n=len(group)
            for rank,p in enumerate(group):
                value=(float(p["rotating_rms_radius_mm"])-float(p["standing_rms_radius_mm"])) if key=="delta_rms_radius_mm" else float(p[key])
                xxv=e+stable_jitter(p,rank,n)
                ao.marker(xxv,value,"circle" if balanced_launch(p) else "square",CLASS_COLOR[p["difference_class"]],size=.42,fill=True,lw=.14)
        ao.xlabel("E⊥ (eV)",y=yy+hh+7,size=6.5)
        ao.ylabel("Δr (mm)",x=xx-3,size=6.3)
    if layout=="double":
        orbital_panel(dybox,"(b)","Δ maximum radius","delta_max_radius_mm",(-8.5,2.5),[-8,-4,0,2])
        orbital_panel(drbox,"(c)","Δ RMS radius","delta_rms_radius_mm",(-6,3),[-6,-3,0,3])
        ly=footer
        legend_item(c,12,ly,"circle",ROT,"Rotating",size=.65,text_size=5.6)
        legend_item(c,38,ly,"circle",STAND,"Standing",size=.65,text_size=5.6)
        legend_item(c,65,ly,"circle",INK,"Filled: escape",size=.58,text_size=5.3)
        legend_item(c,99,ly,"circle",INK,"Open ↑: censored",fill=False,size=.58,text_size=5.3)
        legend_item(c,28,ly+5,"circle",ADVANTAGE,"Rotating only retained",size=.58,text_size=5.3)
        legend_item(c,83,ly+5,"circle",BOTH,"Both retained",size=.58,text_size=5.3)
        c.text(w/2,123,"x offsets separate launches for display; original energy values are unchanged. Circle: balanced; square: adaptive.",size=5.7,color=MUTED)
    else:
        orbital_panel(dybox,"(b)","Δ maximum radius","delta_max_radius_mm",(-8.5,2.5),[-8,-4,0,2])
        orbital_panel(drbox,"(c)","Δ RMS radius","delta_rms_radius_mm",(-6,3),[-6,-3,0,3])
        legend_item(c,10,216,"circle",ROT,"Rotating",size=.62,text_size=5.1)
        legend_item(c,34,216,"circle",STAND,"Standing",size=.62,text_size=5.1)
        legend_item(c,60,216,"circle",INK,"Escape",size=.54,text_size=4.9)
        legend_item(c,77,216,"circle",INK,"Open: censored",fill=False,size=.54,text_size=4.8)
        legend_item(c,18,223,"circle",ADVANTAGE,"Rotating only",size=.55,text_size=5.0)
        legend_item(c,52,223,"circle",BOTH,"Both retained",size=.55,text_size=5.0)
        c.text(w/2,232,"Retained cases are right-censored above 1000 RF. x jitter is display-only; no smoothing.",size=5.3,color=MUTED)
    return c.save()


def write_audit(main,pairs,by_launch,outputs):
    counts=defaultdict(lambda:{"n_matched":0,"rotating_retained":0,"standing_retained":0,
                              "rotating_only_retained":0,"standing_only_retained":0,
                              "both_retained":0,"both_escaped":0})
    for p in pairs:
        e=float(p["energy_eV"]); d=counts[e]; d["n_matched"]+=1
        for m in ("rotating","standing"):
            d[f"{m}_retained"]+=p[f"{m}_status"]=="retained"
        d[p["difference_class"]]+=1
    # Readable launch design summary and exact trace provenance for Fig. 3.
    traces=get_case_traces(main)
    trace_audit={}
    for model,(row,trace) in traces.items():
        trace_audit[model]={"source_file":str(file_path_from_v4(row["sample_csv"]).relative_to(V4.parents[2])).replace("\\","/"),
            "n_saved_samples":len(trace),"last_period":float(trace[-1]["rf_period"]),
            "trajectory_status":row["status"],"escape_time_RF":float(row["escape_time_RF"]) if row["escape_time_RF"] else None,
            "rmax_mm":float(row["max_radius_mm"]),"rrms_mm":float(row["rms_radius_mm"])}
    sources={}
    for path in (V4/"equal_peak_main_grid.csv",V4/"matched_pairs_equal_peak.csv"):
        sources[path.name]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
    (HERE/"figure_data_audit.json").write_text(json.dumps({
        "source":"V4 equal_peak_main_grid.csv and matched_pairs_equal_peak.csv",
        "source_checksums":sources,
        "renderer":"Pillow raster + ReportLab vector PDF + SVG primitives; Cambria fonts",
        "matplotlib":"not installed in rendering environment; package installation could not reach the package index",
        "no_new_rf_integrations":True,"matched_initial_conditions":len(by_launch),
        "horizon_RF":1000,"normalization":"equal peak voltage; 250 V rotating and 250 V standing",
        "balanced_grid_definition":"Eperp=1,2,3 eV; phase=0,90,180,270 deg; azimuth=0,22.5,45,67.5 deg; r0=1,2 mm; tangent sign=-,+",
        "balanced_pairs_per_energy":{str(e):sum(abs(k[1]-e)<1e-12 for k in by_launch) for e in (1.,2.,3.)},
        "adaptive_samples":"All other energy shells are V4 localized/adaptive refinements, plotted with square markers in energy and first-passage figures; they are not mixed into balanced-grid phase/azimuth panels.",
        "shell_counts":{format(e,"g"):v for e,v in sorted(counts.items())},
        "censoring":"retained trajectories are right-censored at 1000 RF; no event time assigned",
        "population_inference":"none; all fractions and orbital comparisons describe the calculated deterministic grid",
        "figure3_trace_audit":trace_audit,"exports":outputs
    },indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    main_rows,pairs,by_launch=load_and_validate()
    outputs=[]
    for layout in ("double","single"):
        outputs.append(figure1(pairs,layout))
        outputs.append(figure2(pairs,layout))
        outputs.append(figure3(main_rows,layout))
        outputs.append(figure4(pairs,layout))
    write_audit(main_rows,pairs,by_launch,outputs)
    print(json.dumps({"figures":len(outputs),"formats":["PDF","SVG","PNG"],"png_dpi":PNG_DPI,
                      "matched_starts":len(by_launch),"outputs":outputs},indent=2))


if __name__=="__main__":
    main()
