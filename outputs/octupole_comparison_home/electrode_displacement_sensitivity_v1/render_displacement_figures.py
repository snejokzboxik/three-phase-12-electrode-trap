"""Render displacement pilot figures from existing, validated data only.

Exports vector PDF/SVG and 600-dpi PNG using the shared publication canvas from
the V4 figure package. No FEM solve or RF integration is performed here.
"""
from __future__ import annotations
import csv, importlib.util, json, math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIG=HERE/"figures"
FIG.mkdir(exist_ok=True)
shared=ROOT/"outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py"
spec=importlib.util.spec_from_file_location("v4figurecanvas",shared)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.FIGURES=FIG
Canvas,Axes=mod.FigureCanvas,mod.Axes
INK, MUTED, GRID=mod.INK,mod.MUTED,mod.GRID
COLORS={"radial_out_10um":"#0072B2","radial_in_10um":"#D55E00",
"tangential_plus_10um":"#009E73","tangential_minus_10um":"#CC79A7",
"out_10":"#0072B2","out_25":"#56B4E9","out_50":"#E69F00","out_100":"#D55E00",
"rotating":"#0072B2","standing":"#D55E00","h35":"#0072B2","h50":"#D55E00"}

def rows_csv(p):
    with p.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def legend(c,x,y,color,label,marker="circle"):
    c.marker(x,y,marker,color,.7,True,lw=.18); c.text(x+2,y,label,size=6.2,ha="left",color=INK)
def fmt(v): return f"{v:g}"

def fig_field():
    data=rows_csv(HERE/"processed/field_sensitivity.csv")
    c=Canvas("Fig1_field_perturbation",178,118,"double")
    c.text(89,7,"Electrode displacement and field perturbation",size=11,weight="bold")
    c.text(89,14,"Full-domain P4 fields; 144 deterministic azimuths per radius; markers show calculated shells",size=7,color=MUTED)
    specs=[("A","Displacement direction at 10 µm",["radial_out_10um","radial_in_10um","tangential_plus_10um","tangential_minus_10um"]),
           ("B","Radial outward amplitude",["out_10","out_25","out_50","out_100"])]
    for j,(tag,title,patterns) in enumerate(specs):
        xx=17+j*86; yy=29; ww=70; hh=62
        ax=Axes(c,xx,yy,ww,hh,(0,7),(0,125),[0,1,2,3,4,5,6,7],[0,25,50,75,100,125],xfmt=fmt,yfmt=lambda v:f"{v:g}")
        mod.panel_title(c,xx,yy-6,tag,title,8)
        sub=[]
        for pat in patterns:
            chosen=pat
            if j==1:
                chosen="radial_out_"+pat.split("_")[-1]+"um"
            rr=[r for r in data if r["pattern"]==chosen and r["resolution"]=="h50"]
            if not rr: continue
            # Keep only exact angular-ring RMS values; intentionally do not connect shells.
            key=pat if j==1 else chosen
            for r in rr:
                val=100*float(r["field_delta_rms_fraction_nominal"])
                ax.marker(float(r["radius_mm"]),val,"circle",COLORS[key],size=.72,fill=True,lw=.18)
            label=({"radial_out_10um":"radial +10 µm","radial_in_10um":"radial −10 µm","tangential_plus_10um":"tangent +10 µm","tangential_minus_10um":"tangent −10 µm"}.get(pat,pat.replace("out_","")))
            if j==1: label=pat.split("_")[-1]+" µm"
            sub.append((COLORS[key],label))
        ax.xlabel("Radius r (mm)",y=yy+hh+8,size=7)
        ax.ylabel("RMS |ΔE| / |E₀| (%)",x=xx-9,size=6.7)
        for k,(col,label) in enumerate(sub):
            lx=xx+(k%2)*36; ly=101+(k//2)*6
            legend(c,lx,ly,col,label)
    c.text(89,115,"RMS ratio is evaluated on each ring; no angular or radial interpolation is shown.",size=6,color=MUTED)
    return c.save()

def fig_minimum():
    j=json.loads((HERE/"processed/effective_minimum_method_validation.json").read_text(encoding="utf-8"))
    pts=j["all_methods_positions_mm"]
    c=Canvas("Fig2_effective_minimum_validation",178,118,"double")
    c.text(89,7,"Validation of the effective-potential minimum",size=11,weight="bold")
    c.text(89,14,"+10 µm radial displacement of electrode 0 · rotating field · search domain r ≤ 5 mm",size=7,color=MUTED)
    ax=Axes(c,30,29,112,70,(-.65,.65),(-.65,.65),[-.6,-.4,-.2,0,.2,.4,.6],[-.6,-.4,-.2,0,.2,.4,.6],xfmt=fmt,yfmt=fmt)
    for p in sorted(pts,key=lambda q:q["resolution"]=="h35"):
        x,y=p["position_mm"]; res=p["resolution"]
        # Show every independent optimizer result as a point; no connecting contour is inferred.
        ax.marker(x,y,"circle" if res=="h35" else "square",COLORS[res],size=.9,fill=(res=="h35"),lw=.28)
    ax.marker(0,0,"triangle",INK,size=1,fill=True)
    ax.marker(.45811,0,"circle",MUTED,size=.95,fill=False,lw=.28)
    ax.xlabel("x (mm)",size=7.2); ax.ylabel("y (mm)",x=20,size=7.2)
    legend(c,146,40,COLORS["h35"],"h35", "circle")
    legend(c,146,47,COLORS["h50"],"h50", "square")
    c.marker(146,54,"triangle",INK,1,True); c.text(148,54,"origin",size=6.0,ha="left")
    c.marker(146,61,"circle",MUTED,.9,False); c.text(148,61,"secondary minimum",size=5.4,ha="left")
    c.text(146,76,f"h35 r_min: {j['h35_best_radius_mm']:.6f} mm",size=5.1,ha="left")
    c.text(146,82,f"h50 r_min: {j['h50_best_radius_mm']:.6f} mm",size=5.1,ha="left")
    c.text(89,109,"81×81 grid followed by Nelder–Mead, Powell and L-BFGS-B multistart searches; symmetric ±y solutions overlap within mesh precision.",size=6.0,color=MUTED)
    c.text(89,114,"Markers show optimizer endpoints only; the plotted points do not assert a continuous boundary or exact RF null.",size=6.0,color=MUTED)
    return c.save()

def collect_trace(folder, max_rf=12):
    paths=sorted(folder.glob("segment_*.csv"))
    out=[]
    for p in paths:
        for r in rows_csv(p):
            if float(r["rf_period"])<=max_rf: out.append(r)
    return out
def fig_trajectory():
    base=HERE/"trajectories"
    names={"rotating":"nominal_rotating_standard_E3.5eV_tangent_minus_v2",
           "standing":"nominal_standing_standard_E3.5eV_tangent_minus_v2"}
    traces={k:collect_trace(base/v) for k,v in names.items()}
    c=Canvas("Fig3_matched_negative_tangent_trajectories",178,118,"double")
    c.text(89,7,"Matched 3.5 eV trajectories after direction correction",size=11,weight="bold")
    c.text(89,14,"Nominal geometry · 1 mm launch · phase 0 · azimuth π/8 · vφ < 0 · 250 V, 100 kHz",size=7,color=MUTED)
    # x-y view within an explicitly limited 12-period window.
    ax=Axes(c,20,30,60,58,(-9,9),(-9,9),[-8,-4,0,4,8],[-8,-4,0,4,8],xfmt=fmt,yfmt=fmt)
    mod.panel_title(c,20,24,"A","Orbit segment (0–12 RF periods)",8)
    # aperture circle
    circle=[(8*math.cos(i*2*math.pi/180),8*math.sin(i*2*math.pi/180)) for i in range(181)]
    ax.path(circle,MUTED,.35,dash=(1.2,.8))
    for k,rr in traces.items():
        col=COLORS[k]
        pts=[(float(r["x_mm"]),float(r["y_mm"])) for r in rr]
        if len(pts)>1: ax.path(pts,col,.38)
        if pts: ax.marker(*pts[0],"circle",col,.7,True)
    ax.xlabel("x (mm)",y=96,size=7); ax.ylabel("y (mm)",x=12,size=7)
    legend(c,27,103,COLORS["rotating"],"Rotating")
    legend(c,57,103,COLORS["standing"],"Standing")
    # radius against RF period, only actually saved points; 8 mm event aperture.
    ar=Axes(c,101,30,65,58,(0,12),(0,8.7),[0,2,4,6,8,10,12],[0,2,4,6,8],xfmt=fmt,yfmt=fmt)
    mod.panel_title(c,101,24,"B","Radius and first passage",8)
    for k,rr in traces.items():
        pts=[(float(r["rf_period"]),float(r["r_mm"])) for r in rr]
        if len(pts)>1: ar.path(pts,COLORS[k],.42)
    ar.line(0,8,12,8,MUTED,.32,dash=(1.1,.8))
    ar.xlabel("RF periods t / T_RF",y=ar.y+ar.h+8,size=7)
    ar.ylabel("Radius r (mm)",x=92,size=7)
    c.text(133,105,"Standing: first passage ≈10 RF",size=6.0,color=COLORS["standing"])
    c.text(133,110,"Rotating: right-censored at 1000 RF",size=6.0,color=COLORS["rotating"])
    return c.save()

def main():
    outputs=[fig_field(),fig_minimum(),fig_trajectory()]
    (FIG/"figure_manifest.json").write_text(json.dumps({"renderer":"shared Pillow/ReportLab/SVG publication canvas","formats":["PDF","SVG","PNG"],"png_dpi":600,"new_rf_integrations":0,"figures":outputs},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(outputs,indent=2))
if __name__=="__main__": main()
