"""Render V6 journal figures from checkpointed trajectory data only."""
from __future__ import annotations
import csv, json, math, sys, importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"figures"; OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE.parent/"publication_figures_v1"))
spec=importlib.util.spec_from_file_location("pub",HERE.parent/"publication_figures_v1"/"render_publication_figures.py")
pub=importlib.util.module_from_spec(spec); spec.loader.exec_module(pub)
pub.FIGURES=OUT
Canvas,Axes=pub.FigureCanvas,pub.Axes
BLUE="#0072B2"; ORANGE="#D55E00"; GREEN="#009E73"; PURPLE="#AA4499"; GRAY="#59636D"; GOLD="#D69F00"; INK="#202830"; MUTED="#5D6872"; GRID="#D9E0E5"
MODEL={"rotating":BLUE,"standing":ORANGE}
CLASS={"both_retained":GRAY,"rotating_only_retained":GREEN,"standing_only_retained":PURPLE,"both_escaped":GOLD}

def readcsv(p):
    with open(p,encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def readjsonl(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def panel(c,x,y,w,h,yr=(0,1000),yt=(0,250,500,750,1000),xr=(0,22.5),xt=(0,5,10,15,20),yfmt=lambda z:f"{z:g}"):
    return Axes(c,x,y,w,h,xr,yr,xt,yt,xfmt=lambda z:f"{z:g}",yfmt=yfmt)
def save(c):return c.save()

ROWS=readjsonl(HERE/"trajectory_checkpoint.jsonl")
def csv_summary():
    # Primary plotted classes are drawn from the full, matched launch table.
    return {r["job_id"]:r for r in ROWS}

def figure1():
    c=Canvas("figure1_escape_time_by_azimuth",178,126,"double_column")
    c.text(12,7,"A  N48: rotating",size=8,weight="bold",ha="left");c.text(99,7,"B  N48: standing",size=8,weight="bold",ha="left")
    c.text(12,66,"C  pure m = 4: rotating",size=8,weight="bold",ha="left");c.text(99,66,"D  pure m = 4: standing",size=8,weight="bold",ha="left")
    boxes=[(22,14),(109,14),(22,73),(109,73)]
    for (rep,model),(x,y) in zip((("N48","rotating"),("N48","standing"),("pure_m4","rotating"),("pure_m4","standing")),boxes):
        ax=panel(c,x,y,62,39,yr=(0,1050),yt=(0,250,500,750,1000));
        ax.xlabel("Initial azimuth θ₀ (deg)",y=y+44,size=6.8)
        if x==22:ax.ylabel("First-passage time (RF periods)",x=x-12,size=6.5)
        candidates=[r for r in ROWS if r["job"]["representation"]==rep and r["job"]["model"]==model and float(r["job"]["phase_deg"])==0 and r["job"]["stage"] in ("scan","pilot","refine") and 0<=float(r["job"]["azimuth_deg"])<=22.5]
        # One exact record per launch state; adaptive midpoints remain separate.
        seen=set()
        for r in candidates:
            j=r["job"]; az=float(j["azimuth_deg"]); direction=int(j["direction"]); stage=j["stage"]
            key=(round(az,12),direction)
            if key in seen:continue
            seen.add(key); xoff=az+(0.035 if direction>0 else -0.035)
            if r["status"]=="retained":
                ax.marker(xoff,1000,"triangle",MODEL[model],size=.82,fill=False,lw=.3)
            else:ax.marker(xoff,float(r["escape_time_RF"]),"circle" if direction>0 else "square",MODEL[model],size=.55,fill=True,lw=.2)
            if stage=="refine":ax.marker(xoff,float(r["right_censored_at_RF"] or r["escape_time_RF"] or 998),"square",INK,size=1.25,fill=False,lw=.25)
    c.text(12,121,"Open triangles at 1000 RF periods are right-censored survivors. Circles = +vθ; squares = −vθ; open-square overlays mark adaptive midpoints; no boundary interpolation.",size=6.0,color=MUTED,ha="left")
    return save(c)

def figure2():
    c=Canvas("figure2_paired_azimuth_phase_outcomes",178,100,"double_column")
    c.text(12,7,"A  Full h35 N48 field",size=8,weight="bold",ha="left");c.text(99,7,"B  Pure analytical m = 4",size=8,weight="bold",ha="left")
    pairs={}
    for r in ROWS:
        j=r["job"]; rep=j["representation"]
        if j["stage"] not in ("scan","pilot","phase-check"):continue
        k=(rep,round(float(j["azimuth_deg"]),10),round(float(j["phase_deg"]),8),int(j["direction"]))
        pairs.setdefault(k,{})[j["model"]]=r
    outcome={}
    for k,v in pairs.items():
        if set(v)!={"rotating","standing"}:continue
        a,b=v["rotating"]["status"],v["standing"]["status"]
        cl="both_retained" if a==b=="retained" else "both_escaped" if a==b=="escaped" else "rotating_only_retained" if a=="retained" else "standing_only_retained"
        outcome[k]=(cl,v)
    for rep,x in (("N48",22),("pure_m4",109)):
        ax=Axes(c,x,15,62,50,(0,22.5),(-.45,2.45),(0,5,10,15,20),(0,1,2),xfmt=lambda z:f"{z:g}",yfmt=lambda z:{0:"0°",1:"90°",2:"180°"}[int(z)])
        ax.xlabel("Initial azimuth θ₀ (deg)",y=70,size=6.8)
        if x==22:ax.ylabel("Initial RF phase",x=9,size=6.6)
        for (rr,az,phase,d),(cl,v) in outcome.items():
            if rr!=rep or phase not in (0.,90.,180.):continue
            # The phase-zero layer is the complete balanced 65-point scan;
            # later layers show only the deliberately selected controls.
            yy={0.:0,90.:1,180.:2}[phase]+(.065 if d>0 else -.065)
            stage=v["rotating"]["job"]["stage"]
            if phase==0 and abs(az/ (22.5/64)-round(az/(22.5/64)))<1e-7: shape="circle" if d>0 else "square"
            else: shape="triangle" if d>0 else "square"
            ax.marker(az,yy,shape,CLASS[cl],size=.55 if phase==0 else .82,fill=(phase==0),lw=.28)
    items=[("both_retained",GRAY,"both retained"),("rotating_only_retained",GREEN,"rotating retained only"),("standing_only_retained",PURPLE,"standing retained only"),("both_escaped",GOLD,"both escaped")]
    xx=18
    for key,col,label in items:
        c.rect(xx,83,2.4,2.4,col);c.text(xx+3.4,84.2,label,size=6.3,ha="left");xx+=43
    c.text(12,94,"Phase 0°: full 65-angle grid; 90° and 180°: selected checks only. Circle/triangle = +vθ; square = −vθ. Deterministic outcomes, not probabilities.",size=6,color=MUTED,ha="left")
    return save(c)

def load_diagnostics():
    p=HERE/"phase_space"; meta=readjsonl(p/"phase_space_checkpoint.jsonl"); out={}
    for r in meta:
        tr=np.load(p/"traces"/(r["job_id"]+".npz"))
        out[r["job"]["diagnostic_label"]]=(r,tr)
    return out

def circle_xy(ax,r=8):
    pts=[(r*math.cos(t),r*math.sin(t)) for t in np.linspace(0,2*math.pi,181)];ax.path(pts,color=INK,lw=.38,dash=(1.2,.8))

def figure3(diag):
    c=Canvas("figure3_matched_xy_and_radial_motion",178,91,"double_column")
    c.text(12,7,"A  Stroboscopic x–y trajectory",size=8,weight="bold",ha="left");c.text(100,7,"B  Radius versus time",size=8,weight="bold",ha="left")
    ax=Axes(c,22,15,63,59,(-8.5,8.5),(-8.5,8.5),(-8,0,8),(-8,0,8),xfmt=lambda z:f"{z:g}",yfmt=lambda z:f"{z:g}")
    ax.xlabel("x (mm)",y=78,size=7);ax.ylabel("y (mm)",x=10,size=7);circle_xy(ax)
    labels=[("N48_early_escape_rotating","rotating"),("N48_early_escape_standing","standing")]
    traces=[]
    for label,model in labels:
        r,tr=diag[label];s=tr["strobe"];pts=[(z[1]*1e3,z[2]*1e3) for z in s]
        ax.path(pts,color=MODEL[model],lw=.35);ax.marker(*pts[0],"circle",MODEL[model],size=.9)
        ax.marker(*pts[-1],"triangle",MODEL[model],size=.9,fill=(r["status"]=="escaped"));traces.append((r,s,model))
    ax2=Axes(c,110,15,58,59,(0,1000),(0,8.5),(0,250,500,750,1000),(0,2,4,6,8),xfmt=lambda z:f"{z:g}",yfmt=lambda z:f"{z:g}")
    ax2.xlabel("Time (RF periods)",y=78,size=7);ax2.ylabel("Radius r (mm)",x=99,size=7)
    ax2.line(0,8,1000,8,color=INK,lw=.35,dash=(1.2,.8))
    for r,s,model in traces:
        pts=[(float(z[0]),math.hypot(z[1],z[2])*1e3) for z in s];ax2.path(pts,color=MODEL[model],lw=.42)
    c.line(119,86,124,86,BLUE,.6);c.text(126,86,"rotating 250 V",size=6.3,ha="left")
    c.line(146,86,151,86,ORANGE,.6);c.text(153,86,"standing 353.553 V",size=6.3,ha="left")
    c.text(12,89,"Matched launch: 2 eV, r₀=1 mm, θ₀=22.5°, phase 0°, +vθ. Points are once-per-RF stroboscopic samples; dashed circle/line: 8-mm aperture.",size=5.9,color=MUTED,ha="left")
    return save(c)

def figure4(diag):
    c=Canvas("figure4_jacobi_and_finite_time_dynamics",178,128,"double_column")
    c.text(12,7,"A  Pure rotating m = 4: Jacobi drift",size=7.7,weight="bold",ha="left");c.text(99,7,"B  Full N48: stroboscopic projection",size=7.7,weight="bold",ha="left")
    c.text(12,67,"C  Finite-time tangent growth",size=7.7,weight="bold",ha="left");c.text(99,67,"D  Standing-wave H and Lz variation",size=7.7,weight="bold",ha="left")
    # Panel A, log absolute drift in the derived Jacobi quantity.
    r,tr=diag["deep_rotating_jacobi"];h=tr["hamiltonian_lz"][:,1];lz=tr["hamiltonian_lz"][:,2];q=1.602176634e-19;om=2*math.pi*100000.;K=h-om/4*lz/q;dk=np.abs(K-K[0]);
    vals=dk[dk>0];ymin=max(1e-16,float(vals.min())*.5) if len(vals) else 1e-16;ymax=max(1e-12,float(vals.max())*2) if len(vals) else 1e-12
    a=Axes(c,22,14,62,40,(0,1000),(ymin,ymax),(0,250,500,750,1000),(1e-15,1e-12,1e-9),xfmt=lambda z:f"{z:g}",yfmt=lambda z:f"{z:.0e}",ylog=True);a.xlabel("Time (RF periods)",y=59,size=6.3);a.ylabel("|K − K₀| (eV)",x=10,size=6.3)
    a.path([(float(row[0]),float(max(d,ymin))) for row,d in zip(tr["hamiltonian_lz"],dk)],color=BLUE,lw=.4)
    # Panel B: (x, vx) phase projection, both matched full-field trajectories.
    b=Axes(c,109,14,62,40,(-8.5,8.5),(-120,120),(-8,0,8),(-100,0,100),xfmt=lambda z:f"{z:g}",yfmt=lambda z:f"{z:g}");b.xlabel("x (mm)",y=59,size=6.3);b.ylabel("vₓ (m s⁻¹)",x=97,size=6.3)
    for label,model in (("N48_early_escape_rotating","rotating"),("N48_early_escape_standing","standing")):
        rr,tt=diag[label];ss=tt["strobe"];b.path([(z[1]*1e3,z[3]) for z in ss[::5]],color=MODEL[model],lw=.35)
    # Panel C, cumulative log tangent growth sampled every ten RF cycles.
    d=Axes(c,22,74,62,39,(0,1000),(-5,45),(0,250,500,750,1000),(-0,10,20,30,40),xfmt=lambda z:f"{z:g}",yfmt=lambda z:f"{z:g}");d.xlabel("Time (RF periods)",y=119,size=6.2);d.ylabel("Cumulative log growth",x=10,size=6.2)
    for label,col in (("N48_boundary_standing",ORANGE),("N48_early_escape_standing",PURPLE),("N48_late_escape_standing",GOLD),("matched_pure_standing",GRAY)):
        if label not in diag:continue
        rr,tt=diag[label];d.path([(float(z[0]),float(z[1])) for z in tt["growth"]],color=col,lw=.38)
    # Panel D, fractional changes in H and Lz for standing cases.
    e=Axes(c,109,74,62,39,(0,1000),(-120,120),(0,250,500,750,1000),(-100,0,100),xfmt=lambda z:f"{z:g}",yfmt=lambda z:f"{z:g}");e.xlabel("Time (RF periods)",y=119,size=6.2);e.ylabel("Fractional change",x=97,size=6.2)
    for label,col in (("N48_early_escape_standing",ORANGE),("matched_pure_standing",BLUE)):
        if label not in diag:continue
        rr,tt=diag[label];z=tt["hamiltonian_lz"];H=z[:,1];L=z[:,2];
        if len(H)>1 and H[0]!=0:e.path([(float(v[0]),float(v[1]/H[0]-1)) for v in z],color=col,lw=.4)
        if len(L)>1 and L[0]!=0:e.path([(float(v[0]),float(v[2]/L[0]-1)) for v in z],color=col,lw=.4,dash=(1.0,.7))
    c.text(12,124,"C: orange boundary, purple 22.5°, gold late escape, gray pure m=4. D: solid ΔH/H₀; dashed ΔLz/Lz₀ (orange N48; blue pure m=4). Finite-time growth alone does not establish chaos.",size=5.4,color=MUTED,ha="left")
    return save(c)

def main():
    figure1();figure2();diag=load_diagnostics();figure3(diag);figure4(diag)
    manifest={"figures":[p.name for p in sorted(OUT.iterdir()) if p.suffix in (".pdf",".svg",".png")],"png_dpi":600,"source":"V6 checkpointed CSV/JSON and stroboscopic diagnostics","interpolation":"none; only observed states and samples are plotted"}
    (OUT/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
