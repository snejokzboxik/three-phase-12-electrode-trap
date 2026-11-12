"""Compact sans-serif publication figures for the V9 pilot (no integrations)."""
from __future__ import annotations
import csv,importlib.util,json,math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]; FIG=HERE/"figures"; FIG.mkdir(exist_ok=True)
SRC=REPO/"outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py"
spec=importlib.util.spec_from_file_location("v9_canvas",SRC);pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Use the user's preferred clean sans-serif style across raster, PDF and SVG.
ARIAL=Path("C:/Windows/Fonts/arial.ttf"); ARIAL_BOLD=Path("C:/Windows/Fonts/arialbd.ttf")
pdfmetrics.registerFont(TTFont("PubArial",str(ARIAL)))
pdfmetrics.registerFont(TTFont("PubArial-Bold",str(ARIAL_BOLD)))
pdfmetrics.registerFontFamily("PubArial",normal="PubArial",bold="PubArial-Bold",italic="PubArial",boldItalic="PubArial-Bold")
pub.register_fonts=lambda:(ARIAL,ARIAL_BOLD)
_init=pub.FigureCanvas.__init__
def _sans_init(self,*args,**kwargs):
    _init(self,*args,**kwargs)
    self.svg[2]='<style>text{font-family:Arial,Helvetica,sans-serif;fill:#20252B}</style>'
pub.FigureCanvas.__init__=_sans_init
pub.FIGURES=FIG;pub.PNG_DPI=600
Canvas,Axes=pub.FigureCanvas,pub.Axes
INK="#20252B";MUTED="#5D6872";GRID="#D9DEE3";TEAL="#36A89A";BLUE="#2F6B8A";ORANGE="#D97706";RED="#C63D4F";PURPLE="#7A6BB7";GREY="#89939D"
GCOLOR={0.0:GREY,10.0:BLUE,100.0:TEAL,1000.0:ORANGE,3000.0:RED}
LAUNCH_COLOR={"deep_0deg":TEAL,"early_escape_22p5deg":RED,"boundary_negative":BLUE,"boundary_positive":ORANGE}
LAUNCH_NAME={"deep_0deg":"0° deep","early_escape_22p5deg":"22.5°","boundary_negative":"9.613°, vθ<0","boundary_positive":"10.272°, vθ>0"}
GAMMAS=(0.0,10.0,100.0,1000.0,3000.0)
def save(c):return c.save()
def read_rows():
    p=HERE/"processed/trajectory_results_v9.csv"
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def traces_for(label,model):
    out=[]
    for g in GAMMAS:
        rows=[r for r in read_rows() if r["launch_label"]==label and r["model"]==model and float(r["gamma_s_inv"])==g]
        if not rows:continue
        p=HERE/rows[0]["trace_file"]
        if not p.exists():continue
        z=np.load(p);out.append((g,z))
    return out
def gamma_xticks():return ([0,1,2,3,4],lambda x:{0:"0",1:"10",2:"100",3:"1000",4:"3000"}[int(round(x))])
def line_legend(c,x,y,label,dash=None,color=INK):
    c.line(x,y,x+3,y,color,.32,dash=dash);c.text(x+4,y,label,size=5.8,ha="left",color=INK)

def figure_escape():
    rows=read_rows(); c=Canvas("V9_Fig1_escape_time_vs_friction",178,112,"double_column")
    pub.panel_title(c,9,8,"(a)","First-passage time versus drag coefficient",8.5)
    xt,xform=gamma_xticks()
    for pi,model in enumerate(("rotating","standing")):
        x=22+pi*82; ax=Axes(c,x,20,68,61,(-.25,4.25),(0,1050),xt,[0,250,500,750,1000],xfmt=xform,yfmt=lambda y:f"{y:g}")
        c.text(x+1,16,"Rotating" if model=="rotating" else "Standing",size=7.2,weight="bold",ha="left")
        for row in rows:
            if row["model"]!=model:continue
            g=float(row["gamma_s_inv"]); xx=GAMMAS.index(g)
            y=float(row["escape_time_RF"]) if row["escape_time_RF"] else 1000
            color=LAUNCH_COLOR[row["launch_label"]]
            offset={"deep_0deg":-.18,"early_escape_22p5deg":-.06,"boundary_negative":.06,"boundary_positive":.18}[row["launch_label"]]
            ax.marker(xx+offset,y,"circle",color,size=.85,fill=bool(row["escape_time_RF"]),lw=.22)
        if pi==0:ax.ylabel("First escape time (RF periods)",x=12,size=7)
        ax.xlabel("Drag coefficient γ (s^-1)",y=91,size=7)
    for k,key in enumerate(LAUNCH_COLOR):pub.legend_item(c,14+k*43,98,"circle",LAUNCH_COLOR[key],LAUNCH_NAME[key],size=.65,text_size=6)
    pub.legend_item(c,18,105,"circle",INK,"open marker = retained to 1000 RF periods",fill=False,size=.65,text_size=5.8)
    return save(c)

def figure_deep_timeseries():
    c=Canvas("V9_Fig2_deep_launch_dynamics",178,146,"double_column")
    pub.panel_title(c,9,8,"(b)","Deep launch: radius, kinetic energy and drag loss",8.5)
    all_deep=[tr for model0 in ("rotating","standing") for _,tr in traces_for("deep_0deg",model0)]
    emax=max([1.0]+[max(float(np.max(z["kinetic_eV"])),float(np.max(z["friction_dissipated_eV"]))) for z in all_deep])
    emax=max(1.0,math.ceil(emax*1.04));eticks=np.linspace(0,emax,5).tolist()
    for col,model in enumerate(("rotating","standing")):
        x=22+col*82
        ax=Axes(c,x,22,68,34,(0,1000),(0,8),[0,250,500,750,1000],[0,2,4,6,8],yfmt=lambda y:f"{y:g}")
        c.text(x+1,18,"Rotating" if model=="rotating" else "Standing",size=7,weight="bold",ha="left")
        for g,z in traces_for("deep_0deg",model):
            t=z["time_RF"];r=np.hypot(z["state"][:,0],z["state"][:,1])*1000
            ax.path(list(zip(t[::4],r[::4])),GCOLOR[g],.28)
        if col==0:ax.ylabel("Radius r (mm)",x=13,size=7)
        ax.xlabel("Time (RF periods)",y=62,size=7)
        ae=Axes(c,x,94,68,34,(0,1000),(0,emax),[0,250,500,750,1000],eticks,yfmt=lambda y:f"{y:g}")
        for g,z in traces_for("deep_0deg",model):
            t=z["time_RF"]; k=z["kinetic_eV"];d=z["friction_dissipated_eV"]
            ae.path(list(zip(t[::4],k[::4])),GCOLOR[g],.28)
            ae.path(list(zip(t[::4],d[::4])),GCOLOR[g],.25,dash=(1.1,.7))
        if col==0:ae.ylabel("Energy (eV)",x=13,size=7)
        ae.xlabel("Time (RF periods)",y=139,size=7)
    for k,g in enumerate(GAMMAS):pub.legend_item(c,16+k*28,72,"circle",GCOLOR[g],f"γ={g:g}",size=.55,text_size=5.8)
    line_legend(c,52,84,"K solid; drag loss dashed")
    return save(c)

def figure_stroboscopic():
    c=Canvas("V9_Fig3_stroboscopic_phase_space",178,102,"double_column")
    pub.panel_title(c,9,8,"(c)","RF-synchronous phase-space samples, 100–1000 periods",8.5)
    for col,model in enumerate(("rotating","standing")):
        x=23+col*82;ax=Axes(c,x,21,68,58,(-8,8),(-75,75),[-8,-4,0,4,8],[-60,-30,0,30,60])
        c.text(x+1,17,"Rotating" if model=="rotating" else "Standing",size=7,weight="bold",ha="left")
        for g,z in traces_for("deep_0deg",model):
            t=z["time_RF"];s=z["state"]
            idx=np.arange(0,len(t),4);idx=idx[(t[idx]>=100)&(t[idx]<=1000)]
            # Thin deterministically for legibility; sample times remain integer RF periods.
            idx=idx[::max(1,len(idx)//160)]
            for n in idx:ax.marker(float(s[n,0]*1e3),float(s[n,2]),"circle",GCOLOR[g],.37,True,.12)
        if col==0:ax.ylabel("Velocity vₓ (m s^-1)",x=13,size=7)
        ax.xlabel("Position x (mm)",y=88,size=7)
    for k,g in enumerate(GAMMAS):pub.legend_item(c,23+k*29,96,"circle",GCOLOR[g],f"γ={g:g}",size=.55,text_size=5.8)
    return save(c)

def figure_lyapunov():
    files=sorted((HERE/"lyapunov").glob("*.json"))
    if not files:return None
    records=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    c=Canvas("V9_Fig4_finite_time_lyapunov",178,96,"double_column")
    pub.panel_title(c,9,8,"(d)","Finite-time Lyapunov spectrum after transient",8.5)
    ax=Axes(c,24,22,62,57,(100,1000),(-4000,1500),[100,400,700,1000],[-4000,-2000,0,1000],yfmt=lambda y:f"{y:g}")
    ax2=Axes(c,108,22,59,57,(.5,4.5),(-3500,500),[1,2,3,4],[-3000,-2000,-1000,0],yfmt=lambda y:f"{y:g}")
    for i,r in enumerate(records):
        name=r.get("job",{}).get("model","case");color=BLUE if name=="rotating" else ORANGE
        conv=r.get("convergence_checkpoints",[])
        pts=[(float(q["end_RF"]),float(q["lyapunov_s_inv"][0])) for q in conv]
        if pts:ax.path(pts,color,.42);[(ax.marker(a,b,"circle",color,.8,True,.2)) for a,b in pts]
        if "lyapunov_spectrum_s_inv" in r:
            specv=r["lyapunov_spectrum_s_inv"]
            for n,val in enumerate(specv,1):ax2.marker(n,val,"circle" if name=="rotating" else "square",color,.8,True,.2)
    ax.line(100,-2000,1000,-2000,INK,.25,dash=(2,1));ax.ylabel("Largest exponent (s⁻¹)",x=13,size=6.7);ax.xlabel("Analysis end (RF periods)",y=85,size=6.7)
    ax2.ylabel("Exponent (s⁻¹)",x=100,size=6.7);ax2.xlabel("Spectrum index",y=85,size=6.7)
    pub.legend_item(c,27,92,"circle",BLUE,"rotating",size=.55,text_size=6);pub.legend_item(c,70,92,"square",ORANGE,"standing",size=.55,text_size=6)
    c.text(119,92,"Check: Σλ = −2γ",size=5.8,ha="left",color=INK)
    return save(c)

def main():
    made=[figure_escape(),figure_deep_timeseries(),figure_stroboscopic(),figure_lyapunov()]
    manifest={"style":"compact white multipanel; Arial sans-serif; restrained teal/blue/orange palette; panel tags; no data interpolation","exports":"PDF, SVG, 600-dpi PNG","figures":[x for x in made if x]}
    (FIG/"figure_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
