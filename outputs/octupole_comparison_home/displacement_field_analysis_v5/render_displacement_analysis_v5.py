"""Render analysis figures from the already validated displacement pilot data."""
from __future__ import annotations
import csv, importlib.util, json, math
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
OLD=ROOT/"outputs/octupole_comparison_home/electrode_displacement_sensitivity_v1"
FIG=HERE/"figures"; FIG.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location("pubcanvas",ROOT/"outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py")
pub=importlib.util.module_from_spec(spec); spec.loader.exec_module(pub); pub.FIGURES=FIG
Canvas,Axes=pub.FigureCanvas,pub.Axes
ROT,STAND,ADV,INK,MUTED=pub.ROT,pub.STAND,pub.ADVANTAGE,pub.INK,pub.MUTED
COLORS={10:"#0072B2",25:"#009E73",50:"#E69F00",100:"#D55E00"}

def csv_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def txt(c,x,y,s,size=6.2,color=INK,ha="center",weight="normal"):c.text(x,y,s,size=size,color=color,ha=ha,weight=weight)
def marker(c,x,y,color,shape="circle",size=.8,fill=True):c.marker(x,y,shape,color,size,fill,.22)
def legend(c,x,y,color,label,shape="circle",size=.75):
    c.marker(x,y,shape,color,size,True,.18); c.text(x+2,y,label,size=5.8,ha="left",color=INK)
def save(c,name,desc):
    info=c.save(); p=FIG/"figure_manifest.json"
    d=json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"figures":[]}
    d["figures"].append({"name":name,"exports":info,"description":desc,"source":"existing displacement pilot P4 field and processed CSV/JSON data"})
    p.write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8")

def figure_scaling():
    rows=csv_rows(HERE/"effective_minimum_scaling.csv")
    c=Canvas("Fig1_displacement_minimum_scaling_v5",178,113,"double")
    txt(c,89,6,"Effective-potential minimum displacement under radial electrode motion",10.5,weight="bold")
    txt(c,89,13,"Four existing h50 field solutions; the 10 µm point also has an independent h35 check",6.5,MUTED)
    ax=Axes(c,26,27,128,65,(.9,2.1),(.3,1.2),[1,math.log10(25),math.log10(50),2],[.3,.4,.6,.8,1.0],
        xfmt=lambda v:{1:"10",round(math.log10(25),3):"25",round(math.log10(50),3):"50",2:"100"}.get(round(v,3),f"{10**v:g}"),yfmt=lambda v:f"{v:g}",ylog=True)
    # The only x coordinates are the four measured amplitudes; the dashed curve is the explicit Delta^(1/3) prediction.
    for row in rows:
        d=float(row["electrode_displacement_um"]); r=float(row["minimum_radius_mm"]); x=math.log10(d)
        marker(c,ax.X(x),ax.Y(r),COLORS[int(d)],"circle",1.15,True)
    d=[10,25,50,100]; guide=[float(rows[0]["minimum_radius_mm"])*(x/10)**(1/3) for x in d]
    ax.path([(math.log10(x),y) for x,y in zip(d,guide)],MUTED,.3,(1.2,.8))
    validation=json.loads((OLD/"processed/effective_minimum_method_validation.json").read_text(encoding="utf-8"))
    for res,key,shape in (("h35","h35_best_radius_mm","square"),("h50","h50_best_radius_mm","triangle")):
        marker(c,ax.X(1.0),ax.Y(float(validation[key])),INK,shape,.8,False)
    ax.xlabel("Electrode-center displacement Δ (µm; logarithmic scale)",y=101,size=6.1)
    ax.ylabel("Radius of located minimum r_min (mm)",x=16,size=6.1)
    legend(c,122,25,MUTED,"Δ^(1/3) guide", "circle",.55)
    legend(c,122,31,INK,"h35 / h50, Δ = 10 µm","square",.55)
    txt(c,89,109,"Log–log fit to the four h50 radial-out points: exponent 0.326; guide is normalized at 10 µm. No data are interpolated.",5.8,MUTED)
    save(c,"Fig1_displacement_minimum_scaling_v5","Located effective-minimum radius and a sampled cubic-root scaling guide.")

def figure_harmonics():
    harmonic=csv_rows(OLD/"processed/azimuthal_harmonics.csv")
    sc=json.loads((HERE/"displacement_analysis_summary_v5.json").read_text(encoding="utf-8"))
    c=Canvas("Fig2_symmetry_breaking_harmonics_v5",178,111,"double")
    txt(c,89,6,"Symmetry-breaking dipole and nominal octupole harmonics",10.6,weight="bold")
    txt(c,89,13,"Complex-potential Fourier magnitudes on deterministic rings; radial-out displacement, h50",6.4,MUTED)
    x0,y,w,h=18,27,65,62
    modes=[(1,"m = 1 · induced dipole",ROT),(4,"m = 4 · nominal octupole",STAND)]
    for j,(mode,title,col) in enumerate(modes):
        x=x0+j*80; pub.panel_title(c,x,y-4,chr(65+j),title,7.0)
        ax=Axes(c,x+7,y+4,57,48,(.35,2.15),(1e-9,2e-4),[.5,1,1.5,2],[1e-9,1e-7,1e-5,1e-4],xfmt=lambda v:f"{v:g}",
            yfmt=lambda v:{1e-9:"10⁻⁹",1e-7:"10⁻⁷",1e-5:"10⁻⁵",1e-4:"10⁻⁴"}.get(v,f"{v:g}"),ylog=True)
        g=[r for r in harmonic if r["pattern"]=="radial_out_10um" and r["resolution"]=="h50" and int(r["azimuthal_order_m"])==mode and float(r["radius_mm"]) in (.5,1.,2.)]
        for r in g: ax.marker(float(r["radius_mm"]),float(r["displaced_abs_coefficient_V_per_V"]),"circle",col,.95,True)
        exponent=next(q["potential_coefficient_radial_exponent"] for q in sc["harmonic_radial_order"] if q["azimuthal_mode_m"]==mode)
        txt(c,x+38,y+56,f"measured radial exponent = {exponent:.3f}",5.6,col)
        ax.xlabel("Radius r (mm)",y=y+59,size=5.8)
        ax.ylabel("|H_m| (V/V)",x=x-1,size=5.7)
    txt(c,89,101,"In the local harmonic expansion, Φ₁ ∝ r gives an approximately constant dipolar field; Φ₄ ∝ r⁴ gives E₄ ∝ r³.",5.8,MUTED)
    txt(c,89,107,"At r = 0.5, 1 and 2 mm, the m = 1 to m = 4 electric-field amplitude ratio is about 1.44×10³, 1.80×10² and 22.5.",5.8,MUTED)
    save(c,"Fig2_symmetry_breaking_harmonics_v5","Measured m=1 and m=4 complex-potential harmonics on three sampled radii.")

def figure_stationary():
    validation=json.loads((OLD/"processed/effective_minimum_method_validation.json").read_text(encoding="utf-8"))
    j=json.loads((HERE/"stationary_structure_v5.json").read_text(encoding="utf-8"))
    c=Canvas("Fig3_stationary_points_and_curvature_v5",178,122,"double")
    txt(c,89,6,"Stationary structure of the 10 µm radial-out effective potential",10.3,weight="bold")
    txt(c,89,13,"Existing full-domain P4 fields; located minima, finite-difference saddles and local curvature",6.4,MUTED)
    pub.panel_title(c,15,23,"A","Located stationary points",7.1)
    ax=Axes(c,25,31,61,65,(-.55,.55),(-.55,.55),[-.5,-.25,0,.25,.5],[-.5,-.25,0,.25,.5],xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v:g}")
    for res,color,shape in (("h35",ROT,"square"),("h50",STAND,"circle")):
        p=next(x["position_mm"] for x in validation["all_methods_positions_mm"] if x["resolution"]==res and x["radius_mm"]<.45 and x["E2"]==min(y["E2"] for y in validation["all_methods_positions_mm"] if y["resolution"]==res and y["radius_mm"]<.45))
        for pos in (p,[p[0],-p[1]]): ax.marker(pos[0],pos[1],shape,color,.8,res=="h50",.2)
    ax.marker(.45811,0,"triangle",ADV,.95,True,.2)
    for s in j["stationary_saddle_refinements"]:
        if s["pair"] not in ("mirror-minimum connecting ridge","global-to-secondary connecting ridge"): continue
        p=s["stationary_candidate_mm"]
        ax.marker(p[0],p[1],"triangle",INK,.65,s["resolution"]=="h50",.2)
    ax.xlabel("x (mm)",y=100,size=5.9); ax.ylabel("y (mm)",x=20,size=5.9)
    legend(c,91,35,ROT,"h35","square",.55); legend(c,91,41,STAND,"h50","circle",.55)
    legend(c,91,47,ADV,"secondary local minimum","triangle",.55)
    legend(c,91,53,INK,"saddle candidates","triangle",.55)
    # Panel B: local curvature eigenvalues versus finite-difference step, no fitted trend.
    pub.panel_title(c,109,23,"B","Local curvature versus spatial step",7.1)
    txt(c,145,27,"circle: λ_min · triangle: λ_max",4.8,MUTED)
    ax2=Axes(c,122,31,47,65,(0,110),(1e-5,3e-5),[10,25,50,100],[1e-5,1.5e-5,2e-5,2.5e-5,3e-5],
        xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v/1e-5:.1f}",ylog=False)
    for res,color,shape in (("h35",ROT,"square"),("h50",STAND,"circle")):
        data=[x for x in j["curvature_results"] if x["pattern"]=="radial_out_10um" and x["resolution"]==res and x["stationary_candidate"]=="global_minimum_branch_1" and x["step_um"] in (25,50,100)]
        for eig_i,style in ((0,shape),(1,"triangle")):
            ax2.path([(x["step_um"],x["curvature_eigenvalues_eV_per_mm2"][eig_i]) for x in data],color,.25)
            for x in data: ax2.marker(x["step_um"],x["curvature_eigenvalues_eV_per_mm2"][eig_i],style,color,.55,res=="h50",.18)
    ax2.xlabel("Finite-difference step (µm)",y=104,size=5.7)
    ax2.ylabel("Hessian eigenvalue (10⁻⁵ eV/mm²)",x=116,size=5.4)
    txt(c,89,111,"Global minima form a reflection-related pair; a second, higher local minimum lies near (0.458 mm, 0). Saddle markers have opposite-sign curvature eigenvalues.",5.8,MUTED)
    txt(c,89,117,"Finite-element field gradients are elementwise; curvature is therefore reported as scale-dependent local finite differences.",5.8,MUTED)
    save(c,"Fig3_stationary_points_and_curvature_v5","Mirror-related minima, a higher local minimum, saddle candidates and finite-difference curvature checks.")

def main():
    figure_scaling(); figure_harmonics(); figure_stationary()
    print(json.dumps({"figures":3,"formats":["PDF","SVG","600-dpi PNG"],"directory":str(FIG)},indent=2))
if __name__=="__main__":main()
