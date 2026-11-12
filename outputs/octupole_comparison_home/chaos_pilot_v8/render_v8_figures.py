"""Publication figures using the repository's Pillow/ReportLab vector canvas."""
import csv,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; REPO=HERE.parents[2]; FIG=HERE/"figures"; FIG.mkdir(exist_ok=True)
PUB=REPO/"outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py"
spec=importlib.util.spec_from_file_location("v8_publication_canvas",PUB);pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)
pub.FIGURES=FIG;pub.PNG_DPI=600
Canvas,Axes=pub.FigureCanvas,pub.Axes
INK="#202830"; MUTED="#5D6872"; GRID="#D9E0E5"; TEAL="#177E89"; RED="#D1495B"; BLUE="#2A6F97"; GOLD="#C48B13"; PURPLE="#8B5CA5"; GRAY="#66717D"
PHASE_COLOR={0.0:BLUE,45.0:PURPLE,90.0:RED,135.0:GOLD,180.0:TEAL,225.0:GRAY,270.0:"#B05B35",315.0:"#50815A"}

def readcsv(name):
    p=HERE/"processed"/name
    if not p.exists(): return []
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def save(c): return c.save()
def stage_shape(stage): return "circle" if stage in ("map","scan") else ("triangle" if stage=="phase-check" else "square")

def fig1_map():
    rows=readcsv("azimuth_phase_map.csv");c=Canvas("V8_Fig1_azimuth_phase_outcomes",178,110,"double_column")
    pub.panel_title(c,10,8,"A","Standing N48 finite-time escape map",9)
    for i,d in enumerate((-1,1)):
        x=23+i*81;ax=Axes(c,x,19,67,55,(8,12),(0,315),[8,9,10,11,12],[0,45,90,135,180,225,270,315],xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v:g}")
        for r in rows:
            if int(r["vtheta_sign"])!=d:continue
            color=TEAL if r["outcome"]=="retained" else RED
            ax.marker(float(r["theta0_deg"]),float(r["rf_phase_deg"]),stage_shape(r.get("source_stage","")),color,size=0.85,fill=True,lw=0.22)
        ax.xlabel("Initial azimuth θ₀ (deg)",y=83,size=7.2)
        if i==0:ax.ylabel("Initial RF phase φRF (deg)",x=12,size=7.2)
        c.text(x+1,16,"vθ < 0" if d<0 else "vθ > 0",size=7.5,weight="bold",ha="left")
    pub.legend_item(c,20,93,"circle",TEAL,"retained to 1000 RF periods",size=.75,text_size=6.5)
    pub.legend_item(c,70,93,"circle",RED,"escaped through 8 mm",size=.75,text_size=6.5)
    pub.legend_item(c,119,93,"circle",INK,"balanced grid / V6 scan",size=.65,text_size=6.2)
    pub.legend_item(c,147,93,"square",INK,"adaptive",size=.65,text_size=6.2)
    c.text(105,105,"Selected phase checks use triangles; blank locations were not calculated. Points are not interpolated.",size=5.8,color=MUTED)
    return save(c)

def fig2_escape():
    rows=readcsv("azimuth_phase_map.csv");c=Canvas("V8_Fig2_escape_times",178,103,"double_column")
    pub.panel_title(c,10,8,"B","First-passage time near the local boundary",9)
    for i,d in enumerate((-1,1)):
        x=24+i*81;ax=Axes(c,x,19,67,55,(8,12),(0,1050),[8,9,10,11,12],[0,250,500,750,1000],xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v:g}")
        ax.line(8,1000,12,1000,GRAY,.22,dash=(1.5,1.2))
        for r in rows:
            if int(r["vtheta_sign"])!=d:continue
            theta=float(r["theta0_deg"]); phase=float(r["rf_phase_deg"])
            if r["outcome"]=="escaped":
                ax.marker(theta,float(r["escape_time_RF"]),stage_shape(r.get("source_stage","")),PHASE_COLOR.get(phase,GRAY),size=.8,fill=True,lw=.2)
            else:
                ax.marker(theta,1000,"triangle",TEAL,size=.78,fill=False,lw=.28)
        ax.xlabel("Initial azimuth θ₀ (deg)",y=83,size=7.2)
        if i==0:ax.ylabel("First escape time (RF periods)",x=13,size=7.2)
        c.text(x+1,16,"vθ < 0" if d<0 else "vθ > 0",size=7.5,weight="bold",ha="left")
    for k,ph in enumerate((0,90,180,270)):
        pub.legend_item(c,25+35*k,92,"circle",PHASE_COLOR[float(ph)],f"φ={ph}°",size=.62,text_size=6.1)
    pub.legend_item(c,25,99,"triangle",TEAL,"retained: right-censored at 1000 RF",fill=False,size=.62,text_size=6.0)
    c.text(99,99,"Adaptive intermediate phases are included only at their calculated points.",size=5.8,color=MUTED)
    return save(c)

def fig3_sensitivity():
    rows=readcsv("paired_sensitivity_summary.csv");c=Canvas("V8_Fig3_initial_condition_sensitivity",100,91,"single_column")
    pub.panel_title(c,8,8,"C","Paired outcome sensitivity pilot",8.5)
    xvals=[math.log10(float(r["delta_deg"])) for r in rows];xmin=min(xvals)-.2 if xvals else -2.8;xmax=max(xvals)+.2 if xvals else -0.6
    ax=Axes(c,22,17,67,43,(xmin,xmax),(0,1),xvals and sorted(set(xvals)),[0,.25,.5,.75,1],xfmt=lambda v:f"{10**v:g}",yfmt=lambda v:f"{v:g}")
    for coord,color,shape in (("azimuth",BLUE,"circle"),("phase",RED,"square")):
        g=sorted([r for r in rows if r["coordinate"]==coord],key=lambda r:float(r["delta_deg"]))
        for r in g:
            x=math.log10(float(r["delta_deg"]));y=float(r["fraction_different"])
            ax.marker(x,y,shape,color,size=.9,fill=True,lw=.2)
            c.text(ax.X(x),ax.Y(y)-2.4,f"{r['n_different']}/{r['n_pairs']}",size=5.6,color=color)
    ax.xlabel("Coordinate perturbation δ (deg; logarithmic scale)",y=68,size=7.1);ax.ylabel("Pair outcome disagreement",x=12,size=7.1)
    pub.legend_item(c,28,77,"circle",BLUE,"azimuth perturbation",size=.7,text_size=6.2)
    pub.legend_item(c,67,77,"square",RED,"RF-phase perturbation",size=.7,text_size=6.2)
    c.text(50,86,"Only 4 pairs per scale; descriptive counts, no uncertainty-exponent fit.",size=6,color=MUTED)
    return save(c)

def fig4_tangent():
    tab=readcsv("ftle_summary.csv");c=Canvas("V8_Fig4_FTLE_and_stroboscopic",178,104,"double_column")
    pub.panel_title(c,10,8,"D","Finite-time tangent growth and stroboscopic projections",9)
    left=Axes(c,24,19,67,55,(0,3000),(0,.06),[0,1000,2000,3000],[0,.02,.04,.06],xfmt=lambda v:f"{v:g}",yfmt=lambda v:f"{v:.2f}")
    chosen={"N48_rotating_deep_3000":BLUE,"N48_standing_deep_3000":RED,"pure_m4_standing_deep_3000":GRAY,"standing_near_boundary_retained":PURPLE}
    for r in tab:
        label=r["label"]
        if label not in chosen:continue
        p=HERE/"tangent_traces"/(r["job_id"]+".npz")
        if r["source"]=="v6":p=REPO/"outputs/octupole_comparison_home/chaos_escape_v6/phase_space/traces"/(r["job_id"]+".npz")
        if not p.exists():continue
        g=np.load(p)["growth"];pts=[(float(t),float(log/t) if t>0 else 0) for t,log in g if t<=3000]
        left.path(pts,color=chosen[label],lw=.42)
    left.xlabel("Elapsed time (RF periods)",y=83,size=7.1);left.ylabel("ln‖δX(t)‖ / t (per RF period)",x=13,size=7)
    names=[("N48_rotating_deep_3000","N48 rotating"),("N48_standing_deep_3000","N48 standing"),("pure_m4_standing_deep_3000","pure m=4 standing"),("standing_near_boundary_retained","boundary retained")]
    for i,(key,label) in enumerate(names):
        xx=24+(i%2)*42; yy=92+(i//2)*5
        c.line(xx-.8,yy,xx+.8,yy,chosen[key],.5);c.text(xx+2.4,yy,label,size=5.9,ha="left")
    # Strobe map: x and vx, with all four-dimensional states retained in NPZ.
    groups=[r for r in tab if r["label"] in ("N48_rotating_deep_3000","N48_standing_deep_3000","pure_m4_standing_deep_3000")]
    arr=[]
    for r in groups:
        z=np.load(HERE/"tangent_traces"/(r["job_id"]+".npz"))["strobe"]
        if len(z):arr.append(z)
    xmax=max([np.max(np.abs(a[:,1]))*1e3 for a in arr]+[1.0]);vmax=max([np.max(np.abs(a[:,3])) for a in arr]+[1.0])
    st=Axes(c,110,19,55,55,(-xmax*1.05,xmax*1.05),(-vmax*1.05,vmax*1.05),[-xmax,0,xmax],[-vmax,0,vmax],xfmt=lambda v:f"{v:.2g}",yfmt=lambda v:f"{v:.2g}")
    for r,color in zip(groups,[BLUE,RED,GRAY]):
        z=np.load(HERE/"tangent_traces"/(r["job_id"]+".npz"))["strobe"]
        for xx,vx in zip(z[:,1]*1e3,z[:,3]):st.marker(float(xx),float(vx),"circle",color,size=.20,fill=True,lw=.10)
    st.xlabel("Stroboscopic x (mm)",y=83,size=7.1);st.ylabel("Stroboscopic vx (m/s)",x=101,size=7.0)
    c.text(111,16,"Integer RF periods; one 2D projection of the full 4D section.",size=5.7,color=MUTED)
    c.text(90,101,"Escaping trajectories terminate at aperture crossing; positive finite-time growth alone does not establish chaos.",size=5.8,color=MUTED)
    return save(c)

def fig5_resolution():
    comp=json.loads((HERE/"processed"/"h35_h50_comparison.json").read_text(encoding="utf-8"));c=Canvas("V8_Fig5_field_resolution_sensitivity",178,100,"double_column")
    pub.panel_title(c,8,8,"E","Resolution sensitivity of narrow angular outcomes",8.5)
    for i,d in enumerate((-1,1)):
        g=[r for r in comp if int(r["direction"])==d];lo=min(float(r["theta0_deg"]) for r in g)-.001;hi=max(float(r["theta0_deg"]) for r in g)+.001
        # show a little more context around the relevant bracket
        margin=(hi-lo)*.18;lo-=margin;hi+=margin
        ax=Axes(c,29+i*74,18,56,42,(lo,hi),(-.5,1.5),[lo,(lo+hi)/2,hi],[0,1],xfmt=lambda v:f"{v:.4f}",yfmt=lambda v:"h35" if v==0 else ("h50" if v==1 else ""))
        for r in g:
            x=float(r["theta0_deg"]);color=TEAL if r["h35_status"]=="retained" else RED
            ax.marker(x,0,"circle",color,size=.85,fill=True,lw=.22)
            color50=TEAL if r["h50_status"]=="retained" else RED
            ax.marker(x,1,"square",color50,size=.9,fill=True,lw=.22)
            ax.line(x,0,x,1,GRID,.35,dash=(.7,.5))
        ax.xlabel("Initial azimuth θ₀ (deg)",y=66,size=6.8)
        c.text(30+i*74,15,"vθ < 0" if d<0 else "vθ > 0",size=7,weight="bold",ha="left")
    pub.legend_item(c,35,77,"circle",TEAL,"retained",size=.7,text_size=6.2);pub.legend_item(c,63,77,"circle",RED,"escaped",size=.7,text_size=6.2)
    c.text(106,84,"Rows: h35 (lower), h50 (upper); dashed links pair identical launches.",size=6,color=MUTED)
    c.text(89,94,"Only one tested positive-direction classification changed between fields; this warns against reading it as a fractal feature.",size=5.8,color=MUTED)
    return save(c)

if __name__=="__main__":
    manifests=[fig1_map(),fig2_escape(),fig3_sensitivity(),fig4_tangent(),fig5_resolution()]
    (FIG/"figure_manifest.json").write_text(json.dumps(manifests,indent=2)+"\n",encoding="utf-8")
    print("Rendered",len(manifests),"figures to",FIG)
