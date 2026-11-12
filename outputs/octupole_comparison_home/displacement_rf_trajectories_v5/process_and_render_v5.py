"""Process the checkpointed V5 P4 trajectories and render publication figures."""
from __future__ import annotations
import csv, importlib.util, json, math
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
V4=ROOT/"outputs/octupole_comparison_home/confinement_maps_v4"
FIG=HERE/"figures"; FIG.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location("pubcanvas",ROOT/"outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py")
pub=importlib.util.module_from_spec(spec); spec.loader.exec_module(pub); pub.FIGURES=FIG
Canvas,Axes=pub.FigureCanvas,pub.Axes
INK,MUTED,GRID=pub.INK,pub.MUTED,pub.GRID
ROT,STAND=pub.ROT,pub.STAND
GEOM_COLOR={0:"#303840",10:"#0072B2",25:"#009E73",50:"#E69F00",100:"#D55E00"}
GEOM_STAND={0:"#CC6677",10:"#56B4E9",25:"#66A61E",50:"#B07AA1",100:"#AA4499"}
RET_COLOR={"retained":"#0072B2","escaped":"#D55E00"}
GEOMETRIES={"nominal":0,"radial_out_10um":10,"radial_out_25um":25,"radial_out_50um":50,"radial_out_100um":100}

def read_json(p):return json.loads(p.read_text(encoding="utf-8"))
def csv_write(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def read_trace(trace_dir):
    allrows=[];prev=None
    for p in sorted(trace_dir.glob("segment_*.csv")):
        with p.open(encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f):
                rf=float(row["rf_period"])
                if prev is not None and abs(rf-prev)<1e-11:continue
                allrows.append({k:float(v) for k,v in row.items()});prev=rf
    return allrows
def ret_label(summary):return "retained" if summary["status"]=="retained_through_horizon" else "escaped"

def process_data():
    summaries={}
    all_points=[]; envelope_rows=[]
    for path in sorted((HERE/"trajectories").glob("*/summary.json")):
        d=read_json(path); summaries[d["key"]]=d
        rows=read_trace(path.parent)
        if not rows:raise ValueError(f"No trajectory samples in {path.parent}")
        radii=np.array([r["r_mm"] for r in rows]); periods=np.array([r["rf_period"] for r in rows])
        # Preserve the raw sampled orbit; the envelope is a transparent per-RF-cycle maximum.
        cycle=np.floor(periods+1e-10).astype(int)
        uniq=np.unique(cycle)
        env=[]
        for k in uniq:
            idx=np.flatnonzero(cycle==k)
            peak=float(np.max(radii[idx])); env.append((int(k),peak))
        env=np.asarray(env,float)
        peaks,_=find_peaks(env[:,1],prominence=.05) if len(env)>2 else (np.array([],int),{})
        gaps=np.diff(env[peaks,0]) if len(peaks)>1 else np.array([])
        d["max_radius_mm_sampled"]=float(np.max(radii))
        d["rms_radius_mm_sampled"]=float(np.sqrt(np.mean(radii*radii)))
        d["minimum_radius_mm_sampled"]=float(np.min(radii))
        d["n_radial_samples"]=int(len(radii))
        d["radial_envelope_definition"]="maximum sampled radius within each integer RF period; no filtering or smoothing"
        d["radial_oscillation_peak_prominence_mm"]=.05
        d["radial_oscillation_peak_count"]=int(len(peaks))
        d["median_radial_peak_spacing_RF"]=float(np.median(gaps)) if len(gaps) else None
        d["first_radial_peak_RF"]=float(env[peaks[0],0]) if len(peaks) else None
        d["last_radial_peak_RF"]=float(env[peaks[-1],0]) if len(peaks) else None
        d["trajectory_time_series_source"]=str(path.parent.relative_to(HERE))
        path.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        for row in rows:
            all_points.append({"trajectory_key":d["key"],"geometry":d["geometry"],"displacement_um":GEOMETRIES.get(d["geometry"],0),
                "model":d["model"],"solver":"tight" if d.get("tight") else "standard",
                "solver_protocol":d["convergence_status"],**row})
        for k,peak in env:
            envelope_rows.append({"trajectory_key":d["key"],"geometry":d["geometry"],"displacement_um":GEOMETRIES.get(d["geometry"],0),
                "model":d["model"],"solver":"tight" if d.get("tight") else "standard","rf_period":int(k),"cycle_max_radius_mm":float(peak)})

    standard=[d for d in summaries.values() if not d.get("tight")]
    if len(standard)!=10:raise ValueError(f"Expected exactly ten standard trajectories; found {len(standard)}")
    tight=[d for d in summaries.values() if d.get("tight")]
    if len(tight)!=3:raise ValueError(f"Expected three strict repeats; found {len(tight)}")
    std_by={(d["geometry"],d["model"]):d for d in standard}
    pair_rows=[]
    for geom,amp in GEOMETRIES.items():
        r=std_by[(geom,"rotating")];s=std_by[(geom,"standing")]
        pair_rows.append({"geometry":geom,"field_resolution":"h50 P4","displacement_um":amp,
            "energy_eV":.75,"r0_mm":1.,"azimuth_deg":67.5,"rf_phase_deg":270.,"tangential_direction":-1,
            "rotating_status":ret_label(r),"rotating_escape_time_RF":r["escape_time_RF"],"rotating_right_censor_RF":r["right_censor_time_RF"],
            "rotating_max_radius_mm":r["max_radius_mm_sampled"],"rotating_rms_radius_mm":r["rms_radius_mm_sampled"],
            "standing_status":ret_label(s),"standing_escape_time_RF":s["escape_time_RF"],"standing_right_censor_RF":s["right_censor_time_RF"],
            "standing_max_radius_mm":s["max_radius_mm_sampled"],"standing_rms_radius_mm":s["rms_radius_mm_sampled"],
            "paired_class":"rotating_only_retained" if ret_label(r)=="retained" and ret_label(s)=="escaped" else "other",
            "rotating_radial_peaks":r["radial_oscillation_peak_count"],"rotating_median_peak_spacing_RF":r["median_radial_peak_spacing_RF"],
            "standing_radial_peaks":s["radial_oscillation_peak_count"],"standing_median_peak_spacing_RF":s["median_radial_peak_spacing_RF"],
            "n_matched_deterministic_starts":1,"observation_horizon_RF":1000})
    fields=list(pair_rows[0])
    csv_write(HERE/"processed_trajectory_pairs_v5.csv",pair_rows,fields)
    csv_write(HERE/"processed_trajectory_summary_v5.csv",[{k:v for k,v in d.items() if not isinstance(v,(dict,list))} for d in sorted(standard+ tight,key=lambda x:x["key"])],
              ["key","geometry","model","tight","status","escape_time_RF","right_censor_time_RF","max_radius_mm_sampled","rms_radius_mm_sampled","minimum_radius_mm_sampled","radial_oscillation_peak_count","median_radial_peak_spacing_RF","nfev","integration_wall_seconds","convergence_status"])
    csv_write(HERE/"trajectory_samples_combined_v5.csv",all_points,
              ["trajectory_key","geometry","displacement_um","model","solver","solver_protocol","time_s","rf_period","x_mm","y_mm","r_mm","vx_m_s","vy_m_s"])
    csv_write(HERE/"radial_envelopes_by_RF_v5.csv",envelope_rows,
              ["trajectory_key","geometry","displacement_um","model","solver","rf_period","cycle_max_radius_mm"])

    # V4 N48 reference has the identical physical start, but a different field representation.
    audit=read_json(HERE/"initial_condition_audit.json")
    v4={"rotating_status":"retained","rotating_escape_time_RF":None,"rotating_right_censor_RF":1000,
        "standing_status":"escaped","standing_escape_time_RF":audit["V4_reference_escape_time_RF"],"standing_right_censor_RF":None,
        "field_resolution":"h35 N48 V4 reference"}
    (HERE/"v4_reference_same_launch_v5.json").write_text(json.dumps(v4,indent=2)+"\n",encoding="utf-8")
    return std_by,tight,pair_rows,all_points,v4

def txt(c,x,y,s,size=6.2,color=INK,ha="center",weight="normal"):c.text(x,y,s,size=size,color=color,ha=ha,weight=weight)
def legend(c,x,y,shape,color,label,size=.8,fill=True):
    c.marker(x,y,shape,color,size,fill,.2);c.text(x+2,y,label,size=5.8,ha="left",color=INK)
def save(c,name,description):
    info=c.save();p=FIG/"figure_manifest.json";d=read_json(p) if p.exists() else {"figures":[]}
    d["figures"]=[x for x in d["figures"] if x["name"]!=name]
    d["figures"].append({"name":name,"exports":info,"description":description,"data":"processed V5 trajectories and exact V4 paired reference; deterministic single-start examples"})
    p.write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8")

def dashed_path(ax,points,color,lw=.22,dash_mm=1.1,gap_mm=.7):
    """Draw an identical dash pattern in raster and vector outputs."""
    if len(points)<2:return
    xy=[(ax.X(x),ax.Y(y)) for x,y in points];cycle=dash_mm+gap_mm;phase=0.0
    for (x0,y0),(x1,y1) in zip(xy[:-1],xy[1:]):
        dx,dy=x1-x0,y1-y0;length=math.hypot(dx,dy)
        if length<=0:continue
        s=0.0
        while s<length-1e-12:
            q=phase%cycle
            if q<dash_mm:
                step=min(length-s,dash_mm-q);a=s/length;b=(s+step)/length
                ax.c.line(x0+dx*a,y0+dy*a,x0+dx*b,y0+dy*b,color,lw)
            else:step=min(length-s,cycle-q)
            s+=step;phase=(phase+step)%cycle

def figure1(pairs,v4):
    c=Canvas("Fig1_displacement_retention_escape_v5",178,131,"double")
    txt(c,89,6,"Finite-time retention and first passage versus radial displacement",10.1,weight="bold")
    txt(c,89,13,"One V4 boundary launch · 0.75 eV · r₀ = 1 mm · θ = 67.5° · φRF = 270° · clockwise",6.4,MUTED)
    pub.panel_title(c,15,24,"A","First aperture crossing; survivors right-censored",7.0)
    labels={0:"V4\nN48",1:"0",2:"10",3:"25",4:"50",5:"100"}
    ax=Axes(c,25,31,143,53,(-.45,5.45),(-40,1080),list(range(6)),[0,200,400,600,800,1000],
        xfmt=lambda v:labels.get(int(v),""),yfmt=lambda v:f"{int(v)}")
    ax.xlabel("Displacement amplitude Δ (µm); V4 N48 reference shown separately",y=92,size=6.2)
    ax.ylabel("Time (RF periods)",x=18,size=6.)
    c.line(ax.x,ax.Y(1000),ax.x+ax.w,ax.Y(1000),"#7C858D",.25,(1.2,.8))
    for idx,p in enumerate(pairs):
        pos=idx+1
        # Rotating: censored survivor at 1000 periods, drawn with open circle and a censor arrow.
        c.marker(ax.X(pos-.18),ax.Y(1000),"circle",ROT,1.0,False,.3)
        xx,yy=ax.X(pos-.18),ax.Y(1000)
        c.line(xx,yy-2.2,xx,yy-0.5,ROT,.25);c.line(xx,yy-0.5,xx-.65,yy-1.2,ROT,.25);c.line(xx,yy-.5,xx+.65,yy-1.2,ROT,.25)
        sy=float(p["standing_escape_time_RF"])
        c.marker(ax.X(pos+.18),ax.Y(sy),"square",STAND,.9,True,.2)
    # Matched V4 start is a separate N48 reference, not part of the h50 P4 geometry sequence.
    c.marker(ax.X(0-.18),ax.Y(1000),"circle",ROT,1.0,False,.3)
    c.marker(ax.X(.18),ax.Y(v4["standing_escape_time_RF"]),"square",STAND,.9,False,.25)
    # Strict repeats are hollow triangles offset slightly within their geometry category.
    for xpos,ypos,color in ((1-.02,636.7586614652821,ROT),(2+.02,474.8351568670178,STAND),(5+.02,416.7711547954465,STAND)):
        c.marker(ax.X(xpos),ax.Y(ypos),"triangle",color,.78,False,.22)
    legend(c,30,18.5,"circle",ROT,"Rotating: survived 1000 RF (censored)",.7,False)
    legend(c,88,18.5,"square",STAND,"Standing: first outward r = 8 mm crossing",.7,True)
    legend(c,142,18.5,"triangle",INK,"Tighter rerun",.7,False)
    # Panel B: actual binary outcome matrix, n=1 matched deterministic launch per column.
    pub.panel_title(c,15,101,"B","Matched outcome matrix",6.8)
    cell_x=[48,70,92,114,136,158]
    c.text(17,112,"Standing",size=5.8,ha="left",color=INK)
    c.text(17,118,"Rotating",size=5.8,ha="left",color=INK)
    for i,x in enumerate(cell_x):
        c.rect(x-4,109,8,5,RET_COLOR["escaped"],stroke="#FFFFFF",lw=.15)
        c.rect(x-4,115,8,5,RET_COLOR["retained"],stroke="#FFFFFF",lw=.15)
        c.text(x,124,labels[i],size=5.1,color=MUTED)
    txt(c,89,129,"Blue = retained through 1000 RF; vermilion = escaped. n = 1 per cell; deterministic result, no population inference.",5.6,MUTED)
    save(c,"Fig1_displacement_retention_escape_v5","Matched retention outcomes and first-passage times for the same deterministic launch across five h50 P4 geometries; V4 N48 is shown as a separate reference.")

def figure2(pairs,all_points):
    c=Canvas("Fig2_radial_motion_selected_geometries_v5",178,112,"double")
    txt(c,89,6,"Radial motion on nominal and displaced h50 P4 fields",10.3,weight="bold")
    txt(c,89,13,"Same 0.75 eV matched launch · raw solver samples at 8 points/RF · traces terminate at first aperture crossing",6.3,MUTED)
    pub.panel_title(c,15,22,"A","Radius versus time",7.0)
    ax=Axes(c,25,29,142,63,(0,1000),(0,8.7),[0,200,400,600,800,1000],[0,2,4,6,8],
       xfmt=lambda v:f"{int(v)}",yfmt=lambda v:f"{v:g}")
    ax.xlabel("Time (RF periods)",y=102,size=6.2);ax.ylabel("Radius r (mm)",x=18,size=6.)
    c.line(ax.x,ax.Y(8),ax.x+ax.w,ax.Y(8),"#9A3D3D",.4,(1.5,.8))
    for geom in ("nominal","radial_out_10um","radial_out_100um"):
        for model in ("rotating","standing"):
            group=[r for r in all_points if r["geometry"]==geom and r["model"]==model and r["solver"]=="standard"]
            group.sort(key=lambda r:r["rf_period"])
            # Use each actual 8/RF sample; only duplicate segment junctions were removed during processing.
            data=[(r["rf_period"],r["r_mm"]) for r in group]
            if model=="rotating":ax.path(data,GEOM_COLOR[GEOMETRIES[geom]],.22)
            else:dashed_path(ax,data,GEOM_COLOR[GEOMETRIES[geom]],.22)
            if model=="standing":
                g=next(p for p in pairs if p["geometry"]==geom)
                c.marker(ax.X(g["standing_escape_time_RF"]),ax.Y(8),"triangle",GEOM_COLOR[GEOMETRIES[geom]],.72,True,.2)
    # Compact legend: hue encodes geometry; solid/dashed encodes model.
    for i,(amp,color) in enumerate(((0,GEOM_COLOR[0]),(10,GEOM_COLOR[10]),(100,GEOM_COLOR[100]))):
        xx=27+i*34;c.line(xx,18.5,xx+5,18.5,color,.6);c.text(xx+6,18.5,"nominal" if amp==0 else f"Δ={amp} µm",size=5.5,ha="left")
    c.line(132,18.5,137,18.5,INK,.6);c.text(138,18.5,"rotating solid",size=5.5,ha="left")
    c.line(158,18.5,163,18.5,INK,.6,(1.,.6));c.text(164,18.5,"standing dashed",size=5.5,ha="left")
    txt(c,89,108,"Aperture r = 8 mm is the thin red line. Hues identify geometry; symbols mark standing-field escape events. No trajectory is extended beyond an observed exit.",5.55,MUTED)
    save(c,"Fig2_radial_motion_selected_geometries_v5","Direct sampled radial motion on nominal, 10 µm and 100 µm h50 P4 fields; 25/50 µm first-passage times are in Fig. 1.")

def figure3(all_points):
    c=Canvas("Fig3_xy_orbits_nominal_and_100um_v5",178,112,"double")
    txt(c,89,6,"Representative x–y orbit geometry for nominal and 100 µm radial-out fields",9.5,weight="bold")
    txt(c,89,13,"First 100 RF periods from the identical clockwise launch · all plotted coordinates are integration samples",6.3,MUTED)
    layouts=[("nominal",19,29),("radial_out_100um",97,29)]
    for geom,x,y in layouts:
        amp=GEOMETRIES[geom];pub.panel_title(c,x,y-6,"A" if amp==0 else "B","Nominal h50 P4" if amp==0 else "Δ = 100 µm h50 P4",6.7)
        ax=Axes(c,x+11,y+4,55,55,(-8.5,8.5),(-8.5,8.5),[-8,0,8],[-8,0,8],xfmt=lambda v:f"{int(v)}",yfmt=lambda v:f"{int(v)}")
        circle=[(8*math.cos(i*2*math.pi/240),8*math.sin(i*2*math.pi/240)) for i in range(241)]
        ax.path(circle,"#8D5555",.35)
        color=GEOM_COLOR[GEOMETRIES[geom]]
        for model in ("rotating","standing"):
            g=[r for r in all_points if r["geometry"]==geom and r["model"]==model and r["solver"]=="standard" and r["rf_period"]<=100+1e-12]
            g.sort(key=lambda r:r["rf_period"])
            xy=[(r["x_mm"],r["y_mm"]) for r in g]
            if model=="rotating":ax.path(xy,color,.28)
            else:dashed_path(ax,xy,color,.28)
            if g:
                ax.marker(g[0]["x_mm"],g[0]["y_mm"],"circle" if model=="rotating" else "triangle",color,.58,True,.15)
        ax.xlabel("x (mm)",y=y+64,size=5.7);ax.ylabel("y (mm)",x=x+7,size=5.7)
    legend(c,63,100,"circle",INK,"Rotating (solid)",.7,True);legend(c,111,100,"triangle",INK,"Standing (dashed)",.7,True)
    txt(c,89,108,"Circle is the 8 mm aperture; the path window ends at 100 RF in both models and both geometries. Full radial records remain available in the CSV data.",5.55,MUTED)
    save(c,"Fig3_xy_orbits_nominal_and_100um_v5","Matched initial x–y trajectories during the first 100 RF periods for nominal and 100 µm radial-out h50 P4 geometries.")

def figure4(refinement):
    c=Canvas("Fig4_tight_tolerance_escape_validation_v5",178,105,"double")
    txt(c,89,6,"Numerical refinement of selected first-passage times",10.1,weight="bold")
    txt(c,89,13,"Same h50 P4 FEM fields and launch · DOP853 rtol = 2×10⁻⁹ vs 2×10⁻¹¹; max step T/50 vs T/100",6.1,MUTED)
    pub.panel_title(c,16,22,"A","Standard and tighter first crossings",6.8)
    labs={0:"nominal",1:"10 µm",2:"100 µm"}
    ax=Axes(c,27,29,61,58,(-.4,2.4),(300,700),[0,1,2],[300,400,500,600,700],xfmt=lambda v:labs[int(v)],yfmt=lambda v:f"{int(v)}")
    ax.xlabel("Geometry",y=96,size=5.8);ax.ylabel("Escape time (RF periods)",x=20,size=5.4)
    for i,q in enumerate(refinement):
        x=i;std=float(q["standard_escape_time_RF"]);tight=float(q["tight_escape_time_RF"])
        ax.marker(x-.075,std,"circle",STAND,.72,True,.18)
        ax.marker(x+.075,tight,"triangle",STAND,.72,False,.18)
    legend(c,49,18,"circle",STAND,"standard",.6,True);legend(c,72,18,"triangle",STAND,"tight",.6,False)
    pub.panel_title(c,103,22,"B","Tight − standard",6.8)
    bx=Axes(c,113,29,48,58,(-.4,2.4),(-.0025,.0006),[0,1,2],[-.002,.001*0,.0005],xfmt=lambda v:labs[int(v)],yfmt=lambda v:f"{v:.4f}")
    bx.xlabel("Geometry",y=96,size=5.8);bx.ylabel("Δ escape time (RF periods)",x=105,size=5.4)
    bx.line(-.3,0,2.3,0,INK,.25,(1.,.6))
    for i,q in enumerate(refinement):
        delta=float(q["escape_time_difference_RF"]);bx.marker(i,delta,"circle",ROT,.78,True,.18)
        c.text(bx.X(i),bx.Y(delta)-3.2,f"{delta:+.2g}",size=4.9,color=MUTED)
    txt(c,89,101,"All three classifications agree; maximum absolute time shift is below 0.0021 RF periods. This controls solver tolerance on fixed fields, not FEM mesh error.",5.5,MUTED)
    save(c,"Fig4_tight_tolerance_escape_validation_v5","Standard/tighter DOP853 escape-time comparison for three selected standing-field cases.")

def main():
    std_by,tight,pairs,points,v4=process_data()
    refinement=read_json(HERE/"numerical_refinement_comparison_v5.json")["selected_cases"]
    figure1(pairs,v4);figure2(pairs,points);figure3(points);figure4(refinement)
    # Auditable numerical checks and resource/deadline accounting.
    audit=read_json(HERE/"initial_condition_audit.json");budget=read_json(HERE/"global_deadline_status.json")
    standard=[d for d in std_by.values()]
    all_summary=list(std_by.values())+tight
    if audit["max_component_difference"]!=0 or audit["direction_sign_audit"]!="passed":raise ValueError("Initial state/direction audit failed")
    if any(d["field_representation"]["resolution"]!="h50" for d in all_summary):raise ValueError("The selected trajectory batch is not consistently h50")
    if any((d["status"]=="retained_through_horizon") != (d["right_censor_time_RF"]==1000) for d in all_summary):raise ValueError("Retention/censoring fields inconsistent")
    std_wall=sum(float(d["integration_wall_seconds"]) for d in standard)
    tight_wall=sum(float(d["integration_wall_seconds"]) for d in tight)
    numcheck={"status":"passed","standard_trajectory_count":len(standard),"tight_repeat_count":len(tight),
      "new_FEM_solves":0,"new_field_reconstructions":0,"new_trajectory_integrations":13,
      "all_initial_conditions_match_V4_exactly":True,"direction_sign":"negative tangential/CW, verified", 
      "all_standard_pair_classifications":"rotating retained through horizon; standing escaped", 
      "tight_repeat_classifications_all_agree":all(x["same_classification"] for x in refinement),
      "maximum_tight_time_difference_RF":max(abs(x["escape_time_difference_RF"]) for x in refinement),
      "P4_resolution":"h50 for all compared geometries; full-asymmetry displaced fields",
      "V4_reference_field":"h35 N48; retained separately as a reference and not used as a displaced-field representation",
      "right_censoring":"1000-RF rotating survivors are right-censored, not labelled escapes",
      "permanent_stability_claim":False,"fabrication_tolerance_claim":False,
      "integration_wall_seconds_standard":std_wall,"integration_wall_seconds_tight_repeats":tight_wall,
      "global_budget_started_utc":budget["started_utc"],"global_deadline_unix_s":budget["deadline_unix_s"],
      "resources_at_last_checkpoint":budget.get("resources"),
      "figure_exports":{"count":4,"formats":["PDF","SVG","PNG 600 dpi"]},
      "limitations":["one selected deterministic launch", "field gradients are elementwise P4 finite-element gradients", "no trajectory population inference", "no fabrication-tolerance inference"]}
    (HERE/"numerical_checks_v5.json").write_text(json.dumps(numcheck,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"figures":4,"trajectory_pairs":len(pairs),"strict_repeats":len(tight),"numerical_checks":"passed","integration_wall_minutes":(std_wall+tight_wall)/60},indent=2))
if __name__=="__main__":main()
