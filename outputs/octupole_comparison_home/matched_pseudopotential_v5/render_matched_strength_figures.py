"""Publication figures for the balanced V5 matched-leading-pseudopotential data."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
FIG=HERE/"figures"; FIG.mkdir(parents=True,exist_ok=True)
RENDERER=ROOT/"outputs/octupole_comparison_home/publication_figures_v1/render_publication_figures.py"
spec=importlib.util.spec_from_file_location("pubcanvas",RENDERER)
pub=importlib.util.module_from_spec(spec); spec.loader.exec_module(pub); pub.FIGURES=FIG
Canvas,Axes=pub.FigureCanvas,pub.Axes
ROT,STAND,ADV,BOTH,INK,MUTED,GRID=pub.ROT,pub.STAND,pub.ADVANTAGE,pub.BOTH, pub.INK,pub.MUTED,pub.GRID

def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def fmt(v): return f"{v:g}"
def ctext(c,x,y,text,size=7,color=INK,ha="center",weight="normal",rotation=0): c.text(x,y,text,size=size,color=color,ha=ha,weight=weight,rotation=rotation)
def legend(c,x,y,color,label,shape="circle",size=.8):
    c.marker(x,y,shape,color,size,True,.18); c.text(x+2.1,y,label,size=6.2,ha="left",color=INK)
def save_meta(name,files,description):
    path=HERE/"figures"/"figure_manifest.json"
    d=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"figures":[]}
    d["figures"]=[x for x in d["figures"] if x["name"]!=name]
    d["figures"].append({"name":name,"exports":files,"description":description,
        "source":"processed/matched_strength_balanced_pairs.csv and checkpointed V4/V5 trajectory samples"})
    path.write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8")

def figure_energy(data):
    c=Canvas("Fig1_matched_strength_energy_retention",178,106,"double")
    ctext(c,89,6,"Finite-time retention at matched leading pseudopotential",10.8,weight="bold")
    ctext(c,89,13,"250 V rotating · 353.553 V standing · 100 kHz · 1000 RF periods · 8 mm aperture",6.8,MUTED)
    ctext(c,89,19,"Balanced deterministic grid; each energy shell has n = 64 paired starts",6.5,MUTED)
    panels=[("A","Rotating", "rotating"),("B","Standing", "standing"),("C","Paired outcome counts", "difference")]
    xstarts=[19,76,133]
    for x,(tag,title,mode) in zip(xstarts,panels):
        pub.panel_title(c,x,25,tag,title,7.6)
        ax=Axes(c,x,31,39,50,(.5,3.5),(0,1),[1,2,3],[0,.25,.5,.75,1],xfmt=fmt,yfmt=lambda v:f"{v:.2g}")
        ax.xlabel("Initial transverse energy E⊥ (eV)",y=91,size=6.2)
        ax.ylabel("Calculated-grid retention fraction",x=x-11,size=5.9)
        for e in (1.,2.,3.):
            group=[r for r in data if float(r["energy_eV"])==e]
            if mode=="rotating": val=sum(r["rotating_status"]=="retained" for r in group)/len(group); color=ROT
            elif mode=="standing": val=sum(r["standing_status"]=="retained" for r in group)/len(group); color=STAND
            else:
                ronly=sum(r["difference_class"]=="rotating_only_retained" for r in group)
                sonly=sum(r["difference_class"]=="standing_only_retained" for r in group)
                val=ronly/len(group); color=ADV
                barx=ax.X(e)-4.2; by=ax.Y(val)
                c.rect(barx,by,8.4,ax.Y(0)-by,color=color,stroke=None)
                ctext(c,ax.X(e),ax.Y(val)-2,f"R-only {ronly}",5.4,ADV)
                if sonly: ctext(c,ax.X(e),ax.Y(0)-2,f"S-only {sonly}",5.0,"#AA4499")
                continue
            barx=ax.X(e)-4.2; by=ax.Y(val)
            c.rect(barx,by,8.4,ax.Y(0)-by,color=color,stroke=None)
            ctext(c,ax.X(e),by-2,f"{sum((r['rotating_status'] if mode=='rotating' else r['standing_status'])=='retained' for r in group)}/64",5.8,INK)
    ctext(c,89,101,"Fractions summarize only the sampled grid; they are not population probabilities. Retained = no aperture crossing by 10 ms.",5.9,MUTED)
    files=c.save(); save_meta("Fig1_matched_strength_energy_retention",files,"Exact retention counts on the three calculated energy shells; no interpolation.")

def figure_phase(data):
    c=Canvas("Fig2_matched_strength_phase_azimuth",178,151,"double")
    ctext(c,89,6,"Paired retention outcomes across RF phase and launch azimuth",10.5,weight="bold")
    ctext(c,89,13,"Tiles show rotating-only survivors among the two launch radii (n = 2 per energy × phase × azimuth × direction tile)",6.5,MUTED)
    palette={0:"#E8EDF0",1:"#8CCDB0",2:ADV}
    x0,y0,pw,ph,gx,gy=23,25,68,32,8,4
    azimuths=[0.,22.5,45.,67.5]; phases=[0.,90.,180.,270.]
    for ei,e in enumerate((1.,2.,3.)):
        for di,direction in enumerate((-1.,1.)):
            x=x0+di*(pw+gx); y=y0+ei*(ph+gy)
            ctext(c,x,y-2,f"E_perp = {e:g} eV · tangential sign {int(direction):+d}",7.0,INK,ha="left",weight="bold")
            aw=(pw-11)/4; ah=(ph-8)/4; left=x+9; top=y+3
            for pi,phase in enumerate(phases):
                for ai,az in enumerate(azimuths):
                    cell=[r for r in data if float(r["energy_eV"])==e and float(r["direction"])==direction
                        and float(r["phase_deg"])==phase and float(r["azimuth_deg"])==az]
                    if len(cell)!=2: raise ValueError(f"Expected two radius samples per phase tile, found {len(cell)}")
                    count=sum(r["difference_class"]=="rotating_only_retained" for r in cell)
                    rx=left+ai*aw; ry=top+(3-pi)*ah
                    c.rect(rx,ry,aw-.5,ah-.5,palette[count],stroke="#FFFFFF",lw=.18)
                    ctext(c,rx+(aw-.5)/2,ry+(ah-.5)/2,f"{count}/2",5.6,"#FFFFFF" if count==2 else INK,weight="bold")
            ctext(c,x+3,top+2*ah,"φ_RF (°)",5.5,MUTED,ha="center",rotation=90)
            for pi,phase in enumerate(phases):
                ctext(c,left-1.4,top+(3-pi+.5)*ah,f"{phase:g}",5.0,MUTED,ha="right")
            for ai,az in enumerate(azimuths): ctext(c,left+(ai+.5)*aw,top+4*ah+2,f"{az:g}",4.8,MUTED)
            ctext(c,left+2*aw,top+4*ah+5.5,"Launch azimuth θ₀ (°)",5.4,MUTED)
    ctext(c,10,148,"Key: tile count = rotating retained / standing escaped; remaining paired starts retained in both models.",5.8,MUTED,ha="left")
    for k,(n,label) in enumerate(((0,"0/2"),(1,"1/2"),(2,"2/2"))):
        c.rect(134+k*12,146,6,2.5,palette[n],stroke="#FFFFFF",lw=.15); ctext(c,143+k*12,147.3,label,5.6,INK,ha="left")
    files=c.save(); save_meta("Fig2_matched_strength_phase_azimuth",files,"Discrete paired-outcome tiles; no interpolation or adaptive samples.")

def source_trace_path(value):
    p=Path(value)
    if p.is_file(): return p
    normalized=value.replace("/","\\")
    marker="outputs\\octupole_comparison_home\\confinement_maps_v4\\"
    ix=normalized.lower().find(marker.lower())
    if ix>=0: return ROOT/Path(normalized[ix:].replace("\\","/"))
    raise FileNotFoundError(value)

def trace_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))

def figure_trajectory(data):
    choices=[r for r in data if float(r["energy_eV"])==1.0 and float(r["r0_mm"])==1.0
        and r["difference_class"]=="rotating_only_retained"]
    chosen=max(choices,key=lambda r:float(r["standing_escape_time_RF"]))
    reuse=rows(HERE/"rotating_v4_reused_trajectories.csv")
    rotmeta=next(x for x in reuse if x["condition_id"]==chosen["condition_id"])
    rotpath=source_trace_path(rotmeta["sampled_trace"])
    jobrows=[json.loads(x) for x in (HERE/"trajectory_checkpoint.jsonl").read_text(encoding="utf-8").splitlines()]
    stand=next(x for x in jobrows if x["job"].get("condition_id")==chosen["condition_id"])
    spath=HERE/"trajectories"/(hashlib.sha256(stand["job_id"].encode()).hexdigest()[:16]+".csv")
    rr,sr=trace_rows(rotpath),trace_rows(spath)
    c=Canvas("Fig3_matched_strength_representative_trajectory",178,113,"double")
    ctext(c,89,6,"Matched trajectory pair: rotating field retained, standing field escaped",10.4,weight="bold")
    ctext(c,89,13,f"E⊥ = {chosen['energy_eV']} eV · r₀ = {chosen['r0_mm']} mm · φ_RF = {float(chosen['phase_deg']):g}° · θ₀ = {float(chosen['azimuth_deg']):g}° · sign vθ = {int(float(chosen['direction'])):+d}",6.2,MUTED)
    specs=[(15,27,"A","Rotating: 250 V",rr,ROT),(99,27,"B","Standing: 353.553 V",sr,STAND)]
    for x,y,tag,title,trace,color in specs:
        pub.panel_title(c,x,y,tag,title,7.2)
        ax=Axes(c,x+8,y+6,61,47,(-8,8),(-8,8),[-8,-4,0,4,8],[-8,-4,0,4,8],xfmt=fmt,yfmt=fmt)
        pts=[(float(z["x_mm"]),float(z["y_mm"])) for z in trace]
        # All saved trajectory samples are shown; no curve reconstruction is performed.
        ax.path(pts,color=color,lw=.20)
        # Aperture boundary (geometric reference, explicitly identified).
        c.polyline([(ax.X(8*math.cos(t)),ax.Y(8*math.sin(t))) for t in [2*math.pi*i/180 for i in range(181)]],INK,.32)
        ax.xlabel("x (mm)",y=y+56,size=6.0); ax.ylabel("y (mm)",x=x+2,size=6.0)
    # Shared radius timeline; survivor is shown with a right-censor marker.
    x,y,w,h=28,91,122,11
    c.line(x,y+h,x+w,y+h,INK,.25); c.line(x,y,x,y+h,INK,.25)
    for tick in (0,250,500,750,1000):
        xx=x+tick/1000*w; c.line(xx,y,xx,y+h,GRID,.16); ctext(c,xx,y+h+2.5,f"{tick}",5.5,MUTED)
    rax=(0,8); ctext(c,15,y+h/2,"r (mm)",5.8,INK,rotation=90)
    for trace,color,model in ((rr,ROT,"rotating"),(sr,STAND,"standing")):
        pts=[]
        for z in trace:
            t=float(z["rf_period"]); rad=float(z["r_mm"])
            pts.append((x+t/1000*w,y+h-(rad/8)*h))
        c.polyline(pts,color,.25)
    ctext(c,152,y+2,"Rotating: survivor",5.7,ROT,ha="left")
    ctext(c,152,y+7,f"Standing: escape at {float(chosen['standing_escape_time_RF']):.1f} RF",5.7,STAND,ha="left")
    ctext(c,89,109,"Time in RF periods; rotating trace is right-censored at 1000 RF. Standing trace ends at first outward crossing.",5.8,MUTED)
    # Validate the selected trace endpoints against the stored classifications.
    if max(float(z["r_mm"]) for z in rr)>8.000001 or abs(float(sr[-1]["r_mm"])-8)>2e-3:
        raise ValueError("Representative trajectory does not match aperture/event convention")
    files=c.save(); save_meta("Fig3_matched_strength_representative_trajectory",files,"Exact saved xy samples and radius trace for a preregistered grid example.")

def figure_escape(data):
    c=Canvas("Fig4_matched_strength_escape_and_orbit",178,124,"double")
    ctext(c,89,6,"First passage and radial dynamics in matched starts",10.6,weight="bold")
    ctext(c,89,13,"Individual deterministic cases; event times are outward crossings, survivors are right-censored at 1000 RF",6.4,MUTED)
    ax=Axes(c,19,27,82,69,(0,192),(1,1000),[0,64,128,192],[1,10,100,1000],xfmt=lambda v:{0:"0",64:"1 eV",128:"2 eV",192:"3 eV"}.get(v,f"{v:g}"),yfmt=lambda v:f"{v:g}",ylog=True)
    ax.xlabel("Paired initial condition (ordered by energy)",y=104,size=6.0)
    ax.ylabel("First passage / censoring time (RF periods)",x=9,size=6.0)
    ordered=sorted(data,key=lambda r:(float(r["energy_eV"]),float(r["r0_mm"]),float(r["phase_deg"]),float(r["azimuth_deg"]),float(r["direction"])))
    for i,r in enumerate(ordered):
        for offset,model,color in ((-.15,"rotating",ROT),(.15,"standing",STAND)):
            status=r[model+"_status"]
            t=float(r[model+"_escape_time_RF"]) if status=="escaped" else 1000.0
            ax.marker(i+offset,t,"circle" if status=="escaped" else "triangle",color,size=.43,fill=True,lw=.15)
    for edge in (64,128): c.line(ax.X(edge),27,ax.X(edge),96,INK,.3,dash=(1.0,.8))
    legend(c,116,34,ROT,"rotating event", "circle",.55); legend(c,116,40,STAND,"standing event","circle",.55)
    legend(c,116,46,ROT,"rotating right-censored","triangle",.55); legend(c,116,52,STAND,"standing right-censored","triangle",.55)
    ctext(c,115,62,"No escape is assigned to censored runs.",5.8,MUTED)
    # All measured paired stroboscopic RMS differences, un-smoothed.
    bx=118; by=73; bw=47; bh=27
    ctext(c,bx,by-4,"Stroboscopic radial RMS difference",6.3,INK,ha="left",weight="bold")
    vals=[float(r["delta_stroboscopic_rms_mm_rot_minus_stand"]) for r in data]
    lo=min(vals); hi=max(vals); pad=max((hi-lo)*.08,.03); lo-=pad;hi+=pad
    xs=[1,2,3]; ymin=by; ymax=by+bh
    c.line(bx,ymax,bx+bw,ymax,INK,.25); c.line(bx,ymin,bx,ymax,INK,.25)
    for v in (lo,(lo+hi)/2,hi):
        yy=ymax-(v-lo)/(hi-lo)*bh
        c.line(bx,yy,bx+bw,yy,GRID,.14)
        ctext(c,bx-1.3,yy,f"{v:.2f}",4.7,MUTED,ha="right")
    if lo < 0 < hi:
        yy=ymax-(0-lo)/(hi-lo)*bh
        c.line(bx,yy,bx+bw,yy,INK,.22,dash=(1,.7))
    for e in xs:
        cx=bx+(e-.5)/3*bw
        c.line(cx,ymin,cx,ymax,GRID,.14); ctext(c,cx,ymax+1.8,f"{e:g}",5.1,MUTED)
        group=[r for r in data if float(r["energy_eV"])==e]
        for r in group:
            val=float(r["delta_stroboscopic_rms_mm_rot_minus_stand"])
            yy=ymax-(val-lo)/(hi-lo)*bh
            # Deterministic balanced-grid observations; small index jitter separates exact overlaps.
            jitter=(int(float(r["phase_deg"]))//90*4+int(float(r["azimuth_deg"])*2)+int(float(r["direction"])>0))%13
            c.marker(cx+(jitter-6)*.21,yy,"circle",ADV,size=.36,fill=True,lw=.12)
    ctext(c,bx+bw/2,ymax+4,"Initial E⊥ (eV)",5.7,MUTED)
    ctext(c,bx+bw/2,by+bh+6.5,"Δr_RMS = rotating − standing (mm)",5.0,MUTED)
    ctext(c,89,119,"The RMS comparison uses integer-RF stroboscopic samples; escape traces include the event point for max-radius diagnostics.",5.7,MUTED)
    files=c.save(); save_meta("Fig4_matched_strength_escape_and_orbit",files,"Event/censoring marks and unsmoothed paired stroboscopic radial differences.")

def main():
    data=rows(HERE/"processed/matched_strength_balanced_pairs.csv")
    if len(data)!=192: raise ValueError(f"Expected 192 matched cases, got {len(data)}")
    figure_energy(data); figure_phase(data); figure_trajectory(data); figure_escape(data)
    print(json.dumps({"figure_count":4,"formats":["PDF","SVG","600-dpi PNG"],"output":str(FIG)},indent=2))

if __name__=="__main__": main()
