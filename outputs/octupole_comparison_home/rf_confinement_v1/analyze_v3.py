"""Summarize V3 checkpoints and render vector/raster figures without extra deps."""
from __future__ import annotations
import csv, hashlib, html, json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.rf_reconstruction import RFReconstruction
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas as pdfcanvas

AMU=1.66053906660e-27; QE=1.602176634e-19; AP=8e-3
COLORS={"rotating":"#0072B2","standing":"#D55E00","free":"#555555","green":"#009E73","red":"#D55E00"}

class Figure:
    def __init__(self,name,w=1200,h=800):
        self.name=name;self.w=w;self.h=h;self.scale=.72
        self.img=Image.new("RGB",(w,h),"white");self.d=ImageDraw.Draw(self.img)
        fontpath=Path("C:/Windows/Fonts/arial.ttf")
        self.font=ImageFont.truetype(str(fontpath),18) if fontpath.exists() else ImageFont.load_default()
        self.fontsmall=ImageFont.truetype(str(fontpath),15) if fontpath.exists() else self.font
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
        self.pdf=pdfcanvas.Canvas(str(HERE/(name+".pdf")),pagesize=(w*self.scale,h*self.scale))
    def line(self,x1,y1,x2,y2,color="#111",width=2,dash=None):
        self.d.line((x1,y1,x2,y2),fill=color,width=max(1,int(width)))
        ds=f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{ds}/>')
        self.pdf.setStrokeColor(color);self.pdf.setLineWidth(width*self.scale)
        if dash:self.pdf.setDash(*[float(v)*self.scale for v in dash.split(",")])
        else:self.pdf.setDash()
        self.pdf.line(x1*self.scale,(self.h-y1)*self.scale,x2*self.scale,(self.h-y2)*self.scale)
    def circle(self,x,y,r,color,outline=None,width=1):
        self.d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline=outline or color,width=max(1,int(width)))
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{color}" stroke="{outline or color}" stroke-width="{width}"/>')
        self.pdf.setFillColor(color);self.pdf.setStrokeColor(outline or color);self.pdf.setLineWidth(width*self.scale)
        self.pdf.circle(x*self.scale,(self.h-y)*self.scale,r*self.scale,stroke=1,fill=1)
    def rect(self,x,y,w,h,fill,outline=None,width=1):
        self.d.rectangle((x,y,x+w,y+h),fill=fill,outline=outline or fill,width=max(1,int(width)))
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{outline or fill}" stroke-width="{width}"/>')
        self.pdf.setFillColor(fill);self.pdf.setStrokeColor(outline or fill);self.pdf.rect(x*self.scale,(self.h-y-h)*self.scale,w*self.scale,h*self.scale,stroke=1,fill=1)
    def text(self,x,y,s,size=18,color="#111",anchor="start",rotate=0,bold=False):
        s=str(s);weight="bold" if bold else "normal";safe=html.escape(s);pdf_x=x
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" transform="rotate({rotate} {x} {y})">{safe}</text>')
        font=self.font if size>=17 else self.fontsmall
        if anchor=="middle": x-=self.d.textbbox((0,0),s,font=font)[2]/2
        elif anchor=="end": x-=self.d.textbbox((0,0),s,font=font)[2]
        if rotate:
            bb=self.d.textbbox((0,0),s,font=font);im=Image.new("RGBA",(max(2,bb[2]-bb[0]+4),max(2,bb[3]-bb[1]+4)),(255,255,255,0));ImageDraw.Draw(im).text((2,2),s,font=font,fill=color)
            im=im.rotate(rotate,expand=True);self.img.paste(im,(int(x),int(y-im.height/2)),im)
        else:self.d.text((x,y-size),s,font=font,fill=color)
        self.pdf.saveState();self.pdf.setFillColor(color);self.pdf.setFont("Helvetica-Bold" if bold else "Helvetica",size*self.scale)
        if rotate:self.pdf.translate(pdf_x*self.scale,(self.h-y)*self.scale);self.pdf.rotate(-rotate);self.pdf.drawString(0,0,s)
        elif anchor=="middle":self.pdf.drawCentredString(pdf_x*self.scale,(self.h-y)*self.scale,s)
        elif anchor=="end":self.pdf.drawRightString(pdf_x*self.scale,(self.h-y)*self.scale,s)
        else:self.pdf.drawString(pdf_x*self.scale,(self.h-y)*self.scale,s)
        self.pdf.restoreState()
    def polyline(self,pts,color,width=2):
        if len(pts)<2:return
        self.d.line(pts,fill=color,width=max(1,int(width)),joint="curve")
        self.svg.append('<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)+f'" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
        p=self.pdf.beginPath();p.moveTo(pts[0][0]*self.scale,(self.h-pts[0][1])*self.scale)
        for x,y in pts[1:]:p.lineTo(x*self.scale,(self.h-y)*self.scale)
        self.pdf.setStrokeColor(color);self.pdf.setLineWidth(width*self.scale);self.pdf.drawPath(p)
    def save(self):
        self.svg.append('</svg>');(HERE/(self.name+".svg")).write_text('\n'.join(self.svg),encoding="utf-8")
        self.img.save(HERE/(self.name+".png"),dpi=(180,180));self.pdf.showPage();self.pdf.save()


def read_results():
    path=HERE/"trajectory_checkpoint.jsonl"; latest={}
    if not path.exists():return []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r=json.loads(line)
            if r.get("status") not in ("retained","escaped"):continue
            k=(r["job_id"],r["job"]["horizon_RF"])
            latest[k]=r
        except Exception:pass
    return list(latest.values())


def read_failures():
    path=HERE/"trajectory_checkpoint.jsonl";bad=[]
    if not path.exists():return bad
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r=json.loads(line)
            if r.get("status") not in ("retained","escaped"):
                bad.append({"job_id":r.get("job_id"),"status":r.get("status"),"stage":r.get("job",{}).get("stage"),
                    "error":r.get("error"),"job_json":json.dumps(r.get("job",{}),sort_keys=True)})
        except Exception as exc:
            bad.append({"job_id":"journal_parse_error","status":"invalid_json","stage":"input_output_audit","error":repr(exc),"job_json":""})
    return bad


def flatten(r):
    j=r["job"]; horizon=j["horizon_RF"]/j["frequency_Hz"]
    tball=r["ballistic_escape_time_s"]
    return {"job_id":r["job_id"],"model":j["model"],"field":j["field"],"normalization":j.get("normalization","equal_peak_voltage"),
        "voltage_V":j["voltage_V"],"frequency_Hz":j["frequency_Hz"],"mz_Th":j["mz_Th"],"r0_mm":j["r0_mm"],
        "energy_meV":j["energy_meV"],"azimuth_rad":j["azimuth_rad"],"phase_rad":j["phase_rad"],"direction":j["direction"],
        "velocity_kind":j["velocity_kind"],"horizon_RF":j["horizon_RF"],"status":r["status"],
        "ballistic_escape_periods":tball*j["frequency_Hz"],"ballistic_escape_horizon_class":("escaped" if tball<=horizon else "retained"),
        "RF_escape_periods":r.get("escape_periods"),"max_radius_mm":r["max_radius_m"]*1e3,"rms_radius_mm":r["rms_radius_m"]*1e3,
        "radial_min_mm":r["radial_min_m"]*1e3,"nfev":r["nfev"],"wall_s":r["wall_s"],"sample_csv":str((HERE/"trajectories"/(hashlib.sha256(r["job_id"].encode()).hexdigest()[:16]+".csv")).relative_to(ROOT))}


def write_data(rows):
    flat=[flatten(r) for r in rows]
    if flat:
        with (HERE/"trajectory_results.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    paired={}
    for r in rows:
        j=r["job"]
        if j["field"]!="h35" or j["normalization"]!="equal_peak_voltage":continue
        k=tuple(j[x] for x in ("voltage_V","frequency_Hz","mz_Th","r0_mm","energy_meV","azimuth_rad","phase_rad","direction","velocity_kind","horizon_RF"))
        paired.setdefault(k,{})[j["model"]]=r
    pair_rows=[]
    for key,p in paired.items():
        if set(p)!={"rotating","standing"}:continue
        a,b=p["rotating"],p["standing"]
        pair_rows.append({"voltage_V":key[0],"frequency_Hz":key[1],"mz_Th":key[2],"r0_mm":key[3],"energy_meV":key[4],"azimuth_rad":key[5],"phase_rad":key[6],"direction":key[7],"velocity_kind":key[8],"horizon_RF":key[9],
            "rotating_status":a["status"],"standing_status":b["status"],"delta_max_radius_mm":(a["max_radius_m"]-b["max_radius_m"])*1e3,
            "delta_rms_radius_mm":(a["rms_radius_m"]-b["rms_radius_m"])*1e3,
            "rotating_escape_periods":a.get("escape_periods"),"standing_escape_periods":b.get("escape_periods")})
    if pair_rows:
        with (HERE/"matched_pair_differences.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(pair_rows[0]));w.writeheader();w.writerows(pair_rows)
    equal_rot={};matched_stand={}
    for r in rows:
        j=r["job"]
        if j.get("field")!="h35" or j.get("frequency_Hz")!=100000. or j.get("mz_Th")!=100000. or j.get("horizon_RF")!=1000 or j.get("velocity_kind")!="tangent":continue
        k=tuple(j[z] for z in ("r0_mm","energy_meV","azimuth_rad","phase_rad","direction"))
        if j["model"]=="rotating" and j["voltage_V"]==250.:equal_rot[k]=r
        if j["model"]=="standing" and j.get("normalization")=="matched_pseudopotential_at_0p5mm":matched_stand[k]=r
    matched_rows=[]
    for k,b in matched_stand.items():
        a=equal_rot.get(k)
        if a is None:continue
        matched_rows.append({"r0_mm":k[0],"energy_meV":k[1],"azimuth_rad":k[2],"phase_rad":k[3],"direction":k[4],
            "rotating_voltage_V":250.,"standing_voltage_V":b["job"]["voltage_V"],"rotating_status":a["status"],"standing_status":b["status"],
            "delta_max_radius_mm":(a["max_radius_m"]-b["max_radius_m"])*1e3,"delta_rms_radius_mm":(a["rms_radius_m"]-b["rms_radius_m"])*1e3,
            "rotating_job_id":a["job_id"],"standing_job_id":b["job_id"]})
    if matched_rows:
        with (HERE/"matched_pseudopotential_differences.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(matched_rows[0]));w.writeheader();w.writerows(matched_rows)
    return flat,pair_rows,matched_rows


def write_science_tables(rows):
    """Compact, inspectable tables for the verified boundary and long runs."""
    frontier=[r for r in rows if r["job"].get("stage")=="near_aperture_boundary"
        and r["job"].get("field")=="h35" and r["job"].get("horizon_RF")==1000]
    by={}
    for r in frontier:
        j=r["job"];key=tuple(j[k] for k in ("r0_mm","energy_meV","azimuth_rad","phase_rad","direction"))
        by.setdefault(key,{})[j["model"]]=r
    paired=[]
    for key,d in sorted(by.items()):
        if set(d)!={"rotating","standing"}:continue
        a,b=d["rotating"],d["standing"]
        paired.append({"r0_mm":key[0],"energy_eV":key[1]/1000,"azimuth_rad":key[2],"rf_phase_rad":key[3],"tangential_direction_sign":key[4],
            "rf_off_escape_RF":a["ballistic_escape_periods"],"horizon_RF":1000,
            "rotating_status":a["status"],"rotating_escape_RF":a.get("escape_periods"),"rotating_max_radius_mm":a["max_radius_m"]*1e3,"rotating_rms_radius_mm":a["rms_radius_m"]*1e3,
            "standing_status":b["status"],"standing_escape_RF":b.get("escape_periods"),"standing_max_radius_mm":b["max_radius_m"]*1e3,"standing_rms_radius_mm":b["rms_radius_m"]*1e3})
    if paired:
        with (HERE/"frontier_matched_pairs.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    longrows=[]
    for r in rows:
        j=r["job"]
        if j.get("stage") not in ("long_horizon","long_horizon_high_energy") or j.get("horizon_RF")!=10000:continue
        longrows.append({"model":j["model"],"field":j["field"],"normalization":j.get("normalization"),"voltage_V":j["voltage_V"],"frequency_Hz":j["frequency_Hz"],
            "mz_Th":j["mz_Th"],"r0_mm":j["r0_mm"],"energy_meV":j["energy_meV"],"azimuth_rad":j["azimuth_rad"],"phase_rad":j["phase_rad"],"direction":j["direction"],
            "rf_off_escape_RF":r["ballistic_escape_periods"],"rf_status":r["status"],"rf_escape_RF":r.get("escape_periods"),"simulated_RF":r["final_time_s"]*j["frequency_Hz"],
            "max_radius_mm":r["max_radius_m"]*1e3,"rms_radius_mm":r["rms_radius_m"]*1e3,**slow_motion_metrics(r)})
    if longrows:
        with (HERE/"long_horizon_summary.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(longrows[0]));w.writeheader();w.writerows(longrows)
    return paired,longrows


def write_selected_trajectory_samples(rows):
    """Export compact, reproducible traces; full per-job traces stay local."""
    chosen=[]
    selectors=(
        ("10meV_1000RF","baseline_pilot","rotating",2.,10.,0.,0.,1.,1000,0.5),
        ("10meV_1000RF","baseline_pilot","standing",2.,10.,0.,0.,1.,1000,0.5),
        ("1eV_10000RF","long_horizon_high_energy","rotating",1.,1000.,math.pi/4,0.,1.,10000,5.),
        ("1eV_early_escape","near_aperture_boundary","standing",1.,1000.,math.pi/4,0.,1.,1000,0.125),
    )
    for label,stage,model,r0,e,az,phase,direction,horizon,stride in selectors:
        r=next((x for x in rows if x["job"].get("stage")==stage and x["job"].get("model")==model
            and x["job"].get("r0_mm")==r0 and x["job"].get("energy_meV")==e
            and x["job"].get("azimuth_rad")==az and x["job"].get("phase_rad")==phase
            and x["job"].get("direction")==direction and x["job"].get("horizon_RF")==horizon),None)
        if r is None:continue
        j=r["job"]; limit=1000. if label=="10meV_1000RF" else (r.get("escape_periods") or 1000.)
        samples=load_trajectory(r);nearest={}
        for z in samples:
            t=float(z["rf_period"])
            if t>limit+1e-10:continue
            bucket=round(t/stride)
            if abs(t-bucket*stride)<1e-7:nearest[bucket]=z
        for z in nearest.values():
            chosen.append({"sample_kind":"RF_on","trace":label,"model":model,"mz_Th":j["mz_Th"],"voltage_V":j["voltage_V"],"frequency_Hz":j["frequency_Hz"],
                "r0_mm":r0,"energy_meV":e,"azimuth_rad":az,"phase_rad":phase,"direction":direction,"rf_period":float(z["rf_period"]),
                "time_s":float(z["time_s"]),"x_mm":float(z["x_mm"]),"y_mm":float(z["y_mm"]),"r_mm":float(z["r_mm"]),"vx_m_s":float(z["vx_m_s"]),"vy_m_s":float(z["vy_m_s"])})
    # Analytical RF-off path matched to the 1-eV high-energy launch.
    from run_v3 import ballistic_escape_time,make_initial
    initial=make_initial(100000.,1.,1000.,math.pi/4,0.,1.,"tangent")
    tc=ballistic_escape_time(initial);f_rf=100000.
    for t in np.linspace(0.,tc,361):
        x=initial[0]+initial[2]*t;y=initial[1]+initial[3]*t
        chosen.append({"sample_kind":"RF_off_analytic","trace":"1eV_ballistic","model":"free","mz_Th":100000.,"voltage_V":0.,"frequency_Hz":f_rf,
            "r0_mm":1.,"energy_meV":1000.,"azimuth_rad":math.pi/4,"phase_rad":0.,"direction":1.,"rf_period":float(t*f_rf),"time_s":float(t),
            "x_mm":float(x*1e3),"y_mm":float(y*1e3),"r_mm":float(math.hypot(x,y)*1e3),"vx_m_s":float(initial[2]),"vy_m_s":float(initial[3])})
    if chosen:
        with (HERE/"selected_trajectory_samples.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(chosen[0]));w.writeheader();w.writerows(chosen)
    return len(chosen)


def pseudo(q,model,r,theta,U,m,omega):
    xy=np.array([[r*math.cos(theta),r*math.sin(theta)]])
    ec,es=q.field_quadratures(xy)
    strength=np.sum(ec*ec+es*es,axis=1)[0] if model=="rotating" else np.sum(es*es,axis=1)[0]
    return QE**2*U**2*strength/(4*m*omega**2)


def turning_radius(q,model,r0,energy_mev,theta=0.,U=250.,freq=100000.):
    m=100000*AMU;om=2*math.pi*freq
    v=math.sqrt(2*energy_mev*1e-3*QE/m);L=m*r0*v
    etot=.5*m*v*v+pseudo(q,model,r0,theta,U,m,om)
    def eff(r):return pseudo(q,model,r,theta,U,m,om)+L*L/(2*m*r*r)
    grid=np.linspace(r0+1e-7,7.9e-3,300)
    vals=[eff(x)-etot for x in grid]
    for a,b,fa,fb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
        if fa*fb<0:return brentq(lambda x:eff(x)-etot,a,b)
    return math.nan


def fig_pseudopotential():
    q=RFReconstruction.load(ROOT/"outputs/abc_validation/direct_fem_validation/v2/h35_P4/diagnostic_N48.json").unit_quadratures
    f=Figure("figure_1_effective_potential")
    L,R,T,B=110,50,105,100; pw=f.w-L-R;ph=f.h-T-B
    xmax=5.5;ymax=160
    X=lambda r:L+r/xmax*pw
    Y=lambda e:T+ph-e/ymax*ph
    f.text(600,42,"Leading pseudopotential and launch energy — 100 kTh, 250 V, 100 kHz",24,"#111","middle",bold=True)
    for y in range(0,161,20):
        yy=Y(y);f.line(L,yy,f.w-R,yy,"#dddddd",1);f.text(L-12,yy+6,str(y),15,"#333","end")
    for x in range(0,6):
        xx=X(x);f.line(xx,T,xx,T+ph,"#eeeeee",1);f.text(xx,T+ph+28,str(x),15,"#333","middle")
    f.line(L,T,L,T+ph,"#111",2);f.line(L,T+ph,f.w-R,T+ph,"#111",2)
    f.text(35,T+ph/2,"Energy (meV)",18,"#111","middle",-90);f.text(L+pw/2,f.h-28,"Radius (mm)",18,"#111","middle")
    styles=(("rotating",COLORS["rotating"]),("standing",COLORS["standing"]))
    rgrid=np.linspace(.05e-3,xmax*1e-3,600)
    for model,color in styles:
        pts=[]
        for r in rgrid:
            e=pseudo(q,model,r,0.,250.,100000*AMU,2*math.pi*100000)/QE*1e3
            if e<=ymax:pts.append((X(r*1e3),Y(e)))
        f.polyline(pts,color,4)
        f.line(820,75+(0 if model=="rotating" else 30),875,75+(0 if model=="rotating" else 30),color,4)
        f.text(890,82+(0 if model=="rotating" else 30),model+" (U0=250 V)",16,color)
    r0=2e-3; E=10.
    for model,color in styles:
        e0=pseudo(q,model,r0,0.,250.,100000*AMU,2*math.pi*100000)/QE*1e3
        turn=turning_radius(q,model,r0,E)
        e0tot=E+e0
        # Tangential launch: centrifugal term raises the radial effective barrier.
        f.line(X(0),Y(e0tot),X(xmax),Y(e0tot),color,2,"9,6")
        f.circle(X(r0*1e3),Y(e0tot),6,color)
        if math.isfinite(turn):
            f.circle(X(turn*1e3),Y(e0tot),7,color,outline="#111",width=2)
            f.text(X(turn*1e3)+12,Y(e0tot)-10,f"{turn*1e3:.2f} mm",14,color)
    f.text(145,730,"10 meV tangential launch at r0=2 mm: dashed total-energy levels; circles mark r0 and predicted outer turn.",15,"#333")
    f.text(145,750,"Diagnostic high-frequency approximation; full-RF trajectories determine retention.",15,"#555")
    f.save()


def load_trajectory(row):
    # Main analysis passes raw checkpoint records; metric helpers and CSV
    # consumers may instead pass the flattened export.
    if "sample_csv" not in row:
        row=flatten(row)
    path=ROOT/row["sample_csv"]
    arr=list(csv.DictReader(path.open(encoding="utf-8")))
    return arr


def slow_motion_metrics(result):
    arr=load_trajectory(flatten(result))
    t=np.asarray([float(x["rf_period"]) for x in arr]);x=np.asarray([float(z["x_mm"]) for z in arr]);y=np.asarray([float(z["y_mm"]) for z in arr]);
    t,idx=np.unique(t,return_index=True);x=x[idx];y=y[idx]
    job=result["job"];phase=job["phase_rad"]
    centers=[];micro=[]
    periods=np.floor(t+1e-7).astype(int)
    for k in np.unique(periods):
        ix=np.where(periods==k)[0]
        if len(ix)<6:continue
        frac=t[ix]-k;ph=2*np.pi*t[ix]+phase
        A=np.column_stack((np.ones(len(ix)),frac-.5,np.cos(ph),np.sin(ph)))
        bx=np.linalg.lstsq(A,x[ix],rcond=None)[0];by=np.linalg.lstsq(A,y[ix],rcond=None)[0]
        centers.append((k,bx[0],by[0],math.hypot(bx[0],by[0])))
        micro.append(math.sqrt(bx[2]**2+bx[3]**2+by[2]**2+by[3]**2))
    if not centers:return {"slow_radial_period_RF":None,"slow_rmin_mm":None,"slow_rmax_mm":None,"micromotion_p95_mm":None,"micromotion_max_mm":None}
    c=np.asarray(centers);from scipy.signal import find_peaks
    peaks,_=find_peaks(c[:,3],prominence=.1,distance=30)
    troughs,_=find_peaks(-c[:,3],prominence=.1,distance=30)
    extrema=np.sort(np.r_[c[peaks,0],c[troughs,0]])
    # Same-sense radial maxima are separated by one full slow radial cycle.
    pperiod=float(np.mean(np.diff(c[peaks,0]))) if len(peaks)>1 else None
    angle=np.unwrap(np.arctan2(y,x))
    return {"slow_radial_period_RF":pperiod,"slow_rmin_mm":float(np.min(c[:,3])),"slow_rmax_mm":float(np.max(c[:,3])),
        "slow_maxima_n":int(len(peaks)),"micromotion_median_mm":float(np.median(micro)),
        "micromotion_p95_mm":float(np.quantile(micro,.95)),"micromotion_max_mm":float(np.max(micro)),
        "net_azimuth_sweep_rad":float(angle[-1]-angle[0])}


def fig_trajectories(rows):
    target=[]
    for r in rows:
        j=r["job"]
        if j["voltage_V"]==250. and j["frequency_Hz"]==100000. and j["r0_mm"]==2. and j["energy_meV"]==10. and j["azimuth_rad"]==0. and j["phase_rad"]==0. and j["direction"]==1. and j["velocity_kind"]=="tangent" and j["horizon_RF"]==1000:
            target.append(r)
    by={x["job"]["model"]:x for x in target}
    if not {"rotating","standing"}<=by.keys():return
    f=Figure("figure_2_matched_trajectories")
    L,R,T,B=110,50,105,100;pw=f.w-L-R;ph=f.h-T-B
    X=lambda t:L+t/1000*pw;Y=lambda r:T+ph-r/8.4*ph
    f.text(600,42,"Matched full-RF trajectories and RF-off control",24,"#111","middle",bold=True)
    for y in range(0,9,2):
        yy=Y(y);f.line(L,yy,f.w-R,yy,"#dddddd",1);f.text(L-12,yy+6,str(y),15,"#333","end")
    for x in range(0,1001,200):
        xx=X(x);f.line(xx,T,xx,T+ph,"#eeeeee",1);f.text(xx,T+ph+28,str(x),15,"#333","middle")
    f.line(L,T,L,T+ph,"#111",2);f.line(L,T+ph,f.w-R,T+ph,"#111",2)
    f.text(35,T+ph/2,"Radius (mm)",18,"#111","middle",-90);f.text(L+pw/2,f.h-28,"Time (RF periods)",18,"#111","middle")
    f.line(L,Y(8),f.w-R,Y(8),"#CC3311",3,"10,7");f.text(f.w-R-8,Y(8)-10,"8 mm aperture",15,"#CC3311","end")
    for model,color in (("rotating",COLORS["rotating"]),("standing",COLORS["standing"])):
        arr=load_trajectory(by[model]);pts=[(float(z["rf_period"]),float(z["r_mm"])) for z in arr]
        f.polyline([(X(t),Y(r)) for t,r in pts[::2]],color,2)
    initial=by["rotating"]["initial_state"];x0,y0,vx,vy=initial
    tesc=by["rotating"]["ballistic_escape_time_s"]*100000
    free_pts=[]
    for tt in np.linspace(0.,tesc,400):
        tx=tt/100000
        free_pts.append((X(tt),Y(math.hypot(x0+vx*tx,y0+vy*tx)*1e3)))
    f.polyline(free_pts,COLORS["free"],2)
    f.circle(X(tesc),Y(8),6,COLORS["free"])
    for i,(label,color) in enumerate((("Rotating, 250 V",COLORS["rotating"]),("Standing, 250 V",COLORS["standing"]),(f"RF-off; exits at {tesc:.1f} RF",COLORS["free"]))):
        yy=58+i*26;f.line(650,yy,700,yy,color,3,"5,5" if i==2 else None);f.text(710,yy+7,label,16,color)
    # Launch metadata is recorded in trajectory_results.csv and the report;
    # keeping it off the plot avoids colliding with the aperture marker.
    f.save()


def fig_map(flat):
    rows=[r for r in flat if r["field"]=="h35" and r["normalization"]=="equal_peak_voltage" and r["voltage_V"]==250. and r["frequency_Hz"]==100000. and r["velocity_kind"]=="tangent" and r["horizon_RF"]==1000]
    f=Figure("figure_3_verified_confinement_map")
    f.text(600,42,"Calculated finite-time confinement points — equal peak voltage",24,"#111","middle",bold=True)
    # Model panels, radius vertical and energy horizontal; fill is retained fraction.
    panels=(("rotating",90),("standing",650)); energies=sorted(set(r["energy_meV"] for r in rows)); radii=sorted(set(r["r0_mm"] for r in rows))
    elo,ehi=math.log10(min(energies)),math.log10(max(energies))
    for model,x0 in panels:
        left=x0+70;top=155;pw=420;ph=450
        f.text(left+pw/2,112,model,21,COLORS[model],"middle",bold=True)
        f.line(left,top,left,top+ph,"#111",2);f.line(left,top+ph,left+pw,top+ph,"#111",2)
        tick_energies=[e for e in (5.,10.,50.,200.,1000.,3000.) if min(energies)<=e<=max(energies)]
        for e in tick_energies:
            xx=left+(math.log10(e)-elo)/(ehi-elo)*pw
            f.line(xx,top,xx,top+ph,"#eeeeee",1);f.text(xx,top+ph+25,f"{e:g}",14,"#333","middle")
        for rad in radii:
            yy=top+ph-(rad-min(radii))/(max(radii)-min(radii))*ph
            f.line(left,yy,left+pw,yy,"#eeeeee",1);f.text(left-15,yy+5,f"{rad:g}",14,"#333","end")
        agg={}
        for r in rows:
            if r["model"]!=model:continue
            k=(r["energy_meV"],r["r0_mm"]);agg.setdefault(k,[]).append(r)
        for (e,rad),group in agg.items():
            p=sum(x["status"]=="retained" for x in group)/len(group)
            xx=left+(math.log10(e)-elo)/(ehi-elo)*pw
            yy=top+ph-(rad-min(radii))/(max(radii)-min(radii))*ph
            color="#009E73" if p==1 else "#E69F00" if p>0 else "#D55E00"
            f.circle(xx,yy,11,color,outline="#111" if group[0]["ballistic_escape_horizon_class"]=="escaped" else color,width=1);f.text(xx,yy+5,f"{p:.1f}",12,"white","middle",bold=True)
        f.text(left+pw/2,top+ph+62,"Initial transverse energy (meV)",16,"#111","middle")
        f.text(left-52,top+ph/2,"r0 (mm)",16,"#111","middle",-90)
    f.text(600,755,"Directly evaluated cells only; labels show P_ret. Green=1, amber=partial, red=0; no interpolation.",15,"#444","middle")
    f.text(600,780,"250 V, 100 kHz, 100 kTh, 1000 RF periods; deterministic phase/azimuth/direction launches.",15,"#444","middle")
    f.save()


def fig_differences(pairs):
    rows=[r for r in pairs if r["voltage_V"]==250. and r["frequency_Hz"]==100000. and r["horizon_RF"]==1000 and r["velocity_kind"]=="tangent"]
    f=Figure("figure_4_rotating_minus_standing")
    L,R,T,B=110,50,100,100;pw=f.w-L-R;ph=f.h-T-B
    f.text(600,42,"Paired dynamics difference: rotating − standing",24,"#111","middle",bold=True)
    energies=[r["energy_meV"] for r in rows] or [1];maxe=max(energies);minlog=math.log10(min(energies));maxlog=math.log10(maxe);minr=min([r["r0_mm"] for r in rows] or [0]);maxr=max([r["r0_mm"] for r in rows] or [1])
    dmax=max([abs(r["delta_max_radius_mm"]) for r in rows] or [1]);dmax=max(.1,dmax)
    X=lambda e:L+(math.log10(e)-minlog)/(maxlog-minlog)*pw;Y=lambda d:T+ph/2-d/(2*dmax)*ph
    for dy in np.linspace(-dmax,dmax,9):
        yy=Y(dy);f.line(L,yy,f.w-R,yy,"#dddddd" if abs(dy)>1e-12 else "#555",1 if abs(dy)>1e-12 else 2)
        f.text(L-10,yy+5,f"{dy:.2f}",14,"#444","end")
    for e in sorted(set(energies)):
        xx=X(e);f.line(xx,T,xx,T+ph,"#eeeeee",1);f.text(xx,T+ph+25,f"{e:g}",14,"#333","middle")
    colors={1.:"#0072B2",2.:"#E69F00",3.:"#009E73"}
    for r in rows:
        x=X(r["energy_meV"]);y=Y(r["delta_max_radius_mm"]);f.circle(x,y,5,colors.get(r["r0_mm"],"#444"))
    f.line(L,T,L,T+ph,"#111",2);f.line(L,T+ph,f.w-R,T+ph,"#111",2)
    f.text(35,T+ph/2,"Δ maximum radius (mm)",18,"#111","middle",-90);f.text(L+pw/2,f.h-28,"Initial transverse kinetic energy (meV)",18,"#111","middle")
    for i,(rad,c) in enumerate(colors.items()):f.circle(880,140+i*30,5,c);f.text(895,146+i*30,f"r0={rad:g} mm",14,c)
    f.text(600,755,"Every dot is an exactly matched initial condition. No smoothing; positive means rotating has larger excursion.",15,"#444","middle")
    f.save()


def fig_matched(matched):
    if not matched:return
    f=Figure("figure_5_matched_pseudopotential")
    L,R,T,B=120,60,105,100;pw=f.w-L-R;ph=f.h-T-B
    values=[r["delta_max_radius_mm"] for r in matched];extent=max(.1,max(abs(v) for v in values))
    X=lambda r:L+(r-.5)/(3.-.5)*pw;Y=lambda d:T+ph/2-d/(2*extent)*ph
    f.text(600,42,"Paired dynamics at matched leading pseudopotential strength",24,"#111","middle",bold=True)
    for d in np.linspace(-extent,extent,9):
        yy=Y(d);f.line(L,yy,f.w-R,yy,"#dddddd" if abs(d)>1e-12 else "#555",1 if abs(d)>1e-12 else 2);f.text(L-10,yy+5,f"{d:.2f}",14,"#444","end")
    for r in (.5,1.,2.,3.):
        xx=X(r);f.line(xx,T,xx,T+ph,"#eeeeee",1);f.text(xx,T+ph+25,f"{r:g}",14,"#333","middle")
    for row in matched:
        x=X(row["r0_mm"])+(row["direction"]*4);y=Y(row["delta_max_radius_mm"])
        f.circle(x,y,6,COLORS["green"] if row["rotating_status"]==row["standing_status"]=="retained" else COLORS["red"])
    f.line(L,T,L,T+ph,"#111",2);f.line(L,T+ph,f.w-R,T+ph,"#111",2)
    f.text(35,T+ph/2,"Δ maximum radius (mm)",18,"#111","middle",-90);f.text(L+pw/2,f.h-28,"Initial radius (mm)",18,"#111","middle")
    f.text(600,755,"Rotating 250 V versus standing 353.55 V; same initial E=10 meV, phase, azimuth and velocity sign.",15,"#444","middle")
    f.text(600,780,"Reference match at r=0.5 mm; points are the 24 directly paired deterministic launches.",15,"#444","middle")
    f.save()


def horizon_envelope(rows):
    data=[]
    for r in rows:
        j=r["job"]
        if j.get("horizon_RF")!=10000 or j.get("r0_mm")!=2. or j.get("azimuth_rad")!=0. or j.get("phase_rad")!=0. or j.get("direction")!=1.:continue
        for seg in r.get("segments",[]):
            data.append({"model":j["model"],"rf_period":seg["final_time_s"]*j["frequency_Hz"],"status":seg["status"],
                "interval_min_radius_mm":seg["radial_min_m"]*1e3,"interval_max_radius_mm":seg["max_radius_m"]*1e3,
                "nfev":seg["nfev"],"job_id":r["job_id"]})
    if data:
        with (HERE/"long_horizon_envelope.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        fig=Figure("figure_6_long_horizon_envelope")
        L,R,T,B=110,50,105,100;pw=fig.w-L-R;ph=fig.h-T-B
        X=lambda t:L+t/10000*pw;Y=lambda r:T+ph-r/8.4*ph
        fig.text(600,42,"300-period radial envelope through 10000 RF periods",24,"#111","middle",bold=True)
        for y in (0,2,4,6,8):
            yy=Y(y);fig.line(L,yy,fig.w-R,yy,"#dddddd",1);fig.text(L-10,yy+5,str(y),14,"#444","end")
        for t in range(0,10001,2000):
            xx=X(t);fig.line(xx,T,xx,T+ph,"#eeeeee",1);fig.text(xx,T+ph+25,str(t),14,"#333","middle")
        fig.line(L,T,L,T+ph,"#111",2);fig.line(L,T+ph,fig.w-R,T+ph,"#111",2);fig.line(L,Y(8),fig.w-R,Y(8),"#CC3311",2,"9,6")
        fig.text(35,T+ph/2,"Radius (mm)",18,"#111","middle",-90);fig.text(L+pw/2,fig.h-28,"Time (RF periods)",18,"#111","middle")
        for model,color in (("rotating",COLORS["rotating"]),("standing",COLORS["standing"])):
            group=sorted((x for x in data if x["model"]==model),key=lambda x:x["rf_period"])
            for key,dash,label in (("interval_max_radius_mm",None,"max"),("interval_min_radius_mm","8,5","min")):
                fig.polyline([(X(x["rf_period"]),Y(x[key])) for x in group],color,3 if key.startswith("interval_max") else 2)
                if dash:
                    for a,b in zip(group[:-1],group[1:]):fig.line(X(a["rf_period"]),Y(a[key]),X(b["rf_period"]),Y(b[key]),color,2,dash)
            yy=65+(0 if model=="rotating" else 26);fig.line(690,yy,740,yy,color,3);fig.text(750,yy+6,model+" max",15,color)
            yy+=14;fig.line(690,yy,740,yy,color,2,"8,5");fig.text(750,yy+6,model+" min",15,color)
        fig.text(600,755,"10 meV, r0=2 mm, azimuth=0, RF phase=0, positive tangential launch; segment extrema only.",15,"#444","middle")
        fig.text(600,780,"Two matched runs; each point summarizes one completed 300-period integration segment.",15,"#444","middle")
        fig.save()
    return data


def fig_high_energy(rows):
    """Paired 1 eV trajectory: early escape comparison and long RF envelope."""
    target={}
    for r in rows:
        j=r["job"]
        if (j.get("field")=="h35" and j.get("r0_mm")==1. and j.get("energy_meV")==1000.
            and j.get("azimuth_rad")==math.pi/4 and j.get("direction")==1.
            and j.get("voltage_V")==250. and j.get("frequency_Hz")==100000.
            and j.get("stage") in ("near_aperture_boundary","long_horizon_high_energy")):
            target[(j["model"],j["phase_rad"])]=r
    rot=target.get(("rotating",0.));stand=target.get(("standing",0.))
    if not rot or not stand:return
    f=Figure("figure_7_high_energy_confinement",1400,850)
    f.text(700,42,"Matched 1 eV launch: rotating RF remains bounded while standing RF and RF-off escape",23,"#111","middle",bold=True)
    panels=((100,120,530,550,60.,"First 60 RF periods"),(770,120,530,550,10000.,"Rotating field through 10,000 RF periods"))
    data={"rotating":load_trajectory(rot),"standing":load_trajectory(stand)}
    tball=rot["ballistic_escape_periods"]
    x0,y0,vx,vy=rot["initial_state"]
    for left,top,pw,ph,xmax,title in panels:
        f.text(left+pw/2,top-28,title,19,"#222","middle",bold=True)
        X=lambda t:left+t/xmax*pw
        Y=lambda r:top+ph-r/8.5*ph
        for rr in (0,2,4,6,8):
            yy=Y(rr);f.line(left,yy,left+pw,yy,"#dddddd",1);f.text(left-12,yy+5,str(rr),14,"#444","end")
        f.line(left,top,left,top+ph,"#111",2);f.line(left,top+ph,left+pw,top+ph,"#111",2)
        f.line(left,Y(8),left+pw,Y(8),"#CC3311",2,"9,6")
        f.text(left+pw-5,Y(8)-8,"8 mm",14,"#CC3311","end")
        for model,color in (("rotating",COLORS["rotating"]),("standing",COLORS["standing"])):
            arr=data[model]
            pts=[(float(z["rf_period"]),float(z["r_mm"])) for z in arr if float(z["rf_period"])<=xmax]
            if xmax>100 and model=="rotating":
                centers={}
                for z in arr:
                    t=float(z["rf_period"])
                    if t>xmax:continue
                    k=math.floor(t);centers.setdefault(k,[]).append((float(z["x_mm"]),float(z["y_mm"])))
                slow=[]
                for k,xy in sorted(centers.items()):
                    cx=float(np.mean([p[0] for p in xy]));cy=float(np.mean([p[1] for p in xy]))
                    slow.append((k+.5,math.hypot(cx,cy)))
                f.polyline([(X(t),Y(r)) for t,r in slow],color,2)
            elif len(pts)>1:
                f.polyline([(X(t),Y(r)) for t,r in pts[::max(1,len(pts)//1800)]],color,2)
        if xmax<=100:
            for tt in np.linspace(0.,tball,150):
                sec=tt/100000.
                free_r=math.hypot(x0+vx*sec,y0+vy*sec)*1e3
                f.circle(X(tt),Y(free_r),1.6,COLORS["free"])
            f.circle(X(tball),Y(8),6,COLORS["free"])
        f.text(left+pw/2,top+ph+30,"Time (RF periods)",15,"#222","middle")
    for i,(label,color) in enumerate((("rotating RF",COLORS["rotating"]),("standing RF",COLORS["standing"]),("RF-off",COLORS["free"]))):
        xx=485+i*170;f.line(xx,755,xx+36,755,color,3);f.text(xx+44,761,label,15,color)
    f.text(700,798,f"100 kTh; r0=1 mm; E⊥=1 eV; azimuth=π/4; phase=0; positive tangential velocity. RF-off exits at {tball:.3f} RF; standing RF exits at {stand['escape_periods']:.3f} RF.",15,"#333","middle")
    f.text(700,825,"Right panel shows the RF-period averaged rotating trajectory through 10,000 RF periods.",15,"#555","middle")
    f.save()


def main():
    rows=read_results();flat,pairs,matched=write_data(rows)
    boundary_rows,long_rows=write_science_tables(rows)
    selected_samples=write_selected_trajectory_samples(rows)
    failures=read_failures()
    if failures:
        with (HERE/"auxiliary_failures.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(failures[0]));w.writeheader();w.writerows(failures)
    else:
        (HERE/"auxiliary_failures.csv").write_text("job_id,status,stage,error,job_json\n",encoding="utf-8")
    envelope=horizon_envelope(rows)
    motion_rows=[]
    for r in rows:
        j=r["job"]
        if j.get("field")=="h35" and j["voltage_V"]==250. and j["frequency_Hz"]==100000. and j["energy_meV"]==10. and j["horizon_RF"]==1000 and j["velocity_kind"]=="tangent":
            motion_rows.append({"job_id":r["job_id"],"model":j["model"],"r0_mm":j["r0_mm"],"azimuth_rad":j["azimuth_rad"],"phase_rad":j["phase_rad"],"direction":j["direction"],"status":r["status"],"ballistic_escape_periods":r["ballistic_escape_periods"],"max_radius_mm":r["max_radius_m"]*1e3,"rms_radius_mm":r["rms_radius_m"]*1e3,**slow_motion_metrics(r)})
    if motion_rows:
        with (HERE/"motion_metrics.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(motion_rows[0]));w.writeheader();w.writerows(motion_rows)
    # Each grid cell retains the actual denominator and direct point set.
    grouped={}
    for r in flat:
        k=tuple(r[x] for x in ("model","normalization","voltage_V","frequency_Hz","mz_Th","r0_mm","energy_meV","velocity_kind","horizon_RF"))
        grouped.setdefault(k,[]).append(r)
    summary=[]
    for k,a in grouped.items():
        summary.append({"model":k[0],"normalization":k[1],"voltage_V":k[2],"frequency_Hz":k[3],"mz_Th":k[4],"r0_mm":k[5],"energy_meV":k[6],"velocity_kind":k[7],"horizon_RF":k[8],
            "n":len(a),"n_retained":sum(x["status"]=="retained" for x in a),"n_escaped":sum(x["status"]=="escaped" for x in a),"n_ballistic_escaped":sum(x["ballistic_escape_horizon_class"]=="escaped" for x in a),
            "mean_max_radius_mm":float(np.mean([x["max_radius_mm"] for x in a])),"mean_rms_radius_mm":float(np.mean([x["rms_radius_mm"] for x in a]))})
    for fname,data in (("confinement_grid_summary.csv",summary),):
        if data:
            with (HERE/fname).open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    j35=[x for x in rows if x["job"]["field"]=="h35"]
    fig_pseudopotential();fig_trajectories(j35);fig_map(flat);fig_differences(pairs);fig_matched(matched);fig_high_energy(rows)
    out={"completed_unique_job_horizons":len(rows),"matched_equal_peak_pairs":len(pairs),"matched_pseudopotential_pairs":len(matched),"verified_frontier_pairs":len(boundary_rows),"selected_10000_RF_runs":len(long_rows),"selected_trajectory_samples":selected_samples,"long_horizon_envelope_segments":len(envelope),"statuses":{s:sum(r["status"]==s for r in rows) for s in ("retained","escaped")},"auxiliary_failures":len(failures),"summaries":summary}
    (HERE/"analysis_summary.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in out.items() if k!="summaries"},indent=2))

if __name__=="__main__":main()
