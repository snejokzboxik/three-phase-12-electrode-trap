"""Publication figure renderer using the repository's Pillow/ReportLab backend."""
from __future__ import annotations
import csv, html, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas as pdfcanvas

from analyze_v4 import OUT, collect, write_csv

ROT="#0072B2"; STAND="#D55E00"; RET="#009E73"; ESC="#CC3311"
DIFF={"both_retained":"#777777","rotating_only_retained":"#009E73",
      "standing_only_retained":"#AA4499","both_escaped":"#D55E00"}
MODEL={"rotating":ROT,"standing":STAND}; STATUS={"retained":RET,"escaped":ESC}


class Figure:
    """Small, dependency-light vector plot canvas with a high-resolution PNG."""
    def __init__(self, name, root, w, h):
        self.name=name; self.root=root; self.w=w; self.h=h; self.pdf_scale=.24
        self.img=Image.new("RGB",(w,h),"white"); self.d=ImageDraw.Draw(self.img)
        fp=Path("C:/Windows/Fonts/arial.ttf")
        self.fontpath=fp if fp.exists() else None
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>']
        self.pdf=pdfcanvas.Canvas(str(root/(name+".pdf")),pagesize=(w*self.pdf_scale,h*self.pdf_scale))
        self.root.mkdir(parents=True,exist_ok=True)
    def line(self,x1,y1,x2,y2,color="#111111",width=2,dash=None):
        self.d.line((x1,y1,x2,y2),fill=color,width=max(1,int(width)))
        ds=f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{ds}/>')
        self.pdf.setStrokeColor(color); self.pdf.setLineWidth(width*self.pdf_scale)
        self.pdf.setDash(*([float(v)*self.pdf_scale for v in dash.split(",")] if dash else []))
        self.pdf.line(x1*self.pdf_scale,(self.h-y1)*self.pdf_scale,x2*self.pdf_scale,(self.h-y2)*self.pdf_scale)
    def circle(self,x,y,r,color,outline=None,width=1):
        outline=outline or color
        self.d.ellipse((x-r,y-r,x+r,y+r),fill=color,outline=outline,width=max(1,int(width)))
        self.svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{color}" stroke="{outline}" stroke-width="{width}"/>')
        self.pdf.setFillColor(color); self.pdf.setStrokeColor(outline); self.pdf.setLineWidth(width*self.pdf_scale)
        self.pdf.circle(x*self.pdf_scale,(self.h-y)*self.pdf_scale,r*self.pdf_scale,stroke=1,fill=1)
    def triangle(self,x,y,r,color,outline=None,orientation="up",fill=True):
        if orientation=="up": pts=[(x,y-r),(x-r,y+r),(x+r,y+r)]
        elif orientation=="right": pts=[(x+r,y),(x-r,y-r),(x-r,y+r)]
        else: pts=[(x-r,y-r),(x+r,y),(x-r,y+r)]
        c=color if fill else "white"; o=outline or color
        self.d.polygon(pts,fill=c); self.d.line(pts+[pts[0]],fill=o,width=2)
        self.svg.append(f'<polygon points="'+' '.join(f'{a:.2f},{b:.2f}' for a,b in pts)+f'" fill="{c}" stroke="{o}" stroke-width="2"/>')
        p=self.pdf.beginPath(); p.moveTo(pts[0][0]*self.pdf_scale,(self.h-pts[0][1])*self.pdf_scale)
        for a,b in pts[1:]:p.lineTo(a*self.pdf_scale,(self.h-b)*self.pdf_scale)
        p.close();self.pdf.setFillColor(c);self.pdf.setStrokeColor(o);self.pdf.setLineWidth(2*self.pdf_scale);self.pdf.drawPath(p,stroke=1,fill=1)
    def square(self,x,y,r,color,outline=None,fill=True):
        c=color if fill else "white"; o=outline or color
        self.d.rectangle((x-r,y-r,x+r,y+r),fill=c,outline=o,width=2)
        self.svg.append(f'<rect x="{x-r}" y="{y-r}" width="{2*r}" height="{2*r}" fill="{c}" stroke="{o}" stroke-width="2"/>')
        self.pdf.setFillColor(c);self.pdf.setStrokeColor(o);self.pdf.setLineWidth(2*self.pdf_scale)
        self.pdf.rect((x-r)*self.pdf_scale,(self.h-y-r)*self.pdf_scale,2*r*self.pdf_scale,2*r*self.pdf_scale,stroke=1,fill=1)
    def text(self,x,y,s,size=30,color="#111111",anchor="start",bold=False,rotate=0):
        s=str(s); safe=html.escape(s); weight="bold" if bold else "normal"
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" transform="rotate({rotate} {x} {y})">{safe}</text>')
        font=ImageFont.truetype(str(self.fontpath),size) if self.fontpath else ImageFont.load_default()
        bb=self.d.textbbox((0,0),s,font=font); xx=x-(bb[2]-bb[0])/2 if anchor=="middle" else x-(bb[2]-bb[0]) if anchor=="end" else x
        if rotate:
            im=Image.new("RGBA",(max(2,bb[2]-bb[0]+8),max(2,bb[3]-bb[1]+8)),(255,255,255,0));ImageDraw.Draw(im).text((4,4),s,font=font,fill=color)
            im=im.rotate(rotate,expand=True);self.img.paste(im,(int(xx),int(y-im.height/2)),im)
        else:self.d.text((xx,y-size),s,font=font,fill=color)
        self.pdf.saveState();self.pdf.setFillColor(color);self.pdf.setFont("Helvetica-Bold" if bold else "Helvetica",size*self.pdf_scale)
        if rotate:self.pdf.translate(x*self.pdf_scale,(self.h-y)*self.pdf_scale);self.pdf.rotate(-rotate);self.pdf.drawString(0,0,s)
        elif anchor=="middle":self.pdf.drawCentredString(x*self.pdf_scale,(self.h-y)*self.pdf_scale,s)
        elif anchor=="end":self.pdf.drawRightString(x*self.pdf_scale,(self.h-y)*self.pdf_scale,s)
        else:self.pdf.drawString(x*self.pdf_scale,(self.h-y)*self.pdf_scale,s)
        self.pdf.restoreState()
    def polyline(self,pts,color,width=2,dash=None):
        if len(pts)<2:return
        self.d.line(pts,fill=color,width=max(1,int(width)),joint="curve")
        ds=f' stroke-dasharray="{dash}"' if dash else ""
        self.svg.append(f'<polyline points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)+f'" fill="none" stroke="{color}" stroke-width="{width}"{ds}/>')
        p=self.pdf.beginPath();p.moveTo(pts[0][0]*self.pdf_scale,(self.h-pts[0][1])*self.pdf_scale)
        for x,y in pts[1:]:p.lineTo(x*self.pdf_scale,(self.h-y)*self.pdf_scale)
        self.pdf.setStrokeColor(color);self.pdf.setLineWidth(width*self.pdf_scale)
        self.pdf.setDash(*([float(v)*self.pdf_scale for v in dash.split(",")] if dash else []));self.pdf.drawPath(p)
    def save(self):
        self.svg.append("</svg>"); (self.root/(self.name+".svg")).write_text("\n".join(self.svg),encoding="utf-8")
        self.img.save(self.root/(self.name+".png"),dpi=(300,300));self.pdf.showPage();self.pdf.save()


class Ax:
    def __init__(self,f,left,top,w,h,xmin,xmax,ymin,ymax):
        self.f=f;self.l=left;self.t=top;self.w=w;self.h=h;self.xmin=xmin;self.xmax=xmax;self.ymin=ymin;self.ymax=ymax
    def X(self,x):return self.l+(x-self.xmin)/(self.xmax-self.xmin)*self.w
    def Y(self,y):return self.t+self.h-(y-self.ymin)/(self.ymax-self.ymin)*self.h
    def grid(self,xticks,yticks,xformat=lambda x:f"{x:g}",yformat=lambda y:f"{y:g}"):
        for x in xticks:
            xx=self.X(x);self.f.line(xx,self.t,xx,self.t+self.h,"#e4e8eb",1);self.f.line(xx,self.t+self.h,xx,self.t+self.h+9,"#222",2);self.f.text(xx,self.t+self.h+39,xformat(x),24,"#444","middle")
        for y in yticks:
            yy=self.Y(y);self.f.line(self.l,yy,self.l+self.w,yy,"#e4e8eb",1);self.f.line(self.l-9,yy,self.l,yy,"#222",2);self.f.text(self.l-16,yy+8,yformat(y),23,"#444","end")
        self.f.line(self.l,self.t,self.l,self.t+self.h,"#222",2);self.f.line(self.l,self.t+self.h,self.l+self.w,self.t+self.h,"#222",2)
    def xlab(self,s):self.f.text(self.l+self.w/2,self.t+self.h+80,s,28,"#222","middle")
    def ylab(self,s):self.f.text(self.l-100,self.t+self.h/2,s,28,"#222","middle",rotate=90)
    def title(self,s):self.f.text(self.l+self.w/2,self.t-28,s,30,"#222","middle",bold=True)
    def dot(self,x,y,color,r=7,marker="o",filled=True):
        xx,yy=self.X(x),self.Y(y)
        if marker=="o":self.f.circle(xx,yy,r,color if filled else "white",color,2)
        elif marker=="^":self.f.triangle(xx,yy,r,color,orientation="up",fill=filled)
        elif marker==">":self.f.triangle(xx,yy,r,color,orientation="right",fill=filled)
        elif marker=="s":self.f.square(xx,yy,r,color,fill=filled)
    def line(self,x1,y1,x2,y2,color="#222",width=2,dash=None):self.f.line(self.X(x1),self.Y(y1),self.X(x2),self.Y(y2),color,width,dash)
    def path(self,pts,color,width=2):self.f.polyline([(self.X(x),self.Y(y)) for x,y in pts],color,width)


def pair_key(p):return (p["r0_mm"],p["energy_eV"],p["azimuth_rad"],p["phase_rad"],p["direction"])
def read_trace(path):
    with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def save_text(root,name,data): (root/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def figure_a(main,pairs,root):
    azs=sorted({round(r["azimuth_rad"],12) for r in main})
    f=Figure("Figure_A_retention_energy_phase",root,2700,1800)
    f.text(1350,72,"Retention by transverse energy and initial RF phase",48,"#111", "middle",True)
    lefts=(250,1080,1910); top0=180; pw=650; ph=270; rowgap=90
    for c,title in enumerate(("Rotating: 250 V; phase on y-axis","Standing: 250 V","Matched outcome difference")):
        f.text(lefts[c]+pw/2,140,title,34,"#111","middle",True)
    diff={(p["r0_mm"],p["energy_eV"],p["azimuth_rad"],p["phase_rad"],p["direction"]):p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["field"]=="h35"}
    for ri,az in enumerate(azs):
        top=top0+ri*(ph+rowgap)
        f.text(20,top+ph/2-8,"azimuth",25,"#222","start",True)
        f.text(20,top+ph/2+30,f"{math.degrees(az):.1f} deg",25,"#222","start")
        for ci,model in enumerate(("rotating","standing","difference")):
            ax=Ax(f,lefts[ci],top,pw,ph,.35,3.7,-30,300);ax.grid([1,2,3],[0,90,180,270],lambda x:f"{x:g}",lambda y:f"{y:g}")
            if model!="difference":
                records=[r for r in main if r["model"]==model and abs(r["azimuth_rad"]-az)<1e-10]
                for r in records:
                    x=r["energy_eV"]+(-.022 if r["r0_mm"]==1 else .022);y=math.degrees(r["phase_rad"])+(2.5 if r["direction"]>0 else -2.5)
                    ax.dot(x,y,STATUS[r["status"]],7 if r["r0_mm"]==1 else 11,"o" if r["direction"]>0 else "^")
            else:
                records=[p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["field"]=="h35" and abs(p["azimuth_rad"]-az)<1e-10]
                for p in records:
                    x=p["energy_eV"]+(-.022 if p["r0_mm"]==1 else .022);y=math.degrees(p["phase_rad"])+(2.5 if p["direction"]>0 else -2.5)
                    ax.dot(x,y,DIFF[p["difference_class"]],7 if p["r0_mm"]==1 else 11,"o" if p["direction"]>0 else "^")
            if ri==len(azs)-1:ax.xlab("Initial transverse energy (eV)")
    leg_y=1655
    x=245
    for color,label in ((RET,"retained to 1000 RF"),(ESC,"escaped")):
        f.circle(x,leg_y,9,color);f.text(x+22,leg_y+9,label,25);x+=250
    for marker,label in (("o","positive tangent"),("^","negative tangent")):
        if marker=="o":f.circle(x,leg_y,9,"#333")
        else:f.triangle(x,leg_y,10,"#333")
        f.text(x+22,leg_y+9,label,25);x+=280
    for color,label in (("#555","r0=1 mm"),("#555","r0=2 mm")):
        if label.endswith("1 mm"):f.circle(x,leg_y,7,color)
        else:f.circle(x,leg_y,11,color)
        f.text(x+22,leg_y+9,label,25);x+=200
    x=500;diff_y=1725
    for cls,color in DIFF.items():
        f.circle(x,diff_y,8,color);f.text(x+18,diff_y+9,cls.replace("_"," "),22);x+=470
    f.text(1350,1780,"Calculated markers only. Rows are azimuths; shapes show tangential sign; size separates r0=1 and 2 mm. A small x-offset separates radii visually. No interpolation.",23,"#444","middle")
    f.save()


def figure_b(pairs,root):
    data=[p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["field"]=="h35" and p["rotating_voltage_V"]==250 and p["standing_voltage_V"]==250]
    energies=sorted({p["energy_eV"] for p in data});f=Figure("Figure_B_retention_fraction_by_energy",root,2100,1250)
    f.text(1050,70,"Retention fraction on identical designed ensembles",45,"#111","middle",True)
    ax=Ax(f,230,180,1740,800,min(energies)-.25,max(energies)+.25,-.04,1.32);ax.grid([.5,1,2,3,4],[0,.2,.4,.6,.8,1.],lambda x:f"{x:g}",lambda y:f"{y:.1f}")
    ax.xlab("Initial transverse energy (eV)");ax.ylab("Designed-grid retained fraction")
    for model,offset in (("rotating",-.035),("standing",.035)):
        for i,e in enumerate(energies):
            group=[p for p in data if p["energy_eV"]==e];n=len(group)
            if not n:continue
            k=sum(p[f"{model}_status"]=="retained" for p in group);frac=k/n
            ax.dot(e+offset,frac,MODEL[model],11,"o")
            if model=="rotating":label_frac=1.16+.08*(i%2)
            else:label_frac=frac-.12 if frac>.85 else min(1.12,frac+.09)
            f.text(ax.X(e+offset),ax.Y(label_frac),f"{k}/{n}",18,MODEL[model],"middle")
    for i,(model,color) in enumerate((("rotating",ROT),("standing",STAND))):
        f.line(1180+i*330,1080,1240+i*330,1080,color,7);f.text(1260+i*330,1089,f"{model.title()} 250 V",25,color)
    f.text(1050,1135,"Calculated shells (eV): "+", ".join(f"{e:g}" for e in energies),19,"#444","middle")
    f.text(1050,1200,"Observation horizon: 1000 RF periods. Counts describe this deterministic matched grid, not a population probability.",22,"#444","middle")
    f.save()


def figure_c(pairs,root):
    data=[p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["field"]=="h35" and .35<=p["energy_eV"]<=5.]
    launches=sorted({(p["azimuth_rad"],p["phase_rad"],p["direction"]) for p in data});ids={k:i for i,k in enumerate(launches)}
    write_csv(root/"launch_index_key.csv",[{"launch_id":i,"azimuth_deg":math.degrees(a),"phase_deg":math.degrees(ph),"tangential_velocity_sign":int(d)} for i,(a,ph,d) in enumerate(launches)])
    f=Figure("Figure_C_high_energy_boundary",root,2700,1800);f.text(1350,65,"High-energy confinement boundary in the sampled launch grid",45,"#111","middle",True)
    positions=((170,180),(1440,180),(170,970),(1440,970));pw=1080;ph=570
    for ri,radius in enumerate((1.,2.)):
        for ci,model in enumerate(("rotating","standing")):
            left,top=positions[ri*2+ci];ax=Ax(f,left,top,pw,ph,.35,5.,-.8,max(2,len(launches)-.2))
            ax.grid([.5,1,2,3,4,5],list(range(0,len(launches),4)),lambda x:f"{x:g}",lambda y:f"{int(y)}")
            ax.title(f"{model.title()}; r0={radius:g} mm");ax.xlab("Initial transverse energy (eV)");ax.ylab("Launch ID (see key CSV)")
            groups=defaultdict(list)
            for p in data:
                if p["r0_mm"]!=radius:continue
                i=ids[(p["azimuth_rad"],p["phase_rad"],p["direction"])];groups[i].append(p)
                ax.dot(p["energy_eV"],i,STATUS[p[f"{model}_status"]],6,"o")
            for i,seq in groups.items():
                seq=sorted(seq,key=lambda x:x["energy_eV"])
                for a,b in zip(seq,seq[1:]):
                    if a[f"{model}_status"]!=b[f"{model}_status"]:ax.line(a["energy_eV"],i,b["energy_eV"],i,"#222",3)
    f.circle(360,1720,9,RET);f.text(385,1728,"retained",24);f.circle(610,1720,9,ESC);f.text(635,1728,"escaped",24);f.line(910,1720,975,1720,"#222",3);f.text(1000,1728,"observed adjacent transition bracket",24)
    f.text(1350,1780,"Dots are calculated states. Brackets join only adjacent sampled energies for one exact launch with different outcomes; no continuous boundary is inferred.",22,"#444","middle")
    f.save()


def figure_d(main,pairs,root):
    cand=[p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["rotating_status"]=="retained" and p["standing_status"]=="escaped"]
    p=next((p for p in cand if p["energy_eV"]==1. and p["r0_mm"]==1. and abs(p["azimuth_rad"]-math.pi/4)<1e-10 and abs(p["phase_rad"])<1e-10 and p["direction"]==1.),None)
    if p is None and cand:p=sorted(cand,key=lambda x:(abs(x["energy_eV"]-1),abs(x["r0_mm"]-1)))[0]
    if p is None:return
    lookup={(r["model"],r["r0_mm"],r["energy_eV"],r["azimuth_rad"],r["phase_rad"],r["direction"]):r for r in main}
    traces={}
    for m in ("rotating","standing"):
        traces[m]=read_trace(lookup[(m,p["r0_mm"],p["energy_eV"],p["azimuth_rad"],p["phase_rad"],p["direction"])] ["sample_csv"])
    f=Figure("Figure_D_matched_trajectory",root,2500,1350);f.text(1250,65,f"Matched {p['energy_eV']:g}-eV start: rotating survives; standing escapes",44,"#111","middle",True)
    ax=Ax(f,180,180,850,850,-8.5,8.5,-8.5,8.5);ax.grid([-8,-4,0,4,8],[-8,-4,0,4,8]);ax.title("Transverse x-y orbit, first 50 RF")
    circle=[(8*math.cos(t),8*math.sin(t)) for t in np.linspace(0,2*math.pi,400)];ax.path(circle,"#222",4)
    for m,tr in traces.items():
        early=[x for x in tr if float(x["rf_period"])<=50]
        stride=max(1,len(early)//2500);ax.path([(float(x["x_mm"]),float(x["y_mm"])) for x in early[::stride]],MODEL[m],3)
    ax.xlab("x (mm)");ax.ylab("y (mm)")
    axr=Ax(f,1220,180,1100,850,0,10,0,8.6);axr.grid([0,2,4,6,8,10],[0,2,4,6,8],lambda x:f"{x:g}",lambda y:f"{y:g}");axr.title("Radius versus time")
    axr.xlab("Time (ms)");axr.ylab("Radius (mm)");axr.line(0,8,10,8,"#222",4,"12,8")
    for m,tr in traces.items():
        buckets=defaultdict(list)
        for z in tr:buckets[round(float(z["rf_period"]))].append((float(z["time_s"])*1000,float(z["r_mm"])))
        pts=[(sum(v[0] for v in bucket)/len(bucket),sum(v[1] for v in bucket)/len(bucket)) for _,bucket in sorted(buckets.items())]
        axr.path(pts,MODEL[m],3)
    # Show raw RF micromotion in a compact inset, while the main curve shows
    # the per-period radial envelope used for the slow orbital comparison.
    inset=Ax(f,1750,260,500,300,0,.05,5,8.5);inset.grid([0,.01,.02,.03,.04,.05],[5,6,7,8],lambda x:f"{x:g}",lambda y:f"{y:g}")
    inset.line(0,8,.05,8,"#222",3,"10,7")
    for m,tr in traces.items():inset.path([(float(x["time_s"])*1000,float(x["r_mm"])) for x in tr if float(x["time_s"])*1000<=.05],MODEL[m],3)
    f.text(2000,225,"Raw RF micro-motion",22,"#333","middle",True)
    f.line(500,1160,560,1160,ROT,7);f.text(580,1168,"Rotating RF",26,ROT);f.line(830,1160,890,1160,STAND,7);f.text(910,1168,"Standing RF",26,STAND)
    f.text(1250,1240,f"100 kTh; 250 V each; 100 kHz; r0={p['r0_mm']:g} mm; azimuth={math.degrees(p['azimuth_rad']):.1f} deg; phase={math.degrees(p['phase_rad']):.0f} deg; tangential sign={p['direction']:+.0f}. Standing first passage: {p['standing_escape_time_RF']:.3f} RF.",23,"#444","middle")
    f.text(1250,1290,"Rotating survives the finite 1000-RF horizon; this does not establish permanent stability. The dashed circle is the 8-mm aperture.",23,"#444","middle")
    f.save();save_text(root,"Figure_D_case.json",p)


def figure_e(pairs,root):
    data=[p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["field"]=="h35" and .35<=p["energy_eV"]<=5.]
    cond=sorted({(p["energy_eV"],p["r0_mm"],p["azimuth_rad"],p["phase_rad"],p["direction"]) for p in data});ids={k:i for i,k in enumerate(cond)}
    write_csv(root/"escape_condition_index_key.csv",[{"condition_id":i,"energy_eV":e,"r0_mm":r,"azimuth_deg":math.degrees(a),"phase_deg":math.degrees(ph),"tangential_velocity_sign":int(d)} for i,(e,r,a,ph,d) in enumerate(cond)])
    f=Figure("Figure_E_escape_times_censored",root,2600,1800);f.text(1300,65,"Matched first-passage times; retained starts are right-censored",46,"#111","middle",True)
    for ci,m in enumerate(("rotating","standing")):
        left=210+ci*1280;ax=Ax(f,left,170,1030,1390,math.log10(.7),math.log10(1700),len(cond),-1)
        for t in (.7,1,3,10,30,100,300,1000):
            x=math.log10(t);ax.line(x,len(cond),x,-1,"#e4e8eb",1);f.text(ax.X(x),ax.t+ax.h+40,f"{t:g}",22,"#444","middle")
        for i in range(0,len(cond),max(1,len(cond)//16)):
            y=len(cond)-i;ax.line(math.log10(.7),y,math.log10(1700),y,"#f0f1f2",1);f.text(ax.l-14,ax.Y(y)+7,str(i),19,"#444","end")
        ax.line(math.log10(1000),len(cond),math.log10(1000),-1,"#444",3,"12,8")
        ax.line(math.log10(.7),len(cond),math.log10(.7),-1,"#222",2);ax.line(math.log10(.7),-1,math.log10(1700),-1,"#222",2)
        ax.title(m.title());ax.ylab("Matched start ID (key CSV)");ax.xlab("First escape time (RF periods; logarithmic)")
        for p in data:
            i=ids[(p["energy_eV"],p["r0_mm"],p["azimuth_rad"],p["phase_rad"],p["direction"])];y=len(cond)-i
            t=p[f"{m}_escape_time_RF"]
            if t is None:ax.dot(math.log10(1000),y,MODEL[m],7,">",False)
            else:ax.dot(math.log10(max(.7,t)),y,MODEL[m],6,"o")
    f.text(1300,1725,"Filled circles: observed first outward crossing. Open right-pointing triangles: 1000-RF survivors censored at the horizon, never counted as escapes.",23,"#444","middle")
    f.save()


def figure_f(pairs,root):
    matched=[p for p in pairs if p["normalization"]=="matched_leading_pseudopotential_at_0p5mm" and p["field"]=="h35"]
    matched.sort(key=lambda p:(p["energy_eV"],p["r0_mm"],p["azimuth_rad"],p["phase_rad"],p["direction"]))
    selected=[]
    for m in matched:
        eq=next((p for p in pairs if p["normalization"]=="equal_peak_voltage" and p["field"]=="h35"
            and p["r0_mm"]==m["r0_mm"] and p["energy_eV"]==m["energy_eV"] and p["azimuth_rad"]==m["azimuth_rad"]
            and p["phase_rad"]==m["phase_rad"] and p["direction"]==m["direction"]),None)
        if eq:selected.append(eq)
    selected=selected[:4]
    if not selected or not matched:return
    f=Figure("Figure_F_equal_peak_vs_matched_strength",root,2400,1300);f.text(1200,60,"High-energy differences under separate voltage normalizations",44,"#111","middle",True)
    data_rows=[{"selected_start_id":i+1,"energy_eV":p["energy_eV"],"r0_mm":p["r0_mm"],
        "azimuth_deg":math.degrees(p["azimuth_rad"]),"phase_deg":math.degrees(p["phase_rad"]),
        "tangential_direction":int(p["direction"])} for i,p in enumerate(selected)]
    xlo=min(p["energy_eV"] for p in selected)-.5;xhi=max(p["energy_eV"] for p in selected)+.5
    for ci,(protocol,title) in enumerate((("equal_peak_voltage","Equal peak: 250 V / 250 V"),("matched_leading_pseudopotential_at_0p5mm","Matched leading term: 250 V / 353.553 V"))):
        left=220+ci*1160;ax=Ax(f,left,180,930,620,xlo,xhi,-.6,len(selected)-.4);ax.grid(sorted({p["energy_eV"] for p in selected}),list(range(len(selected))),lambda x:f"{x:g}",lambda y:f"{int(y)+1}")
        ax.title(title);ax.xlab("Initial transverse energy (eV)");ax.ylab("Selected matched start ID")
        group=[p for p in pairs if p["normalization"]==protocol and p["field"]=="h35"]
        for i,ref in enumerate(selected):
            pair_row=data_rows[i]
            for model,dy,mark in (("rotating",-.15,"o"),("standing",.15,"s")):
                p=next((p for p in group if p["r0_mm"]==ref["r0_mm"] and abs(p["energy_eV"]-ref["energy_eV"])<1e-12 and p["azimuth_rad"]==ref["azimuth_rad"] and p["phase_rad"]==ref["phase_rad"] and p["direction"]==ref["direction"]),None)
                if p:
                    state=p[f"{model}_status"];ax.dot(ref["energy_eV"],i+dy,STATUS[state],12,mark,state=="retained")
                    pair_row[f"{protocol}_{model}_status"]=state
    f.text(1200,990,"Filled = retained to 1000 RF; open = escaped. Marker circles = rotating; squares = standing. Rows are identical launch states in both panels.",22,"#444","middle")
    f.text(1200,1060,"The alternate protocol matches the leading pseudopotential term at r=0.5 mm only; full multipole pseudopotentials need not match.",21,"#444","middle")
    write_csv(root/"Figure_F_selected_cases.csv",data_rows);f.save()


def rebuild_v3(rows,root):
    sub=root/"v3_rebuilt";sub.mkdir(parents=True,exist_ok=True)
    data=[r for r in rows if r["source"]=="V3 reused" and r["horizon_RF"]==1000]
    f=Figure("V3_retention_overview_rebuilt",sub,2100,1100);f.text(1050,60,"V3 confinement map rebuilt from its saved classifications",42,"#111","middle",True)
    for ci,m in enumerate(("rotating","standing")):
        left=220+ci*970;ax=Ax(f,left,170,780,700,0,3.2,.4,3.2);ax.grid([.5,1,2,3],[.5,1,2,3],lambda x:f"{x:g}",lambda y:f"{y:g}");ax.title(m.title());ax.xlab("Initial transverse energy (eV)");ax.ylab("Initial radius (mm)")
        for r in data:
            if r["model"]!=m:continue
            ax.dot(r["energy_eV"],r["r0_mm"],STATUS[r["status"]],10,"o" if r["direction"]>0 else "^")
    f.text(1050,1010,"Every marker is a V3 calculated state. Green: survived 1000 RF; red: escaped. Shape: tangential velocity sign.",23,"#444","middle");f.save()
    candidates=[]
    for r in data:
        if r["energy_eV"]==1 and r["r0_mm"]==1 and r["model"]=="rotating" and r["status"]=="retained":
            st=next((x for x in data if x["model"]=="standing" and x["energy_eV"]==1 and x["r0_mm"]==1 and x["azimuth_rad"]==r["azimuth_rad"] and x["phase_rad"]==r["phase_rad"] and x["direction"]==r["direction"] and x["status"]=="escaped"),None)
            if st:candidates.append((r,st))
    if not candidates:return
    rot,stand=candidates[0];f=Figure("V3_high_energy_pair_rebuilt",sub,2400,1200);f.text(1200,60,"V3 matched 1-eV full-RF trajectory pair rebuilt",42,"#111","middle",True)
    ax=Ax(f,180,170,850,800,-8.5,8.5,-8.5,8.5);ax.grid([-8,-4,0,4,8],[-8,-4,0,4,8]);ax.title("Transverse orbit")
    pts=[(8*math.cos(t),8*math.sin(t)) for t in np.linspace(0,2*math.pi,400)];ax.path(pts,"#222",4);ax.xlab("x (mm)");ax.ylab("y (mm)")
    ar=Ax(f,1220,170,1030,800,0,60,0,8.6);ar.grid([0,10,20,30,40,50,60],[0,2,4,6,8],lambda x:f"{x:g}",lambda y:f"{y:g}");ar.title("Radius and first passage");ar.xlab("Time (RF periods)");ar.ylab("Radius (mm)");ar.line(0,8,60,8,"#222",4,"12,8")
    for r,c in ((rot,ROT),(stand,STAND)):
        tr=read_trace(r["sample_csv"]);ax.path([(float(x["x_mm"]),float(x["y_mm"])) for x in tr[::max(1,len(tr)//3000)]],c,3)
        ar.path([(float(x["rf_period"]),float(x["r_mm"])) for x in tr if float(x["rf_period"])<=60],c,3)
    f.text(1200,1080,f"V3 reused traces; azimuth={math.degrees(rot['azimuth_rad']):.1f} deg, phase={math.degrees(rot['phase_rad']):.1f} deg, sign={rot['direction']:+.0f}; standing first passage {stand['escape_time_RF']:.3f} RF.",23,"#444","middle");f.save()


def main():
    rows,main,pairs,stats=collect();root=OUT/"figures";root.mkdir(parents=True,exist_ok=True)
    figure_a(main,pairs,root);figure_b(pairs,root);figure_c(pairs,root);figure_d(main,pairs,root);figure_e(pairs,root);figure_f(pairs,root);rebuild_v3(rows,root)
    manifest={"renderer":"Pillow drawing with ReportLab vector PDF and hand-authored SVG","png_dpi":300,
      "palette":{"rotating":ROT,"standing":STAND,"retained":RET,"escaped":ESC},"formats":["PDF","SVG","PNG"],
      "interpolation":"none; only actual calculated states plotted; transition bars connect adjacent calculated energies at the same launch",
      "retention_fraction":"deterministic designed-grid fraction, not population probability",
      "censoring":"survivors shown as right-censored at 1000 RF; not treated as escapes",
      "V3_refreshed":["V3_retention_overview_rebuilt","V3_high_energy_pair_rebuilt"]}
    save_text(root,"figure_manifest.json",manifest)
    print(json.dumps({"pdf_figures":sorted(p.name for p in root.glob("*.pdf")),"reused_V3_points":sum(r["source"]=="V3 reused" for r in rows),"pairs":len(pairs)},indent=2))


if __name__=="__main__":main()
