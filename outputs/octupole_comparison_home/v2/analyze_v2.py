import csv,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO

P=Path(__file__).resolve().parent
ROT='rotating'; STA='standing_equal_peak'
COL={ROT:'#1769aa',STA:'#d55e00'}
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
SM=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
TINY=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',11)

def load():
    allrows=[]
    for ln in (P/'trajectory_checkpoint.jsonl').read_text(encoding='utf8').splitlines():
        try:allrows.append(json.loads(ln))
        except Exception:pass
    latest={}
    for r in allrows:
        if r.get('status') in ('retained','escaped'):latest[r['job_id']]=r
    return list(latest.values()),allrows

def writecsv(name,rows):
    if rows:
        with (P/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def savefig(name,title,xlab,ylab,panels,xlim,ylim,xticks,yticks,*,pointsize=5,zero=False):
    W=1450; L,R,T,B=112,40,84,88; gapx,gapy=80,90
    nrow=max(len(panels),1);ncol=1
    # panels is a list of dicts, arrange two columns for paired model comparisons
    if len(panels)>1 and len(panels)%2==0:ncol=2;nrow=len(panels)//2
    H=max(930,int(280*nrow+200))
    pw=(W-L-R-gapx*(ncol-1))/ncol; ph=(H-T-B-gapy*(nrow-1))/nrow
    im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>']
    def txt(x,y,s,font=SM,anchor='mm',fill='black'):
        d.text((x,y),str(s),font=font,fill=fill,anchor=anchor)
        esc=str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        svg.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-family="Arial" font-size="{font.size}" fill="{fill}">{esc}</text>')
    txt(W/2,35,title,FONT)
    for pi,panel in enumerate(panels):
        rr,cc=divmod(pi,ncol);x0=L+cc*(pw+gapx);y0=T+rr*(ph+gapy)
        x1=x0+pw;y1=y0+ph
        plim=panel.get('ylim',ylim)
        plimx=panel.get('xlim',xlim)
        def X(x):return x0+(x-plimx[0])/(plimx[1]-plimx[0])*pw
        def Y(y):return y1-(y-plim[0])/(plim[1]-plim[0])*ph
        txt((x0+x1)/2,y0-23,panel.get('title',''),SM)
        for xv in panel.get('xticks',xticks):
            xx=X(xv);d.line((xx,y0,xx,y1),fill='#e2e2e2',width=1);svg.append(f'<line x1="{xx:.1f}" y1="{y0:.1f}" x2="{xx:.1f}" y2="{y1:.1f}" stroke="#e2e2e2"/>');txt(xx,y1+17,f'{xv:g}',TINY)
        yts=panel.get('yticks',yticks)
        for yv in yts:
            yy=Y(yv);d.line((x0,yy,x1,yy),fill='#e2e2e2',width=1);svg.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="#e2e2e2"/>');txt(x0-30,yy,f'{yv:.3g}',TINY)
        d.line((x0,y0,x0,y1),fill='black',width=2);d.line((x0,y1,x1,y1),fill='black',width=2)
        svg.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" stroke="black" stroke-width="2"/><line x1="{x0:.1f}" y1="{y1:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="black" stroke-width="2"/>')
        if zero and plim[0]<0<plim[1]:
            yy=Y(0);d.line((x0,yy,x1,yy),fill='#777',width=1);svg.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" stroke="#777"/>')
        txt((x0+x1)/2,y1+43,xlab,SM)
        label=Image.new('RGBA',(180,24),(255,255,255,0));ImageDraw.Draw(label).text((90,12),ylab,font=SM,fill='black',anchor='mm')
        label=label.rotate(90,expand=True);im.paste(label,(int(x0-67-label.width/2),int((y0+y1)/2-label.height/2)),label)
        svg.append(f'<text x="{x0-67:.1f}" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-family="Arial" font-size="{SM.size}" transform="rotate(-90 {x0-67:.1f} {(y0+y1)/2:.1f})">{ylab}</text>')
        for pt in panel.get('points',[]):
            x,y=pt['x'],pt['y']
            if not (plimx[0]<=x<=plimx[1] and plim[0]<=y<=plim[1]):continue
            xx,yy=X(x),Y(y);col=pt.get('color','#333');rad=pointsize
            if 'err_low' in pt and 'err_high' in pt:
                yl=Y(pt['err_low']); yh=Y(pt['err_high'])
                d.line((xx,yh,xx,yl),fill=col,width=2); d.line((xx-4,yh,xx+4,yh),fill=col,width=2); d.line((xx-4,yl,xx+4,yl),fill=col,width=2)
                svg.append(f'<line x1="{xx:.2f}" y1="{yh:.2f}" x2="{xx:.2f}" y2="{yl:.2f}" stroke="{col}" stroke-width=2/><line x1="{xx-4:.2f}" y1="{yh:.2f}" x2="{xx+4:.2f}" y2="{yh:.2f}" stroke="{col}" stroke-width=2/><line x1="{xx-4:.2f}" y1="{yl:.2f}" x2="{xx+4:.2f}" y2="{yl:.2f}" stroke="{col}" stroke-width=2/>')
            d.ellipse((xx-rad,yy-rad,xx+rad,yy+rad),fill=col,outline='black',width=1)
            svg.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="{rad}" fill="{col}" stroke="black"/>')
            if pt.get('label'):txt(xx,yy-10,pt['label'],TINY,fill=col)
        for li,item in enumerate(panel.get('legend',[])):
            lx=x0+8+li*min(205,pw/max(1,len(panel['legend'])));ly=y0+14
            d.ellipse((lx,ly-5,lx+10,ly+5),fill=item[1],outline='black');txt(lx+18,ly,item[0],TINY,anchor='lm',fill=item[1])
    svg.append('</svg>');(P/f'{name}.svg').write_text('\n'.join(svg),encoding='utf8')
    im.save(P/f'{name}.png',dpi=(240,240));bio=BytesIO();im.save(bio,format='PNG');bio.seek(0)
    c=canvas.Canvas(str(P/f'{name}.pdf'),pagesize=(W,H));c.drawImage(ImageReader(bio),0,0,width=W,height=H);c.showPage();c.save()

def main():
    rows,attempts=load()
    if not rows:raise SystemExit('No valid checkpoint results found')
    # Counts and retention are always based on evaluated finite-time launches.
    groups={}
    for r in rows:
        k=(r['model'],float(r['mz_Th']),round(float(r['r0_m']),12),float(r['energy_eV']))
        groups.setdefault(k,[]).append(r)
    summary=[]
    for (m,mz,r0,e),g in sorted(groups.items()):
        n=len(g);ret=sum(x['status']=='retained' for x in g)
        summary.append({'model':m,'mz_Th':mz,'r0_mm':r0*1e3,'energy_eV':e,'n_valid':n,'retained':ret,'escaped':n-ret,'P_ret':ret/n})
    writecsv('retention_summary.csv',summary)
    # Figure 1: columns are separate field models; rows are near-transition energy shells.
    measured_energies=sorted(set(float(r['energy_eV']) for r in rows if abs(float(r['r0_m'])-.001)<1e-12))
    energies=[e for e in (.0015,.0035,.005,.00625) if any(abs(x-e)<1e-10 for x in measured_energies)]
    panels=[]
    for model in (ROT,STA):
      for e in energies:
        pts=[]
        for mz in sorted(set(float(r['mz_Th']) for r in rows)):
          g=[r for r in rows if r['model']==model and float(r['mz_Th'])==mz and abs(float(r['r0_m'])-.001)<1e-12 and float(r['energy_eV'])==e]
          if g:pts.append({'x':mz/1000,'y':sum(r['status']=='retained' for r in g)/len(g),'label':str(len(g))})
        panels.append({'title':f'{model.replace("_"," ")} · E={e:.5f} eV','points':pts})
    savefig('figure_1_mass_retention','Finite-time retention vs physical m/z · r₀ = 1 mm','m/z (kTh)','Retained fraction',panels,(40,210),(0,1),[50,100,150,200],[0,.25,.5,.75,1])
    # Radial map and transition audit use one fixed four-state baseline at every
    # radius. Locally enriched 24-state samples are preserved separately below.
    radial_base=[r for r in rows if float(r['energy_eV'])==0 and
      float(r['azimuth_rad']) in (0.,math.pi/2) and float(r['rf_phase_rad']) in (0.,math.pi)]
    rb={}
    for r in radial_base:
      k=(r['model'],float(r['mz_Th']),round(float(r['r0_m']),12));rb.setdefault(k,[]).append(r)
    radial_summary=[]
    for (m,mz,r0),g in sorted(rb.items()):
      n=len(g);ret=sum(x['status']=='retained' for x in g)
      radial_summary.append({'model':m,'mz_Th':mz,'r0_mm':r0*1e3,'n_baseline':n,'retained':ret,'escaped':n-ret,'P_ret':ret/n})
    writecsv('radial_acceptance_grid.csv',radial_summary)
    enriched=[]
    for s in summary:
      if s['energy_eV']==0 and s['r0_mm']>=7:
       enriched.append({'model':s['model'],'mz_Th':s['mz_Th'],'r0_mm':s['r0_mm'],'n_valid':s['n_valid'],'retained':s['retained'],'P_ret':s['P_ret'],'ensemble':'all evaluated launches; locally enriched boundary'})
    writecsv('radial_boundary_enriched.csv',enriched)
    # Figure 2: direct baseline grid plus a separately scaled view of every
    # evaluated outer-boundary point; no interpolation or connecting contour.
    p2=[]
    for model in (ROT,STA):
      off=-.8 if model==ROT else .8
      pts=[]
      for s in radial_summary:
        if s['model']!=model or s['n_baseline']==0:continue
        col=('#14804a' if s['P_ret']==1 else '#b42318' if s['P_ret']==0 else '#cf8a00')
        pts.append({'x':s['mz_Th']/1000+off,'y':s['r0_mm'],'color':col})
      p2.append({'title':model.replace('_',' ')+' · all direct baseline points (n=4)', 'points':pts,
                 'legend':[('all retained','#14804a'),('mixed','#cf8a00'),('all escaped','#b42318')]})
      pts=[]
      for s in enriched:
        if s['model']!=model:continue
        col=('#14804a' if s['P_ret']==1 else '#b42318' if s['P_ret']==0 else '#cf8a00')
        pts.append({'x':s['mz_Th']/1000+off,'y':s['r0_mm'],'color':col,'label':f"{s['retained']}/{s['n_valid']}"})
      p2.append({'title':model.replace('_',' ')+' · outer edge, direct measured points', 'points':pts,
                 'ylim':(6.95,7.98),'yticks':[7.0,7.125,7.25,7.375,7.5,7.625,7.75,7.875],
                 'legend':[('all retained','#14804a'),('mixed','#cf8a00'),('all escaped','#b42318')]})
    savefig('figure_2_mass_radius','Mass–radius acceptance · zero transverse launch energy; point labels = retained / evaluated','m/z (kTh)','Initial radius r₀ (mm)',p2,(40,210),(0,8),[50,100,150,200],[0,2,4,6,8],pointsize=6)
    # Figure 3: energy curves as direct points, separately for each mass and model.
    p3=[]
    masses=sorted(set(float(r['mz_Th']) for r in rows if abs(float(r['r0_m'])-.001)<1e-12))
    for model in (ROT,STA):
      for mz in masses:
        pts=[]
        for e in sorted(set(float(r['energy_eV']) for r in rows)):
          g=[r for r in rows if r['model']==model and float(r['mz_Th'])==mz and abs(float(r['r0_m'])-.001)<1e-12 and float(r['energy_eV'])==e]
          if g:pts.append({'x':e*1000,'y':sum(r['status']=='retained' for r in g)/len(g)})
        p3.append({'title':f'{model.replace("_"," ")} · {mz/1000:g} kTh · n=16/shell','points':pts})
    savefig('figure_3_energy_acceptance','Energy-dependent retention · r₀ = 1 mm, full-RF, 300 periods','Transverse energy (meV)','Retained fraction',p3,(.5,10.5),(0,1),[1,2,3,4,5,6,7,8,9,10],[0,.25,.5,.75,1])
    # Pair trajectories on the exact physical initial-condition key.
    pair={}
    for r in rows:
      k=(float(r['mz_Th']),float(r['r0_m']),float(r['energy_eV']),float(r['azimuth_rad']),float(r['rf_phase_rad']),int(r['velocity_sign']))
      pair.setdefault(k,{})[r['model']]=r
    pd=[];obs={k:[] for k in ('max_radius_m','mean_radius_m','rms_radius_m','radial_span_m','net_angle_rad')};esc=[]
    for k,ab in pair.items():
      if ROT not in ab or STA not in ab:continue
      a,b=ab[ROT],ab[STA];common={'mz_Th':k[0],'r0_mm':k[1]*1e3,'energy_eV':k[2], 'azimuth_rad':k[3], 'rf_phase_rad':k[4], 'velocity_sign':k[5],
        'rotating_status':a['status'],'standing_status':b['status'],'paired_status_delta':(1 if a['status']=='retained' else 0)-(1 if b['status']=='retained' else 0)}
      for f in obs:
        delta=float(a[f])-float(b[f]);common['delta_'+f]=delta;obs[f].append((k[0],k[2],delta))
      if a['status']=='escaped' and b['status']=='escaped':
        delta=float(a['escape_period'])-float(b['escape_period']);common['delta_escape_period_rot_minus_stand']=delta;esc.append((k[0],k[2],delta))
      pd.append(common)
    writecsv('paired_dynamics.csv',pd)
    # Keep the machine-readable paired summary synchronized with the full checkpoint.
    mismatch_groups={}
    for z in pd:
      if z['paired_status_delta']==0:continue
      k=(float(z['mz_Th']),round(float(z['r0_mm']),9),float(z['energy_eV']))
      g=mismatch_groups.setdefault(k,{'mz_Th':k[0],'r0_mm':k[1],'energy_eV':k[2],'rotating_retains_standing_escapes':0,'rotating_escapes_standing_retains':0})
      if z['paired_status_delta']>0:g['rotating_retains_standing_escapes']+=1
      else:g['rotating_escapes_standing_retains']+=1
    energy_summary=[]
    for e in sorted(set(float(z['energy_eV']) for z in pd if float(z['mz_Th'])==100000 and abs(float(z['r0_mm'])-1)<1e-9)):
      g=[z for z in pd if float(z['mz_Th'])==100000 and abs(float(z['r0_mm'])-1)<1e-9 and float(z['energy_eV'])==e]
      if not g:continue
      states={}
      for z in g:
        k=z['rotating_status']+'|'+z['standing_status'];states[k]=states.get(k,0)+1
      def mm(field,scale=1.):
        vals=[float(z[field])*scale for z in g if z.get(field) not in ('',None)]
        return [min(vals),max(vals)] if vals else None
      energy_summary.append({'energy_eV':e,'n_pairs':len(g),'classification_pairs':states,
        'delta_max_radius_mm_minmax':mm('delta_max_radius_m',1000.),'delta_rms_radius_mm_minmax':mm('delta_rms_radius_m',1000.),
        'delta_angle_rad_minmax':mm('delta_net_angle_rad'),'both_escape_delta_period_minmax':mm('delta_escape_period_rot_minus_stand')})
    (P/'paired_dynamics_summary.json').write_text(json.dumps({'pairs':len(pd),'mismatched_pairs':sum(x['paired_status_delta']!=0 for x in pd),
      'mismatches_by_condition':list(mismatch_groups.values()),'100k_energy':energy_summary,
      'outer_radial_pairs':sum(float(x['energy_eV'])==0 and float(x['r0_mm'])>=7 for x in pd)},indent=2)+'\n',encoding='utf-8')
    dyn=[]
    for f,label in [('max_radius_m','Δ maximum radius (mm)'),('mean_radius_m','Δ mean radius (mm)'),
                    ('rms_radius_m','Δ RMS radius (mm)'),('radial_span_m','Δ radial span (mm)'),
                    ('net_angle_rad','Δ net angular sweep (rad)')]:
      pts=[{'x':e*1000,'y':d*(1000 if f.endswith('_m') else 1)} for mz,e,d in obs[f] if mz==100000 and .0005<=e<=.0105]
      lim=max([abs(z['y']) for z in pts]+[1e-12])*1.15 if pts else 1
      dyn.append({'title':label+' · 100 kTh','points':pts,'ylim':(-lim,lim),'yticks':[-lim,0,lim]})
    ep=[{'x':e*1000,'y':d} for mz,e,d in esc if mz==100000 and .0005<=e<=.0105]
    lim=max([abs(z['y']) for z in ep]+[1e-12])*1.15 if ep else 1
    dyn.append({'title':'Δ first escape time · 100 kTh (RF periods)','points':ep,'ylim':(-lim,lim),'yticks':[-lim,0,lim]})
    savefig('figure_4_paired_dynamics','Paired rotating − standing observables · same starts','Initial transverse energy (meV)','Difference (rotating − standing)',dyn,(.5,10.5),(-1,1),[1,2,3,4,5,6,7,8,9,10],[-1,0,1],zero=True)
    # Radial-launch paired dynamics: report the median and central 80% range
    # across matched phases/azimuths, keeping exact radius in the CSV.
    radial_metrics=[('delta_max_radius_m','Δ max radius (mm)',1000.),('delta_mean_radius_m','Δ mean radius (mm)',1000.),
                    ('delta_rms_radius_m','Δ RMS radius (mm)',1000.),('delta_radial_span_m','Δ radial span (mm)',1000.),
                    ('delta_net_angle_rad','Δ angular sweep (rad)',1.),('delta_escape_period_rot_minus_stand','Δ first escape (RF periods; both escaped)',1.)]
    mass_colors={50000.:'#1769aa',100000.:'#d55e00',150000.:'#6a3d9a',200000.:'#14804a'}
    rad_groups={}
    for row in pd:
      if abs(float(row['energy_eV']))>1e-15:continue
      for keym,_,scale in radial_metrics:
        if keym not in row:continue
        val=float(row[keym])*scale
        rad_groups.setdefault((float(row['mz_Th']),float(row['r0_mm']),keym),[]).append(val)
    radial_summary_metrics=[];rad_panels=[]
    for keym,label,scale in radial_metrics:
      pts=[]
      for (mz,r0,metric),vals in sorted(rad_groups.items()):
        if metric!=keym:continue
        arr=sorted(vals);n=len(arr);med=float(__import__('statistics').median(arr));lo=arr[max(0,int(.1*(n-1)))];hi=arr[min(n-1,int(.9*(n-1))) ]
        # Small mass-coded dodge reveals coincident radii; the CSV retains exact r0.
        dodge={50000.:-.012,100000.:-.004,150000.:.004,200000.:.012}[mz]
        pts.append({'x':r0+dodge,'y':med,'err_low':lo,'err_high':hi,'color':mass_colors[mz]})
        radial_summary_metrics.append({'mz_Th':mz,'r0_mm':r0,'metric':keym,'n_pairs':n,'median':med,'p10_empirical':lo,'p90_empirical':hi,
          'rotating_escape_count':sum(x['rotating_status']=='escaped' for x in pd if float(x['mz_Th'])==mz and abs(float(x['r0_mm'])-r0)<1e-9 and float(x['energy_eV'])==0),
          'standing_escape_count':sum(x['standing_status']=='escaped' for x in pd if float(x['mz_Th'])==mz and abs(float(x['r0_mm'])-r0)<1e-9 and float(x['energy_eV'])==0)})
      maxabs=max([max(abs(p['err_low']),abs(p['err_high'])) for p in pts]+[1e-9])*1.15 if pts else 1
      rad_panels.append({'title':label,'points':pts,'ylim':(-maxabs,maxabs),'yticks':[-maxabs,0,maxabs],
        'legend':[(f'{m/1000:g} kTh',c) for m,c in mass_colors.items()]})
    writecsv('paired_radial_dynamics_summary.csv',radial_summary_metrics)
    savefig('figure_6_paired_radial_dynamics','Paired radial-launch dynamics · rotating − standing; median and empirical P10–P90','Initial radius r₀ (mm)','Paired difference',rad_panels,(0,8),(-1,1),[0,2,4,6,8],[-1,0,1],pointsize=4,zero=True)
    # Energy threshold intervals using observed shell fractions only; no interpolated point estimates.
    thresholds=[]
    for m in (ROT,STA):
     for mz in masses:
      series=[]
      for e in sorted(set(float(r['energy_eV']) for r in rows)):
       g=[r for r in rows if r['model']==m and float(r['mz_Th'])==mz and abs(float(r['r0_m'])-.001)<1e-12 and float(r['energy_eV'])==e]
       if g:series.append((e,sum(x['status']=='retained' for x in g)/len(g),len(g)))
      for t in (.5,.9):
       crossing=[]
       for (e0,p0,_),(e1,p1,_) in zip(series,series[1:]):
        if (p0-t)*(p1-t)<0 or p0==t or p1==t:crossing.append((e0,e1))
       thresholds.append({'model':m,'mz_Th':mz,'threshold_retention':t,'bracket_intervals_eV':json.dumps(crossing),
         'nonmonotonic_observed':any(series[i+1][1]>series[i][1] for i in range(len(series)-1))})
    writecsv('energy_threshold_intervals.csv',thresholds)
    # Observed radius brackets (all retained to first non-unity adjacent sample); flag reversals.
    radial=[]
    for m in (ROT,STA):
     for mz in sorted(set(float(r['mz_Th']) for r in radial_summary)):
      s=sorted((x['r0_mm'],x['P_ret']) for x in radial_summary if x['model']==m and x['mz_Th']==mz)
      brackets=[(s[i][0],s[i+1][0]) for i in range(len(s)-1) if s[i][1]!=s[i+1][1]]
      radial.append({'model':m,'mz_Th':mz,'sampled_radius_fraction':json.dumps(s),'observed_transition_brackets_mm':json.dumps(brackets),
        'nonmonotonic_observed':any(s[i+1][1]>s[i][1] for i in range(len(s)-1))})
    writecsv('radial_transition_brackets.csv',radial)
    status={'valid_unique_jobs':len(rows),'journal_lines':len(attempts),'unresolved_attempts':sum(str(r.get('status','')).startswith('unresolved') for r in attempts),
      'paired_initial_conditions':len(pd),'all_retention_pairs_equal':sum(x['paired_status_delta']!=0 for x in pd)==0,
      'retention_summary_rows':len(summary),'notes':'Direct sample markers only; no interpolation or binary-outcome connecting lines.'}
    (P/'analysis_status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
    print(json.dumps(status))

if __name__=='__main__':main()

