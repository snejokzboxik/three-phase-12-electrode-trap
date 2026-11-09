import csv,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
P=Path(__file__).resolve().parent
meta=json.loads((P/'trajectory_geometry_pair_corrected.json').read_text(encoding='utf8'))
rows=list(csv.DictReader((P/'trajectory_geometry_pair_corrected_timeseries.csv').open(encoding='utf8',newline='')))
series={}
for model in ('rotating','standing_equal_peak'):
 series[model]=[r for r in rows if r['model']==model]
colors={'rotating':'#1769aa','standing_equal_peak':'#d55e00'}
max_t=max(float(r['rf_period']) for r in rows)*1.08
W,H=1450,900;im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);sm=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14);tiny=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
d.text((W/2,28),'Full-RF paired trajectory · new diagnostic launch at 100 kTh',font=font,fill='black',anchor='mm')
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="{W/2}" y="34" text-anchor="middle" font-family="Arial" font-size="19">Full-RF paired trajectory · new diagnostic launch at 100 kTh</text>']
for i,m in enumerate(('rotating','standing_equal_peak')):
 x=820+i*290;res=next(r for r in meta['results'] if r['model']==m);lab=(f"{m} · escaped at {float(res['escape_period']):.4g} RF" if res['status']=='escaped' else f"{m} · retained to 300 RF");d.line((x,54,x+28,54),fill=colors[m],width=4);d.text((x+36,54),lab,font=sm,fill='black',anchor='lm');svg.append(f'<line x1="{x}" y1="54" x2="{x+28}" y2="54" stroke="{colors[m]}" stroke-width="4"/><text x="{x+36}" y="59" font-family="Arial" font-size="14">{lab}</text>')
panels=[('x(t)','x_mm',(-8,8)),('y(t)','y_mm',(-8,8)),('r(t)','r_mm',(0,8)),('x–y trajectory','xy',(-8,8))]
note='Same launch: r₀=7.75 mm, azimuth=30°, RF phase=45°, zero speed. Time panels zoom to the aperture events.'
d.text((W/2,76),note,font=sm,fill='black',anchor='mm');svg.append(f'<text x="{W/2}" y="81" text-anchor="middle" font-family="Arial" font-size="14">{note}</text>')
L,R,T,B=105,40,108,70;gx,gy=100,90;pw=(W-L-R-gx)/2;ph=(H-T-B-gy)/2
for i,(title,key,ylim) in enumerate(panels):
 rr,cc=divmod(i,2);x0=L+cc*(pw+gx);y0=T+rr*(ph+gy);x1=x0+pw;y1=y0+ph
 d.text(((x0+x1)/2,y0-19),title,font=sm,fill='black',anchor='mm');svg.append(f'<text x="{(x0+x1)/2}" y="{y0-14}" text-anchor="middle" font-family="Arial" font-size="14">{title}</text>')
 if key=='xy':xlim=(-8,8);xticks=(-8,0,8);xlabel='x (mm)'
 else:xlim=(0,max_t);xticks=(0,max_t/3,2*max_t/3,max_t);xlabel='RF periods'
 for tick in xticks:
  xx=x0+(tick-xlim[0])/(xlim[1]-xlim[0])*pw;d.line((xx,y0,xx,y1),fill='#e6e6e6');d.text((xx,y1+15),f'{tick:.4g}',font=tiny,fill='black',anchor='mm');svg.append(f'<line x1="{xx:.2f}" y1="{y0:.2f}" x2="{xx:.2f}" y2="{y1:.2f}" stroke="#e6e6e6"/><text x="{xx:.2f}" y="{y1+30:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{tick:.4g}</text>')
 for tick in (ylim[0],(ylim[0]+ylim[1])/2,ylim[1]):
  yy=y1-(tick-ylim[0])/(ylim[1]-ylim[0])*ph;d.line((x0,yy,x1,yy),fill='#e6e6e6');d.text((x0-12,yy),f'{tick:g}',font=tiny,fill='black',anchor='rm');svg.append(f'<line x1="{x0:.2f}" y1="{yy:.2f}" x2="{x1:.2f}" y2="{yy:.2f}" stroke="#e6e6e6"/><text x="{x0-10:.2f}" y="{yy+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:g}</text>')
 d.line((x0,y0,x0,y1),fill='black',width=2);d.line((x0,y1,x1,y1),fill='black',width=2)
 d.text(((x0+x1)/2,y1+42),xlabel,font=sm,fill='black',anchor='mm');d.text((x0-65,(y0+y1)/2),'mm',font=sm,fill='black',anchor='mm')
 svg.append(f'<text x="{(x0+x1)/2:.2f}" y="{y1+54:.2f}" text-anchor="middle" font-family="Arial" font-size="14">{xlabel}</text><text x="{x0-65:.2f}" y="{(y0+y1)/2:.2f}" text-anchor="middle" font-family="Arial" font-size="14">mm</text>')
 for m,data in series.items():
  if key=='xy':xs=[float(r['x_mm']) for r in data];ys=[float(r['y_mm']) for r in data]
  else:xs=[float(r['rf_period']) for r in data];ys=[float(r[key]) for r in data]
  pts=[(x0+(x-xlim[0])/(xlim[1]-xlim[0])*pw,y1-(y-ylim[0])/(ylim[1]-ylim[0])*ph) for x,y in zip(xs,ys)]
  if len(pts)>1:d.line(pts,fill=colors[m],width=3)
  for px,py in (pts[0],pts[-1]):d.ellipse((px-4,py-4,px+4,py+4),fill=colors[m])
  coords=' '.join(f'{x:.2f},{y:.2f}' for x,y in pts);svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[m]}" stroke-width="2.5"/>')
svg.append('</svg>')
(P/'figure_7_trajectory_geometry_corrected.svg').write_text('\n'.join(svg),encoding='utf-8')
im.save(P/'figure_7_trajectory_geometry_corrected.png',dpi=(240,240));bio=BytesIO();im.save(bio,format='PNG');bio.seek(0);c=canvas.Canvas(str(P/'figure_7_trajectory_geometry_corrected.pdf'),pagesize=(W,H));c.drawImage(ImageReader(bio),0,0,width=W,height=H);c.showPage();c.save()
print('plotted saved time-series without reintegration')
