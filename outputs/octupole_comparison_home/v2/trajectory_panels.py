import csv,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
P=Path(__file__).resolve().parent;SRC=P.parent/'v1'/'representative_trajectories.csv'
FONT=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);SM=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
COL={'rotating':'#1769aa','standing':'#d55e00','dx':'#1769aa','dy':'#d55e00'}
def main():
 rows=list(csv.DictReader(SRC.open(encoding='utf8')));W,H=1650,760;L,R,T,B=95,38,92,100;gap=62;pw=(W-L-R-2*gap)//3;ph=H-T-B
 im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>']
 d.text((W/2,35),'Matched v1 full-RF trajectories · 100 kTh, r₀=1 mm, 300 RF periods',fill='black',font=FONT,anchor='mm');svg.append('<text x="825" y="42" text-anchor="middle" font-size="22">Matched v1 full-RF trajectories · 100 kTh, r₀=1 mm, 300 RF periods</text>')
 def panel(j,title,xlab,ylab,xlim,ylim,xt,yt,series):
  x0=L+j*(pw+gap);y0=T;x1=x0+pw;y1=y0+ph
  def X(x):return x0+(x-xlim[0])/(xlim[1]-xlim[0])*pw
  def Y(y):return y1-(y-ylim[0])/(ylim[1]-ylim[0])*ph
  d.text(((x0+x1)/2,y0-23),title,fill='black',font=FONT,anchor='mm');svg.append(f'<text x="{(x0+x1)/2}" y="{y0-18}" text-anchor="middle" font-size="18">{title}</text>')
  for v in xt:
   xx=X(v);d.line((xx,y0,xx,y1),fill='#e4e4e4');d.text((xx,y1+15),f'{v:g}',font=SM,fill='black',anchor='ma');svg.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="#e4e4e4"/><text x="{xx:.1f}" y="{y1+30}" text-anchor="middle" font-size="13">{v:g}</text>')
  for v in yt:
   yy=Y(v);d.line((x0,yy,x1,yy),fill='#e4e4e4');d.text((x0-9,yy),f'{v:g}',font=SM,fill='black',anchor='rm');svg.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e4e4e4"/><text x="{x0-9}" y="{yy+4:.1f}" text-anchor="end" font-size="13">{v:g}</text>')
  d.rectangle((x0,y0,x1,y1),outline='black',width=2);svg.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="black"/>')
  for label,xs,ys,color in series:
   pts=[(X(x),Y(y)) for x,y in zip(xs,ys) if xlim[0]<=x<=xlim[1] and ylim[0]<=y<=ylim[1]]
   if len(pts)>1:
    d.line(pts,fill=color,width=2)
    svg.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in pts[::max(1,len(pts)//1800)])}" fill="none" stroke="{color}" stroke-width="2"/>')
   lx=x0+12;ly=y0+18+22*len([1 for s in series if s[0]<label]);d.line((lx,ly,lx+26,ly),fill=color,width=3);d.text((lx+33,ly),label,fill=color,font=SM,anchor='lm')
   svg.append(f'<text x="{lx+34}" y="{ly+4}" fill="{color}" font-size="13">{label}</text>')
  d.text(((x0+x1)/2,H-19),xlab,fill='black',font=SM,anchor='mm');lab=Image.new('RGBA',(170,24),(255,255,255,0));ImageDraw.Draw(lab).text((85,12),ylab,font=SM,fill='black',anchor='mm');lab=lab.rotate(90,expand=True);im.paste(lab,(int(x0-65-lab.width/2),int((y0+y1)/2-lab.height/2)),lab)
  svg.append(f'<text x="{(x0+x1)/2}" y="{H-10}" text-anchor="middle" font-size="14">{xlab}</text><text x="{x0-64}" y="{(y0+y1)/2}" text-anchor="middle" transform="rotate(-90 {x0-64} {(y0+y1)/2})" font-size="14">{ylab}</text>')
 def get(e,m):return sorted((r for r in rows if r['model']==m and float(r['energy_eV'])==e),key=lambda r:float(r['rf_period']))
 def arr(g,col):
  if col=='x_offset':return [(float(r['x_mm'])-1)*1000 for r in g]
  if col=='y_nm':return [float(r['y_mm'])*1e6 for r in g]
  if col=='y_mm':return [float(r['y_mm']) for r in g]
  if col=='x_um':return [float(r['x_mm'])*1000 for r in g]
  return [float(r['rf_period']) for r in g]
 # Rest: zoom into the tiny actual transverse motion; finite energy: show full escape orbit.
 panel(0,'Rest launch: trajectory zoom','x − 1 mm (µm)','y (nm)',(-3,.5),(-5,5),[-3,-2,-1,0],[ -4,-2,0,2,4],[(m,arr(get(0.,m),'x_offset'),arr(get(0.,m),'y_nm'),COL[m]) for m in ('rotating','standing')])
 panel(1,'Finite launch: 10 meV orbit','x − 1 mm (µm)','y (mm)',(-3.5,.5),(0,8.2),[-3,-2,-1,0],[0,2,4,6,8],[(m,arr(get(.01,m),'x_offset'),arr(get(.01,m),'y_mm'),COL[m]) for m in ('rotating','standing')])
 a,b=get(.01,'rotating'),get(.01,'standing');n=min(len(a),len(b));tt=[float(r['rf_period']) for r in a[:n]];dx=[(float(a[i]['x_mm'])-float(b[i]['x_mm']))*1000 for i in range(n)];dy=[(float(a[i]['y_mm'])-float(b[i]['y_mm']))*1000 for i in range(n)]
 panel(2,'Pairwise model separation: 10 meV','RF periods','Coordinate difference (µm)',(0,190),(-1.6,1.6),[0,50,100,150],[ -1,0,1],[(r'Δx rotating−standing',tt,dx,COL['dx']),(r'Δy rotating−standing',tt,dy,COL['dy'])])
 svg.append('</svg>');(P/'figure_5_trajectory_geometry.svg').write_text('\n'.join(svg),encoding='utf8');im.save(P/'figure_5_trajectory_geometry.png',dpi=(240,240));bio=BytesIO();im.save(bio,format='PNG');bio.seek(0);c=canvas.Canvas(str(P/'figure_5_trajectory_geometry.pdf'),pagesize=(W,H));c.drawImage(ImageReader(bio),0,0,width=W,height=H);c.showPage();c.save()
if __name__=='__main__':main()
