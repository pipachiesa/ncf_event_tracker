"""Re-proyecta un tracking ya generado con la calibracion nueva.

Por frame:
  1. registrar contra el frame de referencia de DOS formas: con features de
     toda la imagen y con features SOLO DEL CESPED. La segunda es la correcta
     en principio (la tribuna y los carteles no estan en el plano del piso),
     pero el cesped tiene poca textura, asi que a veces sale peor.
  2. quedarse con la que mejor alinea las lineas proyectadas contra las
     PINTADAS de ese mismo frame -- un criterio que no necesita ground truth.
  3. si aun asi el frame quedo feo, refinar localmente contra sus lineas.
No necesita GPU: el CSV ya trae las coordenadas de imagen de cada deteccion.
"""
import numpy as np, cv2, csv, os, sys, time, json
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/felipechiesa/Desktop/FootballTrackingDataGeneration/data_cleanup')
from refine_homography import line_mask, cost_map, alignment, _samples, unexplained, segments, CTRL
from scipy.optimize import minimize

VID=os.path.expanduser("~/football_data/matches/clip-test/video.mp4")
IN=os.path.expanduser("~/football_data/matches/clip-test-new/tracking_vit.csv")
OUT=os.path.expanduser("~/football_data/matches/clip-test-new/tracking_recal.csv")
REF=761; STRIDE=2; BAD=30.0; EVERY=5
H_REF=np.load(f"{SC}/H_fifa_761_final.npy")
S=_samples(); SEGS=segments()
sift=cv2.SIFT_create(nfeatures=2500); bf=cv2.BFMatcher()

def field(fr):
    hsv=cv2.cvtColor(fr,cv2.COLOR_BGR2HSV); h,s,v=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    g=((h>25)&(h<95)&(s>60)).astype(np.uint8)
    g=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((25,25),np.uint8))
    g=cv2.morphologyEx(g,cv2.MORPH_OPEN,np.ones((15,15),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(g)
    if n<2: return None
    return ((lab==(1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))))*255).astype(np.uint8)

cap=cv2.VideoCapture(VID); cap.set(cv2.CAP_PROP_POS_FRAMES,(REF-1)*STRIDE)
ok,ref=cap.read(); cap.release()
gray=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
KA,DA=sift.detectAndCompute(gray,None)
KM,DM=sift.detectAndCompute(gray,field(ref))
print(f"referencia: {len(KA)} features totales, {len(KM)} en el cesped",flush=True)

def reg(gray,mask,KR,DR):
    kt,dt=sift.detectAndCompute(gray,mask)
    if dt is None or len(kt)<10: return None
    m=bf.knnMatch(dt,DR,k=2)
    good=[a for a,b in m if len(b.__class__.__mro__)>0 and a.distance<0.75*b.distance]
    if len(good)<12: return None
    src=np.float32([kt[a.queryIdx].pt for a in good]).reshape(-1,1,2)
    dst=np.float32([KR[a.trainIdx].pt for a in good]).reshape(-1,1,2)
    A,mk=cv2.findHomography(src,dst,cv2.RANSAC,3.0)
    if A is None or mk is None or int(mk.sum())<12: return None
    return A

def refine(H0,sc):
    def unpack(v):
        M,_=cv2.findHomography(v.reshape(-1,2).astype(np.float32),CTRL); return M
    def obj(v):
        M=unpack(v); return 1e6 if M is None else sc(M)
    v=cv2.perspectiveTransform(CTRL.reshape(-1,1,2),np.linalg.inv(H0)).reshape(-1,2).ravel().astype(np.float64)
    v=minimize(obj,v,method="Powell",options={"xtol":0.3,"ftol":5e-3,"maxiter":250,"maxfev":250}).x
    M=unpack(v)
    return M if M is not None and sc(M)<sc(H0) else H0

Hs={}; stats={"full":0,"cesped":0,"refinado":0,"sin":0}; scores=[]
cap=cv2.VideoCapture(VID); vf=-1; t0=time.time(); n=0
while True:
    if not cap.grab(): break
    vf+=1
    if vf%(STRIDE*EVERY): continue
    ok,fr=cap.retrieve()
    if not ok: break
    k=vf//STRIDE+1; n+=1
    g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
    mk=line_mask(fr); cost=cost_map(mk); ys,xs=np.nonzero(mk)
    if len(ys)<300: stats["sin"]+=1; continue
    r=np.random.default_rng(0).choice(len(ys),size=min(500,len(ys)),replace=False)
    real=np.stack([xs[r],ys[r]],1).astype(np.float32)
    def sc(H):
        f,nn=alignment(H,cost,S)
        return 1e6 if nn<250 else f+unexplained(H,real,SEGS)
    best=None
    for tag,(mask,KR,DR) in (("full",(None,KA,DA)),("cesped",(field(fr),KM,DM))):
        A=reg(g,mask,KR,DR)
        if A is None: continue
        H=H_REF@A; s=sc(H)
        if best is None or s<best[0]: best=(s,H,tag)
    if best is None: stats["sin"]+=1; continue
    s,H,tag=best; stats[tag]+=1
    if s>BAD:
        H2=refine(H,sc); s2=sc(H2)
        if s2<s: H,s=H2,s2; stats["refinado"]+=1
    Hs[k]=H; scores.append(s)
    if n%100==0: print(f"  {n} frames, {(time.time()-t0)/60:.1f} min",flush=True)
cap.release()
sc_a=np.array(scores)
print(f"\nregistrados {len(Hs)} de {n}   eleccion: {stats}",flush=True)
print(f"costo de alineacion por frame: p50 {np.median(sc_a):.1f}  p90 {np.percentile(sc_a,90):.1f}  max {sc_a.max():.1f}",flush=True)
np.savez_compressed(f"{SC}/H_recal.npz",frames=np.array(sorted(Hs)),H=np.array([Hs[k] for k in sorted(Hs)]))

# Los frames intermedios: interpolar las posiciones en IMAGEN de los cuatro
# puntos de control. La camara se mueve ~0,25 px por frame, asi que el error de
# interpolar sobre 5 frames es despreciable, y evita 5x de trabajo.
ks=sorted(Hs)
ctrl={k:cv2.perspectiveTransform(CTRL.reshape(-1,1,2),np.linalg.inv(Hs[k])).reshape(-1,2) for k in ks}
full={}
for i,k in enumerate(ks):
    full[k]=Hs[k]
    if i+1<len(ks):
        k2=ks[i+1]
        for kk in range(k+1,k2):
            t=(kk-k)/(k2-k)
            c=(1-t)*ctrl[k]+t*ctrl[k2]
            M,_=cv2.findHomography(c.astype(np.float32),CTRL)
            if M is not None: full[kk]=M
Hs=full
print(f"tras interpolar: {len(Hs)} frames con mapa",flush=True)

rows=list(csv.DictReader(open(IN))); out=[]; miss=0
for r in rows:
    k=int(r["Frame"])
    if k not in Hs:
        miss+=1; r["X_Pitch"]="0"; r["Y_Pitch"]="0"; out.append(r); continue
    x=(float(r["X1"])+float(r["X2"]))/2
    y=float(r["Y2"]) if r["Object"]!="ball" else (float(r["Y1"])+float(r["Y2"]))/2
    p=cv2.perspectiveTransform(np.array([[[x,y]]],dtype=np.float32),Hs[k].astype(np.float32)).reshape(2)
    r["X_Pitch"]=f"{p[0]:.1f}"; r["Y_Pitch"]=f"{p[1]:.1f}"; out.append(r)
with open(OUT,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
meta=json.load(open(os.path.expanduser("~/football_data/matches/clip-test-new/tracking.meta.json")))
meta["pitch"]="105x68 reglamentaria (ver data_cleanup/pitch_config.py)"
meta["calibracion"]="re-proyectado el 20-ago desde el mapa de lineas del mosaico"
json.dump(meta,open(OUT.replace(".csv",".meta.json"),"w"),indent=2)
print(f"escrito {OUT}  (filas sin mapa: {miss})",flush=True)
