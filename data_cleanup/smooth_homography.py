"""El mapa por frame salta: la eleccion entre las dos formas de registrar
cambia entre refrescos vecinos y produce discontinuidades. Se suaviza la
trayectoria de los cuatro puntos de control (mediana movil + media movil) y se
re-deriva la homografia desde ahi."""
import numpy as np, cv2, csv, os, sys, json
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/felipechiesa/Desktop/FootballTrackingDataGeneration/data_cleanup')
from refine_homography import CTRL
d=np.load(f"{SC}/H_recal.npz"); ks=list(d["frames"]); Hs=list(d["H"])
C=np.array([cv2.perspectiveTransform(CTRL.reshape(-1,1,2),np.linalg.inv(H)).reshape(-1,2) for H in Hs])
print(f"{len(ks)} refrescos, puntos de control {C.shape}")
def medfilt(a,w=5):
    out=a.copy(); h=w//2
    for i in range(len(a)):
        lo,hi=max(0,i-h),min(len(a),i+h+1)
        out[i]=np.median(a[lo:hi],axis=0)
    return out
def movavg(a,w=3):
    out=a.copy(); h=w//2
    for i in range(len(a)):
        lo,hi=max(0,i-h),min(len(a),i+h+1)
        out[i]=a[lo:hi].mean(axis=0)
    return out
jump=np.linalg.norm(np.diff(C,axis=0),axis=2).mean(1)
print(f"salto de los puntos de control entre refrescos: p50 {np.median(jump):.1f} px  p99 {np.percentile(jump,99):.0f} px")
Cs=movavg(medfilt(C,5),3)
jump2=np.linalg.norm(np.diff(Cs,axis=0),axis=2).mean(1)
print(f"                                    suavizado: p50 {np.median(jump2):.1f} px  p99 {np.percentile(jump2,99):.0f} px")
S={}
for k,c in zip(ks,Cs):
    M,_=cv2.findHomography(c.astype(np.float32),CTRL)
    if M is not None: S[int(k)]=M
ordk=sorted(S)
ctrl={k:cv2.perspectiveTransform(CTRL.reshape(-1,1,2),np.linalg.inv(S[k])).reshape(-1,2) for k in ordk}
full={}
for i,k in enumerate(ordk):
    full[k]=S[k]
    if i+1<len(ordk):
        k2=ordk[i+1]
        for kk in range(k+1,k2):
            t=(kk-k)/(k2-k); c=(1-t)*ctrl[k]+t*ctrl[k2]
            M,_=cv2.findHomography(c.astype(np.float32),CTRL)
            if M is not None: full[kk]=M
IN=os.path.expanduser("~/football_data/matches/clip-test-new/tracking_vit.csv")
OUT=os.path.expanduser("~/football_data/matches/clip-test-new/tracking_recal.csv")
rows=list(csv.DictReader(open(IN))); out=[]; miss=0
for r in rows:
    k=int(r["Frame"])
    if k not in full:
        miss+=1; r["X_Pitch"]="0"; r["Y_Pitch"]="0"; out.append(r); continue
    # Caja en cero = NO hubo deteccion en ese frame. El CSV marca eso con
    # X_Pitch=Y_Pitch=0, y hay que conservarlo: proyectar (0,0) da un punto de
    # cancha cualquiera y convierte "sin pelota" en "pelota a -25 m".
    if (float(r["X1"])==0 and float(r["Y1"])==0
            and float(r["X2"])==0 and float(r["Y2"])==0):
        r["X_Pitch"]="0"; r["Y_Pitch"]="0"; out.append(r); continue
    # BOTTOM_CENTER para todo, igual que main.py. Usar el centro de la caja de
    # la pelota la manda al infinito cuando esta alta en la imagen: unos pocos
    # pixeles cerca del horizonte son cientos de metros en cancha.
    x=(float(r["X1"])+float(r["X2"]))/2; y=float(r["Y2"])
    H=full[k]
    den=H[2,0]*x+H[2,1]*y+H[2,2]
    if den<=1e-6:            # detras del horizonte: no hay punto de piso
        miss+=1; r["X_Pitch"]="0"; r["Y_Pitch"]="0"; out.append(r); continue
    px=(H[0,0]*x+H[0,1]*y+H[0,2])/den; py=(H[1,0]*x+H[1,1]*y+H[1,2])/den
    r["X_Pitch"]=f"{px:.1f}"; r["Y_Pitch"]=f"{py:.1f}"; out.append(r)
with open(OUT,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
print(f"escrito {OUT} (sin mapa: {miss})")
