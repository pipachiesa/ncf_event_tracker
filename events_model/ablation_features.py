"""Ablacion de features: cuanto del AUC viene de las features GEOMETRICAS (que
la calibracion arregla) vs las no-geometricas. Sobre el tracking VIEJO (coords
rotas). Si las no-geometricas solas ya dan el mismo AUC, las geometricas rotas
no aportaban -> arreglarlas daria senal NUEVA. Si dan mucho menos, las
geometricas aportaban aun rotas -> arregladas aportan mas."""
import sys, os, csv, time
sys.path.insert(0,'events_model')
import features as feat
from train import build_rows, load_label_blocks, to_xy, prf
from sklearn.ensemble import HistGradientBoostingClassifier
TRK=os.path.expanduser("~/football_data/matches/spain-france/tracking.csv")
LABELS=["events_model/dataset/spain-france_labeled.csv",
        "events_model/dataset/spain-france_10m_18m_proposed_groundtruth.csv",
        "events_model/dataset/spain-france_26m_34m_V2_groundtruth.csv",
        "events_model/dataset/spain-france_64m_72m_proposed_groundtruth.csv"]
GEO={"net_disp_m","path_len_m","straightness","mean_speed_ms","start_x","start_y",
     "end_x","end_y","dist_goal_start_m","dist_goal_end_m","delta_goal_dist_m",
     "lateral_disp_m","goalward","opp_dist_release_m","opp_dist_receive_m",
     "density_5m_release","density_10m_release","density_5m_receive","density_10m_receive",
     "cam_x","cam_y","dist_penalty_spot_m","release_dir_cos","release_speed_delta",
     "release_speed_out","ball_static_s"}
NOT={"match_id","label","start_frame","end_frame"}
t=time.time()
blocks=load_label_blocks(LABELS)
rows=build_rows(TRK,blocks,"PASS",reviewed=None)
print(f"filas: {len(rows)}  ({sum(r['y'] for r in rows)} PASS)  [{time.time()-t:.0f}s]",flush=True)
allcols=[c for c in feat.FIELDNAMES if c not in NOT]
def auc_cv(cols):
    probs=[]
    for name,_lo,_hi,_l in blocks:
        tr=[r for r in rows if r["block"]!=name]; te=[r for r in rows if r["block"]==name]
        if not te or not tr or sum(r["y"] for r in tr)<5: continue
        Xtr,ytr=to_xy(tr,cols); Xte,yte=to_xy(te,cols)
        clf=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.08,random_state=0)
        clf.fit(Xtr,ytr)
        pte=clf.predict_proba(Xte)[:,1]; probs.extend(zip(pte,yte))
    pos=[q for q,t in probs if t==1]; neg=[q for q,t in probs if t==0]
    w=sum(1 for a in pos for b in neg if a>b); ti=sum(1 for a in pos for b in neg if a==b)
    return (w+0.5*ti)/(len(pos)*len(neg))
geo=[c for c in allcols if c in GEO]; nongeo=[c for c in allcols if c not in GEO]
print(f"\n{'conjunto de features':<34}{'n':>4}{'AUC':>8}",flush=True)
print(f"{'TODAS (coords viejas/rotas)':<34}{len(allcols):>4}{auc_cv(allcols):>8.3f}",flush=True)
print(f"{'solo NO-geometricas':<34}{len(nongeo):>4}{auc_cv(nongeo):>8.3f}",flush=True)
print(f"{'solo geometricas (rotas)':<34}{len(geo):>4}{auc_cv(geo):>8.3f}",flush=True)
