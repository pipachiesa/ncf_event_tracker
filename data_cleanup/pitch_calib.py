"""Calibracion imagen->cancha con PnLCalib (modelo SOTA de registro de cancha).

Reemplaza al detector de keypoints viejo (`martinjolif`, 32 vertices en un parche
de 20 m, 26,9 m de error) por PnLCalib (keypoints + lineas + refinamiento,
0,56 m, automatico, cualquier vista). Ver el bloque PnLCalib en CLAUDE.md.

INSTALACION (una vez, donde corra main.py -- en Colab, dentro del notebook):
    git clone https://github.com/mguti97/PnLCalib.git
    # pesos SV_FT_WC14_kp y SV_FT_WC14_lines de los releases v1.0.0 a PnLCalib/weights/
    pip install -U ultralytics    # trae torch; PnLCalib usa torchvision
Y apuntar PNLCALIB_DIR a ese clon (o pasar pnl_dir).

Uso:
    cal = PnLCalibrator(pnl_dir="/content/PnLCalib", device="cuda")
    H = cal.homography(frame_bgr)      # matriz 3x3 imagen->cancha en CM, o None
"""

import os
import numpy as np


class _MatrixTransformer:
    """Envuelve una matriz imagen->cancha con la interfaz de ViewTransformer."""

    def __init__(self, H):
        self.m = np.asarray(H, dtype=np.float32)

    def transform_points(self, points):
        import cv2
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.m).reshape(-1, 2)


class PnLCalibrator:
    def __init__(self, pnl_dir=None, device="cpu",
                 kp_weights="SV_FT_WC14_kp", line_weights="SV_FT_WC14_lines",
                 kp_threshold=0.34, line_threshold=0.79):
        import sys
        import yaml
        import torch
        self.pnl_dir = pnl_dir or os.environ.get("PNLCALIB_DIR")
        if not self.pnl_dir or not os.path.isdir(self.pnl_dir):
            raise RuntimeError(
                "No encuentro PnLCalib. Cloná github.com/mguti97/PnLCalib y pasá "
                "pnl_dir=... o exportá PNLCALIB_DIR.")
        sys.path.insert(0, self.pnl_dir)
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils.utils_calib import FramebyFrameCalib
        import torchvision.transforms as T
        self._T = T
        self._FramebyFrameCalib = FramebyFrameCalib
        self.device = device
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold

        cfg = yaml.safe_load(open(os.path.join(self.pnl_dir, "config/hrnetv2_w48.yaml")))
        cfg_l = yaml.safe_load(open(os.path.join(self.pnl_dir, "config/hrnetv2_w48_l.yaml")))
        wdir = os.path.join(self.pnl_dir, "weights")
        self.m = get_cls_net(cfg)
        self.m.load_state_dict(torch.load(os.path.join(wdir, kp_weights), map_location=device))
        self.m.to(device).eval()
        self.ml = get_cls_net_l(cfg_l)
        self.ml.load_state_dict(torch.load(os.path.join(wdir, line_weights), map_location=device))
        self.ml.to(device).eval()
        self._resize = T.Resize((540, 960))

    def homography(self, frame_bgr):
        """Matriz 3x3 imagen->cancha en CM (105x68 -> 0..10500, 0..6800), o None."""
        import cv2
        import torch
        import torchvision.transforms.functional as f
        from PIL import Image
        from utils.utils_heatmap import (
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
            complete_keypoints, coords_to_dict)

        h_img, w_img = frame_bgr.shape[:2]
        cam = self._FramebyFrameCalib(iwidth=w_img, iheight=h_img, denormalize=True)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = self._resize(f.to_tensor(Image.fromarray(rgb)).float().unsqueeze(0)).to(self.device)
        with torch.no_grad():
            hm = self.m(t)
            hml = self.ml(t)
        kp = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1, :, :])
        lc = get_keypoints_from_heatmap_batch_maxpool_l(hml[:, :-1, :, :])
        kpd = coords_to_dict(kp, threshold=self.kp_threshold)
        ld = coords_to_dict(lc, threshold=self.line_threshold)
        kpd, ld = complete_keypoints(kpd[0], ld[0], w=960, h=540, normalize=True)
        cam.update(kpd, ld)
        fp = cam.heuristic_voting(refine_lines=True)
        if not fp:
            return None
        cp = fp["cam_params"]
        Q = np.array([[cp['x_focal_length'], 0, cp['principal_point'][0]],
                      [0, cp['y_focal_length'], cp['principal_point'][1]],
                      [0, 0, 1]])
        It = np.eye(4)[:-1]
        It[:, -1] = -np.array(cp['position_meters'])
        P = Q @ (np.array(cp['rotation_matrix']) @ It)
        # cancha(cm, origen esquina) -> imagen. mundo_centrado = (X_cm/100-52.5, Y_cm/100-34)
        Hc2i = np.column_stack([P[:, 0] / 100.0, P[:, 1] / 100.0,
                                P[:, 3] - 52.5 * P[:, 0] - 34 * P[:, 1]])
        try:
            return np.linalg.inv(Hc2i)
        except np.linalg.LinAlgError:
            return None

    def transformer(self, frame_bgr):
        """Como homography() pero devuelve un objeto con .transform_points, o None."""
        H = self.homography(frame_bgr)
        return _MatrixTransformer(H) if H is not None else None
