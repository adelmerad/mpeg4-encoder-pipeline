"""
preprocessing.py
Partie 1 — Pré-traitement
  - Chargement des frames
  - Conversion BGR → YCbCr
  - Chroma subsampling 4:2:0
"""
import numpy as np
import cv2
import os


def load_frames(frames_dir="frames"):
    files = sorted([f for f in os.listdir(frames_dir)
                    if f.endswith(('.png', '.jpg', '.jpeg'))])
    frames = []
    for f in files:
        img = cv2.imread(os.path.join(frames_dir, f))
        if img is not None:
            frames.append(img)
    print(f"✓ {len(frames)} frames chargées")
    return frames


def bgr_to_ycbcr(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    Y  = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]
    return Y, Cb, Cr


def ycbcr_to_bgr(Y, Cb, Cr):
    ycrcb = np.stack([Y, Cr, Cb], axis=2).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def chroma_subsample(Cb, Cr):
    h, w = Cb.shape
    Cb_sub = cv2.resize(Cb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    Cr_sub = cv2.resize(Cr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return Cb_sub, Cr_sub


def chroma_upsample(Cb_sub, Cr_sub, target_h, target_w):
    Cb = cv2.resize(Cb_sub, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    Cr = cv2.resize(Cr_sub, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return Cb, Cr


def preprocess_frames(frames_bgr):
    processed = []
    for frame in frames_bgr:
        Y, Cb, Cr      = bgr_to_ycbcr(frame)
        Cb_sub, Cr_sub = chroma_subsample(Cb, Cr)
        processed.append({'Y': Y, 'Cb': Cb_sub, 'Cr': Cr_sub})

    h, w   = frames_bgr[0].shape[:2]
    avant  = w * h * 3
    apres  = w * h + 2 * (w // 2) * (h // 2)
    print(f"✓ Pré-traitement : {avant} → {apres} valeurs/frame (gain ×{avant/apres:.1f})")
    return processed
