"""
encoder.py
Pipeline complet d'encodage MPEG-4 simplifié.

Utilisation :
  python encoder.py
"""
import numpy as np
import cv2
import os

from preprocessing  import load_frames, preprocess_frames
from intra_coding   import encode_iframe
from inter_coding   import encode_pframe
from entropy_coding import encode_to_bin


def encode_video(frames_dir="frames", output_path="output/video.bin",
                 gop_size=5, fq=1.0, mb_size=16, search_range=8):
    """
    Encode une séquence de frames en fichier .bin.

    Paramètres :
      gop_size     : nombre de frames par GOP (1 I-frame + gop_size-1 P-frames)
      fq           : facteur de qualité (0.5=haute qualité, 4.0=très compressé)
      mb_size      : taille des macroblocks (16×16)
      search_range : fenêtre de recherche pour le block matching (±pixels)
    """
    print("=" * 55)
    print("ENCODEUR MPEG-4 SIMPLIFIÉ")
    print("=" * 55)

    # ── Étape 1 : Charger et pré-traiter ────────────────────────────────
    frames_bgr = load_frames(frames_dir)
    processed  = preprocess_frames(frames_bgr)
    n_frames   = len(processed)

    print(f"\nParamètres :")
    print(f"  GOP size     = {gop_size}")
    print(f"  Fq           = {fq}")
    print(f"  Macroblocks  = {mb_size}×{mb_size}")
    print(f"  Search range = ±{search_range} px")

    # ── Étape 2 : Encoder frame par frame ───────────────────────────────
    encoded_frames   = []
    reconstructed    = []  # frames reconstruites (nécessaires pour les P-frames)
    n_iframes        = 0
    n_pframes        = 0

    print(f"\nEncodage de {n_frames} frames...")

    for idx, frame in enumerate(processed):

        # Décider I-frame ou P-frame selon la position dans le GOP
        is_iframe = (idx % gop_size == 0)

        if is_iframe:
            # ── I-frame ─────────────────────────────────────────────────
            encoded = encode_iframe(frame, fq=fq)
            encoded_frames.append(encoded)

            # Reconstruire pour avoir la référence des P-frames suivantes
            from intra_coding import decode_iframe
            recon = decode_iframe(encoded)
            reconstructed.append(recon)
            n_iframes += 1
            print(f"  Frame {idx:02d} → I-frame")

        else:
            # ── P-frame ─────────────────────────────────────────────────
            ref = reconstructed[-1]  # dernière frame reconstruite = référence
            encoded = encode_pframe(frame, ref, fq=fq,
                                    mb_size=mb_size,
                                    search_range=search_range)
            encoded_frames.append(encoded)

            # Reconstruire ce P-frame pour servir de référence au suivant
            from inter_coding import decode_pframe
            recon = decode_pframe(encoded, ref)
            reconstructed.append(recon)
            n_pframes += 1
            print(f"  Frame {idx:02d} → P-frame")

    print(f"\n  Total : {n_iframes} I-frames + {n_pframes} P-frames")

    # ── Étape 3 : Entropy coding → fichier .bin ─────────────────────────
    print(f"\nEntropy coding...")
    compressed_size = encode_to_bin(encoded_frames, output_path)

    # ── Calcul du taux de compression ───────────────────────────────────
    h, w        = frames_bgr[0].shape[:2]
    raw_size    = w * h * 3 * n_frames
    ratio       = raw_size / compressed_size

    print(f"\n{'=' * 55}")
    print(f"RÉSULTAT FINAL")
    print(f"{'=' * 55}")
    print(f"  Taille brute      : {raw_size / 1024:.1f} Ko")
    print(f"  Taille compressée : {compressed_size / 1024:.1f} Ko")
    print(f"  Taux compression  : ×{ratio:.2f}")
    print(f"{'=' * 55}")

    return encoded_frames, reconstructed


if __name__ == "__main__":
    encode_video(gop_size=5, fq=1.0)
