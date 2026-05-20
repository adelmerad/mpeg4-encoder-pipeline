"""
decoder.py
Pipeline complet de décodage.
Lit le fichier .bin et reconstruit les frames images.

Utilisation :
  python decoder.py
"""
import numpy as np
import cv2
import os

from entropy_coding import decode_from_bin
from intra_coding   import decode_iframe
from inter_coding   import decode_pframe
from preprocessing  import ycbcr_to_bgr, chroma_upsample


def decode_video(input_path="output/video.bin", output_dir="reconstructed"):
    """
    Décode un fichier .bin et sauvegarde les frames reconstruites.
    """
    print("=" * 55)
    print("DÉCODEUR MPEG-4 SIMPLIFIÉ")
    print("=" * 55)

    os.makedirs(output_dir, exist_ok=True)

    # ── Étape 1 : Lire le fichier .bin ──────────────────────────────────
    encoded_frames = decode_from_bin(input_path)

    # ── Étape 2 : Décoder frame par frame ───────────────────────────────
    reconstructed = []
    print(f"\nDécodage de {len(encoded_frames)} frames...")

    for idx, encoded in enumerate(encoded_frames):

        if encoded['type'] == 'I':
            recon = decode_iframe(encoded)
            print(f"  Frame {idx:02d} → I-frame décodée")

        else:
            ref   = reconstructed[-1]
            recon = decode_pframe(encoded, ref)
            print(f"  Frame {idx:02d} → P-frame décodée")

        reconstructed.append(recon)

        # ── Étape 3 : Reconvertir YCbCr → BGR et sauvegarder ────────────
        h, w         = recon['Y'].shape
        Cb_up, Cr_up = chroma_upsample(recon['Cb'], recon['Cr'], h, w)
        bgr          = ycbcr_to_bgr(recon['Y'], Cb_up, Cr_up)

        path = os.path.join(output_dir, f"frame_{idx:03d}.png")
        cv2.imwrite(path, bgr)

    print(f"\n✓ {len(reconstructed)} frames sauvegardées dans '{output_dir}/'")
    return reconstructed


if __name__ == "__main__":
    decode_video()
