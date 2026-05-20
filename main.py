"""
main.py
Point d'entrée principal — lance tout le pipeline :
  1. Génération des frames
  2. Encodage complet
  3. Décodage complet
  4. Évaluation + visualisation
  5. Analyse Fq (pour le rapport)
"""
import os
import numpy as np

from generate_frames import generate_frames
from preprocessing   import load_frames, preprocess_frames
from encoder         import encode_video
from decoder         import decode_video
from entropy_coding  import decode_from_bin
from evaluation      import evaluate, visualize_pipeline, compute_psnr
from preprocessing   import ycbcr_to_bgr, chroma_upsample
import cv2


def main():

    # ── Étape 0 : Générer les frames de test ────────────────────────────
    if not os.path.exists("frames") or len(os.listdir("frames")) == 0:
        generate_frames()
    else:
        print("✓ Frames déjà présentes dans 'frames/'")

    # ── Étape 1 : Encodage ──────────────────────────────────────────────
    encoded_frames, reconstructed_ycbcr = encode_video(
        frames_dir   = "frames",
        output_path  = "output/video.bin",
        gop_size     = 5,
        fq           = 1.0,
        mb_size      = 16,
        search_range = 8
    )

    # ── Étape 2 : Décodage ──────────────────────────────────────────────
    decode_video(
        input_path = "output/video.bin",
        output_dir = "reconstructed"
    )

    # ── Étape 3 : Évaluation ────────────────────────────────────────────
    frames_bgr      = load_frames("frames")
    compressed_size = os.path.getsize("output/video.bin")

    ratio, psnr_list, n_i, n_p = evaluate(
        frames_bgr, reconstructed_ycbcr, encoded_frames, compressed_size
    )

    # ── Étape 4 : Analyse Fq (pour le rapport) ──────────────────────────
    print("\nAnalyse Fq pour le rapport...")
    processed  = preprocess_frames(frames_bgr)
    fq_values_list   = [0.5, 1.0, 2.0, 3.0, 4.0]
    ratio_list       = []
    psnr_avg_list    = []

    for fq in fq_values_list:
        # Encoder avec ce Fq
        enc_frames, recon_ycbcr = encode_video(
            frames_dir   = "frames",
            output_path  = f"output/video_fq{fq}.bin",
            gop_size     = 5,
            fq           = fq,
            mb_size      = 16,
            search_range = 8
        )
        cs  = os.path.getsize(f"output/video_fq{fq}.bin")
        raw = frames_bgr[0].shape[0] * frames_bgr[0].shape[1] * 3 * len(frames_bgr)
        ratio_list.append(raw / cs)

        # PSNR moyen
        psnrs = []
        for orig, rec in zip(frames_bgr, recon_ycbcr):
            h, w         = rec['Y'].shape
            Cb_up, Cr_up = chroma_upsample(rec['Cb'], rec['Cr'], h, w)
            rec_bgr      = ycbcr_to_bgr(rec['Y'], Cb_up, Cr_up)
            psnrs.append(compute_psnr(orig, rec_bgr))
        psnr_avg_list.append(np.mean(psnrs))
        print(f"  Fq={fq:.1f} → ratio={raw/cs:.2f}×, PSNR={np.mean(psnrs):.1f} dB")

    # ── Étape 5 : Visualisation complète ────────────────────────────────
    print("\nGénération de la visualisation...")
    visualize_pipeline(
        frames_bgr, processed, encoded_frames, reconstructed_ycbcr,
        psnr_list, ratio,
        fq_values=(fq_values_list, ratio_list, psnr_avg_list)
    )

    print("\n" + "=" * 55)
    print("PROJET TERMINÉ ✓")
    print("=" * 55)
    print("  Fichiers générés :")
    print("  output/video.bin                 ← vidéo compressée")
    print("  output/pipeline_visualization.png← figure rapport")
    print("  reconstructed/frame_*.png        ← frames reconstruites")


if __name__ == "__main__":
    main()
