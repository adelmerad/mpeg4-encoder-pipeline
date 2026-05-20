"""
evaluation.py
Partie 5 — Evaluation & Visualisation
  5a : PSNR, taux de compression, breakdown I/P frames
  5b : Figure matplotlib avec toutes les étapes du pipeline
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from preprocessing  import load_frames, preprocess_frames, ycbcr_to_bgr, chroma_upsample
from intra_coding   import encode_iframe, decode_iframe, get_q_matrix
from inter_coding   import encode_pframe, decode_pframe, block_matching
from scipy.fft      import dctn


# ── 5a : Métriques ──────────────────────────────────────────────────────────

def compute_psnr(original, reconstructed):
    orig  = original.astype(np.float32)
    recon = reconstructed.astype(np.float32)
    mse   = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def evaluate(frames_bgr, reconstructed_ycbcr, encoded_frames, compressed_size):
    print("\n" + "=" * 55)
    print("ÉVALUATION")
    print("=" * 55)

    n_frames   = len(frames_bgr)
    h, w       = frames_bgr[0].shape[:2]
    raw_size   = w * h * 3 * n_frames
    ratio      = raw_size / compressed_size
    n_iframes  = sum(1 for f in encoded_frames if f['type'] == 'I')
    n_pframes  = sum(1 for f in encoded_frames if f['type'] == 'P')

    print(f"  Frames totales   : {n_frames} ({n_iframes} I + {n_pframes} P)")
    print(f"  Taille brute     : {raw_size / 1024:.1f} Ko")
    print(f"  Taille compressée: {compressed_size / 1024:.1f} Ko")
    print(f"  Taux compression : ×{ratio:.2f}")

    psnr_list = []
    for orig_bgr, recon_ycbcr in zip(frames_bgr, reconstructed_ycbcr):
        rh, rw       = recon_ycbcr['Y'].shape
        Cb_up, Cr_up = chroma_upsample(recon_ycbcr['Cb'], recon_ycbcr['Cr'], rh, rw)
        recon_bgr    = ycbcr_to_bgr(recon_ycbcr['Y'], Cb_up, Cr_up)
        psnr_list.append(compute_psnr(orig_bgr, recon_bgr))

    print(f"\n  PSNR moyen       : {np.mean(psnr_list):.2f} dB")
    print(f"  PSNR min         : {np.min(psnr_list):.2f} dB")
    print(f"  PSNR max         : {np.max(psnr_list):.2f} dB")
    print(f"  (>35 dB = bonne qualité visuelle)")

    return ratio, psnr_list, n_iframes, n_pframes


# ── 5b : Visualisation ──────────────────────────────────────────────────────

def visualize_pipeline(frames_bgr, processed, encoded_frames, reconstructed_ycbcr,
                       psnr_list, ratio, fq_values=None):

    fig = plt.figure(figsize=(20, 26))
    fig.suptitle("Pipeline MPEG-4 Simplifié — Visualisation Complète",
                 fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(6, 5, figure=fig, hspace=0.55, wspace=0.35)

    # ── 1. Frames originales ─────────────────────────────────────────────
    ax_t1 = fig.add_subplot(gs[0, :])
    ax_t1.axis('off')
    ax_t1.text(0.5, 0.5, "1 — Frames originales (sélection)",
               ha='center', va='center', fontsize=13, fontweight='bold',
               transform=ax_t1.transAxes)

    indices = [0, 4, 9, 14, 19]
    for k, idx in enumerate(indices):
        ax = fig.add_subplot(gs[1, k])
        rgb = cv2.cvtColor(frames_bgr[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ftype = encoded_frames[idx]['type']
        color = 'red' if ftype == 'I' else 'blue'
        ax.set_title(f"Frame {idx} [{ftype}]", fontsize=8, color=color)
        ax.axis('off')

    # ── 2. Canaux YCbCr ──────────────────────────────────────────────────
    ax_t2 = fig.add_subplot(gs[2, :])
    ax_t2.axis('off')
    ax_t2.text(0.5, 0.5, "2 — Canaux YCbCr de la Frame 0",
               ha='center', va='center', fontsize=13, fontweight='bold',
               transform=ax_t2.transAxes)

    frame0   = processed[0]
    orig_rgb = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2RGB)

    ax_orig = fig.add_subplot(gs[3, 0])
    ax_orig.imshow(orig_rgb)
    ax_orig.set_title("Original BGR", fontsize=9)
    ax_orig.axis('off')

    channels = [
        (frame0['Y'],  'Y — Luminance',     'gray'),
        (frame0['Cb'], 'Cb — Chroma bleu',  'Blues'),
        (frame0['Cr'], 'Cr — Chroma rouge', 'Reds'),
    ]
    for k, (ch, title, cmap) in enumerate(channels):
        ax = fig.add_subplot(gs[3, k + 1])
        ax.imshow(ch, cmap=cmap)
        ax.set_title(f"{title}\n{ch.shape[1]}×{ch.shape[0]}", fontsize=8)
        ax.axis('off')

    # ── 3. DCT & Quantification (bloc 8×8) ──────────────────────────────
    ax_t3 = fig.add_subplot(gs[4, :])
    ax_t3.axis('off')
    ax_t3.text(0.02, 0.5, "3 — DCT & Quantification  |  4 — Vecteurs de mouvement  |  5 — Résidus & Reconstruction",
               ha='left', va='center', fontsize=11, fontweight='bold',
               transform=ax_t3.transAxes)

    block_raw = frame0['Y'][16:24, 16:24].astype(np.float32)
    block_dct = dctn(block_raw - 128, norm='ortho')
    q_mat     = get_q_matrix(1.0)     
    block_qnt = np.round(block_dct / q_mat).astype(np.int16)

    labels = ["Pixels bruts", "Coefficients DCT", "Quantifiés (Fq=1)"]
    blocks = [block_raw, block_dct, block_qnt]
    fmts   = [".0f", ".0f", "d"]
    for k in range(3):
        ax = fig.add_subplot(gs[5, k])
        ax.imshow(blocks[k], cmap='RdBu_r', aspect='auto')
        ax.set_title(labels[k], fontsize=8)
        for ii in range(8):
            for jj in range(8):
                val = blocks[k][ii, jj]
                txt = f"{int(val)}" if fmts[k] == "d" else f"{val:.0f}"
                ax.text(jj, ii, txt, ha='center', va='center',
                        fontsize=4.5, color='black')
        ax.set_xticks([])
        ax.set_yticks([])

    # ── 4. Vecteurs de mouvement ─────────────────────────────────────────
    pframe_idx = next((i for i, f in enumerate(encoded_frames) if f['type'] == 'P'), 1)
    ax4 = fig.add_subplot(gs[5, 3])
    pf_rgb = cv2.cvtColor(frames_bgr[pframe_idx], cv2.COLOR_BGR2RGB)
    ax4.imshow(pf_rgb, alpha=0.75)
    mv   = encoded_frames[pframe_idx]['motion_vectors']
    mb   = 16
    n_h, n_w = mv.shape[:2]
    for i in range(n_h):
        for j in range(n_w):
            dy, dx = mv[i, j]
            y0 = i * mb + mb // 2
            x0 = j * mb + mb // 2
            if abs(dy) + abs(dx) > 0:
                ax4.annotate("", xy=(x0+dx, y0+dy), xytext=(x0, y0),
                             arrowprops=dict(arrowstyle="->", color='yellow', lw=1.2))
    ax4.set_title(f"MV — P-frame {pframe_idx}", fontsize=8)
    ax4.axis('off')

    # ── 5. Résidu visuel ─────────────────────────────────────────────────
    recon_f   = reconstructed_ycbcr[pframe_idx]
    h, w      = recon_f['Y'].shape
    Cb_up, Cr_up = chroma_upsample(recon_f['Cb'], recon_f['Cr'], h, w)
    recon_bgr = ycbcr_to_bgr(recon_f['Y'], Cb_up, Cr_up)
    residual  = np.abs(frames_bgr[pframe_idx].astype(np.float32) -
                       recon_bgr.astype(np.float32)).mean(axis=2).astype(np.uint8)

    ax5 = fig.add_subplot(gs[5, 4])
    ax5.imshow(residual, cmap='hot')
    ax5.set_title(f"Résidu |orig−recon|\nframe {pframe_idx}", fontsize=8)
    ax5.axis('off')

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/pipeline_visualization.png", dpi=120, bbox_inches='tight')
    print("\n✓ Visualisation sauvegardée → output/pipeline_visualization.png")

    # ── Graphique séparé : Fq analysis ──────────────────────────────────
    if fq_values is not None:
        fq_list, ratio_list, psnr_avg_list = fq_values
        fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
        fig2.suptitle("Analyse Fq — Compression vs Qualité", fontsize=14)

        ax_a.plot(fq_list, ratio_list, 'bo-', lw=2, ms=8)
        ax_a.set_xlabel("Facteur Fq")
        ax_a.set_ylabel("Taux de compression (×)")
        ax_a.set_title("Compression ratio vs Fq")
        ax_a.grid(True, alpha=0.3)
        for x, y in zip(fq_list, ratio_list):
            ax_a.annotate(f"×{y:.1f}", (x, y), textcoords="offset points",
                          xytext=(0, 8), ha='center', fontsize=9)

        ax_b.plot(fq_list, psnr_avg_list, 'ro-', lw=2, ms=8)
        ax_b.axhline(y=35, color='green', ls='--', label='Seuil qualité (35 dB)')
        ax_b.set_xlabel("Facteur Fq")
        ax_b.set_ylabel("PSNR moyen (dB)")
        ax_b.set_title("PSNR moyen vs Fq")
        ax_b.legend()
        ax_b.grid(True, alpha=0.3)
        for x, y in zip(fq_list, psnr_avg_list):
            ax_b.annotate(f"{y:.1f} dB", (x, y), textcoords="offset points",
                          xytext=(0, 8), ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig("output/fq_analysis.png", dpi=120, bbox_inches='tight')
        print("✓ Graphique Fq sauvegardé  → output/fq_analysis.png")

    plt.close('all')
