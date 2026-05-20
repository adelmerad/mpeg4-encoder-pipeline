"""
inter_coding.py
Partie 3 — Inter-frame Coding (P-frames)

Block matching avec MSE selon le TP8 (Mme DAHMANE) :
  MSE = (1/N²) × Σ Σ (Bcourant(i,j) − Bprecedent(i,j))²
"""
import numpy as np
from scipy.fft import dctn, idctn
from intra_coding import get_q_matrix


def block_matching(current_Y, ref_Y, mb_size=16, search_range=8):
    """
    Pour chaque macroblock 16x16 de current_Y, cherche le bloc le plus
    similaire dans ref_Y avec la mesure MSE (TP8).
    Retourne les vecteurs de mouvement (dy, dx) pour chaque bloc.
    """
    H, W           = current_Y.shape
    n_h            = H // mb_size
    n_w            = W // mb_size
    motion_vectors = np.zeros((n_h, n_w, 2), dtype=np.int16)

    for i in range(n_h):
        for j in range(n_w):
            y0        = i * mb_size
            x0        = j * mb_size
            cur_block = current_Y[y0:y0+mb_size, x0:x0+mb_size].astype(np.float32)

            best_mse  = float('inf')
            best_dy, best_dx = 0, 0

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    ry = y0 + dy
                    rx = x0 + dx

                    if ry < 0 or rx < 0 or ry + mb_size > H or rx + mb_size > W:
                        continue

                    ref_block = ref_Y[ry:ry+mb_size, rx:rx+mb_size].astype(np.float32)

                    # MSE selon formule TP8
                    mse = np.mean((cur_block - ref_block) ** 2)

                    if mse < best_mse:
                        best_mse = mse
                        best_dy  = dy
                        best_dx  = dx

            motion_vectors[i, j] = [best_dy, best_dx]

    return motion_vectors


def compute_residual(current_Y, ref_Y, motion_vectors, mb_size=16):
    """
    Calcule le résidu = bloc courant - meilleur bloc (TP8).
    résidu = bloc courant - meilleur bloc
    """
    H, W     = current_Y.shape
    n_h      = H // mb_size
    n_w      = W // mb_size
    residual = np.zeros((H, W), dtype=np.float32)

    for i in range(n_h):
        for j in range(n_w):
            y0 = i * mb_size
            x0 = j * mb_size
            dy, dx = motion_vectors[i, j]

            ry = int(max(0, min(y0 + dy, H - mb_size)))
            rx = int(max(0, min(x0 + dx, W - mb_size)))

            cur_block  = current_Y[y0:y0+mb_size, x0:x0+mb_size].astype(np.float32)
            pred_block = ref_Y[ry:ry+mb_size, rx:rx+mb_size].astype(np.float32)
            residual[y0:y0+mb_size, x0:x0+mb_size] = cur_block - pred_block

    return residual


def encode_residual(residual, fq=1.0):
    q      = get_q_matrix(fq)
    H, W   = residual.shape
    H_pad  = int(np.ceil(H / 8) * 8)
    W_pad  = int(np.ceil(W / 8) * 8)
    padded = np.zeros((H_pad, W_pad), dtype=np.float32)
    padded[:H, :W] = residual

    n_h    = H_pad // 8
    n_w    = W_pad // 8
    blocks = np.zeros((n_h, n_w, 8, 8), dtype=np.int16)

    for i in range(n_h):
        for j in range(n_w):
            block        = padded[i*8:(i+1)*8, j*8:(j+1)*8]
            dct_b        = dctn(block, norm='ortho')
            blocks[i, j] = np.floor(dct_b / q).astype(np.int16)

    return blocks, (H, W)


def decode_residual(blocks, original_shape, fq=1.0):
    q              = get_q_matrix(fq)
    H_orig, W_orig = original_shape
    n_h, n_w       = blocks.shape[:2]
    recon          = np.zeros((n_h * 8, n_w * 8), dtype=np.float32)

    for i in range(n_h):
        for j in range(n_w):
            dct_b = blocks[i, j].astype(np.float32) * q
            recon[i*8:(i+1)*8, j*8:(j+1)*8] = idctn(dct_b, norm='ortho')

    return recon[:H_orig, :W_orig]


def reconstruct_pframe_Y(ref_Y, motion_vectors, residual_decoded, mb_size=16):
    H, W  = ref_Y.shape
    n_h   = H // mb_size
    n_w   = W // mb_size
    recon = np.zeros((H, W), dtype=np.float32)

    for i in range(n_h):
        for j in range(n_w):
            y0 = i * mb_size
            x0 = j * mb_size
            dy, dx = motion_vectors[i, j]

            ry = int(max(0, min(y0 + dy, H - mb_size)))
            rx = int(max(0, min(x0 + dx, W - mb_size)))

            recon[y0:y0+mb_size, x0:x0+mb_size] = \
                ref_Y[ry:ry+mb_size, rx:rx+mb_size].astype(np.float32)

    recon += residual_decoded
    return np.clip(recon, 0, 255).astype(np.uint8)


def encode_pframe(current_ycbcr, ref_ycbcr_recon, fq=1.0, mb_size=16, search_range=8):
    from intra_coding import encode_channel

    cur_Y = current_ycbcr['Y']
    ref_Y = ref_ycbcr_recon['Y']

    mv                    = block_matching(cur_Y, ref_Y, mb_size, search_range)
    residual              = compute_residual(cur_Y, ref_Y, mv, mb_size)
    res_blocks, res_shape = encode_residual(residual, fq)

    q        = get_q_matrix(fq)
    Cb_b, Cb_s = encode_channel(current_ycbcr['Cb'], q)
    Cr_b, Cr_s = encode_channel(current_ycbcr['Cr'], q)

    return {
        'type': 'P', 'fq': fq,
        'motion_vectors': mv,
        'res_blocks': res_blocks, 'res_shape': res_shape,
        'Cb_blocks': Cb_b, 'Cb_shape': Cb_s,
        'Cr_blocks': Cr_b, 'Cr_shape': Cr_s,
    }


def decode_pframe(encoded, ref_ycbcr_recon):
    from intra_coding import decode_channel

    fq    = encoded['fq']
    q     = get_q_matrix(fq)
    ref_Y = ref_ycbcr_recon['Y']
    mv    = encoded['motion_vectors']

    res_decoded = decode_residual(encoded['res_blocks'], encoded['res_shape'], fq)
    Y_recon     = reconstruct_pframe_Y(ref_Y, mv, res_decoded)

    Cb = decode_channel(encoded['Cb_blocks'], q, encoded['Cb_shape'])
    Cr = decode_channel(encoded['Cr_blocks'], q, encoded['Cr_shape'])

    return {'Y': Y_recon, 'Cb': Cb, 'Cr': Cr}
