"""
intra_coding.py
Partie 2 — Intra-frame Coding (I-frames)

Matrice de quantification selon la formule du TP7 (Mme DAHMANE) :
  Q(i,j) = 1 + (1+i+j) * Fq
"""
import numpy as np
from scipy.fft import dctn, idctn


def get_q_matrix(fq=1.0):
    """
    Matrice de quantification 8x8 selon la formule du cours :
    Q(i,j) = 1 + (1+i+j) * Fq
    """
    Q = np.fromfunction(lambda i, j: 1 + (1 + i + j) * fq, (8, 8), dtype=int)
    return Q.astype(np.float32)


def encode_channel(channel, q_matrix):
    H, W   = channel.shape
    H_pad  = int(np.ceil(H / 8) * 8)
    W_pad  = int(np.ceil(W / 8) * 8)
    padded = np.zeros((H_pad, W_pad), dtype=np.float32)
    padded[:H, :W] = channel

    n_h    = H_pad // 8
    n_w    = W_pad // 8
    blocks = np.zeros((n_h, n_w, 8, 8), dtype=np.int16)

    for i in range(n_h):
        for j in range(n_w):
            block        = padded[i*8:(i+1)*8, j*8:(j+1)*8] - 128.0
            dct_b        = dctn(block, norm='ortho')
            blocks[i, j] = np.floor(dct_b / q_matrix).astype(np.int16)

    return blocks, (H, W)


def decode_channel(blocks, q_matrix, original_shape):
    H_orig, W_orig = original_shape
    n_h, n_w       = blocks.shape[:2]
    recon          = np.zeros((n_h * 8, n_w * 8), dtype=np.float32)

    for i in range(n_h):
        for j in range(n_w):
            dct_b = blocks[i, j].astype(np.float32) * q_matrix
            block = idctn(dct_b, norm='ortho') + 128.0
            recon[i*8:(i+1)*8, j*8:(j+1)*8] = block

    return np.clip(recon[:H_orig, :W_orig], 0, 255).astype(np.uint8)


def encode_iframe(frame_ycbcr, fq=1.0):
    q = get_q_matrix(fq)

    Y_b,  Y_s  = encode_channel(frame_ycbcr['Y'],  q)
    Cb_b, Cb_s = encode_channel(frame_ycbcr['Cb'], q)
    Cr_b, Cr_s = encode_channel(frame_ycbcr['Cr'], q)

    return {
        'type': 'I', 'fq': fq,
        'Y_blocks':  Y_b,  'Y_shape':  Y_s,
        'Cb_blocks': Cb_b, 'Cb_shape': Cb_s,
        'Cr_blocks': Cr_b, 'Cr_shape': Cr_s,
    }


def decode_iframe(encoded):
    fq = encoded['fq']
    q  = get_q_matrix(fq)

    Y  = decode_channel(encoded['Y_blocks'],  q, encoded['Y_shape'])
    Cb = decode_channel(encoded['Cb_blocks'], q, encoded['Cb_shape'])
    Cr = decode_channel(encoded['Cr_blocks'], q, encoded['Cr_shape'])

    return {'Y': Y, 'Cb': Cb, 'Cr': Cr}
