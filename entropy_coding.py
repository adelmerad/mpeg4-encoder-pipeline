"""
entropy_coding.py
Partie 4 — Entropy Coding
  - Sérialiser toutes les données encodées
  - Compression lossless avec zlib
  - Écriture dans un fichier .bin
  - Décodeur correspondant
"""
import numpy as np
import pickle
import zlib
import os


def encode_to_bin(encoded_frames, output_path="output/video.bin"):
    """
    Sérialise et compresse toutes les frames encodées dans un fichier .bin.

    Étapes :
      1. pickle  → convertit les objets Python en bytes
      2. zlib    → compression lossless (comme gzip) sur ces bytes
      3. écriture dans le fichier .bin
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sérialisation avec pickle
    raw_bytes = pickle.dumps(encoded_frames)

    # Compression avec zlib (niveau 9 = compression maximale)
    compressed = zlib.compress(raw_bytes, level=9)

    # Écriture
    with open(output_path, 'wb') as f:
        f.write(compressed)

    original_size   = sum(
        f['Y_blocks'].nbytes + f['Cb_blocks'].nbytes + f['Cr_blocks'].nbytes
        if f['type'] == 'I'
        else f['res_blocks'].nbytes + f['Cb_blocks'].nbytes + f['Cr_blocks'].nbytes
        for f in encoded_frames
    )
    compressed_size = os.path.getsize(output_path)

    print(f"✓ Fichier binaire écrit → {output_path}")
    print(f"  Taille compressée : {compressed_size / 1024:.1f} Ko")
    print(f"  Taille coefficients : {original_size / 1024:.1f} Ko")

    return compressed_size


def decode_from_bin(input_path="output/video.bin"):
    """
    Lit le fichier .bin et décompresse pour retrouver les frames encodées.
    """
    with open(input_path, 'rb') as f:
        compressed = f.read()

    raw_bytes      = zlib.decompress(compressed)
    encoded_frames = pickle.loads(raw_bytes)

    print(f"✓ Fichier binaire lu → {input_path}")
    print(f"  {len(encoded_frames)} frames décodées")

    return encoded_frames
