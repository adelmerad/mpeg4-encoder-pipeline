"""
generate_frames.py
Génère 20 frames PNG synthétiques simulant une courte vidéo.
"""
import numpy as np
import cv2
import os


def generate_frames(output_dir="frames", n_frames=20, width=192, height=144):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Génération de {n_frames} frames ({width}x{height})...")

    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for x in range(width):
            val = int(50 + (x / width) * 80 + i * 1.5) % 180
            frame[:, x] = [val, val + 30, 60]

        sq_x = int((i / n_frames) * (width - 40))
        cv2.rectangle(frame, (sq_x, 20), (sq_x + 35, 55), (0, 0, 200), -1)

        cx = width // 2
        cy = int(20 + (i / n_frames) * (height - 60))
        cv2.circle(frame, (cx, cy), 18, (200, 80, 0), -1)

        cv2.putText(frame, f"Frame {i:02d}", (5, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}.png"), frame)

    print(f"✓ {n_frames} frames sauvegardées dans '{output_dir}/'")
    print(f"  Taille totale brute : {width * height * 3 * n_frames / 1024:.1f} Ko")


if __name__ == "__main__":
    generate_frames()
