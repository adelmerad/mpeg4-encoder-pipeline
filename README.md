# Simplified MPEG-4 Video Encoder Pipeline

**M1 Informatique — IL | USTHB | Module Multimédia | 2025/2026**

---

## Description

This project implements a simplified but complete MPEG-4-like video encoder pipeline in Python.  
It takes a folder of sequential image frames as input and produces a compressed binary file (`.bin`)  
that can be decoded back into the original frames.

The pipeline covers five stages:

| Part | Component | Description |
|------|-----------|-------------|
| 1 | Pre-processing | BGR → YCbCr conversion + 4:2:0 chroma subsampling |
| 2 | Intra-frame coding (I-frames) | DCT-based spatial compression per block |
| 3 | Inter-frame coding (P-frames) | Motion estimation (MSE) + residual DCT coding |
| 4 | Entropy coding | Lossless compression with zlib → `.bin` file |
| 5 | Evaluation & Visualisation | PSNR metrics + full pipeline matplotlib figure |

---

## Project Structure

```
mpeg4-encoder/
├── generate_frames.py   # Generates synthetic test frames
├── preprocessing.py     # Part 1 : BGR→YCbCr, chroma subsampling
├── intra_coding.py      # Part 2 : DCT, quantisation, I-frame encode/decode
├── inter_coding.py      # Part 3 : block matching (MSE), residual, P-frame encode/decode
├── entropy_coding.py    # Part 4 : pickle serialisation + zlib compression
├── encoder.py           # Full encoding pipeline
├── decoder.py           # Full decoding pipeline
├── evaluation.py        # Part 5 : PSNR, metrics, matplotlib visualisation
├── main.py              # Entry point : runs all stages + Fq analysis
├── output/
│   └── video.bin        # Compressed output file
└── reconstructed/       # Reconstructed frames after decoding
```

---

## Requirements

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Generate test frames (if not already present)
```bash
python generate_frames.py
```

### 2. Run the full pipeline
```bash
python main.py
```

This single command will:
- Generate 20 synthetic test frames (192×144 px)
- Encode the video (GOP=5, Fq=1.0)
- Decode and reconstruct all frames
- Compute PSNR and compression metrics
- Analyse Fq values (0.5 → 4.0)
- Generate the pipeline visualisation figure

### 3. Encode only
```bash
python encoder.py
```

### 4. Decode only
```bash
python decoder.py
```

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gop_size` | 5 | Frames per Group of Pictures (1 I-frame + N P-frames) |
| `fq` | 1.0 | Quantisation factor — higher = more compression, less quality |
| `mb_size` | 16 | Macroblock size for motion estimation (16×16 pixels) |
| `search_range` | 8 | Motion search window (±8 pixels) |

---

## Quantisation Matrix

Following the formula from the course (TP7) :

```
Q(i, j) = 1 + (1 + i + j) × Fq
```

---

## Motion Estimation

Block matching uses **MSE** (Mean Squared Error) as similarity measure, as defined in TP8 :

```
MSE = (1/N²) × Σ Σ (Bcurrent(i,j) − Bprevious(i,j))²
```

The block with the lowest MSE gives the motion vector (dx, dy).

---

## Results (default parameters : Fq=1.0, GOP=5)

| Metric | Value |
|--------|-------|
| Raw size | 1 620.0 Ko |
| Compressed size | 67.2 Ko |
| Compression ratio | × 24.09 |
| Mean PSNR | 33.22 dB |
| Min PSNR | 32.20 dB |
| Max PSNR | 34.64 dB |
| I-frames | 4 (20%) |
| P-frames | 16 (80%) |

### Compression ratio vs Fq

| Fq | Compression ratio | Mean PSNR |
|----|------------------|-----------|
| 0.5 | × 21.1 | 34.1 dB |
| 1.0 | × 24.1 | 33.2 dB |
| 2.0 | × 26.8 | 31.1 dB |
| 3.0 | × 28.8 | 29.1 dB |
| 4.0 | × 29.7 | 27.3 dB |

---

## Output Files

After running `main.py` :

```
output/
├── video.bin                    ← compressed video bitstream
├── video_fq0.5.bin              ← compressed at Fq=0.5
├── video_fq1.0.bin              ← compressed at Fq=1.0
├── video_fq2.0.bin              ← compressed at Fq=2.0
├── video_fq3.0.bin              ← compressed at Fq=3.0
├── video_fq4.0.bin              ← compressed at Fq=4.0
├── pipeline_visualization.png   ← full pipeline figure
└── fq_analysis.png              ← compression ratio vs Fq plot

reconstructed/
└── frame_000.png ... frame_019.png   ← decoded frames
```

---

