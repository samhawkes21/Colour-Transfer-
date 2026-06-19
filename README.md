# Optimal Transport-Based Colour Transfer

This project implements an image colour transfer algorithm using entropic optimal transport (Sinkhorn iterations). The method transfers the colour distribution of a target image onto a source image while preserving spatial structure using pixel-level feature embeddings.

---

## Overview

This project:

- Represents each pixel as a joint feature of **RGB colour + spatial coordinates**
- Solves an **entropic optimal transport problem**
- Uses the **Sinkhorn algorithm (log-domain)** for numerical stability
- Performs **barycentric projection** to transfer colour statistics
- Refines mapping using **nearest-neighbour assignment**

---

## Method Summary

1. **Feature Construction**
   - Each pixel is represented as:
     ```
     [R, G, B, w_pos * x, w_pos * y]
     ```
   - This encodes both colour and spatial structure.

2. **Optimal Transport**
   - Compute cost matrix using squared Euclidean distance.
   - Solve entropic OT problem:
     ```
     min_P ⟨P, C⟩ + ε KL(P)
     ```
   - Using Sinkhorn iterations in log space.

3. **Colour Transfer**
   - Compute barycentric projection from target to source.
   - Map colours back using nearest-neighbour assignment.

---

## Project Structure

```text
optimal-transport-colour-transfer/
│
├── src/
│   ├── main.py                     # Entry point (runs experiments)
│   ├── optimal_transport.py        # Sinkhorn + cost computation
│   ├── pixels.py                   # Feature extraction + NN mapping
│   ├── images.py                   # Image I/O utilities
│   ├── utilities.py                # Optimal transport utility functions
│
├── data/
│   ├── input/                      # Input images
│   └── output/                     # Generated results
│
├── Figure_1.png                    # Example of results from project
├── Optimal Transport Thesis.pdf    # My bachelor's thesis, which provides a thorough explanation of optimal transport
├── Theory Overview.pdf             # Brief explanation of the theory used in this project
├── requirements.txt
└── README.md
