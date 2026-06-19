"""
Executable script for batch color transfer experiments.

This main module runs multiple instances of the global optimal transport
color transfer pipeline using predefined source/target image pairs and
parameter configurations.

Each experiment applies the `colour_transfer` function with fixed OT
hyperparameters and varying spatial weighting (`w_pos`) or image pairs.
The results are saved as separate output images for comparison.
"""

from optimal_transport import colour_transfer

if __name__ == "__main__":

    runs = [
        ("Field_Colour.jpg", "Orange_Purple.jpg", "Output_1.jpg", 0.5),
        ("Field_Colour.jpg", "Orange_Purple.jpg", "Output_2.jpg", 1.0),
        ("Field_Colour.jpg", "Field_Mono.jpg", "Output_3.jpg", 0.5),
        ("Field_Mono.jpg", "Field_Colour.jpg", "Output_4.jpg", 0.5),
        ("Orange_Purple.jpg", "Field_Colour.jpg", "Output_5.jpg", 0.5),
    ]

    for src, tgt, out, w_pos in runs:
        colour_transfer(
            src, tgt, out,
            eps=10.0,
            sinkhorn_iters=100,
            sinkhorn_tol=1e-6,
            check_every=25,
            w_pos=w_pos,
            ns=6000,
            nt=6000,
            seed=0,
            cost_chunk=2048,
            nn_chunk=8192
        )