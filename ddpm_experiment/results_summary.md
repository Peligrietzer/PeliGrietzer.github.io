# Results Summary: Diffusion Manifold Projection Experiment

## Motivation

There is a known result for idealized autoencoders: if you train an autoencoder *f* on a large dataset D_m, then generate reconstructions f(D_m), a new autoencoder trained on only *n* of those reconstructions (n << m) achieves performance close to the original — i.e. f(f(D_m)_n) ≈ f(D_m) — while f(D_n) trained on *n* raw samples is much worse. Intuitively, the reconstructed samples live on the learned manifold, so they are more "information-dense" for learning the compression schema.

This experiment tests whether an analogous phenomenon holds for diffusion models (DDPMs).

## Method

**Dataset**: Fashion-MNIST (60,000 training images, 10,000 test images).

**Teacher**: A lightweight U-Net DDPM (base channels=64, channel multipliers=[1,2,2], 2 residual blocks per resolution, attention at the lowest resolution, sinusoidal time embeddings, ~2.5M parameters) trained on 54,000 images with 6,000 held out for validation. Linear noise schedule, 1000 timesteps, AdamW with lr=2e-4, EMA decay=0.9999. Trained with early stopping (patience=10 epochs).

**Diffusion "reconstruction"** (manifold projection): For each input image, add noise to an intermediate timestep t_proj, then denoise back to t=0 using the teacher. This is the diffusion analogue of autoencoder reconstruction — partial noising + denoising projects the input toward the learned score function's implicit manifold.

**Experimental conditions**: For each n ∈ {500, 1000, 2000, 5000, 10000, 20000}, we train three student DDPMs (same architecture as teacher, from scratch, fixed 15,000 gradient steps):
- **student_raw**: trained on *n* raw Fashion-MNIST samples
- **student_proj**: trained on *n* projected (reconstructed) samples at t_proj=200
- **student_gen**: trained on *n* purely generated samples from the teacher

**Evaluation**: FID (Fréchet Inception Distance) computed from 10,000 generated samples vs. the full Fashion-MNIST test set, using clean-fid.

## Results

*[To be filled after running the experiment]*

| n | FID (raw) | FID (proj) | FID (gen) | Teacher FID |
|---|-----------|------------|-----------|-------------|
| 500 | | | | |
| 1000 | | | | |
| 2000 | | | | |
| 5000 | | | | |
| 10000 | | | | |
| 20000 | | | | |

![FID Comparison](results/fid_comparison.png)

## Discussion

### Choice of t_proj

The projection timestep t_proj controls how strongly the teacher's manifold shapes the reconstructed samples:

- **t_proj too low** (e.g., 10–50): The image is barely perturbed and recovered almost identically. The "reconstructed" samples are essentially raw data, providing no manifold-projection benefit.
- **t_proj too high** (e.g., 800–900): So much noise is added that the original image is nearly obliterated. Denoising from here is essentially unconditional generation, losing the anchoring to real inputs.
- **t_proj ≈ 200** (our default): At t=200 with a linear schedule from β=1e-4 to β=0.02, the cumulative noise parameter ᾱ_200 ≈ 0.80, meaning about 20% of the signal energy has been replaced by noise. This is enough for the teacher's score function to meaningfully reshape the image toward its learned manifold, while preserving the overall structure and class identity of the original. This is analogous to an autoencoder reconstruction that preserves content while projecting onto the learned representation space.

The optimal t_proj likely depends on teacher quality. A well-trained teacher with a faithful manifold can tolerate higher t_proj; a weaker teacher benefits from staying closer to the data.

### Expected Pattern

Based on the autoencoder analogy and the practical realities of imperfect teacher training:

1. **Very small n (500)**: The advantage of projected samples may be weak or absent, because the student has too few samples to distinguish manifold structure from noise artifacts in the teacher's imperfect manifold.

2. **Medium-small n (1000–5000)**: This is the predicted sweet spot. Here n is large enough for the student to pick up the manifold structure from projected samples, but small enough that raw samples are too sparse to learn the distribution well on their own. We expect student_proj to meaningfully outperform student_raw.

3. **Large n (10000–20000)**: The gap should narrow and eventually vanish, since raw samples become sufficient on their own.

4. **student_gen**: May show a similar pattern but possibly weaker or noisier, since purely generated samples lack the anchoring to real inputs. However, they live entirely on the teacher's implicit manifold, which could be advantageous.

### Interpretation

If manifold-projected samples consistently provide better sample efficiency at medium-small n, this supports the thesis that diffusion models — like autoencoders — learn an implicit manifold whose samples are more information-dense representations of the underlying distribution's structure.

The shape of the advantage curve (where it appears and disappears) is itself informative about the quality of the teacher's learned manifold and the degree to which the partial-noising-denoising procedure succeeds as a manifold projection operator.
