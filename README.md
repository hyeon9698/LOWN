# Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs

[Paper (arXiv:2610.36671)](https://arxiv.org/abs/2610.36671) | [Project page](https://hyeon9698.github.io/LOWN/) | [Video (1 min)](https://hyeon9698.github.io/LOWN/#video) | [🤗 HF Papers](https://huggingface.co/papers/2610.36671)

## News

- **2026-10-30**: [#1 Paper of the day](https://huggingface.co/papers/date/2026-10-30) on 🤗 Hugging Face Daily Papers.
- **2026-10-27**: Paper released on [arXiv](https://arxiv.org/abs/2610.36671).

https://github.com/user-attachments/assets/947c55f1-cf23-46f8-9e4a-3c914347f90a

<p align="center">
  <img src="https://hyeon9698.github.io/LOWN/static/images/teaser.png" width="100%"
       alt="LOWN compared with token pruning, the token budget, and the relative cost and performance on ten benchmarks.">
</p>

**LOWN (Look Only When Needed)** compares text-only predictions with the most recent image-conditioned predictions to decide when to process visual tokens and how many tokens to unmask, without additional training. On ten benchmarks, LOWN retains 99.6% of LLaDA-V's score at 35.3% of the decoding FLOPs on average.

<details>
<summary>Abstract</summary>

Diffusion vision-language models (DVLMs) process their entire input at every denoising step. Since visual tokens make up 89–97% of this input on the benchmarks we study, most of the decoding computation is spent repeatedly processing the same image. Visual token pruning methods partially relieve this burden by removing part of the visual tokens, but they lose the removed information and still process the remaining image at every step. To measure how often the image is actually needed, we compare LLaDA-V's predictions with and without the image at every decoding step and find that removing the image changes a token being unmasked at only 22.8% of steps on average and at as few as 7.9%. We therefore propose LOWN, a decoding method that decides at every step whether to process the image and how many tokens to unmask. LOWN predicts the masked tokens without the image and processes the image only when this prediction disagrees with the last image-conditioned prediction. Agreement certifies the earlier prediction for reuse, and the number of certified tokens sets how many are unmasked in that step. We further observe that the need for the image differs across benchmarks, so LOWN's adaptive decision is needed rather than a fixed schedule. Without training or a tuned threshold, LOWN retains 99.6% of LLaDA-V's score on ten benchmarks at 35.3% of the decoding FLOPs on average. LOWN also generalizes across LLaDA-based DVLMs, composes orthogonally with visual token pruning, and runs under key-value caching for further efficiency gains.

</details>

## Run

Linux, Python 3.11, and one NVIDIA GPU with 32 GB+ VRAM for the default comparison.
The setup uses PyTorch with CUDA 12.6; a compatible NVIDIA driver is required.

```bash
git clone https://github.com/hyeon9698/LOWN.git
cd LOWN
bash setup.sh
bash evaluate.sh --gpu 0
```

Run these commands inside this folder. Setup creates a local `.venv`.
The first run downloads the public LLaDA-V and SigLIP2 weights.
Both baseline and LOWN run on the included image; outputs and FLOPs are saved to `outputs/result.json`.

To use your own image:

```bash
bash evaluate.sh --gpu 0 --method lown --image /path/to/image.jpg --prompt "Describe this image."
```

## COCO bear example

<img src="assets/bear.jpg" width="320" alt="COCO bear">

Prompt: **Describe this image in one sentence.**

**LLaDA-V**

> The image depicts a close-up of a brown bear's face, showcasing its thick fur, prominent nose, and expressive eyes, set against a grassy background.

**LOWN**

> The image depicts a close-up of a brown bear with a black nose, dark eyes, and a thick, dense fur, set against a grassy background.

| Method | TFLOPs | Relative FLOPs |
| --- | ---: | ---: |
| LLaDA-V | 1930.30 | 1.000 |
| LOWN | 812.16 | 0.421 |

Actual outputs from one NVIDIA H200, FP16, SDPA, 32 tokens and 32 steps; [full result](assets/bear.json).
FLOPs are analytical estimates from executed decoder and LM-head shapes (2 FLOPs per multiply-add), excluding vision encoding, projection, and elementwise operations. This is a single-image example.

Source: [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) ([license](llava/LICENSE)); reference: [D3ToM](https://github.com/bcmi/D3ToM-Diffusion-MLLM).
Image: “Brown Bear II” by [Derek John Lee](https://www.flickr.com/photos/derek-john-lee/9138147604/), MS COCO image 285, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) as recorded in COCO metadata.

## Citation

```bibtex
@article{cho2026lown,
  title   = {Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs},
  author  = {Cho, Donghyeon and Kim, Joowon and Shin, Seungho and Yang, Eunho},
  journal = {arXiv preprint arXiv:2610.36671},
  year    = {2026}
}
```
