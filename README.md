# Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs

**LOWN (Look Only When Needed)** compares text-only predictions with the most recent image-conditioned predictions to decide when to process visual tokens and how many tokens to unmask, without additional training.

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
