"""Run the same image and prompt with original LLaDA-V and/or LOWN."""

import argparse
import json
from pathlib import Path
import time


def decoder_flops(config, tokens):
    hidden = config.hidden_size
    kv_hidden = hidden * config.num_key_value_heads // config.num_attention_heads
    projections = 4 * tokens * hidden * (hidden + kv_hidden)
    attention = 4 * tokens * tokens * hidden
    feedforward = 6 * tokens * hidden * config.intermediate_size
    return config.num_hidden_layers * (projections + attention + feedforward)


def head_flops(tokens, hidden, vocabulary):
    return 2 * tokens * hidden * vocabulary


class FlopsCounter:
    """Observe sequence/head lengths without changing model computation.

    Two FLOPs per multiply-add. Excludes vision encoding, the multimodal
    projector, normalization, softmax and other elementwise operations.
    """

    def __init__(self, model):
        self.model = model
        self.forwards = []
        self.head_tokens = []

    def _decoder(self, module, args, kwargs):
        embeddings = kwargs["inputs_embeds"]
        assert embeddings.shape[0] == 1
        self.forwards.append((embeddings.shape[1], kwargs.get("position_ids") is not None))

    def _head(self, module, args):
        self.head_tokens.append(args[0].numel() // module.in_features)

    def __enter__(self):
        self.handles = [
            self.model.get_model().register_forward_pre_hook(self._decoder, with_kwargs=True),
            self.model.lm_head.register_forward_pre_hook(self._head),
        ]
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()

    def result(self):
        if not self.forwards or len(self.forwards) != len(self.head_tokens):
            raise RuntimeError("Decoder and output-head call counts do not match")
        decoder = sum(decoder_flops(self.model.config, n) for n, _ in self.forwards)
        head = sum(head_flops(n, self.model.lm_head.in_features, self.model.lm_head.out_features)
                   for n in self.head_tokens)
        return {
            "full_forwards": sum(not text for _, text in self.forwards),
            "text_only_forwards": sum(text for _, text in self.forwards),
            "decoder_flops": decoder,
            "lm_head_flops": head,
            "total_flops": decoder + head,
            "tflops": (decoder + head) / 1e12,
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="assets/bear.jpg")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--method", choices=("both", "baseline", "lown"), default="both")
    parser.add_argument("--model", default="GSAI-ML/LLaDA-V")
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output", default="outputs/result.json")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= args.length:
        parser.error("Require 1 <= --steps <= --length")
    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    return args


def load_model(checkpoint):
    import torch
    from transformers import AutoTokenizer
    from llava.model.language_model.llava_llada import LlavaLLaDAConfig, LlavaLLaDAModelLM

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    config = LlavaLLaDAConfig.from_pretrained(checkpoint)
    config.use_cache = False
    model = LlavaLLaDAModelLM.from_pretrained(
        checkpoint, config=config, torch_dtype=torch.float16,
        device_map="cuda:0", low_cpu_mem_usage=True, attn_implementation="sdpa",
    ).eval()
    model.resize_token_embeddings(len(tokenizer))
    vision = model.get_vision_tower()
    if not vision.is_loaded:
        vision.load_model(device_map="cuda:0")
    vision.to(device="cuda:0", dtype=torch.float16)
    return tokenizer, model, vision.image_processor


def prepare_inputs(tokenizer, model, processor, image, question):
    import torch
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    messages = [
        {"role": "system", "content": (
            "You are a helpful language and vision assistant. "
            "You are able to understand the visual content that the user provides, "
            "and assist the user with a variety of tasks using natural language."
        )},
        {"role": "user", "content": "<image>\n" + question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    images = [tensor.to("cuda:0", dtype=torch.float16)
              for tensor in process_images([image], processor, model.config)]
    return {"inputs": tokens[None].to("cuda:0"), "images": images, "image_sizes": [image.size]}


def main():
    import torch
    from PIL import Image
    from lown import apply_lown

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required. Select one with --gpu in evaluate.sh.")
    if not Path(args.image).is_file():
        raise SystemExit(f"Image not found: {args.image}")
    torch.manual_seed(0)
    image = Image.open(args.image).convert("RGB")
    tokenizer, model, processor = load_model(args.model)
    inputs = prepare_inputs(tokenizer, model, processor, image, args.prompt)
    methods = ("baseline", "lown") if args.method == "both" else (args.method,)
    result = {
        "image": Path(args.image).name,
        "prompt": args.prompt,
        "generation_length": args.length,
        "steps": args.steps,
        "dtype": "float16",
        "attention": "sdpa",
        "seed": 0,
        "gpu": torch.cuda.get_device_name(0),
        "flops_scope": "dense decoder and executed LM-head matmuls; 2 FLOPs/MAC; excludes vision encoder/projector and elementwise ops",
        "results": {},
    }
    for method in methods:
        if method == "lown":
            apply_lown(model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.no_grad(), FlopsCounter(model) as counter:
            tokens = model.generate(
                **inputs, gen_length=args.length, block_length=args.length, steps=args.steps,
                tokenizer=tokenizer, stopping_criteria=["<|eot_id|>"],
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if bool(tokens.eq(126336).any()):
            raise RuntimeError(f"{method} returned an unresolved MASK token")
        text = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
        measured = counter.result()
        measured.update(text=text, output_tokens=tokens.shape[1], seconds=round(seconds, 3),
                        peak_memory_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3))
        result["results"][method] = measured
        print(f"\n{method.upper()}: {text}\nFLOPs: {measured['tflops']:.3f} TFLOPs", flush=True)
    if len(methods) == 2:
        ratio = result["results"]["lown"]["total_flops"] / result["results"]["baseline"]["total_flops"]
        result["lown_flops_ratio"] = ratio
        print(f"LOWN / baseline FLOPs: {ratio:.3f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
