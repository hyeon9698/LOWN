"""LOWN: Look Only When Needed — minimal LLaDA-V decoding hook.

The last image-conditioned prediction is an anchor. A text-only forward
certifies an agreement prefix of that anchor; otherwise a fresh full forward
resolves disagreements.
"""

from types import MethodType

import torch
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


MASK_ID = 126336
SUPPRESS_TOKEN_IDS = (126081, 126080, 126346, 126347)


def apply_lown(model):
    """Replace generate on one loaded LLaDA-V instance; return that instance."""
    model.generate = MethodType(generate_lown, model)
    return model


def _predict(model, embeddings, length, position_ids=None):
    hidden = model.get_model()(
        inputs_embeds=embeddings, position_ids=position_ids, use_cache=False,
    )[0]
    logits = model.lm_head(hidden[:, -length:]).float()[0]
    logits[:, list(SUPPRESS_TOKEN_IDS)] = -torch.inf
    tokens = logits.argmax(dim=-1)
    scores = logits.double().softmax(dim=-1).gather(1, tokens[:, None])[:, 0]
    return tokens, scores


def _agreement_prefix(active, text_tokens, anchor_tokens, anchor_scores, committed):
    """Walk by anchor confidence; stop at the first mismatch or repetition."""
    order = active[torch.argsort(anchor_scores[active], descending=True, stable=True)]
    accepted = []
    staged = committed.tolist()
    for position in order.tolist():
        token = int(anchor_tokens[position])
        if int(text_tokens[position]) != token:
            break
        if any(
            0 <= neighbor < len(staged) and staged[neighbor] == token
            for neighbor in (position - 1, position + 1)
        ):
            break
        accepted.append(position)
        staged[position] = token
    return torch.tensor(accepted, dtype=torch.long, device=active.device)


def _stop_index(tokens, sequences):
    """Find a committed stop sequence, even if earlier positions remain masked."""
    values = tokens.tolist()
    end = len(values)
    for sequence in sequences:
        for start in range(len(values) - len(sequence) + 1):
            if values[start:start + len(sequence)] == sequence:
                end = min(end, start)
                break
    return end


@torch.no_grad()
def generate_lown(
    model,
    inputs,
    images,
    image_sizes=None,
    *,
    gen_length=128,
    steps=128,
    block_length=None,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    tokenizer=None,
    stopping_criteria=None,
):
    """Generate response IDs [1, length] for one unpadded image/prompt pair.

    Uses greedy, single-block decoding without KV caching or CFG. Stop strings
    are excluded from the result; all positions before a stop are completed.
    """
    if any(type(value) is not int for value in (steps, gen_length)):
        raise ValueError("steps and gen_length must be integers")
    if not 1 <= steps <= gen_length:
        raise ValueError("Require 1 <= steps <= gen_length")
    if block_length is not None and block_length != gen_length:
        raise ValueError("LOWN uses a single block: block_length = gen_length")
    if temperature != 0 or cfg_scale != 0 or remasking != "low_confidence":
        raise ValueError("Use temperature=0, cfg_scale=0, remasking='low_confidence'")
    if inputs.ndim != 2 or inputs.shape[0] != 1:
        raise ValueError("Provide one unpadded prompt with shape [1, sequence_length]")
    if images is None or int(inputs.eq(IMAGE_TOKEN_INDEX).sum()) != 1:
        raise ValueError("Provide one image and one image placeholder in the prompt")

    stops = []
    if stopping_criteria:
        if tokenizer is None:
            raise ValueError("A tokenizer is required for stop strings")
        strings = [stopping_criteria] if isinstance(stopping_criteria, str) else stopping_criteria
        stops = [tokenizer.encode(text, add_special_tokens=False) for text in strings]
        if any(not sequence for sequence in stops):
            raise ValueError("Stop strings must encode to nonempty token sequences")

    # Dummy text labels make the official preparer mark only visual rows IGNORE.
    prepared = model.prepare_inputs_labels_for_multimodal(
        inputs, None, torch.ones_like(inputs, dtype=torch.bool), None,
        torch.zeros_like(inputs), images, ["image"], image_sizes=image_sizes,
    )
    prefix, labels = prepared[4], prepared[5]
    if prefix is None or labels is None or labels.shape != prefix.shape[:2]:
        raise RuntimeError("Multimodal embeddings and labels must align")
    visual = labels[0].eq(IGNORE_INDEX)
    if not bool(visual.any()):
        raise RuntimeError("No visual tokens found in the prepared prompt")

    decoder = model.get_model()
    device = prefix.device
    prefix_length = prefix.shape[1]
    mask = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    canvas = torch.cat((prefix, decoder.embed_tokens(mask).to(prefix.dtype)), dim=1)
    response_positions = torch.arange(prefix_length, canvas.shape[1], device=device)
    # Keep the original absolute positions after removing visual embeddings.
    text_positions = torch.cat((torch.where(~visual)[0], response_positions))
    schedule = model.get_num_transfer_tokens(torch.ones_like(mask, dtype=torch.bool), steps)[0]
    committed = torch.full((gen_length,), -1, dtype=torch.long, device=device)
    anchor_tokens = anchor_scores = None
    end = gen_length

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for width in schedule.tolist():
            active = torch.where(committed[:end] < 0)[0]
            if not active.numel():
                break
            k = min(width, active.numel())

            # 1. Check the anchor using only the prompt and current response.
            text_tokens, _ = _predict(
                model, canvas[:, text_positions], gen_length, text_positions[None],
            )
            accepted = active[:0] if anchor_tokens is None else _agreement_prefix(
                active, text_tokens, anchor_tokens, anchor_scores, committed,
            )

            if accepted.numel() >= k:
                # 2a. Commit the entire certified prefix, possibly more than k.
                positions = accepted
            else:
                # 2b. Discard the proposal; read the image on the unchanged canvas.
                anchor_tokens, anchor_scores = _predict(model, canvas, gen_length)
                order = active[torch.argsort(anchor_scores[active], descending=True, stable=True)]
                differs = text_tokens[order] != anchor_tokens[order]
                positions = torch.cat((order[differs], order[~differs]))[:k]

            # Every committed value comes from a FULL prediction, old or fresh.
            tokens = anchor_tokens[positions]
            if bool(tokens.eq(MASK_ID).any()):
                raise RuntimeError("The model predicted MASK at a commit position")
            committed[positions] = tokens
            canvas[0, prefix_length + positions] = decoder.embed_tokens(tokens).to(canvas.dtype)
            end = _stop_index(committed, stops) if stops else gen_length

    if bool(committed[:end].lt(0).any()):
        raise RuntimeError("The decoding schedule ended with unresolved positions")
    return committed[None, :end]
