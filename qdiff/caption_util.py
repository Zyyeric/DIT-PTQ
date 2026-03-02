import os
import logging
import torch

logger = logging.getLogger(__name__)

PIXART_TXT_FILE_LOC = os.sep.join(["captions", "pixart_samples.txt"])
COCO_VAL_CAPTIONS   = os.sep.join(["captions", "captions_val2017.json"])
COCO_2014_CAPTIONS  = os.sep.join(["captions", "annots_10k.txt"])
HPSV2_CAPTIONS      = os.sep.join(["captions", "hpsv2.txt"])


def get_captions(name, model,
                 coco_9k=False, coco_10k=False,
                 pixart=False, coco2014=False, hpsv2=False):
    """
    Load or compute T5-XXL prompt embeddings for the chosen caption set.

    First call: encodes prompts one-by-one and caches to disk as a .pt file.
    Subsequent calls: loads from the cached .pt file directly.

    Returns: (prompt_embeds, prompt_attention_masks,
               negative_prompt_embeds, negative_prompt_attention_mask)
    """
    # Validate exclusive flags
    flags = [coco_9k, coco_10k, pixart, coco2014, hpsv2]
    if sum(flags) > 1:
        raise ValueError("Only one caption set flag may be set at a time.")

    if   pixart:   tag = f"{name}_pixart"
    elif coco_9k:  tag = f"{name}_coco_10k"
    elif coco_10k: tag = f"{name}_coco_10k"
    elif coco2014: tag = f"{name}_coco2014"
    elif hpsv2:    tag = f"{name}_hpsv2"
    else:          tag = f"{name}_coco_10k"

    embedding_fname = os.sep.join(["captions", tag + ".pt"])
    logger.info(f"get_captions: tag={tag}  cache_file={embedding_fname}")

    # ── Load from cache if available ─────────────────────────────────────────
    if os.path.isfile(embedding_fname):
        logger.info(f"Cache found — loading precomputed embeddings from {embedding_fname}")
        embedding_dict = torch.load(embedding_fname, map_location="cpu")
        prompt_embeds           = embedding_dict['prompt_embeds']
        prompt_attention_masks  = embedding_dict['prompt_attention_masks']
        negative_prompt_embeds  = embedding_dict['negative_prompt_embeds'].to('cuda')
        negative_prompt_attn    = embedding_dict['negative_prompt_attention_mask'].to('cuda')

        logger.info(f"  Loaded: prompt_embeds={tuple(prompt_embeds.shape)}  "
                    f"prompt_attn={tuple(prompt_attention_masks.shape)}  "
                    f"neg_embeds={tuple(negative_prompt_embeds.shape)}")

        # Slice for the correct subset
        if coco_9k:
            prompt_embeds          = prompt_embeds[1000:]
            prompt_attention_masks = prompt_attention_masks[1000:]
            logger.info(f"  coco_9k slice: using prompts 1000..{1000+len(prompt_embeds)}")
        elif not any([pixart, coco_10k, coco2014, hpsv2]):
            # Default: first 1000 (coco_1k)
            prompt_embeds          = prompt_embeds[:1000]
            prompt_attention_masks = prompt_attention_masks[:1000]
            logger.info(f"  coco_1k slice: using first 1000 prompts")

        logger.info(f"  Final prompt_embeds shape: {tuple(prompt_embeds.shape)}")
        return prompt_embeds, prompt_attention_masks, negative_prompt_embeds, negative_prompt_attn

    # ── Compute and cache ────────────────────────────────────────────────────
    logger.info(f"Cache not found — computing prompt embeddings (this may take a while)...")

    if pixart:
        with open(PIXART_TXT_FILE_LOC, 'r') as f:
            prompt_list = [item.strip() for item in f.readlines()]
        logger.info(f"  Loaded {len(prompt_list)} PixArt prompts from {PIXART_TXT_FILE_LOC}")
    elif coco2014:
        with open(COCO_2014_CAPTIONS, 'r') as f:
            prompt_list = [item.strip() for item in f.readlines()]
        logger.info(f"  Loaded {len(prompt_list)} COCO-2014 prompts from {COCO_2014_CAPTIONS}")
    elif hpsv2:
        with open(HPSV2_CAPTIONS, 'r') as f:
            prompt_list = [item.strip() for item in f.readlines()]
        logger.info(f"  Loaded {len(prompt_list)} HPSv2 prompts from {HPSV2_CAPTIONS}")
    else:
        import json
        with open(COCO_VAL_CAPTIONS) as f:
            captions = json.load(f)
        captions_list = captions['annotations'][:10000]
        prompt_list   = [c['caption'] for c in captions_list]
        logger.info(f"  Loaded {len(prompt_list)} COCO-val prompts from {COCO_VAL_CAPTIONS}")

    # NOTE: captions are encoded one-by-one here because encode_prompt
    # returns the negative embedding only on the first call.
    # This is slow but only runs once — result is cached to disk.
    # Future improvement: batch encode with a single T5 call.
    logger.info(f"  Encoding {len(prompt_list)} prompts one-by-one via T5-XXL "
                f"(first-run only — will be cached)...")
    pes, pams, npe, npam = [], [], None, None

    with torch.no_grad():
        for i, prompt in enumerate(prompt_list):
            if i % 500 == 0:
                logger.info(f"  Encoding prompt {i}/{len(prompt_list)}: {prompt[:60]}...")
            if i == 0:
                pe, pam, npe, npam = model.encode_prompt(prompt=prompt)
            else:
                pe, pam, _, _ = model.encode_prompt(prompt=prompt)
            pes.append(pe.to('cpu'))
            pams.append(pam.to('cpu'))

    pes  = torch.cat(pes,  dim=0)
    pams = torch.cat(pams, dim=0)
    logger.info(f"  Encoding done: pes={tuple(pes.shape)}  pams={tuple(pams.shape)}")

    embedding_dict = {
        'prompt_embeds':                  pes,
        'prompt_attention_masks':         pams,
        'negative_prompt_embeds':         npe,
        'negative_prompt_attention_mask': npam,
    }
    os.makedirs(os.path.dirname(embedding_fname), exist_ok=True)
    torch.save(embedding_dict, embedding_fname)
    logger.info(f"  Cached embeddings saved to {embedding_fname}")

    npe  = npe.to("cuda")
    npam = npam.to("cuda")

    if coco_9k:
        return pes[1000:], pams[1000:], npe, npam
    elif not any([pixart, coco_10k, coco2014, hpsv2]):
        return pes[:1000], pams[:1000], npe, npam
    return pes, pams, npe, npam


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing caption_util...")
    caption_list = get_captions("test", model=None, pixart=True)
    logger.info(f"Result shapes: {[tuple(t.shape) for t in caption_list]}")