import os
import json
import torch

PIXART_TXT_FILE_LOC = os.sep.join(["captions", "pixart_samples.txt"])
COCO_VAL_CAPTIONS = os.sep.join(["captions", "captions_val2017.json"])
COCO_2014_CAPTIONS = os.sep.join(["captions", "annots_10k.txt"])
HPSV2_CAPTIONS = os.sep.join(["captions", "hpsv2.txt"])


def get_coco_subset_spec(coco_9k=False, coco_10k=False):
    if coco_9k:
        return {"skip": 1000, "take": 9000}
    if coco_10k:
        return {"skip": 0, "take": 10000}
    return {"skip": 0, "take": 1000}


def resolve_coco_unique_captions(coco_9k=False, coco_10k=False):
    spec = get_coco_subset_spec(coco_9k=coco_9k, coco_10k=coco_10k)
    with open(COCO_VAL_CAPTIONS) as f:
        captions = json.load(f)

    unique_captions = []
    seen_image_ids = set()
    for ann in captions["annotations"]:
        image_id = ann["image_id"]
        if image_id in seen_image_ids:
            continue
        seen_image_ids.add(image_id)
        unique_captions.append(ann["caption"])
        if len(unique_captions) >= spec["skip"] + spec["take"]:
            break

    start = spec["skip"]
    end = start + spec["take"]
    selected = unique_captions[start:end]
    if len(selected) < spec["take"]:
        raise RuntimeError(
            f"Resolved only {len(selected)} unique COCO captions; expected {spec['take']}."
        )
    return selected


def get_captions(name, model, coco_9k=False, coco_10k=False, pixart=False, coco2014=False, hpsv2=False):
    if pixart:
        assert not coco_9k
        assert not coco_10k
        assert not coco2014
        assert not hpsv2
        tag = "_".join([name, "pixart"])
    elif coco_9k:
        assert not coco_10k
        assert not pixart
        assert not coco2014
        assert not hpsv2
        tag = "_".join([name, "coco_10k"])
    elif coco_10k:
        assert not coco_9k
        assert not pixart
        assert not coco2014
        assert not hpsv2
        tag = "_".join([name, "coco_10k"])
    elif coco2014:
        assert not coco_9k
        assert not pixart
        assert not coco_10k
        assert not hpsv2
        tag = "_".join([name, "coco2014"])
    elif hpsv2:
        assert not coco_9k
        assert not pixart
        assert not coco_10k
        assert not coco2014
        tag = "_".join([name, "hpsv2"])
    else:
        tag = "_".join([name, "coco_10k"])
    
    # Check if precomputed prompt embedding file already exists
    embedding_fname = os.sep.join(["captions", tag+".pt"])
    print("Checking for file ", embedding_fname)
    if os.path.isfile(embedding_fname):
        print("File found! Loading precomputed prompt embeddings")
        embedding_dict = torch.load(embedding_fname, map_location="cpu")
        prompt_embeds = embedding_dict['prompt_embeds']
        prompt_attention_masks = embedding_dict['prompt_attention_masks']
        if coco_9k:
            prompt_embeds = prompt_embeds[1000:10000]
            prompt_attention_masks = prompt_attention_masks[1000:10000]
        elif not pixart and not coco_10k and not coco2014 and not hpsv2: # coco_1k
            prompt_embeds = prompt_embeds[:1000]
            prompt_attention_masks = prompt_attention_masks[:1000]
        negative_prompt_embeds = embedding_dict['negative_prompt_embeds'].to('cuda')
        negative_prompt_attention_mask = embedding_dict['negative_prompt_attention_mask'].to('cuda')
        return prompt_embeds, prompt_attention_masks, negative_prompt_embeds, negative_prompt_attention_mask
    else:
        print("File not found. Precomputing prompt embeddings...")
        if pixart:
            with open(PIXART_TXT_FILE_LOC, 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        elif coco2014:
            with open(COCO_2014_CAPTIONS, 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        elif hpsv2:
            with open(HPSV2_CAPTIONS, 'r') as f:
                prompt_list = [item.strip() for item in f.readlines()]
        else:
            prompt_list = resolve_coco_unique_captions(coco_9k=coco_9k, coco_10k=coco_10k)
            
        pes, pams, npe, npam = [], [], None, None
        with torch.no_grad():
            for i, prompt in enumerate(prompt_list):
                print(i, prompt)
                if i == 0:
                    pe, pam, npe, npam = model.encode_prompt(prompt=prompt)
                else:
                    pe, pam, _, _ = model.encode_prompt(prompt=prompt)
                    
                pes.append(pe.to('cpu'))
                pams.append(pam.to('cpu'))
            pes = torch.cat(pes, dim=0)
            pams = torch.cat(pams, dim=0)
        embedding_dict = {
            'prompt_embeds': pes,
            'prompt_attention_masks': pams,
            'negative_prompt_embeds': npe,
            'negative_prompt_attention_mask': npam
        }
        print("Saving precomputed prompt embeddings to disk: ", embedding_fname)
        torch.save(embedding_dict, embedding_fname)
        print("Save successful!")
        npe = npe.to("cuda")
        npam = npam.to("cuda")
        return pes, pams, npe, npam

            
if __name__ == "__main__":
    print("Testing")
    caption_list = get_captions(pixart=True)
    print(caption_list)
    print(len(caption_list))
