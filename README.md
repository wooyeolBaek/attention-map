# Cross Attention Map

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/We-Want-GPU/diffusers-cross-attention-map-SDXL-t2i)


## Examples

<!-- <img src="./assets/t2i.png" alt="attn_map">
<img src="./assets/attn_maps.png" alt="attn_map"> -->
<img src="./assets/hf_spaces.png" alt="hf_spaces">

<details>
<summary>6_kangaroo</summary>
<div markdown="1">

<img src="./assets/6_<kangaroo>.png" alt="6_kangaroo">

</div>
</details>


<details>
<summary>10_hoodie</summary>
<div markdown="1">

<img src="./assets/10_<hoodie>.png" alt="10_hoodie">

</div>
</details>


<details>
<summary>13_sunglasses</summary>
<div markdown="1">

<img src="./assets/13_<sunglasses>.png" alt="13_sunglasses">

</div>
</details>





## Initialize
```shell
python -m venv .venv
source .venv/bin/activate
pip install diffusers==0.24.0 accelerate==0.25.0 transformers==4.36.0
```

## Visualize
Visualize Cross Attention Map for Text-to-Image
```shell
python t2i.py
```

## How to use
```python
import torch
from diffusers import StableDiffusionXLPipeline
from utils import (
    cross_attn_init,
    register_cross_attention_hook,
    attn_maps,
    get_net_attn_map,
    resize_net_attn_map,
    save_net_attn_map,
)

cross_attn_init()

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe.unet = register_cross_attention_hook(pipe.unet)
pipe = pipe.to("cuda")

prompt = "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest that says 'SDXL'!."
image = pipe(prompt).images[0]
image.save('test.png')

dir_name = "attn_maps"
net_attn_maps = get_net_attn_map(image.size)
net_attn_maps = resize_net_attn_map(net_attn_maps, image.size)
save_net_attn_map(net_attn_maps, dir_name, pipe.tokenizer, prompt)
```

# TODO
1. Resize attention maps for rectangle images.
