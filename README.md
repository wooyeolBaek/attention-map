# Cross Attention Map

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/We-Want-GPU/diffusers-cross-attention-map-SDXL-t2i)
#### For errors reports or feature requests, please raise an issue :)


## Compatible models
UNet with attn2(cross attention module) is compatible
- [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo)
- ...


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
    attn_maps,
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
    preprocess,
    visualize_and_save_attn_map
)

##### 1. Init modules #####
cross_attn_init()
###########################

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

##### 2. Replace modules and Register hook #####
pipe.unet = set_layer_with_name_and_path(pipe.unet)
pipe.unet = register_cross_attention_hook(pipe.unet)
################################################

height = 512
width = 768
prompt = "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest that says 'SDXL'!."
pipe = pipe.to("cuda:0")
image = pipe(
    prompt,
    height=height,
    width=width,
).images[0]
image.save('test.png')

##### 3. Process and Save attention map #####
attn_map = preprocess(max_height=height, max_width=width)
visualize_and_save_attn_map(attn_map, pipe.tokenizer, prompt)
#############################################
```

## TODO
- Add options for how to combine attention maps for each layers and timsteps
