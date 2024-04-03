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