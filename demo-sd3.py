import torch
from diffusers import StableDiffusion3Pipeline

from utils import (
    attn_maps,
    cross_attn_init,
    init_pipeline,
    save_attention_maps
)

##### 1. Init redefined modules #####
cross_attn_init()
#####################################

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

##### 2. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

# recommend not using batch operations for sd3, as cpu memory could be exceeded.
prompts = [
    # "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
    guidance_scale=4.5,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-sd3.png')

##### 3. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps', unconditional=True)
#############################################