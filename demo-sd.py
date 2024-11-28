import torch
from diffusers import DiffusionPipeline
from utils import (
    attn_maps,
    cross_attn_init,
    init_pipeline,
    save_attention_maps
)

##### 1. Init redefined modules #####
cross_attn_init()
#####################################

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

##### 2. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

prompts = [
    "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-sdxl.png')

##### 3. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps', unconditional=True)
#############################################
