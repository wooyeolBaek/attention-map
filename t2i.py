import torch
from diffusers import DiffusionPipeline
from utils import (
    attn_maps,
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
    save_by_timesteps_and_path,
    save_by_timesteps
)

##### 1. Init modules #####
cross_attn_init()
###########################

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda:0")

##### 2. Replace modules and Register hook #####
pipe.unet = set_layer_with_name_and_path(pipe.unet)
pipe.unet = register_cross_attention_hook(pipe.unet)
################################################

height = 512
width = 768
prompt = "A portrait photo of a kangaroo wearing an orange hoodie and blue sunglasses standing on the grass in front of the Sydney Opera House holding a sign on the chest that says 'SDXL'!."

image = pipe(
    prompt,
    height=height,
    width=width,
    num_inference_steps=15,
).images[0]
image.save('test.png')

##### 3. Process and Save attention map #####
print('resizing and saving ...')

##### 3-1. save by timesteps and path (2~3 minutes) #####
save_by_timesteps_and_path(pipe.tokenizer, prompt, height, width)
#########################################################

##### 3-2. save by timesteps (1~2 minutes) #####
# save_by_timesteps(pipe.tokenizer, prompt, height, width)
################################################