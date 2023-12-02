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

prompt = "A cinematic shot of a black dog with a santa hat, photo booth, adorable, christmas atmosphere"
image = pipe(prompt).images[0]
image.save('test.png')

dir_name = "attn_maps"
net_attn_maps = get_net_attn_map()
net_attn_maps = resize_net_attn_map(net_attn_maps, image.size)
save_net_attn_map(net_attn_maps, dir_name, pipe.tokenizer, prompt)