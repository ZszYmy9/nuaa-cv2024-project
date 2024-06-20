import json, os, tqdm, torch

import jittor
from diffusers import DiffusionPipeline

max_num = 15
dataset_root = "A"



with torch.no_grad():
    for taskid in tqdm.tqdm(range(0, max_num)):
        pipe = DiffusionPipeline.from_pretrained("stabile_diffusion/stable-diffusion-2-1")
        pipe.load_lora_weights(f"style/style_{taskid:02d}")
        pipe = pipe.to("cuda")

        # load json
        with open(f"{dataset_root}/{taskid:02d}/prompt.json", "r") as file:
            prompts = json.load(file)

        for id, prompt in prompts.items():
            print(prompt)
            image = pipe(prompt + f" in style_{taskid:02d}", num_inference_steps=25).images[0]
            os.makedirs(f"./output/{taskid:02d}", exist_ok=True)
            image.save(f"./output/{taskid:02d}/{prompt}.png")