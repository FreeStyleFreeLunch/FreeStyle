import diffusers
from diffusers import (
    DPMSolverSDEScheduler,
    DDIMScheduler,
    # StableDiffusionXLPipeline,
    UNet2DConditionModel
)
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from pipeline_stable_diffusion_img2img import StableDiffusionXLImg2ImgPipeline
# from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
# from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline

import torch
import logging
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
from PIL import PngImagePlugin, Image
import math
import json
import tqdm
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        [
            "argument parser for using webui functions in generating test images without"
            "launching the UI"
        ]
    )
    # parser.add_argument("--use_webui", action='store_true')
    # parser.add_argument("--webui_host", type=str, default='127.0.0.1')
    # parser.add_argument("--webui_port", type=int, default=None)

    parser.add_argument(
        "--b",
        type=float,
        default="",
        required=True,
        help="hyper-parameter b",
    )

    parser.add_argument(
        "--s",
        type=float,
        default="",
        required=True,
        help="hyper-parameter s",
    )

    parser.add_argument(
        "--n",
        type=int,
        default="",
        required=True,
        help="hyper-parameter n",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        required=True,
        help="model used for evaluation",
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default="",
    )

    parser.add_argument(
        "--unet_name",
        type=str,
        default=None,
        help="unet used for evaluation",
    )
    parser.add_argument(
        "--moe_json",
        type=str,
        default=None,
        help="MoE model dict json used for evaluation",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="the desired width for the generated images",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="the desired height for the generated images",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="DPM++ SDE Karras",
        help="the sampler used in the generation process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123456789,
        help="the random seed used in the generation process",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=35,
        help="the number of diffusion steps used in generation",
    )
    parser.add_argument(
        "--cfg",
        type=int,
        default=7,
        help="the cfg scale used in the generation process",
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default=[],
        nargs="+",
        help="test prompts, will be overrided by json",
    )
    parser.add_argument(
        "--prompt_json", type=str, default=None, help="json file with all test prompts"
    )
    parser.add_argument(
        "--postive_prompt_suffix",
        type=str,
        default="",
        help="common postive prompts appended after each prompt",
    )
    parser.add_argument(
        "--postive_prompt_prefix",
        type=str,
        default="",
        help="common postive prompts appended after each prompt",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="worst quality, low quality, low res, blurry, cropped image, jpeg artifacts, error, ugly, out of frame, deformed, poorly drawn, mutilated, mangled, bad proportions, long neck, missing limb, floating limbs, disconnected limbs, long body, missing arms, malformed limbs, missing legs, extra arms, extra legs, poorly drawn face, cloned face, deformed iris, deformed pupils, deformed hands, twisted fingers, malformed hands, poorly drawn hands, mutated hands, mutilated hands, extra fingers, fused fingers, too many fingers, duplicate, multiple heads, extra limb, duplicate artifacts",
        help="negative prompts used in the generation process",
    )

    parser.add_argument(
        "--num_images_per_prompt",
        default=1,
        type=int,
        help="batch size used in generating images",
    )
    parser.add_argument(
        "--grid", action="store_true", help="whether save the images in grid mode"
    )
    parser.add_argument(
        "--refiner", action="store_true", help="whether user SDXL refiner"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="the output dir used to save all images",
    )
    parser.add_argument(
        "--refimgpath",
        type=str,
        default="./ContentImages/imgs0", 
        #default="/mnt/CV_teamz/users/ligang/diffusers_test/referenceimg/jeep.png",
        help="the reference image path",
    )
    parser.add_argument(
        "--styleimgpath",
        type=str,
        default="",#"/mnt/CV_teamz/users/ligang/diffusers_test/referenceimg/peoples/0004.png",
        help="the path of the style reference img"
    )
    parser.add_argument("--exist_ok", action="store_true", help="")


    args = parser.parse_args()

    if args.prompt_json is not None:
        if len(args.prompts) > 0:
            logging.warning("the prompts will be overrided by the prompt json")
        prompts = json.load(open(args.prompt_json))
        if isinstance(prompts, dict):
            prompts = [key for key, _ in prompts.items()]
        assert isinstance(prompts, list)
        args.prompts = prompts
    logging.info(f"{len(args.prompts)} prompts will be processed in the process")

    return args


def test(args):

    if args.sampler == "DPM++ SDE Karras":
        noise_scheduler = DPMSolverSDEScheduler.from_pretrained(
            args.model_name, subfolder="scheduler", noise_sampler_seed=args.seed
        )
    elif args.sampler == "DDIM":
        noise_scheduler = DDIMScheduler.from_pretrained(
            args.model_name, subfolder="scheduler",noise_sampler_seed=args.seed
        )
    else:
        raise NotImplementedError

    if not args.unet_name:
        args.unet_name = args.model_name
    try:
        unet = UNet2DConditionModel.from_pretrained(args.unet_name, subfolder='unet_ema')
    except:
        unet = UNet2DConditionModel.from_pretrained(args.unet_name, subfolder='unet')
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model_name, scheduler=noise_scheduler, unet=unet.to(dtype=torch.float16), torch_dtype=torch.float16
    )
    unet.set_hyper_parameter(args)
    if args.lora_path:
        print(args.lora_path)
        try:
            pipeline.load_lora_weights(args.lora_path)
        except:
            state_dict, network_alphas = pipeline.lora_state_dict(args.lora_path)
            # print(state_dict)
            # print(network_alphas)
            pipeline.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=pipeline.unet)

            text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
            if len(text_encoder_state_dict) > 0:
                pipeline.load_lora_into_text_encoder(
                    text_encoder_state_dict,
                    network_alphas=network_alphas,
                    text_encoder=pipeline.text_encoder,
                    prefix="text_encoder",
                    lora_scale=pipeline.lora_scale,
                )

            text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
            if len(text_encoder_2_state_dict) > 0:
                pipeline.load_lora_into_text_encoder(
                    text_encoder_2_state_dict,
                    network_alphas=network_alphas,
                    text_encoder=pipeline.text_encoder_2,
                    prefix="text_encoder_2",
                    lora_scale=pipeline.lora_scale,
                )

    if args.refiner:
        pipeline_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "/mnt/CV_teamz/pretrained/stable-diffusion-xl-refiner-1.0"
        )
        pipeline_refiner = pipeline_refiner.to(device)

    pipeline = pipeline.to(device)
    args.output_dir = args.output_dir + '_fp16'


    if not os.path.exists(args.output_dir):
        logging.info(f"{args.output_dir} not exist, create")
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        logging.info(f"{args.output_dir} exists, files may be overwrited")
    # start generate images

    if args.grid:
        logging.info(f"images will be saved in grids")
    

    files = os.listdir(args.refimgpath)
    files = [os.path.join(args.refimgpath, file) for file in files]
    for idx, prompt in enumerate(args.prompts):
               
        #the code for many images 
        for file in files:
            all_images = []
        
            if args.refiner:
                outputs = pipeline(
                    args.postive_prompt_prefix + prompt + args.postive_prompt_suffix,
                    negative_prompt=args.negative_prompt,
                    num_images_per_prompt=args.num_images_per_prompt,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    generator=torch.Generator(device=device).manual_seed(args.seed),
                    output_type="latent",
                    refimgpath=file,
                    styleimgpath=args.styleimgpath,
                ).images
            
                outputs = pipeline_refiner(
                    args.postive_prompt_prefix + prompt + args.postive_prompt_suffix,
                    image=outputs,
                    negative_prompt=args.negative_prompt,
                    num_images_per_prompt=args.num_images_per_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    generator=torch.Generator(device=device).manual_seed(args.seed),
                    refimgpath=file,
                    #styleimgpath=args.styleimgpath,
                )
            else:
                print(args.postive_prompt_prefix + prompt + args.postive_prompt_suffix)
                outputs = pipeline(
                    args.postive_prompt_prefix + prompt + args.postive_prompt_suffix,
                    negative_prompt=args.negative_prompt,
                    num_images_per_prompt=args.num_images_per_prompt,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    generator=torch.Generator(device=device).manual_seed(args.seed),
                    refimgpath=file,
                    styleimgpath=args.styleimgpath,
                )

            # print(outputs)

            images = outputs.images
            all_images = all_images + images

            to_save_images = []
            if args.grid:
                to_save_width = int(math.sqrt(args.num_images_per_prompt))
                to_save_height = args.num_images_per_prompt // to_save_width
                if to_save_height * to_save_width != args.num_images_per_prompt:
                    to_save_width = to_save_width + 1
                for _idx in range(0, len(all_images), args.num_images_per_prompt):
                    new_img = Image.new(
                        "RGB", (to_save_width * args.width, to_save_height * args.height)
                    )
                    for subIdx in range(args.num_images_per_prompt):
                        x_offset = (subIdx % to_save_width) * args.width
                        y_offset = (subIdx // to_save_height) * args.height
                        new_img.paste(all_images[_idx + subIdx], (x_offset, y_offset))
                    to_save_images.append(new_img)
            else:
                logging.info(f"images will be saved in separate images")
                to_save_images = all_images
            
            file=os.path.basename(file)
            filename=file.split(".")[0]
            if not os.path.exists(os.path.join(args.output_dir,filename)):
                os.makedirs(os.path.join(args.output_dir,filename))
            for id, image in enumerate(to_save_images):
                image.save(os.path.join(args.output_dir,filename, f'{idx:05d}_{id:02d}.png'))

        """
        if not args.exist_ok  and  os.path.exists(os.path.join(args.output_dir, f'{idx:05d}_00.png')):
            continue
        
        all_images = []

        if args.refiner:
            outputs = pipeline(
                args.postive_prompt_prefix + prompt + args.postive_prompt_suffix,
                negative_prompt=args.negative_prompt,
                num_images_per_prompt=args.num_images_per_prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                output_type="latent",
                refimgpath=args.refimgpath,
                styleimgpath=args.styleimgpath,
            ).images
            
            outputs = pipeline_refiner(
                args.postive_prompt_prefix + prompt + args.postive_prompt_suffix,
                image=outputs,
                negative_prompt=args.negative_prompt,
                num_images_per_prompt=args.num_images_per_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                refimgpath=args.refimgpath,
                #styleimgpath=args.styleimgpath,
            )
        else:
            print(args.postive_prompt_prefix + prompt + args.postive_prompt_suffix)
            outputs = pipeline(
                args.postive_prompt_prefix + prompt + args.postive_prompt_suffix,
                negative_prompt=args.negative_prompt,
                num_images_per_prompt=args.num_images_per_prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                refimgpath=args.refimgpath,
                styleimgpath=args.styleimgpath,
            )

        # print(outputs)

        images = outputs.images
        all_images = all_images + images

        to_save_images = []
        if args.grid:
            to_save_width = int(math.sqrt(args.num_images_per_prompt))
            to_save_height = args.num_images_per_prompt // to_save_width
            if to_save_height * to_save_width != args.num_images_per_prompt:
                to_save_width = to_save_width + 1
            for _idx in range(0, len(all_images), args.num_images_per_prompt):
                new_img = Image.new(
                    "RGB", (to_save_width * args.width, to_save_height * args.height)
                )
                for subIdx in range(args.num_images_per_prompt):
                    x_offset = (subIdx % to_save_width) * args.width
                    y_offset = (subIdx // to_save_height) * args.height
                    new_img.paste(all_images[_idx + subIdx], (x_offset, y_offset))
                to_save_images.append(new_img)
        else:
            logging.info(f"images will be saved in separate images")
            to_save_images = all_images

        for id, image in enumerate(to_save_images):
                image.save(os.path.join(args.output_dir, f'{idx:05d}_{id:02d}.png'))
        """
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    test(args)
