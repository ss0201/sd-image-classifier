import argparse
import logging
import os
from typing import cast

import torch
import webuiapi
from dotenv import load_dotenv
from PIL import Image, PngImagePlugin

from classify import load_model, predict_classification
from util import get_device, get_val_transform


def generate_images(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)

    api = webuiapi.WebUIApi(host=args.host, port=args.port, use_https=args.https)
    username = os.environ.get("USERNAME")
    password = os.environ.get("PASSWORD")
    if username is not None and password is not None:
        api.set_auth(username, password)
    if args.sd_model is not None:
        api.util_set_model(args.sd_model)
    if args.vae is not None:
        options = {"sd_vae": args.vae}
        api.set_options(options)

    device = get_device()

    prompt = get_prompt(args.prompt, args.prompt_file)
    negative_prompt = get_prompt(args.negative_prompt, args.negative_prompt_file)

    generator = ImageGenerator(
        api=api,
        output_dir=args.output_dir,
        model_path=args.model_path,
        skip_classes=args.skip_classes,
        device=device,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        sampler_name=args.sampler_name,
        steps=args.steps,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
    )

    itr = 0
    while itr < args.iterations or args.iterations == -1:
        logging.info(f"Generating images (iteration {itr})...")
        generator.generate()
        itr += 1


def get_prompt(prompt: str | None, prompt_file: str | None):
    if prompt is not None:
        return prompt
    elif prompt_file is not None:
        with open(prompt_file, "r") as f:
            return f.read()
    else:
        return ""


def get_next_image_id(output_dir: str) -> int:
    last_image_id = 0
    for _, _, files in os.walk(output_dir):
        for filename in files:
            if not filename.endswith(".png"):
                continue
            try:
                image_id = int(os.path.splitext(filename)[0])
            except ValueError:
                continue
            if image_id > last_image_id:
                last_image_id = image_id

    return last_image_id + 1


class ImageGenerator:
    def __init__(
        self,
        api: webuiapi.WebUIApi,
        output_dir: str,
        model_path: str,
        skip_classes: list[str],
        device: torch.device,
        **kwargs,
    ):
        self.api = api
        self.output_dir = output_dir
        self.model_path = model_path
        self.skip_classes = skip_classes
        self.device = device
        self.kwargs = kwargs
        self.model, self.classes, self.resize_to, self.task_type = load_model(
            self.model_path, device
        )
        self.transform = get_val_transform(self.resize_to)

    def generate(self):
        result = self.api.txt2img(**self.kwargs)
        result = cast(webuiapi.WebUIApiResult, result)

        for i, image in enumerate(result.images):
            image = cast(Image.Image, image)

            class_name, _ = predict_classification(
                image,
                self.transform,
                self.model,
                self.classes,
                self.task_type,
                self.device,
            )
            if class_name in self.skip_classes:
                continue

            pnginfo = self.create_pnginfo(result.info, i)
            image_id = get_next_image_id(self.output_dir)
            self.save_image(image, pnginfo, class_name, image_id)
            result.parameters

    def create_pnginfo(self, api_result_info: dict, i: int) -> PngImagePlugin.PngInfo:
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in api_result_info.items():
            if isinstance(key, str) and isinstance(value, str):
                pnginfo.add_text(key, value)
        pnginfo.add_text("parameters", str(api_result_info["infotexts"][i]))
        return pnginfo

    def save_image(
        self,
        image: Image.Image,
        pnginfo: PngImagePlugin.PngInfo,
        class_name: str,
        image_id: int,
    ) -> None:
        class_dir = os.path.join(self.output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        filename = os.path.join(class_dir, f"{image_id:06}.png")
        image.save(filename, pnginfo=pnginfo)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate images using Stable Diffusion.",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="The host running the web UI.",
    )
    parser.add_argument(
        "--port", default=7860, type=int, help="The port number of the web UI."
    )
    parser.add_argument(
        "--https",
        action="store_true",
        help="Use HTTPS to connect to the web UI.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="The path to the model to use for classification.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="The directory to save the generated images.",
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="The number of times to generate images. "
        "Set to -1 for infinite iterations.",
    )
    parser.add_argument(
        "--skip-classes",
        nargs="+",
        default=[],
        help="The classes to skip saving.",
    )

    sd_group = parser.add_argument_group("stable diffusion parameters")
    sd_group.add_argument(
        "--prompt",
        help="The prompt to generate images from. This overrides --prompt-file.",
    )
    sd_group.add_argument(
        "--prompt-file", help="The file containing the prompt to generate images from."
    )
    sd_group.add_argument(
        "--negative-prompt",
        help="The negative prompt. This overrides --negative-prompt-file.",
    )
    sd_group.add_argument(
        "--negative-prompt-file",
        help="The file containing the negative prompt.",
    )
    sd_group.add_argument("--sd-model", help="The name of the stable diffusion model.")
    sd_group.add_argument("--vae", help="The name of the VAE.")
    sd_group.add_argument(
        "--seed",
        default=-1,
        type=int,
        help="The seed to use for image generation.",
    )
    sd_group.add_argument(
        "--cfg-scale", default=7, type=int, help="Classifier free guidance scale."
    )
    sd_group.add_argument(
        "--sampler-name", default="Euler a", help="The name of the sampler."
    )
    sd_group.add_argument("--steps", default=20, type=int, help="The number of steps.")
    sd_group.add_argument(
        "--width", default=512, type=int, help="The width of the generated images."
    )
    sd_group.add_argument(
        "--height", default=512, type=int, help="The height of the generated images."
    )
    sd_group.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="The batch size to use for generating images.",
    )

    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    generate_images(args)


if __name__ == "__main__":
    main()
