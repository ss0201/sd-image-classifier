import argparse
import logging
from typing import cast

import webuiapi
from PIL import Image, PngImagePlugin


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
        "--output-dir",
        default="tmp",
        help="The directory to save the generated images.",
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="The number of times to generate images. "
        "Set to -1 for infinite iterations.",
    )

    sd_group = parser.add_argument_group("stable diffusion parameters")
    sd_group.add_argument(
        "--prompt", required=True, help="The prompt to generate images from."
    )
    sd_group.add_argument("--negative-prompt", default="", help="The negative prompt.")
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

    logging.basicConfig(level=logging.INFO)

    api = webuiapi.WebUIApi(host=args.host, port=args.port)
    i = 0
    while i < args.iterations or args.iterations == -1:
        generate(api, args)
        i += 1


def generate(api: webuiapi.WebUIApi, args: argparse.Namespace):
    result = api.txt2img(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        sampler_name=args.sampler_name,
        steps=args.steps,
        width=args.width,
        height=args.height,
        batch_size=args.batch_size,
    )
    result = cast(webuiapi.WebUIApiResult, result)

    for i, image in enumerate(result.images):
        image = cast(Image.Image, image)

        pnginfo = PngImagePlugin.PngInfo()
        for key, value in result.info.items():
            if isinstance(key, str) and isinstance(value, str):
                pnginfo.add_text(key, str(value))
        pnginfo.add_text("parameters", str(result.info["infotexts"][i]))

        image.save(f"tmp/output{i}.png", pnginfo=pnginfo)


if __name__ == "__main__":
    main()
