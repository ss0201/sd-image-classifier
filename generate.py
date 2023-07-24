import argparse
from typing import cast

import webuiapi
from PIL import Image, PngImagePlugin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--cfg-scale", default=7, type=int)
    parser.add_argument("--sampler-name", default="Euler a")
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--width", default=512, type=int)
    parser.add_argument("--height", default=512, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()

    api = webuiapi.WebUIApi(host=args.host, port=args.port)

    result = api.txt2img(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        sampler_name=args.sampler_name,
        steps=args.steps,
        width=args.width,
        height=args.height,
    )
    result = cast(webuiapi.WebUIApiResult, result)

    for image in result.images:
        image = cast(Image.Image, image)

        pnginfo = PngImagePlugin.PngInfo()
        for key, value in result.info.items():
            if isinstance(key, str) and isinstance(value, str):
                pnginfo.add_text(key, str(value))
        pnginfo.add_text("parameters", result.info["infotexts"][0])

        image.save("tmp/output.png", pnginfo=pnginfo)


if __name__ == "__main__":
    main()
