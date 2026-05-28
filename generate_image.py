from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionXLPipeline


JSON_PATH = Path("marketing_agent_result.json")
OUTPUT_DIR = Path("generated_content")
SDXL_MODEL = "segmind/SSD-1B"


def load_sns_prompt(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sns_content = data.get("sns_content", {})

    required_keys = [
        "image_prompt_for_sdxl",
        "negative_prompt_for_sdxl",
        "image_width",
        "image_height",
    ]

    for key in required_keys:
        if key not in sns_content:
            raise KeyError(f"Missing key in sns_content: {key}")

    return sns_content


def generate_image(sns_content: dict) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on the GPU server.")

    prompt = sns_content["image_prompt_for_sdxl"]
    negative_prompt = sns_content.get("negative_prompt_for_sdxl", "")
    width = int(sns_content.get("image_width", 640))
    height = int(sns_content.get("image_height", 960))

    print("[INFO] Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing() # VRAM 사용량 최적화 옵션

    print("[INFO] Generating image...")
    print(f"[PROMPT] {prompt}")
    print(f"[NEGATIVE] {negative_prompt}")
    print(f"[SIZE] {width}x{height}")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=20,
        guidance_scale=7.0,
    ).images[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"sns_sdxl_{timestamp}.png"
    image.save(output_path)

    return output_path


def main():
    sns_content = load_sns_prompt(JSON_PATH)
    output_path = generate_image(sns_content)
    print(f"[DONE] Image saved to: {output_path}")


if __name__ == "__main__":
    main()