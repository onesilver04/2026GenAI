from __future__ import annotations

import os
import json
import base64
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import httpx


API_KEY = os.getenv("SMU_API_KEY")
BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"

JSON_PATH = Path("marketing_agent_result.json")
OUTPUT_DIR = Path("generated_content")

IMAGE_MODEL = "gemini-2.5-flash-image"
# 대안:
# IMAGE_MODEL = "gpt-image-1-mini"
# IMAGE_MODEL = "gpt-image-1.5"


def load_prompt_from_json(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sns = data["sns_content"]

    return {
        "prompt": sns["image_prompt_for_sdxl"],
        "negative_prompt": sns.get("negative_prompt_for_sdxl", ""),
        "aspect_ratio": "2:3",  # Instagram portrait 느낌
    }


def save_base64_image(data_url: str) -> Path:
    if "," in data_url:
        b64 = data_url.split(",", 1)[1]
    else:
        b64 = data_url

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"gateway_image_{timestamp}.png"

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64))

    return output_path


def main():
    if not API_KEY:
        raise RuntimeError("SMU_API_KEY 환경변수가 없습니다.")

    prompt_data = load_prompt_from_json(JSON_PATH)

    prompt = (
        prompt_data["prompt"]
        + "\nAvoid: "
        + prompt_data["negative_prompt"]
    )

    payload = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "aspect_ratio": prompt_data["aspect_ratio"],
        "number_of_images": 1,
    }

    with httpx.Client(
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=120.0,
    ) as client:
        print("[INFO] Sending request...")
        print(json.dumps(payload, indent=2))

        resp = client.post(
            f"{BASE_URL}/images/generate/",
            json=payload
        )

        print("[INFO] Response status:", resp.status_code)
        print(resp.text[:1000])

        resp.raise_for_status()

        result = resp.json()

    image_url = result["data"][0]["url"]

    if image_url.startswith("data:image"):
        output_path = save_base64_image(image_url)
        print(f"[DONE] Saved image: {output_path}")
    else:
        print("[DONE] Image URL:")
        print(image_url)


if __name__ == "__main__":
    main()