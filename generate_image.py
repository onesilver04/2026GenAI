"""
generate_image.py

marketing_agent_result.json 의 SDXL 프롬프트를 읽어
Hugging Face Inference API로 이미지를 생성합니다.

Requirements:
    pip install requests

Setup:
    export HF_API_KEY="your_huggingface_token"
    # 또는 실행 시 입력 프롬프트에서 입력

API 키 발급 (무료):
    https://huggingface.co → Settings → Access Tokens → New Token

Usage:
    # 기본 (marketing_agent_result.json 자동 참조)
    python generate_image.py

    # 결과 파일 직접 지정
    python generate_image.py --input my_result.json

    # 프롬프트 직접 입력
    python generate_image.py --prompt "your prompt here"
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests


# ── 설정 ──────────────────────────────────────────────────────────────────────
# 모델 변경 시 아래 MODEL_ID만 교체하면 됩니다.
# 대안 모델:
#   "black-forest-labs/FLUX.1-schnell"  (더 빠름)
#   "stabilityai/stable-diffusion-2-1"  (가벼움)
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
API_URL  = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

DEFAULT_STEPS  = 30
DEFAULT_CFG    = 7.5
DEFAULT_WIDTH  = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SEED   = 42


# ── API 호출 ───────────────────────────────────────────────────────────────────
def generate(
    api_key: str,
    positive_prompt: str,
    negative_prompt: str = "",
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int = DEFAULT_SEED,
    output_path: str = "output.png",
):
    print(f"\n모델: {MODEL_ID}")
    print(f"이미지 생성 중... (steps={steps}, cfg={cfg}, seed={seed})")
    print(f"  Positive: {positive_prompt[:80]}...")
    if negative_prompt:
        print(f"  Negative: {negative_prompt[:80]}...")

    payload = {
        "inputs": positive_prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "width": width,
            "height": height,
            "seed": seed,
        },
    }

    # 모델 로딩 중일 경우 재시도
    for attempt in range(1, 6):
        t0 = time.time()
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=120,
        )

        # 모델 워밍업 중 (503)
        if response.status_code == 503:
            wait = response.json().get("estimated_time", 20)
            print(f"  모델 로딩 중... {wait:.0f}초 대기 후 재시도 ({attempt}/5)")
            time.sleep(min(wait, 30))
            continue

        if response.status_code != 200:
            raise RuntimeError(f"API 오류 {response.status_code}: {response.text}")

        # 응답은 바이너리 PNG
        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"생성 완료: {time.time() - t0:.1f}초")
        print(f"저장됨: {output_path}")
        return output_path

    raise RuntimeError("모델 로딩 타임아웃. 잠시 후 다시 시도하세요.")


# ── JSON에서 프롬프트 추출 ──────────────────────────────────────────────────────
def load_prompts_from_json(json_path: str) -> tuple[str, str, str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sns          = data.get("sns_content", {})
    positive     = sns.get("image_prompt_for_sdxl", "")
    negative     = sns.get("negative_prompt_for_sdxl", "")
    product_name = data.get("input", {}).get("product_name", "output")

    if not positive:
        raise ValueError("sns_content.image_prompt_for_sdxl 값이 비어 있습니다.")

    return positive, negative, product_name


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hugging Face SDXL 이미지 생성기")
    parser.add_argument("--input",    "-i", default="marketing_agent_result.json",
                        help="마케팅 에이전트 결과 JSON 파일 경로")
    parser.add_argument("--prompt",   "-p", default=None,
                        help="프롬프트 직접 입력 (지정 시 JSON 무시)")
    parser.add_argument("--negative", "-n", default="",
                        help="네거티브 프롬프트 직접 입력")
    parser.add_argument("--steps",    type=int,   default=DEFAULT_STEPS)
    parser.add_argument("--cfg",      type=float, default=DEFAULT_CFG)
    parser.add_argument("--width",    type=int,   default=DEFAULT_WIDTH)
    parser.add_argument("--height",   type=int,   default=DEFAULT_HEIGHT)
    parser.add_argument("--seed",     type=int,   default=DEFAULT_SEED)
    parser.add_argument("--output",   "-o", default=None,
                        help="출력 파일 경로 (기본값: {product_name}_{timestamp}.png)")
    args = parser.parse_args()

    # API 키 확인
    api_key = os.environ.get("HF_API_KEY", "").strip()
    if not api_key:
        api_key = input("Hugging Face API 토큰을 입력하세요: ").strip()
    if not api_key:
        print("❌ API 키가 필요합니다. https://huggingface.co/settings/tokens 에서 발급하세요.")
        return

    # 프롬프트 결정
    if args.prompt:
        positive     = args.prompt
        negative     = args.negative
        product_name = "output"
    else:
        json_path = Path(args.input)
        if not json_path.exists():
            print(f"❌ JSON 파일을 찾을 수 없습니다: {json_path}")
            return
        positive, negative, product_name = load_prompts_from_json(str(json_path))

    # 출력 경로 결정
    output_path = args.output or f"{product_name.replace(' ', '_').lower()}_{int(time.time())}.png"

    generate(
        api_key=api_key,
        positive_prompt=positive,
        negative_prompt=negative,
        steps=args.steps,
        cfg=args.cfg,
        width=args.width,
        height=args.height,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()