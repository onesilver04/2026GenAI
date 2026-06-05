import json
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

JSON_PATH = Path("marketing_agent_result.json")
OUTPUT_DIR = Path("generated_content")
# 더 빠르고 가벼운 모델을 원한다면 "stabilityai/sdxl-turbo"도 고려해보세요.
SDXL_MODEL = "stabilityai/sdxl-turbo" 

def load_sns_prompt(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 파일이 없습니다: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sns_content", {})

def generate_image(sns_content: dict) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU(CUDA)가 필요합니다.")

    # 1. 파이프라인 로드 및 스케줄러 최적화
    print(f"[INFO] 모델 로딩 중: {SDXL_MODEL}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    
    # 속도 향상을 위한 스케줄러 설정 (기존보다 훨씬 빠름)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # GPU 메모리 최적화 (매우 중요)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    # VRAM이 아주 부족하다면 아래 주석을 해제하세요.
    # pipe.enable_model_cpu_offload() 

    # 2. 파라미터 설정
    prompt = sns_content["image_prompt_for_sdxl"]
    negative_prompt = sns_content.get("negative_prompt_for_sdxl", "text, watermark, low quality, blurry")
    
    # SDXL은 1024x1024에 최적화되어 있습니다. 너무 작거나 비율이 깨지면 속도가 느려질 수 있습니다.
    width = int(sns_content.get("image_width", 1024))
    height = int(sns_content.get("image_height", 1024))

    print(f"[INFO] 이미지 생성 시작... (Steps: 20)")
    
    # 3. 이미지 생성
    # 메모리 캐시 비우기
    torch.cuda.empty_cache()
    
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=20, # 속도를 높이려면 15~20 사이 권장
            guidance_scale=7.0,
        ).images[0]

    # 4. 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"sns_sdxl_{timestamp}.png"
    image.save(output_path)
    
    return output_path

def main():
    try:
        sns_content = load_sns_prompt(JSON_PATH)
        path = generate_image(sns_content)
        print(f"✨ 생성 완료: {path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()