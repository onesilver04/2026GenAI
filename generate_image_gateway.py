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
# IMAGE_MODEL = "gpt-image-1-mini"
# IMAGE_MODEL = "gpt-image-1.5"


def load_prompt_from_json(json_path: Path) -> dict:
    """JSON에서 이미지 생성에 필요한 모든 컨텍스트를 로드합니다."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sns = data["sns_content"]
    input_data = data.get("input", {})

    return {
        # 기본 이미지 프롬프트 (참고용)
        "base_prompt": sns.get("image_prompt_for_sdxl", ""),
        "negative_prompt": sns.get("negative_prompt_for_sdxl", ""),
        # 제품 정보
        "product_category": input_data.get("product_category", ""),
        "product_name": input_data.get("product_name", ""),
        "product_description": input_data.get("product_description", ""),
        "target_group": input_data.get("target_group", ""),
        "brand_tone": input_data.get("brand_tone", ""),
        # 에이전트 분석 결과
        "creative_direction": data.get("creative_direction", {}),
        "campaign_concept": data.get("campaign_concept", {}),
        "marketing_strategy": data.get("marketing_strategy", {}),
        "sns_copy": data.get("sns_copy", {}),
        "aspect_ratio": "2:3",
    }


def build_gateway_prompt(prompt_data: dict) -> str:
    cd = prompt_data.get("creative_direction", {})
    cc = prompt_data.get("campaign_concept", {})
    strategy = prompt_data.get("marketing_strategy", {})
    sns = prompt_data.get("sns_copy", {})

    # --- 씬 기본 요소 ---
    hero_subject    = cd.get("hero_subject", "")
    scene_type      = cd.get("scene_type", "Instagram campaign hero shot")
    visual_concept  = cd.get("visual_concept", "")
    campaign_story  = cd.get("campaign_story", "")
    composition     = cd.get("composition", "")
    background      = cd.get("background", "")
    lighting        = cd.get("lighting", "soft natural light")
    texture_viz     = cd.get("texture_visualization", "")
    mood            = cd.get("mood", "premium, clean, trustworthy")
    color_palette   = ", ".join(cd.get("color_palette", []))
    props           = ", ".join(cd.get("material_and_props", []))

    # --- 캠페인 컨셉 ---
    big_idea            = cc.get("campaign_big_idea", "")
    campaign_message    = cc.get("campaign_message", "")
    visual_story        = cc.get("visual_story", visual_concept)
    emotional_keywords  = ", ".join(cc.get("emotional_keywords", []))
    trust_signals       = cc.get("trust_signals", [])
    selling_points_viz  = cc.get("selling_points_to_visualize",
                                  strategy.get("selling_points", []))

    # --- 고객 인사이트 → 비주얼 솔루션 ---
    insight_lines = [
        f"  - [{iv.get('insight','')}] → {iv.get('visual_solution','')}"
        for iv in cd.get("customer_insight_visualization", [])
    ]
    # --- 브랜드 가치 → 비주얼 솔루션 ---
    brand_lines = [
        f"  - [{bv.get('brand_value','')}] → {bv.get('visual_solution','')}"
        for bv in cd.get("brand_value_visualization", [])
    ]
    # --- 타겟 오디언스 시각 단서 ---
    audience_cues = cd.get("target_audience_visualization", [])
    if isinstance(audience_cues, list):
        audience_cues_str = " / ".join(audience_cues)
    else:
        audience_cues_str = str(audience_cues)

    # --- 회피 항목 (creative_direction 기준) ---
    things_to_avoid = cd.get("things_to_avoid", [])
    avoid_str = "\n- ".join(things_to_avoid) if things_to_avoid else ""

    trust_str = (
        "\n".join(f"  • {t}" for t in trust_signals)
        if isinstance(trust_signals, list)
        else str(trust_signals)
    )
    selling_str = "\n".join(f"  • {s}" for s in selling_points_viz)

    prompt = f"""Professional Instagram skincare campaign photograph.

━━━ CAMPAIGN BRIEF ━━━
Big Idea: {big_idea}
Core Message: {campaign_message}
Emotional Keywords: {emotional_keywords}

━━━ SCENE DIRECTION ━━━
{hero_subject}
{visual_story}
{campaign_story}

Composition: {composition}
Background: {background}
Lighting: {lighting}
Mood: {mood}
Color palette: {color_palette}
Props & materials: {props}
Product texture: {texture_viz}

━━━ PRODUCT IN SCENE ━━━
Category: {prompt_data["product_category"]}
Packaging: sunscreen tube or sun care cream tube (NOT serum bottle, dropper, toner, or transparent bottle)
Placement: naturally integrated into the scene, label visible but not forced
Occupies approximately 45–60% of the visual weight — scene tells the story, product is the anchor

━━━ WHAT TO VISUALLY COMMUNICATE ━━━
Selling points:
{selling_str}

Trust signals:
{trust_str}

Customer concerns → visual solutions:
{chr(10).join(insight_lines)}

Brand values → visual solutions:
{chr(10).join(brand_lines)}

Target audience cues: {audience_cues_str}
Target group: {prompt_data["target_group"]}
Brand tone: {prompt_data["brand_tone"]}

━━━ TECHNICAL SPECS ━━━
Style: {scene_type}, editorial beauty photography, commercial skincare Instagram ad
Depth of field: shallow (soft bokeh background)
Detail: high-detail on product packaging and cream texture
Skin: natural, healthy skin tones — not heavily retouched
Format: 2:3 portrait (Instagram)

━━━ DO NOT INCLUDE ━━━
- {avoid_str}
- Readable text overlays, fake brand logos, or watermarks
- Extra unrelated skincare products as prominent objects
- Serum bottles, dropper bottles, perfume bottles, transparent liquid bottles
- Overly busy or cluttered backgrounds that compete with the product
- Plain white/grey studio floating product render
""".strip()

    return prompt


def print_marketing_summary(prompt_data: dict) -> None:
    """이미지 생성 전 마케팅 전략 요약을 콘솔에 출력합니다."""
    cc = prompt_data.get("campaign_concept", {})
    strategy = prompt_data.get("marketing_strategy", {})
    sns = prompt_data.get("sns_copy", {})

    print("\n" + "=" * 62)
    print("  📊  마케팅 전략 요약")
    print("=" * 62)
    print(f"  🎯 제품  : {prompt_data.get('product_name','')} ({prompt_data.get('product_category','')})")
    print(f"  💡 빅 아이디어 : {cc.get('campaign_big_idea','')}")
    print(f"  📣 핵심 메시지 : {cc.get('campaign_message','')[:180]}")
    print(f"\n  📈 마케팅 방향  : {strategy.get('marketing_direction','')}")
    print(f"  🎨 추천 톤앤매너 : {strategy.get('recommended_tone','')}")
    print("\n  ✅ 주요 셀링포인트")
    for sp in strategy.get("selling_points", []):
        print(f"     • {sp}")
    print("\n  ⚠️  리스크 요인")
    for r in strategy.get("risk_to_avoid", []):
        print(f"     ✗ {r}")
    print("\n  ✍️  SNS 카피 (Instagram)")
    print(f"     제목    : {sns.get('post_title','')}")
    print(f"     후킹    : {sns.get('short_hook','')}")
    print(f"     본문    : {sns.get('main_copy','')[:220]}...")
    print(f"     해시태그 : {' '.join(sns.get('hashtags',[]))}")
    print("=" * 62)


def save_base64_image(data_url: str) -> Path:
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"campaign_image_{timestamp}.png"
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64))
    return output_path


def extract_image_url(result: dict) -> str:
    try:
        return result["data"][0]["url"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Image URL not found in API response. "
            f"Response preview: {json.dumps(result, ensure_ascii=False)[:1000]}"
        ) from exc


def main():
    if not API_KEY:
        raise RuntimeError("SMU_API_KEY 환경변수가 없습니다.")

    prompt_data = load_prompt_from_json(JSON_PATH)

    # ① 마케팅 전략 요약 출력
    print_marketing_summary(prompt_data)

    # ② 캠페인 크리에이티브 브리프 기반 이미지 프롬프트 구성
    prompt = build_gateway_prompt(prompt_data)

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
        print("\n[INFO] 이미지 생성 요청 중...")
        resp = client.post(f"{BASE_URL}/images/generate/", json=payload)
        print(f"[INFO] Response status: {resp.status_code}")
        resp.raise_for_status()
        result = resp.json()

    image_url = extract_image_url(result)

    if image_url.startswith("data:image"):
        output_path = save_base64_image(image_url)
        print(f"\n🖼️  이미지 저장 완료: {output_path}")
    else:
        print(f"\n🖼️  이미지 URL:\n{image_url}")


if __name__ == "__main__":
    main()
