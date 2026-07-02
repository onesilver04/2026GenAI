# marketing_agent/main.py
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import List, Dict, Type


DEFAULT_MLX_MODEL      = os.getenv("MLX_MODEL", "mlx-community/Dolphin3.0-Llama3.1-8B-MLX-6bit")
PERSONA_MODEL          = os.getenv("PERSONA_MODEL", "mlx-community/Dolphin3.0-Llama3.1-8B-MLX-6bit")
STRATEGY_MODEL         = os.getenv("STRATEGY_MODEL", "mlx-community/gemma-4-e2b-it-8bit")
STRATEGY_FALLBACK_MODEL = os.getenv("STRATEGY_FALLBACK_MODEL", "mlx-community/gemma-2-2b-it-4bit")
SNS_COPY_MODEL         = os.getenv("SNS_COPY_MODEL", DEFAULT_MLX_MODEL)
MLX_MAX_TOKENS         = int(os.getenv("MLX_MAX_TOKENS", "2048"))


@dataclass
class CampaignInput:
    product_category: str
    brand_values: str
    product_name: str
    product_description: str
    target_group: str
    brand_tone: str
    platform: str


@dataclass
class PersonaFeedback:
    persona_name: str
    persona_profile: str
    positive_reaction: str
    negative_reaction: str
    purchase_intent: int
    improvement_suggestion: str


@lru_cache(maxsize=4)
def load_mlx_model(model_name: str = DEFAULT_MLX_MODEL):
    with without_local_mlx_shadow():
        from mlx_lm import load
        return load(model_name)


@contextmanager
def without_local_mlx_shadow():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = list(sys.path)
    local_mlx = sys.modules.get("mlx")

    if getattr(local_mlx, "__file__", None) == os.path.join(script_dir, "mlx.py"):
        del sys.modules["mlx"]

    sys.path = [
        path for path in sys.path
        if os.path.abspath(path or os.curdir) != script_dir
    ]
    try:
        yield
    finally:
        sys.path = original_path
        if local_mlx is not None and "mlx" not in sys.modules:
            sys.modules["mlx"] = local_mlx


def build_mlx_prompt(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "has_chat_template", False):
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            )
    return prompt


def call_mlx(
    prompt: str,
    model_name: str = DEFAULT_MLX_MODEL,
    fallback_model_name: str | None = None,
    max_tokens: int = MLX_MAX_TOKENS,
) -> str:
    with without_local_mlx_shadow():
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

    try:
        model, tokenizer = load_mlx_model(model_name)
    except ValueError as exc:
        if fallback_model_name is None:
            raise
        print(f"\nFallback: {model_name} → {fallback_model_name} ({str(exc).splitlines()[0]})\n")
        model, tokenizer = load_mlx_model(fallback_model_name)

    mlx_prompt = build_mlx_prompt(tokenizer, prompt)
    sampler = make_sampler(temp=0.0)
    return generate(
        model, tokenizer, prompt=mlx_prompt,
        max_tokens=max_tokens, sampler=sampler, verbose=False,
    ).strip()


# ─────────────────────────────────────────
# 에이전트 함수
# ─────────────────────────────────────────

def generate_product_concept(
    product_category: str,
    brand_values: str,
    target_group: str,
    brand_tone: str,
) -> Dict:
    prompt = f"""
You are a product naming and branding agent.

Based on the product category, brand values, target audience, and brand tone,
create a marketable product name and a concise product description.

Important context:
- "Brand Values" may be copied from a company's website sections such as ethics
  management, corporate values, mission, ESG principles, sustainability policy,
  social responsibility, or management philosophy.
- Treat Brand Values as the company's strategic and ethical product design
  principles, not just as decorative marketing keywords.
- The product concept must feel naturally aligned with these values through its
  benefits, materials, features, user experience, and positioning.

[Product Category]
{product_category}

[Brand Values]
{brand_values}

[Target Audience]
{target_group}

[Brand Tone]
{brand_tone}

Return ONLY a valid JSON object in the following format.

{{
  "product_name": "A unique and marketable product name",
  "product_description": "A clear product description including key benefits, product features, and target user needs"
}}

Rules:
- The product name must sound like a real brand product.
- The product name should reflect the brand values.
- The product description must explain how the product embodies the brand values
  in concrete product-level choices.
- Convert abstract corporate values into product-relevant benefits.
- Do not mention personal data protection unless the product is a digital service.
- Do not use generic category names such as "Sunscreen" or "Keyboard".
- Return JSON only. Do not include markdown. Do not include explanations outside JSON.
"""
    result = call_mlx(prompt)
    return safe_json_loads(result, dict)


def generate_customer_feedback(user_input: CampaignInput, num_personas: int = 5) -> List[Dict]:
    prompt = f"""
You are a virtual customer simulation agent for marketing research.

Based on the following product and target audience information,
create {num_personas} different customer personas and simulate
how each persona would react to the product and campaign.

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Brand Values]
{user_input.brand_values}

[Target Audience]
{user_input.target_group}

[Brand Tone]
{user_input.brand_tone}

Return ONLY a valid JSON array in the following format.

[
  {{
    "persona_name": "Name",
    "persona_profile": "Age, occupation, interests, buying behavior",
    "positive_reaction": "Positive response",
    "negative_reaction": "Negative response",
    "purchase_intent": 1,
    "improvement_suggestion": "Suggestion for improvement"
  }}
]

Rules:
- purchase_intent must be an integer from 1 to 5.
- Interpret Brand Values as the company's stated ethics, corporate values, mission,
  ESG principles, or management philosophy from its website, and evaluate whether
  the product feels consistent with those values.
- Do not include markdown.
- Do not include explanations outside JSON.
"""
    result = call_mlx(prompt, PERSONA_MODEL)
    return safe_json_loads(result, list)


def analyze_strategy(user_input: CampaignInput, feedbacks: List[Dict]) -> Dict:
    prompt = f"""
You are a marketing strategy analysis agent.

Analyze the following virtual customer feedback and generate
a marketing strategy for the campaign.

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Brand Values]
{user_input.brand_values}

[Target Audience]
{user_input.target_group}

[Virtual Customer Feedback]
{json.dumps(feedbacks, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object in the following format.

{{
  "main_insight": "Summary of key customer reactions",
  "customer_pain_points": ["Pain point 1", "Pain point 2"],
  "selling_points": ["Key selling point 1", "Key selling point 2"],
  "marketing_direction": "Overall marketing direction",
  "message_strategy": "Messaging strategy for SNS marketing",
  "recommended_tone": "Recommended tone and manner",
  "risk_to_avoid": ["Risk factor 1", "Risk factor 2"]
}}

Rules:
- Return JSON only.
- Interpret Brand Values as the company's stated ethics, corporate values, mission,
  ESG principles, or management philosophy from its website, and make the strategy
  reinforce those values without greenwashing or unsupported claims.
- Do not include markdown.
- Do not include explanations outside JSON.
"""
    result = call_mlx(prompt, STRATEGY_MODEL, STRATEGY_FALLBACK_MODEL)
    return safe_json_loads(result, dict)


def generate_campaign_concept(
    user_input: CampaignInput,
    feedbacks: List[Dict],
    strategy: Dict,
) -> Dict:
    prompt = f"""
You are a senior Instagram campaign planner for premium skincare brands.

Your role:
- Read customer personas, their positive reactions, concerns, purchase intent, and improvement suggestions.
- Extract the strongest emotional and strategic campaign message.
- Translate customer insights into an Instagram campaign concept.
- Make sure the campaign addresses customer concerns, not only selling points.
- The campaign should feel emotional, premium, trustworthy, and visually memorable.

[Product Category]
{user_input.product_category}

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Target Audience]
{user_input.target_group}

[Brand Tone]
{user_input.brand_tone}

[Brand Values]
{user_input.brand_values}

[Virtual Customer Feedback]
{json.dumps(feedbacks, ensure_ascii=False, indent=2)}

[Marketing Strategy]
{json.dumps(strategy, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object in the following format.

{{
  "campaign_big_idea": "Core emotional campaign idea",
  "campaign_message": "Main message the Instagram visual should communicate",
  "primary_persona_insight": "Most important customer insight from the personas",
  "customer_concerns_to_address": ["Concern 1", "Concern 2"],
  "selling_points_to_visualize": ["Selling point 1", "Selling point 2"],
  "visual_story": "How the image should tell the campaign story",
  "emotional_keywords": ["keyword1", "keyword2", "keyword3"],
  "trust_signals": ["trust signal 1", "trust signal 2"]
}}

Rules:
- Do not write generic product-photo directions.
- Focus on the campaign story and customer motivation.
- Reflect customer concerns such as price, SPF clarity, sensitive skin trust, and environmental transparency when relevant.
- Avoid unsupported claims.
- Return JSON only. Do not include markdown.
"""
    result = call_mlx(prompt, SNS_COPY_MODEL, DEFAULT_MLX_MODEL)
    return safe_json_loads(result, dict)


def generate_sns_copy(user_input: CampaignInput, strategy: Dict) -> Dict:
    prompt = f"""
You are an Instagram beauty copywriter specialized in clean beauty and skincare advertising.
Your role:
- Turn marketing strategy into natural Instagram copy.
- Write like a real beauty brand, not like a corporate report.
- Use emotionally appealing but realistic language.
- Avoid exaggerated or unsupported claims.

Generate SNS copy optimized for {user_input.platform}.

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Brand Tone]
{user_input.brand_tone}

[Brand Values]
{user_input.brand_values}

[Marketing Strategy]
{json.dumps(strategy, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object in the following format.

{{
  "post_title": "SNS post title",
  "main_copy": "2-4 short emotionally appealing sentences",
  "short_hook": "Short attention-grabbing hook",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
}}

Rules:
- Make it sound like a real Instagram beauty advertisement.
- Do not sound like a corporate introduction.
- Use short, natural, emotionally appealing sentences.
- Emphasize the strongest selling points from the strategy.
- Reflect the target audience naturally.
- Avoid unsupported claims.
- Return JSON only. Do not include markdown.
"""
    result = call_mlx(prompt, SNS_COPY_MODEL, DEFAULT_MLX_MODEL)
    return safe_json_loads(result, dict)


def generate_creative_direction(
    user_input: CampaignInput,
    feedbacks: List[Dict],
    strategy: Dict,
    sns_copy: Dict,
    campaign_concept: Dict,
) -> Dict:
    prompt = f"""
You are an advertising creative director for premium Instagram skincare campaigns.

Your role:
- Translate customer research, campaign concept, and marketing strategy into a visual creative brief.
- Make the image feel like an Instagram campaign, not a plain catalog product photo.
- Balance product visibility with emotional storytelling.
- Show the product, its texture, and the feeling of using it.
- Make customer concerns visible through reassuring visual cues.

[Product Category]
{user_input.product_category}

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Target Audience]
{user_input.target_group}

[Brand Tone]
{user_input.brand_tone}

[Brand Values]
{user_input.brand_values}

[Virtual Customer Feedback]
{json.dumps(feedbacks, ensure_ascii=False, indent=2)}

[Marketing Strategy]
{json.dumps(strategy, ensure_ascii=False, indent=2)}

[SNS Copy]
{json.dumps(sns_copy, ensure_ascii=False, indent=2)}

[Campaign Concept]
{json.dumps(campaign_concept, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object in the following format.

{{
  "visual_concept": "Core visual concept",
  "campaign_story": "How the visual communicates the campaign message",
  "hero_subject": "Main subject of the scene — describe who, what they are doing, where",
  "scene_type": "Instagram campaign hero shot / sensory texture shot / product lifestyle campaign",
  "composition": "How product, subject, props, and background are arranged in the frame",
  "product_visibility": "How the product remains visible without looking like a plain catalog shot",
  "texture_visualization": "How the product formula or cream texture should be shown",
  "customer_insight_visualization": [
    {{
      "insight": "Customer insight or concern",
      "visual_solution": "How to express it visually"
    }}
  ],
  "brand_value_visualization": [
    {{
      "brand_value": "Brand value",
      "visual_solution": "How to express it visually"
    }}
  ],
  "target_audience_visualization": [
    "Visual cue for target audience 1",
    "Visual cue for target audience 2"
  ],
  "background": "Specific background description — location, texture, color",
  "lighting": "Lighting description — direction, quality, time of day",
  "color_palette": ["color1", "color2", "color3"],
  "material_and_props": ["prop1", "prop2", "prop3"],
  "mood": "Overall mood and emotional atmosphere",
  "things_to_avoid": ["avoid item 1", "avoid item 2"]
}}

Rules:
- hero_subject must describe a person in a scene, not just the product alone.
- Do not create a plain floating product render.
- Use one main hero product; allow one small texture element such as a cream smear.
- The product should occupy around 45-60% of the frame, not 80-90%.
- Reflect persona concerns visually: SPF clarity, sensitive skin trust, eco transparency, premium quality.
- Do not use serum bottles, droppers, toner bottles, or transparent liquid bottles.
- Return JSON only. Do not include markdown.
"""
    result = call_mlx(prompt, SNS_COPY_MODEL, DEFAULT_MLX_MODEL)
    return safe_json_loads(result, dict)


def generate_image_prompt(
    user_input: CampaignInput,
    strategy: Dict,
    sns_copy: Dict,
    campaign_concept: Dict,
    creative_direction: Dict,
) -> Dict:
    prompt = f"""
You are an AI image prompt engineer for premium Instagram skincare campaign photography.

Your role:
- Write a vivid, scene-narrative image prompt optimized for modern AI image generation
  models (Gemini Image, GPT-image) — NOT SDXL weighted prompts.
- The prompt should read like a brief to a professional photographer / film director.
- Describe WHO is in the scene, WHAT they are doing, WHERE they are, and
  HOW the product is naturally integrated.
- Include specific sensory details: textures, skin, light quality, colors, expressions.
- The scene should TELL the campaign story, not just SHOW a product.

[Product Category]
{user_input.product_category}

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Target Audience]
{user_input.target_group}

[Brand Tone]
{user_input.brand_tone}

[Marketing Strategy]
{json.dumps(strategy, ensure_ascii=False, indent=2)}

[SNS Copy]
{json.dumps(sns_copy, ensure_ascii=False, indent=2)}

[Campaign Concept]
{json.dumps(campaign_concept, ensure_ascii=False, indent=2)}

[Creative Direction]
{json.dumps(creative_direction, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object in the following format.

{{
  "image_prompt_for_sdxl": "3-5 sentence vivid scene narrative in English",
  "negative_prompt_for_sdxl": "Detailed negative prompt listing concrete visual errors",
  "image_width": 640,
  "image_height": 960
}}

Rules for image_prompt_for_sdxl:
- Write as flowing narrative sentences, NOT a bulleted list of attributes.
- Describe the full scene: person, action, setting, light, product placement, emotion.
- Do NOT start with "A product photo of..." — start with the scene and subject.
- The sunscreen tube should appear naturally in the person's hand or on a surface.
- Convey the campaign emotion and brand values through the scene atmosphere.
- Mention: lighting quality, color palette, composition feel, skin texture, product texture.
- Do NOT ask for text rendering, logos, or brand names on the packaging.
- 3-5 sentences of flowing, vivid description.

Rules for negative_prompt_for_sdxl:
- List specific visual errors: wrong bottle types, extra products, fake text, blurry areas.
- Include: "serum bottle, dropper bottle, toner bottle, transparent liquid bottle, perfume bottle"
- Include: "plain floating product render, white studio background, crowded flat lay"
- Include: "fake readable text, brand logo, watermark, unreadable characters"
- Return JSON only. Do not include markdown.
"""
    result = call_mlx(prompt, SNS_COPY_MODEL, DEFAULT_MLX_MODEL)
    image_prompt = safe_json_loads(result, dict)
    image_prompt["image_width"]  = normalize_sdxl_dimension(image_prompt.get("image_width",  640))
    image_prompt["image_height"] = normalize_sdxl_dimension(image_prompt.get("image_height", 960))
    return image_prompt


# ─────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────

def normalize_sdxl_dimension(value) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return 640
    dimension = max(512, min(1536, dimension))
    return round(dimension / 8) * 8


def safe_json_loads(text: str, expected_type: Type[dict] | Type[list] | None = None):
    text = text.strip()
    try:
        value = json.loads(text)
        if expected_type is None or isinstance(value, expected_type):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if expected_type is not None:
            if isinstance(value, expected_type):
                return value
            continue
        if isinstance(value, (dict, list)):
            return value

    expected_name = expected_type.__name__ if expected_type is not None else "JSON object or array"
    raise ValueError(f"JSON parsing failed. Expected {expected_name}.\nModel output preview:\n{text[:1000]}")


# ─────────────────────────────────────────
# 마케팅 리포트 출력
# ─────────────────────────────────────────

def print_marketing_report(
    user_input: CampaignInput,
    feedbacks: List[Dict],
    strategy: Dict,
    sns_copy: Dict,
    campaign_concept: Dict,
    creative_direction: Dict,
) -> None:
    """마케터를 위한 마케팅 전략 요약 리포트를 콘솔에 출력합니다."""
    avg_intent = (
        sum(f.get("purchase_intent", 0) for f in feedbacks) / len(feedbacks)
        if feedbacks else 0
    )

    print("\n" + "═" * 62)
    print("  🚀  마케팅 에이전트 최종 리포트")
    print("═" * 62)

    print(f"\n  📦  제품 정보")
    print(f"      제품명  : {user_input.product_name}")
    print(f"      카테고리: {user_input.product_category}")
    print(f"      타겟    : {user_input.target_group}")
    print(f"      설명    : {user_input.product_description[:180]}...")

    print(f"\n  👥  가상 고객 피드백  ({len(feedbacks)}명 시뮬레이션)")
    print(f"      평균 구매 의향: {avg_intent:.1f} / 5")
    for fb in feedbacks:
        intent = fb.get("purchase_intent", "-")
        name   = fb.get("persona_name", "")
        pos    = fb.get("positive_reaction", "")[:80]
        neg    = fb.get("negative_reaction", "")[:60]
        print(f"      [{intent}/5] {name}")
        print(f"            + {pos}")
        print(f"            - {neg}")

    print(f"\n  📈  마케팅 전략")
    print(f"      인사이트  : {strategy.get('main_insight','')[:200]}")
    print(f"      방향성    : {strategy.get('marketing_direction','')}")
    print(f"      톤앤매너  : {strategy.get('recommended_tone','')}")
    print(f"      셀링포인트:")
    for sp in strategy.get("selling_points", []):
        print(f"        ✓ {sp}")
    print(f"      고객 페인포인트:")
    for pp in strategy.get("customer_pain_points", []):
        print(f"        ! {pp}")

    print(f"\n  💡  캠페인 컨셉")
    print(f"      빅 아이디어 : {campaign_concept.get('campaign_big_idea','')}")
    print(f"      메시지      : {campaign_concept.get('campaign_message','')[:220]}")
    print(f"      감성 키워드 : {', '.join(campaign_concept.get('emotional_keywords', []))}")

    print(f"\n  🎨  크리에이티브 방향")
    print(f"      씬 타입   : {creative_direction.get('scene_type','')}")
    print(f"      비주얼    : {creative_direction.get('visual_concept','')[:180]}")
    print(f"      무드      : {creative_direction.get('mood','')}")
    print(f"      색상 팔레트: {', '.join(creative_direction.get('color_palette', []))}")

    print(f"\n  ✍️   SNS 카피 (Instagram)")
    print(f"      제목    : {sns_copy.get('post_title','')}")
    print(f"      후킹    : {sns_copy.get('short_hook','')}")
    print(f"      본문    :\n        {sns_copy.get('main_copy','')[:300]}")
    print(f"      해시태그: {' '.join(sns_copy.get('hashtags', []))}")

    print("\n" + "═" * 62)


# ─────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────

def run_pipeline():
    print("\n" + "=" * 62)
    print("  🤖  마케팅 에이전트 파이프라인 시작")
    print("=" * 62)

    product_category = input("제품 카테고리 (예: sunscreen, keyboard, tumbler): ")
    brand_values     = input("브랜드 가치 (홈페이지 기업 가치/경영 철학 등): ")
    target_group     = input("타겟 고객: ")
    brand_tone       = input("브랜드 톤앤매너: ")
    platform         = input("플랫폼 (예: Instagram, TikTok): ")

    print("\n[0] 제품명 및 컨셉 생성 중...")
    product_concept = generate_product_concept(
        product_category=product_category,
        brand_values=brand_values,
        target_group=target_group,
        brand_tone=brand_tone,
    )

    user_input = CampaignInput(
        product_category=product_category,
        brand_values=brand_values,
        product_name=product_concept["product_name"],
        product_description=product_concept["product_description"],
        target_group=target_group,
        brand_tone=brand_tone,
        platform=platform,
    )
    print(f"    → 제품명: {user_input.product_name}")

    print("\n[1] 가상 고객 피드백 생성 중...")
    feedbacks = generate_customer_feedback(user_input)

    print("\n[2] 마케팅 전략 분석 중...")
    strategy = analyze_strategy(user_input, feedbacks)

    print("\n[3] SNS 카피 생성 중...")
    sns_copy = generate_sns_copy(user_input, strategy)

    print("\n[4] 캠페인 컨셉 생성 중...")
    campaign_concept = generate_campaign_concept(user_input, feedbacks, strategy)

    print("\n[5] 크리에이티브 방향 생성 중...")
    creative_direction = generate_creative_direction(
        user_input, feedbacks, strategy, sns_copy, campaign_concept,
    )

    print("\n[6] 이미지 프롬프트 생성 중...")
    image_prompt = generate_image_prompt(
        user_input, strategy, sns_copy, campaign_concept, creative_direction,
    )

    sns_content = {**sns_copy, **image_prompt}

    final_result = {
        "agent_models": {
            "product_concept_agent":           DEFAULT_MLX_MODEL,
            "persona_agent":                   PERSONA_MODEL,
            "strategy_agent":                  STRATEGY_MODEL,
            "strategy_fallback_agent":         STRATEGY_FALLBACK_MODEL,
            "sns_copy_and_image_prompt_agent": SNS_COPY_MODEL,
        },
        "input":                    asdict(user_input),
        "virtual_customer_feedbacks": feedbacks,
        "marketing_strategy":       strategy,
        "sns_copy":                 sns_copy,
        "campaign_concept":         campaign_concept,
        "creative_direction":       creative_direction,
        "sns_content":              sns_content,
    }

    with open("marketing_agent_result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    print("\n[7] marketing_agent_result.json 저장 완료.")

    # ── 마케팅 리포트 출력 ──────────────────────────────────────
    print_marketing_report(
        user_input, feedbacks, strategy, sns_copy, campaign_concept, creative_direction,
    )

    # ── 이미지 자동 생성 ────────────────────────────────────────
    print("\n[8] 홍보 이미지 생성 중...")
    try:
        from generate_image_gateway import main as run_gateway
        run_gateway()
    except Exception as exc:
        print(f"\n[경고] 이미지 자동 생성 실패: {exc}")
        print("→ generate_image_gateway.py를 별도로 실행해주세요.")


if __name__ == "__main__":
    run_pipeline()