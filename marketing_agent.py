from __future__ import annotations

import json
import os
from urllib import error, request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Type


@dataclass
class CampaignInput:
    product_category: str
    brand_values: str
    product_name: str
    product_description: str
    target_group: str
    brand_tone: str
    platform: str


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
PERSONA_MODEL = os.getenv("PERSONA_MODEL", DEFAULT_LLM_MODEL)
STRATEGY_MODEL = os.getenv("STRATEGY_MODEL", "gemma2:9b")
STRATEGY_FALLBACK_MODEL = os.getenv("STRATEGY_FALLBACK_MODEL", DEFAULT_LLM_MODEL)
SNS_COPY_MODEL = os.getenv("SNS_COPY_MODEL", DEFAULT_LLM_MODEL)

# 이미지 프롬프트 전용 모델
IMAGE_PROMPT_MODEL = os.getenv("IMAGE_PROMPT_MODEL", "llama3.1:8b")

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))


def normalize_ollama_model_name(model_name: str) -> str:
    model_name = model_name.strip()
    if "/" in model_name:
        model_name = model_name.rsplit("/", 1)[-1]
    return model_name.lower()


def call_llm(
    prompt: str,
    model_name: str = DEFAULT_LLM_MODEL,
    fallback_model_name: str | None = None,
    max_tokens: int = LLM_MAX_TOKENS,
) -> str:
    model_name = normalize_ollama_model_name(model_name)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body)["response"].strip()

    except (error.URLError, error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        if fallback_model_name is None:
            raise RuntimeError(f"Ollama request failed for {model_name}: {exc}") from exc

        fallback_model_name = normalize_ollama_model_name(fallback_model_name)

        if fallback_model_name == model_name:
            raise RuntimeError(f"Ollama request failed for {model_name}: {exc}") from exc

        print(f"\nOllama request failed for {model_name}. Falling back to {fallback_model_name}.")
        return call_llm(prompt, fallback_model_name, None, max_tokens)


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

        if expected_type is None or isinstance(value, expected_type):
            return value

    raise ValueError(f"JSON parsing failed. Expected {expected_type}. Output preview:\n{text[:1000]}")


def generate_product_concept(
    product_category: str,
    brand_values: str,
    target_group: str,
    brand_tone: str,
) -> Dict:
    prompt = f"""
You are a product naming and branding agent.

Create a marketable product name and product description.

[Product Category]
{product_category}

[Brand Values]
{brand_values}

[Target Audience]
{target_group}

[Brand Tone]
{brand_tone}

Return ONLY valid JSON.

{{
  "product_name": "Unique product name",
  "product_description": "Clear product description"
}}

Rules:
- The product name must sound like a real product.
- The product description must align with brand values.
- Convert abstract corporate values into product-relevant benefits.
- Do not mention personal data protection unless the product is a digital service.
- For physical consumer goods, translate trust and data protection into product safety, ingredient transparency, quality control, and responsible communication.
- Return JSON only.
"""
    result = call_llm(prompt, DEFAULT_LLM_MODEL)
    return safe_json_loads(result, dict)


def generate_customer_feedback(user_input: CampaignInput, num_personas: int = 5) -> List[Dict]:
    prompt = f"""
You are a virtual customer simulation agent.

Create {num_personas} different customer personas and reactions.

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Product Category]
{user_input.product_category}

[Brand Values]
{user_input.brand_values}

[Target Audience]
{user_input.target_group}

Return ONLY valid JSON array.

[
  {{
    "persona_name": "Name",
    "persona_profile": "Age, occupation, lifestyle, buying behavior",
    "positive_reaction": "Positive response",
    "negative_reaction": "Negative response",
    "purchase_intent": 1,
    "improvement_suggestion": "Suggestion"
  }}
]

Rules:
- purchase_intent must be integer from 1 to 5.
- Make customer concerns realistic for the target audience.
- Do not invent unsupported product claims.
- Return JSON only.
"""
    result = call_llm(prompt, PERSONA_MODEL)
    return safe_json_loads(result, list)


def analyze_strategy(user_input: CampaignInput, feedbacks: List[Dict]) -> Dict:
    prompt = f"""
You are a marketing strategy analyst.

Analyze customer feedback and generate a marketing strategy.

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Product Category]
{user_input.product_category}

[Brand Values]
{user_input.brand_values}

[Target Audience]
{user_input.target_group}

[Virtual Customer Feedback]
{json.dumps(feedbacks, ensure_ascii=False, indent=2)}

Return ONLY valid JSON.

{{
  "main_insight": "Summary",
  "customer_pain_points": ["Pain point 1", "Pain point 2"],
  "selling_points": ["Selling point 1", "Selling point 2"],
  "marketing_direction": "Direction",
  "message_strategy": "Message strategy",
  "recommended_tone": "Tone",
  "risk_to_avoid": ["Risk 1", "Risk 2"]
}}

Rules:
- Avoid greenwashing.
- Avoid unsupported claims.
- Strategy must reflect target audience.
- Return JSON only.
"""
    result = call_llm(prompt, STRATEGY_MODEL, STRATEGY_FALLBACK_MODEL)
    return safe_json_loads(result, dict)


def generate_sns_copy(user_input: CampaignInput, strategy: Dict) -> Dict:
    prompt = f"""
You are an Instagram beauty copywriter.

Generate SNS copy for an Instagram sponsored post.

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Product Category]
{user_input.product_category}

[Brand Tone]
{user_input.brand_tone}

[Target Audience]
{user_input.target_group}

[Marketing Strategy]
{json.dumps(strategy, ensure_ascii=False, indent=2)}

Return ONLY valid JSON.

{{
  "post_title": "SNS post title",
  "main_copy": "2-4 short Instagram ad sentences",
  "short_hook": "Short hook",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
}}

Rules:
- Make it sound like a real Instagram sponsored beauty advertisement.
- Do not sound like a corporate introduction.
- Use short, emotionally appealing sentences.
- Avoid unsupported claims.
- Return JSON only.
"""
    result = call_llm(prompt, SNS_COPY_MODEL, DEFAULT_LLM_MODEL)
    return safe_json_loads(result, dict)


def generate_ad_concept(
    user_input: CampaignInput,
    strategy: Dict,
    sns_copy: Dict,
) -> Dict:
    prompt = f"""
You are a senior creative director at a global beauty advertising agency.

Create a visual advertising concept for an Instagram sponsored advertisement.

[Product Category]
{user_input.product_category}

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Brand Tone]
{user_input.brand_tone}

[Target Audience]
{user_input.target_group}

[Marketing Strategy]
{json.dumps(strategy, ensure_ascii=False, indent=2)}

[SNS Copy]
{json.dumps(sns_copy, ensure_ascii=False, indent=2)}

Return ONLY valid JSON.

{{
  "campaign_type": "Instagram Sponsored Advertisement",
  "visual_style": "Visual style",
  "shooting_style": "Shooting style",
  "composition": "Composition",
  "background_style": "Background style",
  "lighting": "Lighting style",
  "mood": "Mood",
  "hero_product_rule": "Rule for main product"
}}

Rules:
- The campaign_type must be exactly "Instagram Sponsored Advertisement".
- The shooting_style must focus on hero product photography.
- The composition must use ONE SINGLE main product.
- Do not suggest flat lay, pattern layout, repeated products, product grid, or catalog arrangement.
- The concept must fit the product category and target audience.
- Return JSON only.
"""
    result = call_llm(prompt, IMAGE_PROMPT_MODEL, DEFAULT_LLM_MODEL)
    return safe_json_loads(result, dict)


def generate_image_prompt(
    user_input: CampaignInput,
    strategy: Dict,
    sns_copy: Dict,
    ad_concept: Dict,
) -> Dict:
    prompt = f"""
You are a professional image prompt engineer for premium beauty advertising.

Your task is to create image generation prompts for ComfyUI / SDXL.

The final image must look like a real Instagram Sponsored Advertisement.

Important:
- The image must show ONE SINGLE HERO PRODUCT.
- The product must be the main focus.
- The product should occupy most of the frame.
- Do not create a pattern, product grid, flat lay, collage, or repeated product layout.

Good Example 1:

Input:
Product category: sunscreen
Target: sensitive skin beauty users
Advertising style: Instagram Sponsored Advertisement

Output:
{{
  "image_prompt_for_sdxl": "Professional Instagram sponsored beauty advertisement. ONE SINGLE HERO PRODUCT. A premium matte SPF50+ sunscreen tube standing upright in the center of the frame. Hero product occupying most of the image. Luxury cosmetic product photography. Soft natural sunlight. Clean beige background. Shallow depth of field. High-end Korean skincare campaign. Editorial beauty photography. Realistic commercial product shot. Premium minimal packaging. Healthy skin glow aesthetic.",
  "negative_prompt_for_sdxl": "multiple products, repeated products, duplicate tubes, flat lay, product grid, catalog photography, collage layout, many tubes, product pattern, text, logo, watermark, blurry, low quality, deformed packaging, serum bottle, dropper bottle, transparent liquid bottle",
  "image_width": 768,
  "image_height": 1024
}}

Good Example 2:

Input:
Product category: luxury perfume
Target: premium fragrance consumers
Advertising style: Instagram Sponsored Advertisement

Output:
{{
  "image_prompt_for_sdxl": "Professional Instagram sponsored luxury fragrance advertisement. ONE SINGLE HERO PRODUCT. A premium perfume bottle centered on a marble pedestal. Product occupies most of the frame. Luxury editorial photography. Soft dramatic lighting. Dark elegant background. Shallow depth of field. High-end commercial product shot. Premium fragrance campaign aesthetic. Minimal composition. Realistic studio photography.",
  "negative_prompt_for_sdxl": "multiple bottles, repeated products, product grid, flat lay, catalog layout, collage, text, logo, watermark, blurry, low quality, distorted bottle, cluttered background",
  "image_width": 768,
  "image_height": 1024
}}

Good Example 3:

Input:
Product category: reusable tumbler
Target: office workers
Advertising style: Instagram Sponsored Advertisement

Output:
{{
  "image_prompt_for_sdxl": "Professional Instagram sponsored lifestyle product advertisement. ONE SINGLE HERO PRODUCT. A premium reusable tumbler placed upright on a modern desk. Product occupies most of the frame. Clean minimalist workspace background. Soft morning sunlight. Shallow depth of field. Commercial product photography. Modern sustainable lifestyle campaign. Calm premium aesthetic. Realistic editorial product shot.",
  "negative_prompt_for_sdxl": "multiple tumblers, repeated products, flat lay, product pattern, product grid, catalog photography, collage layout, text, logo, watermark, blurry, low quality, deformed product, cluttered background",
  "image_width": 768,
  "image_height": 1024
}}

Now create the prompt for this product.

[Product Category]
{user_input.product_category}

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Brand Tone]
{user_input.brand_tone}

[Target Audience]
{user_input.target_group}

[Advertising Concept]
{json.dumps(ad_concept, ensure_ascii=False, indent=2)}

[Marketing Strategy]
{json.dumps(strategy, ensure_ascii=False, indent=2)}

[SNS Copy]
{json.dumps(sns_copy, ensure_ascii=False, indent=2)}

Return ONLY valid JSON.

{{
  "image_prompt_for_sdxl": "English positive prompt",
  "negative_prompt_for_sdxl": "English negative prompt",
  "image_width": 512,
  "image_height": 512
}}

Strict Rules:
- The image must clearly show the product.
- The product packaging must be fully visible.
- The product must be recognizable at first glance.
- Prefer realistic catalog-style product photography over artistic interpretation.
- Avoid abstract artistic effects.
- Avoid floating fragments.
- Avoid surreal compositions.
- Avoid conceptual art.
- The prompt must explicitly include: "ONE SINGLE HERO PRODUCT".
- The prompt must explicitly include: "Instagram sponsored advertisement".
- The prompt must describe professional commercial product photography.
- The prompt must describe the product as the visual focus.
- If product category is sunscreen, describe a matte sunscreen tube or sun care cream tube.
- If product category is sunscreen, include SPF, UV protection, sun care, soft sunlight, and beauty campaign mood.
- Do not describe sunscreen as serum, toner, essence, perfume, mist, transparent liquid bottle, or dropper bottle.
- Do not include readable text, logo, watermark, or letters on the product.
- The negative prompt must include: multiple products, duplicate products, repeated products, flat lay, product grid, catalog photography, collage layout.
- Return JSON only.
"""

    result = call_llm(prompt, IMAGE_PROMPT_MODEL, DEFAULT_LLM_MODEL)
    image_prompt = safe_json_loads(result, dict)

    fixed_negative_terms = [
        "multiple products",
        "duplicate products",
        "repeated products",
        "many tubes",
        "flat lay",
        "product grid",
        "catalog photography",
        "collage layout",
        "pattern layout",
        "text",
        "logo",
        "watermark",
        "blurry",
        "low quality",
        "deformed packaging",
        "cropped product",
        "serum bottle",
        "dropper bottle",
        "transparent liquid bottle",
        "toner bottle",
        "essence bottle",
        "perfume bottle",
    ]

    existing_negative = image_prompt.get("negative_prompt_for_sdxl", "")
    image_prompt["negative_prompt_for_sdxl"] = (
        existing_negative + ", " + ", ".join(fixed_negative_terms)
    )

    image_prompt["image_width"] = int(image_prompt.get("image_width", 768))
    image_prompt["image_height"] = int(image_prompt.get("image_height", 1024))

    return image_prompt


def run_pipeline():
    product_category = input("Product category: ")
    brand_values = input("Brand values: ")
    target_group = input("Target audience: ")
    brand_tone = input("Brand tone: ")
    platform = input("Platform (Instagram, etc): ")

    print("\n[1] Generating product concept...")
    concept = generate_product_concept(
        product_category,
        brand_values,
        target_group,
        brand_tone,
    )

    user_input = CampaignInput(
        product_category=product_category,
        brand_values=brand_values,
        product_name=concept["product_name"],
        product_description=concept["product_description"],
        target_group=target_group,
        brand_tone=brand_tone,
        platform=platform,
    )

    print("[2] Generating customer feedback...")
    feedbacks = generate_customer_feedback(user_input)

    print("[3] Analyzing marketing strategy...")
    strategy = analyze_strategy(user_input, feedbacks)

    print("[4] Generating SNS copy...")
    sns_copy = generate_sns_copy(user_input, strategy)

    print("[5] Generating ad concept...")
    ad_concept = generate_ad_concept(user_input, strategy, sns_copy)

    print("[6] Generating image prompt...")
    image_prompt = generate_image_prompt(
        user_input=user_input,
        strategy=strategy,
        sns_copy=sns_copy,
        ad_concept=ad_concept,
    )

    sns_content = {
        **sns_copy,
        **image_prompt,
    }

    final_result = {
        "agent_models": {
            "product_concept_agent": DEFAULT_LLM_MODEL,
            "persona_agent": PERSONA_MODEL,
            "strategy_agent": STRATEGY_MODEL,
            "strategy_fallback_agent": STRATEGY_FALLBACK_MODEL,
            "sns_copy_agent": SNS_COPY_MODEL,
            "ad_concept_agent": IMAGE_PROMPT_MODEL,
            "image_prompt_agent": IMAGE_PROMPT_MODEL,
        },
        "input": asdict(user_input),
        "virtual_customer_feedbacks": feedbacks,
        "marketing_strategy": strategy,
        "ad_concept": ad_concept,
        "sns_content": sns_content,
    }

    with open("marketing_agent_result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\nCompleted: marketing_agent_result.json saved.")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_pipeline()