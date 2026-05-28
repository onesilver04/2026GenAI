from __future__ import annotations
import json
import os
from urllib import error, request
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from datetime import datetime
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


@dataclass
class PersonaFeedback:
    persona_name: str
    persona_profile: str
    positive_reaction: str
    negative_reaction: str
    purchase_intent: int
    improvement_suggestion: str


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
PERSONA_MODEL = os.getenv("PERSONA_MODEL", DEFAULT_LLM_MODEL)
STRATEGY_MODEL = os.getenv("STRATEGY_MODEL", "gemma2:9b")
STRATEGY_FALLBACK_MODEL = os.getenv("STRATEGY_FALLBACK_MODEL", DEFAULT_LLM_MODEL)
SNS_COPY_MODEL = os.getenv("SNS_COPY_MODEL", DEFAULT_LLM_MODEL)

SDXL_MODEL = os.getenv("SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
SDXL_OUTPUT_DIR = Path(os.getenv("SDXL_OUTPUT_DIR", "generated_content"))


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
            raise RuntimeError(
                f"Ollama request failed for model '{model_name}' at {OLLAMA_URL}: {exc}"
            ) from exc

        fallback_model_name = normalize_ollama_model_name(fallback_model_name)
        if fallback_model_name == model_name:
            raise RuntimeError(
                f"Ollama request failed for model '{model_name}' at {OLLAMA_URL}: {exc}"
            ) from exc

        print(f"\nOllama request failed for {model_name}. Falling back to {fallback_model_name}.")
        print(f"Reason: {str(exc).splitlines()[0]}\n")
        return call_llm(prompt, fallback_model_name, None, max_tokens)


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

    result = call_llm(prompt, PERSONA_MODEL)
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
  "customer_pain_points": [
    "Pain point 1",
    "Pain point 2"
  ],
  "selling_points": [
    "Key selling point 1",
    "Key selling point 2"
  ],
  "marketing_direction": "Overall marketing direction",
  "message_strategy": "Messaging strategy for SNS marketing",
  "recommended_tone": "Recommended tone and manner",
  "risk_to_avoid": [
    "Risk factor 1",
    "Risk factor 2"
  ]
}}

Rules:
- Return JSON only.
- Interpret Brand Values as the company's stated ethics, corporate values, mission,
  ESG principles, or management philosophy from its website, and make the strategy
  reinforce those values without greenwashing or unsupported claims.
- Do not include markdown.
- Do not include explanations outside JSON.
"""

    result = call_llm(prompt, STRATEGY_MODEL, STRATEGY_FALLBACK_MODEL)
    return safe_json_loads(result, dict)

def generate_product_concept(
    product_category: str,
    brand_values: str,
    target_group: str,
    brand_tone: str
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
- For physical consumer goods, translate customer trust and data protection into product safety, ingredient transparency, quality control, and responsible communication.
- Do not use generic category names such as "Sunscreen" or "Keyboard".
- Return JSON only.
- Do not include markdown.
- Do not include explanations outside JSON.
"""

    result = call_llm(prompt)
    return safe_json_loads(result, dict)

# def generate_sns_content(user_input: CampaignInput, strategy: Dict) -> Dict:
#     prompt = f"""
# You are a creative SNS marketing content agent.

# Based on the following marketing strategy,
# generate promotional SNS content optimized for {user_input.platform}.

# [Product Name]
# {user_input.product_name}

# [Brand Tone]
# {user_input.brand_tone}

# [Brand Values]
# {user_input.brand_values}

# [Marketing Strategy]
# {json.dumps(strategy, ensure_ascii=False, indent=2)}

# Return ONLY a valid JSON object in the following format.

# {{
#   "post_title": "SNS post title",
#   "main_copy": "Main SNS promotional copy",
#   "short_hook": "Short attention-grabbing hook",
#   "hashtags": [
#     "#hashtag1",
#     "#hashtag2",
#     "#hashtag3"
#   ],
#   "image_prompt_for_sdxl": "Detailed English prompt for SDXL image generation",
#   "negative_prompt_for_sdxl": "Detailed negative prompt for SDXL",
#   "image_width": 640,
#   "image_height": 640
# }}

# Rules:
# - Choose the image width and height based on the platform.
# - Interpret Brand Values as the company's stated ethics, corporate values, mission,
#   ESG principles, or management philosophy from its website, and keep the copy and
#   visual prompt aligned with them.
# - For Instagram feed posts, use a square or portrait-friendly size.
# - For TikTok, YouTube Shorts, or Reels, use a vertical size.
# - For X/Twitter or Facebook, use a landscape-friendly size.
# - Width and height must be valid SDXL-friendly values.
# - Use multiples of 8.
# - Keep every string concise.
# - Return JSON only.
# - Do not wrap the JSON in ``` fences.
# - Do not include markdown.
# - Do not include explanations outside JSON.
# """

#     result = call_llm(prompt, SNS_COPY_MODEL, DEFAULT_LLM_MODEL)
#     try:
#         sns_content = safe_json_loads(result, dict)
#     except ValueError:
#         sns_content = repair_sns_content(result, user_input, strategy)

#     required_keys = [
#         "post_title",
#         "main_copy",
#         "short_hook",
#         "hashtags",
#         "image_prompt_for_sdxl",
#         "negative_prompt_for_sdxl",
#         "image_width",
#         "image_height"
#     ]

#     for key in required_keys:
#         if key not in sns_content:
#             sns_content[key] = default_sns_value(key, user_input, strategy)

#     sns_content["image_width"] = normalize_sdxl_dimension(sns_content["image_width"])
#     sns_content["image_height"] = normalize_sdxl_dimension(sns_content["image_height"])

#     return sns_content

# sns copy
def generate_sns_copy(user_input: CampaignInput, strategy: Dict) -> Dict:
    prompt = f"""
You are an Instagram beauty copywriter.

Generate SNS copy for {user_input.platform}.

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

Return ONLY a valid JSON object.

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
- Emphasize sensitive skin, gentle protection, honest communication, and transparency.
- Avoid unsupported claims.
- Return JSON only.
"""
    result = call_llm(prompt, SNS_COPY_MODEL, DEFAULT_LLM_MODEL)
    return safe_json_loads(result, dict)

# image prompt
def generate_image_prompt(
    user_input: CampaignInput,
    strategy: Dict,
    sns_copy: Dict
) -> Dict:
    prompt = f"""
You are a visual prompt engineer for SDXL product advertising images.

Create an image prompt for a product advertisement.
The image must visually match the product category: {user_input.product_category}.

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

[Brand Tone]
{user_input.brand_tone}

[Brand Values]
{user_input.brand_values}

[SNS Copy]
{json.dumps(sns_copy, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object.

{{
  "image_prompt_for_sdxl": "English SDXL prompt",
  "negative_prompt_for_sdxl": "Negative prompt",
  "image_width": 640,
  "image_height": 960
}}

Rules:
- The image prompt must be in English.
- Describe a concrete product advertising scene.
- Include product packaging, background, lighting, material, composition, mood, and brand aesthetic.
- Reflect a clean, modern, minimalist, trustworthy skincare brand.
- Do not create vague nature-only scenes.
- Do not include text, readable letters, logo, or watermark.
- The negative prompt must include category-mismatch objects to avoid.
- For sunscreen, include: serum bottle, transparent liquid bottle, dropper bottle, perfume bottle, toner bottle, essence bottle, watery cosmetic bottle.- The product packaging must clearly match the product category.
- If the product category is sunscreen, describe it as a sunscreen tube, sun care cream tube, or matte sunscreen bottle, not as a serum, toner, perfume, mist, or transparent liquid bottle.
- If the product category is sunscreen, include visual cues such as SPF, UV protection, sun care, outdoor sunlight, beach, sports, summer, or active lifestyle.
- Avoid transparent glass serum bottles unless the product category specifically requires that packaging.
- Do not describe watery liquid, dropper bottles, perfume bottles, toner bottles, or essence containers for sunscreen products.
"""
    result = call_llm(prompt, SNS_COPY_MODEL, DEFAULT_LLM_MODEL)
    image_content = safe_json_loads(result, dict)

    image_content["image_width"] = normalize_sdxl_dimension(image_content["image_width"])
    image_content["image_height"] = normalize_sdxl_dimension(image_content["image_height"])

    return image_content

def repair_sns_content(raw_output: str, user_input: CampaignInput, strategy: Dict) -> Dict:
    repair_prompt = f"""
Create one complete valid JSON object for Instagram SNS content.

Use this product and strategy:
{json.dumps({
    "product_name": user_input.product_name,
    "brand_tone": user_input.brand_tone,
    "strategy": strategy
}, ensure_ascii=False)}

The previous model output was invalid or incomplete:
{raw_output[:1500]}

Return ONLY this JSON object. No markdown, no code fences.

{{
  "post_title": "short title",
  "main_copy": "one concise paragraph",
  "short_hook": "short hook",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"],
  "image_prompt_for_sdxl": "concise English SDXL prompt under 60 words",
  "negative_prompt_for_sdxl": "blurry, low quality, distorted, text, logo",
  "image_width": 640,
  "image_height": 640
}}
"""
    repaired = call_llm(repair_prompt, DEFAULT_LLM_MODEL, max_tokens=640)
    return safe_json_loads(repaired, dict)


def default_sns_value(key: str, user_input: CampaignInput, strategy: Dict):
    defaults = {
        "post_title": f"Meet {user_input.product_name}",
        "main_copy": strategy.get("message_strategy", f"Discover {user_input.product_name}."),
        "short_hook": strategy.get("marketing_direction", "Clean protection for everyday skin."),
        "hashtags": ["#cleanbeauty", "#skincare", "#sunscreen", "#sensitiveskin", "#minimalbeauty"],
        "image_prompt_for_sdxl": (
            f"Minimal clean beauty product photo for {user_input.product_name}, "
            "soft natural light, sensitive skin sunscreen, modern trustworthy aesthetic"
        ),
        "negative_prompt_for_sdxl": "blurry, low quality, distorted, text, logo, watermark",
        "image_width": 640,
        "image_height": 640,
    }
    return defaults[key]


def normalize_sdxl_dimension(value) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return 640

    dimension = max(512, min(1536, dimension))
    return round(dimension / 8) * 8


@lru_cache(maxsize=1)
def load_sdxl_pipeline(model_name: str = SDXL_MODEL):
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError as exc:
        raise RuntimeError(
            "SDXL image generation requires diffusers and torch. "
        ) from exc
        
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. SDXL requires the WSL 4090 CUDA environment.")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    return pipe


def generate_sdxl_content(sns_content: Dict) -> Dict:
    image_prompt = sns_content["image_prompt_for_sdxl"]
    negative_prompt = sns_content.get("negative_prompt_for_sdxl", "")

    width = int(sns_content.get("image_width", 640))
    height = int(sns_content.get("image_height", 640))

    try:
        pipe = load_sdxl_pipeline()
        image = pipe(
            prompt=image_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=20,
            guidance_scale=7.0,
        ).images[0]

    except RuntimeError as exc:
        return {
            "model": SDXL_MODEL,
            "status": "skipped",
            "reason": str(exc),
            "image_path": None,
        }

    SDXL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = SDXL_OUTPUT_DIR / f"sdxl_content_{timestamp}.png"
    image.save(image_path)

    return {
        "model": SDXL_MODEL,
        "status": "generated",
        "width": width,
        "height": height,
        "image_path": str(image_path),
    }
    

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
    preview = text[:1000]
    raise ValueError(f"JSON parsing failed. Expected {expected_name}. Model output preview:\n{preview}")


def run_pipeline():

    product_category = input("Product category (e.g., sunscreen, keyboard, tumbler): ")
    brand_values = input(
        "Brand values (e.g., company ethics, corporate values, ESG, sustainability): "
    )
    target_group = input("Target audience: ")
    brand_tone = input("Brand tone: ")
    platform = input("Platform (Instagram, TikTok, etc): ")

    print("\n[0] Generating product name and concept...")
    product_concept = generate_product_concept(
        product_category=product_category,
        brand_values=brand_values,
        target_group=target_group,
        brand_tone=brand_tone
    )

    user_input = CampaignInput(
        product_category=product_category,
        brand_values=brand_values,
        product_name=product_concept["product_name"],
        product_description=product_concept["product_description"],
        target_group=target_group,
        brand_tone=brand_tone,
        platform=platform
    )

    print("\n[1] Generating virtual customer feedback...")
    feedbacks = generate_customer_feedback(user_input)

    print("\n[2] Analyzing marketing strategy...")
    strategy = analyze_strategy(user_input, feedbacks)

    print("\n[3] Generating SNS content...")
    sns_copy = generate_sns_copy(user_input, strategy)

    print("\n[4] Skipping related visual content with SDXL...")
    image_prompt = generate_image_prompt(user_input, strategy, sns_copy)

    sns_content = {
        **sns_copy,
        **image_prompt,
    }
    print("\n[5] Skipping related visual content with SDXL...")
    sdxl_content = {
        "model": SDXL_MODEL,
        "status": "skipped",
        "reason": "Skipped to reduce runtime.",
        "image_path": None,
    }

    final_result = {
        "agent_models": {
            "product_concept_agent": DEFAULT_LLM_MODEL,
            "persona_agent": PERSONA_MODEL,
            "strategy_agent": STRATEGY_MODEL,
            "strategy_fallback_agent": STRATEGY_FALLBACK_MODEL,
            "sns_copy_agent": SNS_COPY_MODEL,
            "related_content_agent": SDXL_MODEL
        },
        "input": asdict(user_input),
        "virtual_customer_feedbacks": feedbacks,
        "marketing_strategy": strategy,
        "sns_content": sns_content,
        "sdxl_content": sdxl_content
    }

    with open("marketing_agent_result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\nCompleted: marketing_agent_result.json saved.")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run_pipeline()
