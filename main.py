# marketing_agent/main.py
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Type
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL          = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.3-70b-versatile")
PERSONA_MODEL          = os.getenv("GROQ_PERSONA_MODEL", DEFAULT_MODEL)
STRATEGY_MODEL         = os.getenv("GROQ_STRATEGY_MODEL", "openai/gpt-oss-120b")
SNS_COPY_MODEL         = os.getenv("GROQ_SNS_COPY_MODEL", DEFAULT_MODEL)
MAX_TOKENS             = int(os.getenv("GROQ_MAX_TOKENS", "2048"))
IMAGE_PROMPT_OUTPUT_DIR = Path("Img Generate Prompt")
MARKETING_RESULT_PATH = Path("marketing_agent_result.json")
MARKETING_RESULT_ARCHIVE_DIR = Path("marketing_agent_results")


@dataclass
class CampaignInput:
    product_category: str
    brand_values: str
    product_name: str
    product_description: str
    target_group: str
    brand_tone: str
    platform: str


def call_groq(prompt: str, model_name: str = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

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
    result = call_groq(prompt)
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
    result = call_groq(prompt, PERSONA_MODEL)
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
    result = call_groq(prompt, STRATEGY_MODEL)
    return safe_json_loads(result, dict)


def generate_campaign_concept(
    user_input: CampaignInput,
    feedbacks: List[Dict],
    strategy: Dict,
) -> Dict:
    prompt = f"""
You are a senior Instagram campaign planner for premium consumer brands across
categories such as beauty, fashion accessories, tech gadgets, home goods,
food & beverage, and other lifestyle products.

Your role:
- Read customer personas, their positive reactions, concerns, purchase intent, and improvement suggestions.
- Extract the strongest emotional and strategic campaign message.
- Translate customer insights into an Instagram campaign concept that fits the
  specific product category given below — do not default to beauty/skincare framing
  unless the product category actually is a beauty or skincare product.
- Make sure the campaign addresses customer concerns, not only selling points.
- The campaign should feel emotional, premium, trustworthy, and visually memorable,
  in a way that is credible for this specific product category.

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
- Reflect customer concerns that are actually relevant to the {user_input.product_category}
  category (for example: durability, craftsmanship, material quality, price/value,
  comfort, performance, sensitivity/safety, or environmental transparency —
  choose whichever apply, rather than assuming skincare-specific concerns like SPF).
- Avoid unsupported claims.
- Return JSON only. Do not include markdown.
"""
    result = call_groq(prompt, SNS_COPY_MODEL)
    return safe_json_loads(result, dict)


def generate_sns_copy(user_input: CampaignInput, strategy: Dict) -> Dict:
    prompt = f"""
You are an Instagram copywriter specialized in premium lifestyle advertising
across product categories (beauty, fashion, accessories, tech, home goods,
food & beverage, etc).
Your role:
- Turn marketing strategy into natural Instagram copy.
- Write like a real brand in the {user_input.product_category} category, not like a corporate report.
- Use emotionally appealing but realistic language suited to this category.
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
- Make it sound like a real Instagram advertisement for this specific product category.
- Do not sound like a corporate introduction.
- Do not default to beauty/skincare vocabulary unless the product category is beauty/skincare.
- Use short, natural, emotionally appealing sentences.
- Emphasize the strongest selling points from the strategy.
- Reflect the target audience naturally.
- Avoid unsupported claims.
- Return JSON only. Do not include markdown.
"""
    result = call_groq(prompt, SNS_COPY_MODEL)
    return safe_json_loads(result, dict)


def generate_creative_direction(
    user_input: CampaignInput,
    feedbacks: List[Dict],
    strategy: Dict,
    sns_copy: Dict,
    campaign_concept: Dict,
) -> Dict:
    prompt = f"""
You are an advertising creative director for premium Instagram campaigns across
product categories (beauty, fashion accessories, tech gadgets, home goods,
food & beverage, and other lifestyle products).

Your role:
- Translate customer research, campaign concept, and marketing strategy into a
  visual creative brief that fits the specific product category given below.
- Make the image feel like an Instagram campaign, not a plain catalog product photo.
- Balance product visibility with emotional storytelling.
- Show the product, its material/texture/finish, and the feeling of using or owning it.
- Make customer concerns visible through reassuring visual cues appropriate to this category.

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
  "texture_visualization": "How the product's material, surface finish, or texture should be shown (e.g., fabric weave, leather grain, metal sheen, cream texture — whichever fits this product)",
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
- hero_subject should generally describe a person interacting with, wearing, or using
  the product in a scene — unless the product category (e.g., furniture, food ingredients,
  raw materials) is more naturally shown as a still-life/object hero shot. Use judgment
  based on the {user_input.product_category} category.
- Do not create a plain floating product render.
- Use one main hero product; allow one small supporting detail shot that reveals its
  material or texture (e.g., a fabric close-up, a leather grain macro, a metal reflection,
  a cream smear — whichever genuinely fits this product, not a default assumption).
- The product should occupy around 45-60% of the frame, not 80-90%.
- Reflect persona concerns visually in a way that fits the {user_input.product_category}
  category (e.g., durability, craftsmanship, comfort, eco transparency, premium quality,
  sensitivity/safety) rather than assuming skincare-specific concerns like SPF clarity.
- things_to_avoid must be tailored to this product category. Only mention avoiding
  items like serum bottles, droppers, or toner bottles if the product actually is a
  liquid cosmetic; otherwise list clichés relevant to this category instead.
- Return JSON only. Do not include markdown.
"""
    result = call_groq(prompt, SNS_COPY_MODEL)
    return safe_json_loads(result, dict)


def generate_image_prompt(
    user_input: CampaignInput,
    strategy: Dict,
    sns_copy: Dict,
    campaign_concept: Dict,
    creative_direction: Dict,
) -> Dict:
    prompt = f"""
You are an AI image prompt engineer for premium Instagram product campaign
photography, working across categories such as beauty, fashion accessories,
tech gadgets, home goods, food & beverage, and other lifestyle products.

Your mission:
Create a highly specific image-generation prompt for a modern image model,
tailored to the {user_input.product_category} category given below.
Do not write a vague prompt, and do not default to skincare/beauty imagery
unless the product category actually is skincare or beauty.
Build the scene carefully like a photographer and art director.

You must internally follow this scene-design workflow before writing the final prompt:
1. Identify the single most important campaign message.
2. Choose one primary persona insight to emphasize.
3. Define one clear hero subject.
4. Define one natural action involving the product.
5. Choose a specific setting and time of day.
6. Decide how the product is physically integrated into the scene.
7. Specify shot type, camera angle, lens feel, and focus behavior.
8. Specify lighting direction, softness, color temperature, and shadows.
9. Add concrete sensory details: relevant human detail (if a person appears), product texture/material, environment.
10. Make sure every detail supports the campaign message.
11. Self-check for physical plausibility and remove vague wording.

Important prompt-engineering requirements:

A. Specificity rules
- Do not use vague phrases such as "beautiful background", "premium mood", "soft lighting", or "natural pose"
  unless they are explained with observable details.
- Replace abstract adjectives with visible evidence.
- Instead of "premium lighting", describe direction, softness, color temperature, and shadow quality.
- Instead of "eco-friendly atmosphere", describe visible materials, reusable objects, restrained packaging, or natural surfaces.
- Instead of "young woman", specify approximate age range, styling, expression, posture, and activity
  (only include a person at all if it fits the {user_input.product_category} category).

B. Translate abstract values into visible evidence
- Trust → calm expression, realistic texture/finish appropriate to the product, a natural handling or using gesture, restrained packaging, believable light.
- Sustainability → natural materials, minimal packaging, reusable or recyclable-looking elements, low-clutter composition.
- Premium quality → intentional composition, refined materials, controlled negative space, elegant light.
- Safety/comfort reassurance (relevant for categories like skincare, baby products, wellness) → gentle handling gesture, relaxed facial expression, natural healthy appearance, no signs of irritation or discomfort.
- Craftsmanship/durability (relevant for categories like fashion accessories, leather goods, furniture, tech hardware) → visible stitching, grain, hinge, joinery, or material detail that signals quality construction.
- Only apply the value-to-evidence mappings above that are actually relevant to {user_input.product_category}; skip the ones that don't fit.

C. Camera direction requirements
- Specify shot type: close-up, medium close-up, medium shot, or wide shot.
- Specify camera angle: eye level, slightly above eye level, low angle, or over-the-shoulder.
- Describe lens feel, such as a 50mm editorial feel or 85mm beauty-campaign feel.
- State what is in focus and what is softly blurred.
- Describe foreground, midground, and background relationships.
- State how much of the frame the product occupies.

D. Physical consistency requirements
- The hand position must match the way the product is actually held, worn, or used.
- Any visible texture (cream, fabric, leather grain, metal finish, liquid, wood grain, etc.) must plausibly come from the visible product itself.
- The product must rest naturally on a hand or surface and must not float.
- Shadows must follow one consistent light direction.
- Scale between any person, hand, and product must be realistic.
- Avoid impossible finger positions or disconnected objects.

E. Output style requirements
- The final image prompt must be 5-7 flowing English sentences.
- It must read like a scene direction for a photographer, not a keyword list.
- It must describe WHO is in the scene (if anyone), WHAT they are doing, WHERE they are, HOW the product appears, and WHY the image feels emotionally persuasive.
- Do not request text rendering, logos, or brand names on the package.

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
  "selected_campaign_message": "The single most important message expressed by the image",
  "primary_persona_insight": "The most important customer insight emphasized visually",
  "subject": {{
    "identity": "Approximate age, styling, and relevant visual characteristics (omit or describe as 'no person' if the category is better shown as an object/still-life)",
    "expression": "Specific facial expression, if a person appears",
    "pose": "Natural body posture, if a person appears",
    "action": "One clear physical action involving the product"
  }},
  "setting": {{
    "location": "Specific place",
    "time_of_day": "Specific time of day",
    "atmosphere": "Visible environmental condition or mood"
  }},
  "product_integration": {{
    "placement": "Exact product position in the scene",
    "interaction": "How the subject physically interacts with the product",
    "frame_ratio": "Approximate visual prominence of the product in the frame"
  }},
  "camera": {{
    "shot_type": "Shot distance",
    "angle": "Camera angle",
    "lens_feel": "Lens and editorial feel",
    "focus": "What is sharp and what is softly blurred",
    "composition": "Foreground, midground, background arrangement"
  }},
  "lighting": {{
    "source": "Main light source",
    "direction": "Direction of light",
    "quality": "Soft or hard light quality",
    "color_temperature": "Warm, neutral, or cool",
    "shadow_behavior": "How the shadows appear"
  }},
  "visual_details": {{
    "human_detail": "Observable skin/hand/styling detail if a person is prominent in the scene, otherwise describe the surface or setting the product rests on",
    "product_texture": "Observable material, surface, or texture detail of the product itself (e.g., leather grain, fabric weave, metal finish, wood grain, cream texture — whichever fits this product)",
    "materials": ["material 1", "material 2", "material 3"],
    "foreground": "Foreground detail",
    "background": "Background detail",
    "color_palette": ["color1", "color2", "color3"]
  }},
  "image_prompt_for_sdxl": "A 5-7 sentence vivid English photographic scene prompt",
  "negative_prompt_for_sdxl": "A detailed negative prompt listing concrete visual failures to avoid",
  "image_width": 640,
  "image_height": 960
}}

Rules for image_prompt_for_sdxl:
- 5-7 flowing English sentences.
- Sentence 1: hero subject, setting, and action.
- Sentence 2: product placement and interaction.
- Sentence 3: subject appearance, facial expression, and relevant physical/material detail
  (skin/hand detail if a person is prominent; otherwise product surface/material detail).
- Sentence 4: camera angle, shot type, lens feel, and focus behavior.
- Sentence 5: lighting, color temperature, and shadows.
- Sentence 6: background, props, materials, and environmental texture.
- Sentence 7 (optional): emotional campaign atmosphere and visual storytelling payoff.
- Do not begin with "A product photo of...".
- Do not write in bullet points.
- Do not include brand names, logos, or text on the packaging.
- Avoid generic adjectives unless they are explained concretely.
- Base every detail on what a real {user_input.product_category} product and its
  typical use context actually look like — do not borrow skincare/beauty imagery
  (creams, droppers, serums, skin application) unless the category genuinely is
  skincare or beauty.

Rules for negative_prompt_for_sdxl:
- Must be concrete and visual, and tailored to what would actually look wrong for
  the {user_input.product_category} category specifically — do not default to
  skincare-bottle assumptions for unrelated categories.
- Include generic product-photography failure modes such as:
  "plain floating product render, white studio background, crowded flat lay"
- Include:
  "fake readable text, brand logo, watermark, unreadable characters"
- Include the following ONLY if a person appears in the scene:
  "extra fingers, deformed hands, disconnected objects, impossible grip, floating product, blurry face"
- Include:
  "inconsistent shadows, unrealistic scale, duplicate products, cluttered composition"
- If, and only if, the product category is a liquid cosmetic (serum, toner, sunscreen,
  perfume, etc.), also include a line excluding generic bottle clichés such as
  "serum bottle, dropper bottle, toner bottle, transparent liquid bottle, perfume bottle".
  Do not include this line for unrelated categories such as wallets, bags, electronics,
  furniture, or food.
- Return JSON only. Do not include markdown.

Before returning the final JSON, silently self-check:
- Is the scene specific rather than vague?
- Is the action physically plausible?
- Is the product clearly visible but naturally integrated?
- Are camera and lighting details explicit?
- Do visual details support the campaign message?
- Does everything genuinely fit {user_input.product_category}, rather than defaulting to skincare/beauty imagery?
Do not output the checklist or analysis. Return JSON only.
"""    
    result = call_groq(prompt, SNS_COPY_MODEL)
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


def save_image_prompt_json(product_name: str, image_prompt: Dict) -> Path:
    """이미지 생성용 프롬프트를 전체 파이프라인 결과와 별도 JSON으로 저장합니다."""
    IMAGE_PROMPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = IMAGE_PROMPT_OUTPUT_DIR / f"image_prompt_{timestamp}.json"
    
    prompt_data = {
        "product_name": product_name,
        "selected_campaign_message": image_prompt.get("selected_campaign_message", ""),
        "primary_persona_insight": image_prompt.get("primary_persona_insight", ""),
        "subject": image_prompt.get("subject", {}),
        "setting": image_prompt.get("setting", {}),
        "product_integration": image_prompt.get("product_integration", {}),
        "camera": image_prompt.get("camera", {}),
        "lighting": image_prompt.get("lighting", {}),
        "visual_details": image_prompt.get("visual_details", {}),
        "negative_prompt_for_sdxl": image_prompt.get("negative_prompt_for_sdxl", ""),
        "image_prompt_for_sdxl": image_prompt.get("image_prompt_for_sdxl", ""),
        "image_width": image_prompt.get("image_width", 640),
        "image_height": image_prompt.get("image_height", 960),
    }
    
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(prompt_data, file, ensure_ascii=False, indent=2)
    return output_path


def save_marketing_result_json(result: Dict) -> Path:
    """최신 결과를 gateway용 파일에 쓰고 실행별 결과를 별도로 보관합니다."""
    MARKETING_RESULT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_path = (
        MARKETING_RESULT_ARCHIVE_DIR
        / f"marketing_agent_result_{timestamp}.json"
    )

    for output_path in (MARKETING_RESULT_PATH, archive_path):
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)

    return archive_path


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
            "product_concept_agent":           DEFAULT_MODEL,
            "persona_agent":                   PERSONA_MODEL,
            "strategy_agent":                  STRATEGY_MODEL,
            "sns_copy_and_image_prompt_agent": SNS_COPY_MODEL,
        },
        "input":                    asdict(user_input),
        "virtual_customer_feedbacks": feedbacks,
        "marketing_strategy":       strategy,
        "sns_copy":                 sns_copy,
        "campaign_concept":         campaign_concept,
        "creative_direction":       creative_direction,
        "image_prompt_design":      image_prompt,
        "sns_content":              sns_content,
    }

    marketing_result_path = save_marketing_result_json(final_result)
    print("\n[7] marketing_agent_result.json 저장 완료.")
    print(f"    실행별 결과 저장 완료: {marketing_result_path}")

    image_prompt_path = save_image_prompt_json(user_input.product_name, image_prompt)
    print(f"    이미지 생성 프롬프트 저장 완료: {image_prompt_path}")

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
