import json
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict


OLLAMA_URL = "http://localhost:11434/api/generate"


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


def call_ollama(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    return response.json()["response"].strip()


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
- Do not include markdown.
- Do not include explanations outside JSON.
"""

    result = call_ollama("qwen3.6:35b", prompt)
    return safe_json_loads(result)


def analyze_strategy(user_input: CampaignInput, feedbacks: List[Dict]) -> Dict:
    prompt = f"""
You are a marketing strategy analysis agent.

Analyze the following virtual customer feedback and generate
a marketing strategy for the campaign.

[Product Name]
{user_input.product_name}

[Product Description]
{user_input.product_description}

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
- Do not include markdown.
- Do not include explanations outside JSON.
"""

    result = call_ollama("qwen3.6:35b", prompt)
    return safe_json_loads(result)

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
- Do not use generic category names such as "Sunscreen" or "Keyboard".
- Return JSON only.
- Do not include markdown.
- Do not include explanations outside JSON.
"""

    result = call_ollama("qwen3.6:35b", prompt)
    return safe_json_loads(result)

def generate_sns_content(user_input: CampaignInput, strategy: Dict) -> Dict:
    prompt = f"""
You are a creative SNS marketing content agent.

Based on the following marketing strategy,
generate promotional SNS content optimized for {user_input.platform}.

[Product Name]
{user_input.product_name}

[Brand Tone]
{user_input.brand_tone}

[Marketing Strategy]
{json.dumps(strategy, ensure_ascii=False, indent=2)}

Return ONLY a valid JSON object in the following format.

{{
  "post_title": "SNS post title",
  "main_copy": "Main SNS promotional copy",
  "short_hook": "Short attention-grabbing hook",
  "hashtags": [
    "#hashtag1",
    "#hashtag2",
    "#hashtag3"
  ],
  "image_prompt_for_sdxl": "Detailed English prompt for SDXL image generation",
  "negative_prompt_for_sdxl": "Detailed negative prompt for SDXL"
}}

Rules:
- The SNS copy must sound natural and engaging.
- The image prompt must be visually descriptive.
- Return JSON only.
- Do not include markdown.
- Do not include explanations outside JSON.
"""

    result = call_ollama("qwen3.6:35b", prompt)
    return safe_json_loads(result)


def safe_json_loads(text: str):
    try:
        return json.loads(text)

    except json.JSONDecodeError:

        start = text.find("[") if "[" in text else text.find("{")
        end = text.rfind("]") if "]" in text else text.rfind("}")

        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])

        raise ValueError(f"JSON parsing failed:\n{text}")


def run_pipeline():

    product_category = input("Product category (e.g., sunscreen, keyboard, tumbler): ")
    brand_values = input("Brand values (e.g., clean beauty, sustainability, comfort): ")
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
    sns_content = generate_sns_content(user_input, strategy)

    final_result = {
        "input": asdict(user_input),
        "virtual_customer_feedbacks": feedbacks,
        "marketing_strategy": strategy,
        "sns_content": sns_content
    }

    with open("marketing_agent_result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print("\nCompleted: marketing_agent_result.json saved.")
    print(json.dumps(final_result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    run_pipeline()
