# app.py
import os, json, base64, httpx
from pathlib import Path
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# ── main.py 에이전트 함수 import ──────────────────────────
from main import (
    CampaignInput,
    generate_product_concept,
    generate_customer_feedback,
    analyze_strategy,
    generate_sns_copy,
    generate_campaign_concept,
    generate_creative_direction,
    generate_image_prompt,
)
# ── generate_image_gateway.py 함수 import ────────────────
from generate_image_gateway import build_gateway_prompt, print_marketing_summary

# ── 설정 ─────────────────────────────────────────────────
BASE_URL    = "https://factchat-cloud.mindlogic.ai/v1/gateway"
IMAGE_MODEL = "gemini-2.5-flash-image"
OUTPUT_DIR  = Path("generated_content")

# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="AI 마케팅 에이전트", page_icon="🚀", layout="wide")
st.title("🚀 AI 마케팅 콘텐츠 자동 생성기")
st.caption("제품 정보를 입력하면 마케팅 전략 · SNS 카피 · 홍보 이미지를 자동 생성합니다.")

# ── 사이드바 ──────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 API 키 설정")
    groq_key = st.text_input("Groq API Key", type="password",
                              value=os.getenv("GROQ_API_KEY", ""),
                              help="console.groq.com에서 무료 발급")
    smu_key  = st.text_input("SMU API Key (이미지 생성용)", type="password",
                              value=os.getenv("SMU_API_KEY", ""))
    st.divider()
    st.header("⚙️ 생성 설정")
    platform = st.selectbox("플랫폼", ["Instagram", "TikTok", "YouTube"])
    gen_image = st.toggle("이미지 생성 포함", value=True)

# ── 입력 폼 ───────────────────────────────────────────────
st.subheader("📋 제품 정보 입력")
col1, col2 = st.columns(2)
with col1:
    category = st.text_input("제품 카테고리", placeholder="sunscreen, keyboard, tumbler...")
    target   = st.text_input("타겟 고객",     placeholder="20-30대 직장 여성, 환경에 관심 있는 MZ세대...")
with col2:
    tone     = st.text_input("브랜드 톤앤매너", placeholder="전문적이고 신뢰감 있는, 친근하고 유머러스한...")
    values   = st.text_area("브랜드 가치 (홈페이지 경영철학/ESG 등 붙여넣기)", height=100)

run_btn = st.button("✨ 마케팅 콘텐츠 생성 시작", type="primary", use_container_width=True)

# ── 메인 파이프라인 ───────────────────────────────────────
if run_btn:
    # 입력 검증
    if not groq_key:
        st.error("Groq API Key를 입력해주세요.")
        st.stop()
    if gen_image and not smu_key:
        st.error("이미지 생성을 위해 SMU API Key를 입력해주세요.")
        st.stop()
    if not category or not values or not target or not tone:
        st.error("모든 항목을 입력해주세요.")
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["SMU_API_KEY"]  = smu_key

    result_data = {}  # 최종 결과를 메모리에 저장

    # ── Step 0: 제품 컨셉 ────────────────────────────────
    with st.status("🤖 AI 에이전트 실행 중...", expanded=True) as status:

        st.write("**[0/6]** 제품명 & 컨셉 생성 중...")
        concept = generate_product_concept(category, values, target, tone)
        product_name = concept["product_name"]
        product_desc = concept["product_description"]
        st.write(f"→ 제품명: **{product_name}**")

        user_input = CampaignInput(
            product_category=category,
            brand_values=values,
            product_name=product_name,
            product_description=product_desc,
            target_group=target,
            brand_tone=tone,
            platform=platform,
        )

        # ── Step 1: 가상 고객 피드백 ─────────────────────
        st.write("**[1/6]** 가상 고객 피드백 생성 중...")
        feedbacks = generate_customer_feedback(user_input)
        avg_intent = sum(f.get("purchase_intent", 0) for f in feedbacks) / len(feedbacks)
        st.write(f"→ {len(feedbacks)}명 시뮬레이션 완료 | 평균 구매 의향: {avg_intent:.1f}/5")

        # ── Step 2: 마케팅 전략 ───────────────────────────
        st.write("**[2/6]** 마케팅 전략 분석 중...")
        strategy = analyze_strategy(user_input, feedbacks)

        # ── Step 3: SNS 카피 ──────────────────────────────
        st.write("**[3/6]** SNS 카피 생성 중...")
        sns_copy = generate_sns_copy(user_input, strategy)

        # ── Step 4: 캠페인 컨셉 ───────────────────────────
        st.write("**[4/6]** 캠페인 컨셉 생성 중...")
        campaign_concept = generate_campaign_concept(user_input, feedbacks, strategy)

        # ── Step 5: 크리에이티브 방향 ─────────────────────
        st.write("**[5/6]** 크리에이티브 방향 생성 중...")
        creative_direction = generate_creative_direction(
            user_input, feedbacks, strategy, sns_copy, campaign_concept
        )

        # ── Step 6: 이미지 프롬프트 ───────────────────────
        st.write("**[6/6]** 이미지 프롬프트 생성 중...")
        image_prompt_data = generate_image_prompt(
            user_input, strategy, sns_copy, campaign_concept, creative_direction
        )
        sns_content = {**sns_copy, **image_prompt_data}

        # ── 결과 딕셔너리 구성 (gateway와 동일한 구조) ───
        result_data = {
            "input": {
                "product_category": category,
                "product_name": product_name,
                "product_description": product_desc,
                "target_group": target,
                "brand_tone": tone,
            },
            "virtual_customer_feedbacks": feedbacks,
            "marketing_strategy": strategy,
            "sns_copy": sns_copy,
            "campaign_concept": campaign_concept,
            "creative_direction": creative_direction,
            "sns_content": sns_content,
        }

        # JSON 파일도 함께 저장 (선택)
        with open("marketing_agent_result.json", "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        status.update(label="✅ 텍스트 생성 완료!", state="complete")

    # ── 결과 출력 ─────────────────────────────────────────
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📦 제품 컨셉", "📊 마케팅 전략", "📱 SNS 카피", "👥 고객 피드백"])

    with tab1:
        st.subheader(product_name)
        st.write(product_desc)

    with tab2:
        st.markdown(f"**핵심 인사이트:** {strategy.get('main_insight','')}")
        st.markdown(f"**마케팅 방향:** {strategy.get('marketing_direction','')}")
        st.markdown(f"**캠페인 빅 아이디어:** {campaign_concept.get('campaign_big_idea','')}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ 셀링포인트**")
            for sp in strategy.get("selling_points", []):
                st.markdown(f"- {sp}")
        with c2:
            st.markdown("**⚠️ 리스크**")
            for r in strategy.get("risk_to_avoid", []):
                st.markdown(f"- {r}")

    with tab3:
        st.markdown(f"### {sns_copy.get('post_title','')}")
        st.info(sns_copy.get("short_hook", ""))
        st.write(sns_copy.get("main_copy", ""))
        st.markdown(" ".join(sns_copy.get("hashtags", [])))

    with tab4:
        for fb in feedbacks:
            with st.expander(f"[{fb.get('purchase_intent')}/5] {fb.get('persona_name','')} — {fb.get('persona_profile','')}"):
                st.markdown(f"✅ **긍정:** {fb.get('positive_reaction','')}")
                st.markdown(f"❌ **부정:** {fb.get('negative_reaction','')}")
                st.markdown(f"💡 **개선 제안:** {fb.get('improvement_suggestion','')}")

    # ── JSON 다운로드 ─────────────────────────────────────
    st.download_button(
        "📥 전체 결과 JSON 다운로드",
        data=json.dumps(result_data, ensure_ascii=False, indent=2),
        file_name=f"{product_name}_marketing.json",
        mime="application/json",
    )

    # ── 이미지 생성 ───────────────────────────────────────
    if gen_image:
        st.divider()
        st.subheader("🖼️ 홍보 이미지 생성")

        # gateway와 동일한 prompt_data 구조 구성
        prompt_data = {
            "base_prompt":          image_prompt_data.get("image_prompt_for_sdxl", ""),
            "negative_prompt":      image_prompt_data.get("negative_prompt_for_sdxl", ""),
            "product_category":     category,
            "product_name":         product_name,
            "product_description":  product_desc,
            "target_group":         target,
            "brand_tone":           tone,
            "creative_direction":   creative_direction,
            "campaign_concept":     campaign_concept,
            "marketing_strategy":   strategy,
            "sns_copy":             sns_copy,
            "aspect_ratio":         "2:3",
        }

        image_prompt_str = build_gateway_prompt(prompt_data)

        payload = {
            "model":            IMAGE_MODEL,
            "prompt":           image_prompt_str,
            "aspect_ratio":     "2:3",
            "number_of_images": 1,
        }

        with st.spinner("🎨 이미지 생성 중 (약 20~40초 소요)..."):
            try:
                with httpx.Client(
                    headers={"Authorization": f"Bearer {smu_key}"},
                    timeout=120.0,
                ) as client:
                    resp = client.post(f"{BASE_URL}/images/generate/", json=payload)
                    resp.raise_for_status()
                    result = resp.json()

                image_url = result["data"][0]["url"]

                if image_url.startswith("data:image"):
                    # base64 → 파일 저장 후 표시
                    b64 = image_url.split(",", 1)[1]
                    img_bytes = base64.b64decode(b64)

                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = OUTPUT_DIR / f"campaign_image_{ts}.png"
                    save_path.write_bytes(img_bytes)

                    st.image(img_bytes, caption=f"{product_name} — 캠페인 이미지", use_container_width=False, width=400)
                    st.download_button("📥 이미지 다운로드", img_bytes,
                                       file_name=save_path.name, mime="image/png")
                else:
                    st.image(image_url, caption=f"{product_name} — 캠페인 이미지", width=400)

            except httpx.HTTPStatusError as e:
                st.error(f"이미지 API 오류: {e.response.status_code} — {e.response.text}")
            except Exception as e:
                st.error(f"이미지 생성 실패: {e}")
