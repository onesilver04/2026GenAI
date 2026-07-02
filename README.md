# 🚀 Marketing Agent: Virtual Customer Feedback

가상 고객(Persona) 시뮬레이션을 기반으로 마케팅 전략, SNS 콘텐츠, 홍보 이미지를 자동 생성하는 멀티 에이전트 시스템

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-LLM_API-orange)
![Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-Image_Generation-4285F4?logo=google)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Motivation

마케팅 캠페인을 기획하기 위해서는 시장 조사, 고객 인터뷰, 페르소나 분석, 전략 수립, 콘텐츠 제작 등의 과정이 필요하다.

그러나 이러한 작업은 많은 시간과 비용이 소요되며, 특히 소규모 기업이나 개인 사업자는 충분한 시장 조사를 수행하기 어렵다.

본 프로젝트는 생성형 AI를 활용하여 가상 고객을 생성하고 제품에 대한 반응을 시뮬레이션함으로써,  
**마케팅 전략 수립부터 SNS 홍보 콘텐츠 제작 및 이미지 생성까지의 과정을 자동화**하는 것을 목표로 한다.

---

## ✨ Features

| 단계 | 기능 | 설명 |
|------|------|------|
| 1 | **Product Concept Generation** | 제품 카테고리 · 브랜드 가치 · 타겟 고객을 바탕으로 제품명과 설명 자동 생성 |
| 2 | **Virtual Customer Simulation** | 서로 다른 가치관을 가진 5명의 가상 고객 페르소나 생성 및 반응 시뮬레이션 |
| 3 | **Marketing Strategy Analysis** | 페르소나 피드백 분석 → Pain Points · Selling Points · 마케팅 방향 도출 |
| 4 | **SNS Content Generation** | Instagram 홍보 카피 · 훅 · 해시태그 자동 생성 |
| 5 | **Campaign Concept & Creative Direction** | 캠페인 빅 아이디어 · 비주얼 스토리 · 컬러 팔레트 · 무드 생성 |
| 6 | **Promotion Image Generation** | 전략 기반 SNS 홍보 이미지 자동 생성 (Gemini 2.5 Flash) |

---

## 🏗️ System Architecture

```
User Input (Streamlit Web UI)
        │
        ▼
Product Concept Agent   ── Llama 3.1 8B (Groq)
        │
        ▼
Virtual Customer Agent  ── Llama 3.1 8B (Groq)
        │
        ▼
Marketing Strategy Agent ── Llama 3.3 70B (Groq)
        │
        ▼
SNS Copy Agent          ── Llama 3.1 8B (Groq)
        │
        ▼
Campaign Concept Agent  ── Llama 3.1 8B (Groq)
        │
        ▼
Creative Director Agent ── Llama 3.1 8B (Groq)
        │
        ▼
Image Prompt Agent      ── Llama 3.1 8B (Groq)
        │
        ▼
Image Generation        ── Gemini 2.5 Flash Image
        │
        ▼
Final Marketing Content (Web UI + 다운로드)
```

---

## 🛠️ Tech Stack

| 분류 | 기술 |
|------|------|
| **Web UI** | Streamlit |
| **LLM API** | Groq (Llama 3.1 8B · Llama 3.3 70B) |
| **Image Generation** | Gemini 2.5 Flash Image |
| **Backend** | Python 3.11+ |
| **Pipeline** | Multi-Agent · JSON Pipeline |
| **HTTP Client** | httpx |
| **환경변수 관리** | python-dotenv |

---

## 📁 Project Structure

```
2026GenAI/
├── app.py                        # Streamlit 웹 UI 진입점
├── main.py                       # 멀티 에이전트 파이프라인
├── generate_image_gateway.py     # 이미지 생성 API 게이트웨이
├── .env                          # API 키 (Git 제외)
├── .gitignore
├── requirements.txt
└── generated_content/            # 생성된 이미지 저장 폴더
```

---

## ⚡ Quickstart

### 1. 레포지토리 클론

```bash
git clone https://github.com/onesilver04/2026GenAI.git
cd 2026GenAI
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate      # Mac / Linux
# .venv\Scripts\activate     # Windows
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. API 키 발급

| API | 발급 경로 | 비용 |
|-----|-----------|------|
| **Groq API Key** | [console.groq.com](https://console.groq.com) → API Keys | 무료 |
| **SMU API Key** | 이미지 생성 게이트웨이 키 | 별도 문의 |

### 5. `.env` 파일 생성

프로젝트 루트에 `.env` 파일을 생성하고 아래 내용을 입력한다.

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
SMU_API_KEY=your_smu_api_key_here
```

> ⚠️ `.env` 파일은 절대 GitHub에 올리지 말 것. `.gitignore`에 포함되어 있음.

### 6. 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 열린다. 열리지 않으면 아래 주소로 직접 접속한다.

```
http://localhost:8501
```

---

## 🖥️ Web UI 사용법

1. **사이드바**에 Groq API Key와 SMU API Key를 입력한다.
2. 이미지 생성 포함 여부를 토글로 선택한다.
3. 아래 항목을 입력한다.

| 입력 항목 | 예시 |
|-----------|------|
| 제품 카테고리 | `sunscreen`, `keyboard`, `tumbler` |
| 타겟 고객 | `20-30대 민감성 피부를 가진 직장 여성` |
| 브랜드 톤앤매너 | `clean, modern, trustworthy` |
| 브랜드 가치 | 홈페이지의 경영철학 · ESG · 미션 문구 붙여넣기 |

4. **✨ 마케팅 콘텐츠 생성 시작** 버튼을 클릭한다.
5. 생성 완료 후 탭별로 결과를 확인하고 JSON 및 이미지를 다운로드한다.

---

## 📊 Example

### Input

```
제품 카테고리 : Sunscreen
브랜드 가치   : Sustainability, Customer Trust
타겟 고객     : 20-30대 민감성 피부를 가진 남녀
톤앤매너      : Clean, Modern, Trustworthy
플랫폼        : Instagram
```

### Generated Personas

| 페르소나 | 구매 의향 | 핵심 반응 |
|---------|-----------|-----------|
| Eco-Conscious Emily | 5/5 | 친환경 패키징에 높은 관심 |
| Sensitive Steve | 4/5 | 민감성 피부 성분 신뢰 |
| Health-Conscious Hannah | 4/5 | 성분 투명성 중요 |
| Budget-Minded Ben | 2/5 | 가격 대비 가치 의문 |
| Beauty-Savvy Bella | 4/5 | 브랜드 감성과 비주얼 중요 |

### Campaign Message

> **"Protect Your Skin, Respect the Planet"**

### Promotion Image

AI가 생성한 Instagram 마케팅 비주얼 — 페르소나 피드백과 브랜드 가치가 반영된 캠페인 이미지

---

## 🤖 Models

| 에이전트 | 모델 | 제공 |
|---------|------|------|
| Product Concept | `llama-3.1-8b-instant` | Groq |
| Persona Generation | `llama-3.1-8b-instant` | Groq |
| Marketing Strategy | `llama-3.3-70b-versatile` | Groq |
| SNS Copy | `llama-3.1-8b-instant` | Groq |
| Campaign Concept | `llama-3.1-8b-instant` | Groq |
| Creative Direction | `llama-3.1-8b-instant` | Groq |
| Image Prompt | `llama-3.1-8b-instant` | Groq |
| Image Generation | `gemini-2.5-flash-image` | Google |

---

## 🔮 Future Work

- [ ] Chain-of-Thought (CoT) Prompting 적용
- [ ] 페르소나 다양성 확대 (문화권 · 연령대 · 라이프스타일)
- [ ] 실제 사용자 검증 및 A/B 테스트
- [ ] 멀티 플랫폼 지원 (TikTok · YouTube · X)
- [ ] 마케팅 성과 평가 지표 연동
- [ ] 더 큰 파운데이션 모델 적용

---

## 📄 License

MIT License
