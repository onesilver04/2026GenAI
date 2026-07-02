# Marketing Agent: Virtual Customer Feedback

가상 고객(Persona) 시뮬레이션을 기반으로 마케팅 전략, SNS 콘텐츠, 홍보 이미지를 자동 생성하는 멀티 에이전트 시스템

## Motivation

마케팅 캠페인을 기획하기 위해서는 시장 조사, 고객 인터뷰, 페르소나 분석, 전략 수립, 콘텐츠 제작 등의 과정이 필요하다.

그러나 이러한 작업은 많은 시간과 비용이 소요되며, 특히 소규모 기업이나 개인 사업자는 충분한 시장 조사를 수행하기 어렵다.

본 프로젝트는 생성형 AI를 활용하여 가상 고객을 생성하고 제품에 대한 반응을 시뮬레이션함으로써, 마케팅 전략 수립부터 SNS 홍보 콘텐츠 제작까지의 과정을 자동화하는 것을 목표로 한다.

---

## Features

### Product Concept Generation

사용자가 입력한 제품 카테고리, 브랜드 가치, 타겟 고객 정보를 바탕으로 제품명과 제품 설명을 생성한다.

### Virtual Customer Simulation

서로 다른 가치관과 구매 기준을 가진 5명의 가상 고객을 생성하고 다음 정보를 시뮬레이션한다.

* Persona Profile
* Positive Reaction
* Negative Reaction
* Purchase Intent
* Improvement Suggestion

### Marketing Strategy Analysis

가상 고객 피드백을 분석하여

* Customer Pain Points
* Selling Points
* Marketing Direction
* Message Strategy
* Marketing Risks

를 도출한다.

### SNS Content Generation

마케팅 전략을 바탕으로 Instagram 홍보용 콘텐츠를 생성한다.

* Post Title
* Main Copy
* Hook
* Hashtags

### Campaign Concept & Creative Direction

생성된 고객 인사이트를 기반으로

* Campaign Big Idea
* Campaign Message
* Visual Story
* Color Palette
* Mood & Composition

을 생성한다.

### Promotion Image Generation

최종적으로 생성된 전략과 크리에이티브 방향을 활용하여 SNS 홍보 이미지를 자동 생성한다.

---

## System Architecture

User Input
↓
Product Concept Agent
(Llama 3.1)

↓

Virtual Customer Agent
(Llama 3.1)

↓

Marketing Strategy Agent
(Gemma)

↓

SNS Copy Agent
(Llama 3.1)

↓

Campaign Concept Agent
(Llama 3.1)

↓

Creative Director Agent
(Llama 3.1)

↓

Image Prompt Agent
(Llama 3.1)

↓

Gemini Flash Image

↓

Final Marketing Content

---

## Example Input

```text
Product Category
: Sunscreen

Brand Values
: Sustainability, Customer Trust

Target Audience
: Skincare-conscious women and men
  in their 20s and 30s
  with sensitive skin

Brand Tone
: Clean, Modern, Trustworthy

Platform
: Instagram
```

---

## Example Output

### Generated Personas

* Eco-Conscious Emily
* Sensitive Steve
* Health-Conscious Hannah
* Budget-Minded Ben
* Beauty-Savvy Bella

### Marketing Strategy

Selling Points

* Eco-friendly packaging
* Sensitive skin protection
* Sustainable ingredients

Pain Points

* Price concerns
* Brand trust
* SPF information

### Campaign Message

Protect Your Skin, Respect the Planet

### Promotion Image

AI-generated Instagram marketing visual aligned with persona feedback and brand values.

---

## Models

| Stage              | Model                      |
| ------------------ | -------------------------- |
| Product Concept    | Dolphin 3.0 (Llama 3.1 8B) |
| Persona Generation | Dolphin 3.0 (Llama 3.1 8B) |
| Marketing Strategy | Gemma   4                  |
| SNS Copy           | Dolphin 3.0 (Llama 3.1 8B) |
| Campaign Concept   | Dolphin 3.0 (Llama 3.1 8B) |
| Creative Direction | Dolphin 3.0 (Llama 3.1 8B) |
| Image Prompt       | Dolphin 3.0 (Llama 3.1 8B) |
| Image Generation   | Gemini 2.5 Flash Image     |

---

## Tech Stack

* Python
* MLX
* Llama 3.1 8B
* Gemma
* Gemini 2.5 Flash Image
* JSON Pipeline
* Multi-Agent Architecture

---

## Future Work

* Chain-of-Thought(CoT) Prompting
* More Diverse Persona Generation
* Real User Validation & A/B Testing
* Multi-Platform Marketing Support
* Marketing Performance Evaluation
* Larger Foundation Models

```
```
