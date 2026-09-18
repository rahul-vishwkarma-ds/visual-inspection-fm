# Visual-Inspection FM: Pitch Deck Study Guide

> [!NOTE]
> This document synthesizes our deep dives into the 8 concept slides. Use this as your master cheat sheet for the hackathon pitch to ensure you and Shahul hit both the business metrics and the technical architecture perfectly.

## Slide 1: The Elevator Pitch
*   **The Concept:** A pre-trained visual-inspection Foundation Model (FM).
*   **The Product:** Delivered as a physical Edge Appliance (for fast, local factory processing) and an API (for cloud sync).
*   **The Hook:** It gets a factory camera inspection-ready in **days, not months**.

## Slide 2: The Customer Pain Point (The Problem)
*   **The Bottleneck:** Currently, every time a factory launches a new product (SKU), it takes 4–8 weeks of engineering and thousands of manually labeled images to program the camera. 
*   **The Financial Cost:** This wastes **€2–4 million annually** in setup costs alone, before even accounting for lawsuits from missed defects.
*   **The Competitor Flaw:** Generic AI vendors (Google/OpenAI) lack industrial data, so they fail to catch rare, microscopic defects.

## Slide 3: The Moat & Business Model
*   **The Moat:** The architecture (Vision Transformers) is a commodity. The true moat is Bosch's proprietary, cross-plant defect data. No startup can copy this.
*   **The Business Model:** Sold as a SaaS subscription (per line, per year).
*   **Federated Learning:** Every time a new customer uses the system and finds a new defect, they send anonymized gradients back to the cloud, making the Master AI smarter for everyone else.

## Slide 4: The Technical Architecture (Your Domain)
*You are not building a model from scratch. You are fine-tuning a pre-trained backbone.*
1.  **Capture:** Take just 50-100 photos of the new customer's specific part.
2.  **Embed (Frozen Backbone):** Pass those photos through the frozen Bosch Foundation Model to extract rich mathematical latent embeddings (e.g., a 512-dimensional vector).
3.  **Fine-Tune the Head:** Attach a blank, lightweight classification head and train *only* that head using the embeddings. 
4.  **Deploy:** Push the updated weights to the Edge Appliance instantly.

> [!TIP]
> **The Bayesian Phase 1 Upgrade:** While the MVP uses a standard linear head, your Phase 1 roadmap is to make this head **Bayesian** (e.g., using Monte Carlo Dropout). This calculates *Epistemic Uncertainty*, triggering a human-in-the-loop when the AI is confused.

## Slide 5: The Go-To-Market Strategy (The Wedge)
*   **Zero Cold-Start:** You don't have to hunt for your first customer. The first customer is Bosch's own internal factory network. You get paid immediately and generate massive training data.
*   **The Target:** Automotive Tier-1 suppliers. Under IATF 16949 standards, a passed inspection is trusted blindly. They need Day-One perfection.
*   **The Killer Stats:** Cuts defect escape rates by 83% and reduces warranty claims by 60%.

## Slide 6: Market Size & The Legal Trigger
*   **TAM:** $42 Billion total machine vision market.
*   **The Legal Ticking Clock:** The **EU AI Act (August 2027)** classifies factory AI as high-risk. It legally mandates visual evidence logging and "human-in-the-loop" fallbacks. Standard AI cannot do this. Your Phase 1 Bayesian architecture solves this compliance nightmare instantly.

## Slide 7: Competitive Whitespace
*   **Why we win:** 
    *   Silicon Valley AI giants own the math, but don't own factory data.
    *   Incumbent camera companies own the factory hardware, but lack Foundation Models.
    *   By partnering with Bosch, this is the only startup that combines cutting-edge Foundation Model AI with trusted German industrial hardware.

## Slide 8: Bosch's Right to Win
*   **Data:** Millions of proprietary factory images.
*   **Research:** BCAI provides the heavy lifting for the master brain.
*   **Hardware Integration:** Instantly plugs into Nexeed software and Bosch Security Systems cameras.
*   **Safety Pedigree:** Startups cannot certify AI for safety-critical aviation/automotive loops. Bosch already holds these certifications (IEC 61508 / IATF 16949), providing a legal halo that allows immediate sales.
