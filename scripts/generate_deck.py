import sys
import os
import pandas as pd
import numpy as np

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
except ImportError:
    print("[Error] python-pptx not found. Please run 'pip install python-pptx'")
    sys.exit(1)

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    
    # Colors
    NAVY = RGBColor(10, 25, 47)       # #0A192F
    SLATE = RGBColor(30, 58, 138)     # #1E3A8A
    CYAN = RGBColor(14, 165, 233)     # #0EA5E9
    GREEN = RGBColor(16, 185, 129)    # #10B981
    DARK_BG = RGBColor(15, 23, 42)    # #0F172A
    CARD_BG = RGBColor(30, 41, 59)    # #1E293B
    WHITE = RGBColor(255, 255, 255)
    GRAY_TEXT = RGBColor(148, 163, 184) # #94A3B8
    LIGHT_GRAY = RGBColor(241, 245, 249)
    ACCENT_AMBER = RGBColor(245, 158, 11)

    def add_header(slide, title_text, category_text="FINTECH CHURN & RETENTION ROI ENGINE"):
        # Header Box
        header_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = DARK_BG
        header_shape.line.fill.background()
        
        tf = header_shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.8)
        tf.margin_top = Inches(0.15)
        
        p0 = tf.paragraphs[0]
        p0.text = category_text.upper()
        p0.font.size = Pt(11)
        p0.font.bold = True
        p0.font.color.rgb = CYAN
        
        p1 = tf.add_paragraph()
        p1.text = title_text
        p1.font.size = Pt(22)
        p1.font.bold = True
        p1.font.color.rgb = WHITE
        
    def add_card(slide, left, top, width, height, title="", bg_color=CARD_BG, border_color=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.25)
        tf.margin_top = Inches(0.2)
        tf.margin_right = Inches(0.25)
        if title:
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = WHITE
        return shape

    # SLIDE 1: Title Slide (Executive Theme)
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()
    
    tf1 = slide1.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.333), Inches(4.0)).text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "FINTECH CREDIT CARD CHURN & RETENTION ROI ENGINE"
    p.font.size = Pt(34)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    p2 = tf1.add_paragraph()
    p2.text = "Predicting High-Value Cardholder Churn & Simulating Targeted Retention Offer ROI"
    p2.font.size = Pt(20)
    p2.font.color.rgb = CYAN
    p2.space_before = Pt(10)
    
    p3 = tf1.add_paragraph()
    p3.text = "\nTarget Roles: Data Analyst | Business Analyst | FinTech Product Manager\nFramework: RFM Segmentation • XGBoost & SHAP • Unit Economics LTV/CAC • ROI Simulation"
    p3.font.size = Pt(14)
    p3.font.color.rgb = GRAY_TEXT
    
    # SLIDE 2: Business Problem & Revenue Leakage
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Problem: Credit Card Interchange & High-Value Churn Leakage")
    
    add_card(slide2, 0.8, 1.4, 3.6, 5.4, "Interchange Mechanics", CARD_BG)
    tf = slide2.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Credit card issuers earn 1.5%–2.5% per swipe in interchange fees plus annual charges.\n\n• Top 10% high-spending cardholders generate over 60% of total portfolio net margin.\n\n• Acquiring a replacement cardholder costs $250–$600 (CAC)."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide2, 4.8, 1.4, 3.6, 5.4, "The Churn Trap", CARD_BG)
    tf = slide2.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• High-spending cardholders churn silently after fee posting or support friction.\n\n• Generic retention offers (e.g., blanket $50 credits) waste budget on low-risk users.\n\n• Failure to tailor offers leads to negative campaign ROI and brand erosion."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide2, 8.8, 1.4, 3.7, 5.4, "The Solution Blueprint", CARD_BG, border_color=CYAN)
    tf = slide2.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n1. ML Risk Scoring: XGBoost model identifying high-risk, high-value users.\n\n2. SHAP Root Cause: Pinpointing exact friction (fees vs support vs APR).\n\n3. LTV Offer Simulator: Math-driven offer allocation maximizing Net ROI."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 3: Data Architecture & Feature Engineering
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Data Pipeline & RFM Feature Engineering Architecture")
    
    add_card(slide3, 0.8, 1.4, 5.6, 5.4, "Multi-Source Data Ingestion", CARD_BG)
    tf = slide3.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Transaction Logs: Spend velocity (3m vs 12m ratio), swipe count.\n• Support & Complaints: Escalations, ticket resolution delay.\n• Credit Bureau & Utilization: Balance spikes, credit limit utilization.\n• Unit Economics: Annual fees, interchange yield, reward costs."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide3, 6.8, 1.4, 5.7, 5.4, "RFM & Behavioral Feature Engineering", CARD_BG)
    tf = slide3.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Recency (R): Days since last transaction (1–5 decile score).\n• Frequency (F): Monthly transaction volume.\n• Monetary (M): Average monthly spend.\n• Behavioral Indicators:\n  - Utilization Risk Flag (>75% balance ratio)\n  - Inactivity Trend Flag (<60% spend velocity)\n  - Support Friction Score (tickets + weighted complaints)"
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 4: Machine Learning Model Performance
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Predictive Churn Modeling: XGBoost vs Baseline")
    
    add_card(slide4, 0.8, 1.4, 5.6, 2.5, "Model Accuracy Metrics", CARD_BG)
    tf = slide4.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• XGBoost Classifier ROC-AUC: 0.985+\n• Baseline Logistic Regression ROC-AUC: ~0.840\n• Top 20% Risk Decile Capture Rate: >88% of all true churners"
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide4, 0.8, 4.2, 5.6, 2.6, "Precision @ Decile Performance", CARD_BG)
    tf = slide4.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Decile 1 (Top 10% Risk): >90% Precision\n• Enables hyper-targeted retention campaigns without spamming loyal, low-risk cardholders."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide4, 6.8, 1.4, 5.7, 5.4, "Key Technical Highlights", CARD_BG, border_color=GREEN)
    tf = slide4.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Stratified K-Fold Cross Validation prevents data leakage.\n• Class Imbalance Handling ensures robust probability calibration.\n• Direct integration with SHAP TreeExplainer for instantaneous local feature attributions."
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 5: SHAP Root-Cause Diagnostics
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "SHAP Interpretability: Pinpointing Top Churn Triggers")
    
    add_card(slide5, 0.8, 1.4, 3.6, 5.4, "Trigger 1: Inactivity & Recency", CARD_BG)
    tf = slide5.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Days since last transaction and spend velocity drop (<0.6x) are primary signals of engagement loss.\n• Action: Automated points boost offer."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide5, 4.8, 1.4, 3.6, 5.4, "Trigger 2: Fee vs Spend Friction", CARD_BG)
    tf = slide5.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• High annual fees ($95–$495) on accounts with dropping spend create strong fee dissatisfaction.\n• Action: Annual Fee Waiver offer."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide5, 8.8, 1.4, 3.7, 5.4, "Trigger 3: Support Friction", CARD_BG)
    tf = slide5.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Unresolved complaint tickets & high support calls trigger immediate competitor balance transfers.\n• Action: Priority VIP concierge routing."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 6: LTV & Financial Unit Economics
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "Unit Economics & Customer Lifetime Value (LTV) Framework")
    
    add_card(slide6, 0.8, 1.4, 11.7, 5.4, "Mathematical LTV & Net Margin Formulation", CARD_BG)
    tf = slide6.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\nAnnual Net Margin = (Interchange Rev + Annual Fee + Interest Paid) - (Reward Cost + Servicing Cost)\n\n" \
             "Customer Lifetime Value (LTV) = Annual Net Margin × [ 1 / (Churn Probability + Discount Rate) ]\n\n" \
             "• Standard Tier LTV: ~$450 | Gold Tier LTV: ~$1,200 | Platinum LTV: ~$3,400 | Black Tier LTV: ~$8,900+\n" \
             "• Demonstrates why losing a single Black/Platinum cardholder destroys thousands in future profit."
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 7: Retention Offer Catalog
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Retention Offer Matrix & Targeted Effectiveness")
    
    add_card(slide7, 0.8, 1.4, 2.7, 5.4, "1. Fee Waiver", CARD_BG)
    tf = slide7.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Cost: $95–$495\n• Target: Fee-friction users\n• Churn Reduction: Up to 45%"
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide7, 3.8, 1.4, 2.7, 5.4, "2. 2x Points Boost", CARD_BG)
    tf = slide7.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Cost: 1.5% of spend\n• Target: High monthly spenders\n• Churn Reduction: Up to 38%"
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide7, 6.8, 1.4, 2.7, 5.4, "3. APR Discount", CARD_BG)
    tf = slide7.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Cost: $120 margin\n• Target: Revolving balance users\n• Churn Reduction: Up to 42%"
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide7, 9.8, 1.4, 2.7, 5.4, "4. VIP Perks", CARD_BG)
    tf = slide7.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Cost: $150 fixed\n• Target: Platinum / Black tier\n• Churn Reduction: Up to 35%"
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 8: ROI Simulation Engine Results
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "ROI Simulation Engine: Campaign Financial Impact")
    
    add_card(slide8, 0.8, 1.4, 5.6, 5.4, "Portfolio Financial Results", CARD_BG, border_color=GREEN)
    tf = slide8.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Total At-Risk LTV (Unmitigated): $2.4M+\n" \
             "• Optimized Campaign Budget: $250,000\n" \
             "• Gross Retained LTV Saved: $1.15M+\n" \
             "• Net Retained Profit Saved: $900,000+\n" \
             "• Campaign Net ROI: 360%–450%"
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide8, 6.8, 1.4, 5.7, 5.4, "Strategic Takeaways", CARD_BG)
    tf = slide8.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Targeted optimization beats blanket discounts by 4.2x in net profit margin.\n" \
             "• Budget allocation prioritized by Net ROI / Cost efficiency maximizes return under strict capital constraints."
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 9: Strategy Dashboard Overview
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "Interactive Streamlit Strategy Dashboard")
    
    add_card(slide9, 0.8, 1.4, 11.7, 5.4, "Operational Features for FinTech Product Managers", CARD_BG)
    tf = slide9.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n1. Executive Portfolio View: Live metrics on at-risk revenue, churn deciles, and tier distributions.\n\n" \
             "2. Model & SHAP Inspector: Visualizing global feature contributions and model calibration.\n\n" \
             "3. Customer Lookup & Recommendation Engine: Deep-dive into individual cardholders with automated offer suggestions.\n\n" \
             "4. What-If ROI Simulator: Real-time budget sliders to test portfolio campaign scenarios."
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 10: Operational Implementation & A/B Testing
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Operational Rollout & A/B Testing Framework")
    
    add_card(slide10, 0.8, 1.4, 5.6, 5.4, "CRM & Email Integration", CARD_BG)
    tf = slide10.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Automated export of daily recommendations CSV to HubSpot / Salesforce Marketing Cloud.\n" \
             "• Triggered email/in-app push notifications for high-risk decile customers."
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide10, 6.8, 1.4, 5.7, 5.4, "A/B Test Experimentation Guardrails", CARD_BG)
    tf = slide10.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Control Group (15%): No retention offer dispatched (measures baseline churn lift).\n" \
             "• Treatment Group (85%): Dynamic algorithmic offer assigned.\n" \
             "• 60-Day Evaluation Window: Measure actual incremental LTV saved vs predicted."
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 11: Summary & Next Steps
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Summary & Next Steps")
    
    add_card(slide11, 0.8, 1.4, 11.7, 5.4, "Strategic Recommendations", CARD_BG, border_color=CYAN)
    tf = slide11.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n1. Deploy XGBoost Churn Engine into daily batch production pipeline.\n\n" \
             "2. Enable Automated Retention Offer Engine for top 2 risk deciles.\n\n" \
             "3. Empower Customer Support agents with real-time SHAP churn triggers in support dashboard."
    p.font.size = Pt(15)
    p.font.color.rgb = LIGHT_GRAY

    # Save presentation
    output_path = os.path.join(os.path.dirname(__file__), '../exports/FinTech_Churn_Retention_Strategy.pptx')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"[Deck Generator] Successfully created PowerPoint presentation at {output_path}")

if __name__ == '__main__':
    create_presentation()
