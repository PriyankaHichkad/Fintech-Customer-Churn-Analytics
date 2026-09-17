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
    
    # Executive Theme Color Palette
    NAVY = RGBColor(15, 23, 42)        # #0F172A
    CARD_BG = RGBColor(30, 41, 59)     # #1E293B
    CYAN = RGBColor(14, 165, 233)      # #0EA5E9
    GREEN = RGBColor(16, 185, 129)     # #10B981
    WHITE = RGBColor(255, 255, 255)
    GRAY_TEXT = RGBColor(148, 163, 184) # #94A3B8
    LIGHT_GRAY = RGBColor(241, 245, 249)

    def add_header(slide, title_text, category_text="FINTECH CREDIT RISK & RETENTION ANALYTICS"):
        header_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.1))
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = NAVY
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

    # SLIDE 1: Title + One-Line Hook
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()
    
    tf1 = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(4.5)).text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "Predicting Credit Card Default Risk & Optimizing Retention Spend"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    p2 = tf1.add_paragraph()
    p2.text = "Prioritizing retention spend across 30,000 real credit card accounts (UCI Dataset)."
    p2.font.size = Pt(20)
    p2.font.color.rgb = CYAN
    p2.space_before = Pt(10)
    
    p3 = tf1.add_paragraph()
    p3.text = "\nHeadline Results: 483% Simulated Net Campaign ROI  •  $2.41M Net Profit Saved under $500k Budget Cap"
    p3.font.size = Pt(16)
    p3.font.bold = True
    p3.font.color.rgb = GREEN
    p3.space_before = Pt(15)

    # SLIDE 2: Business Problem (Framed as Money)
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "The Business Problem: Unit Economics & Capital Allocation")
    
    add_card(slide2, 0.8, 1.4, 3.6, 5.4, "Revenue Mechanics", CARD_BG)
    tf = slide2.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Issuers earn ~1.5%–2.5% per swipe in interchange fees plus annual card charges.\n\n• High-spending and high-limit cardholders drive over 60% of total portfolio net margin.\n\n• Acquiring a replacement customer costs $250–$600 (CAC)."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide2, 4.8, 1.4, 3.6, 5.4, "The Budget Waste", CARD_BG)
    tf = slide2.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Credit default/attrition erases years of accumulated interchange margin.\n\n• Generic retention offers (blanket credits) waste capital on low-risk cardholders.\n\n• With a fixed budget, which accounts do you prioritize to maximize net return?"
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide2, 8.8, 1.4, 3.7, 5.4, "The Strategy Solution", CARD_BG, border_color=CYAN)
    tf = slide2.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n1. ML Default Risk Scoring: Identify high-risk accounts 60-90 days early.\n\n2. SHAP Trigger Diagnostics: Pinpoint friction (delinquency vs utilization vs fees).\n\n3. Knapsack ROI Optimizer: Allocate budget by net profit density."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 3: Data & Approach (Simple Flow Diagram)
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Data Pipeline & Analytics Architecture")
    
    add_card(slide3, 0.8, 1.4, 11.7, 5.4, "From 30,000 Real Statements to Decision-Support Engine", CARD_BG)
    tf = slide3.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n[30,000 Real UCI Accounts]  ──>  [6-Month Statement History]  ──>  [Engineered Features]\n" \
             "                                                             • 6m Utilization Trend\n" \
             "                                                             • Pay-to-Bill Ratio\n" \
             "                                                             • Delinquency Escalation\n" \
             "                                                             • Deterministic RFM Matrix\n\n" \
             "       ┌──────────────────────────────────────────────────────────────┘\n" \
             "       ▼\n" \
             "[XGBoost Churn/Default Model] ──> [SHAP Root-Cause Diagnostics] ──> [Knapsack ROI Optimization Engine]"
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 4: Model Results & SHAP Drivers
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Machine Learning Performance & SHAP Drivers")
    
    add_card(slide4, 0.8, 1.4, 5.6, 5.4, "Model Accuracy Metrics", CARD_BG)
    tf = slide4.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• XGBoost Classifier ROC-AUC: 0.7848\n\n• Baseline Logistic Regression AUC: 0.7721\n\n• Top 20% Risk Decile Capture Rate: >55% of all true default/churn accounts\n\n• Enables hyper-targeted interventions without spamming low-risk cardholders."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide4, 6.8, 1.4, 5.7, 5.4, "Top SHAP Default Triggers", CARD_BG, border_color=CYAN)
    tf = slide4.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n1. PAY_0 (Recent Repayment Delay): Strongest empirical default predictor.\n\n2. Current Utilization Ratio: High utilization (>75%) signals debt stress.\n\n3. 6-Month Pay-to-Bill Ratio: Identifies revolvers paying minimum balance.\n\n4. Credit Limit (LIMIT_BAL): Dictates total potential LTV margin exposure."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 5: From Prediction to Economics (The LTV/CAC Bridge)
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "The LTV Economics Bridge & Offer Catalog")
    
    add_card(slide5, 0.8, 1.4, 5.6, 5.4, "Mathematical LTV Equation", CARD_BG)
    tf = slide5.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\nLTV = Annual Net Margin × [ 1 / (Churn Prob + Discount Rate) ]\n\n" \
             "• Net Margin = (Interchange + Fees + Interest) - (Rewards + Servicing)\n\n" \
             "• Risk Score → Expected LTV Margin Loss → Targeted Offer Selection\n\n" \
             "• Prevents spending $150 offer on an account with only $80 in remaining LTV."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    add_card(slide5, 6.8, 1.4, 5.7, 5.4, "4-Offer Retention Catalog", CARD_BG)
    tf = slide5.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n1. Annual Fee Waiver: Waives $50–$495 fee for fee-dissatisfied users.\n2. 2x Points Boost: 1.5% spend boost for active monthly spenders.\n3. 0% APR Cut / Relief: $120 interest margin discount for revolvers.\n4. VIP Perks & Concierge: $150 perks for Platinum & Black cardholders."
    p.font.size = Pt(13)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 6: ROI Simulation Results
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "ROI Simulation Engine Results (Knapsack Density Optimization)")
    
    add_card(slide6, 0.8, 1.4, 11.7, 5.4, "Portfolio Performance Breakdown ($500k Budget Cap)", CARD_BG, border_color=GREEN)
    tf = slide6.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Total Unmitigated At-Risk LTV Exposure:  $1.48M+\n" \
             "• Total Campaign Budget Allocated:       $500,000\n" \
             "• Targeted High-Risk Accounts:            3,194 / 30,000 (10.6%)\n" \
             "• Gross Retained LTV Saved:               $2.91M+\n" \
             "• Net Retained Profit Saved:              $2.41M+\n" \
             "• Portfolio Campaign Net ROI:             483.0%"
    p.font.size = Pt(15)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 7: Recommendations & What to Do Next
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "Operational Next Steps & Strategic Recommendations")
    
    add_card(slide7, 0.8, 1.4, 11.7, 5.4, "Action Plan for FinTech Product & Growth Teams", CARD_BG, border_color=CYAN)
    tf = slide7.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n1. Pilot Top-Decile Campaign: Roll out automated offer dispatches to top 2 risk deciles.\n\n" \
             "2. Replace Simulated Multipliers with Causal A/B Testing: Calibration of treatment effect assumptions against live production experimental data.\n\n" \
             "3. Extend Model to True Attrition Data: Connect credit default signals to account-closure events when attrition logs become available."
    p.font.size = Pt(15)
    p.font.color.rgb = LIGHT_GRAY

    # SLIDE 8: Dashboard Showcase
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Interactive Streamlit Strategy Dashboard (app.py)")
    
    add_card(slide8, 0.8, 1.4, 11.7, 5.4, "Operational Decision-Support App for Risk & Product Managers", CARD_BG)
    tf = slide8.shapes[-1].text_frame
    p = tf.add_paragraph()
    p.text = "\n• Executive Portfolio Overview: Live metrics on at-risk revenue, churn deciles, and tier distributions.\n\n" \
             "• Machine Learning & SHAP Inspector: Visualizing global feature contributions and model calibration.\n\n" \
             "• Individual Account Lookup: Deep-dive into specific cardholders with automated offer suggestions.\n\n" \
             "• What-If ROI Simulator: Real-time budget sliders to test portfolio campaign scenarios."
    p.font.size = Pt(14)
    p.font.color.rgb = LIGHT_GRAY

    # Save presentation
    output_path = os.path.join(os.path.dirname(__file__), '../exports/FinTech_Churn_Retention_Strategy.pptx')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"[Deck Generator] Successfully created 8-Slide PowerPoint presentation at {output_path}")

if __name__ == '__main__':
    create_presentation()
