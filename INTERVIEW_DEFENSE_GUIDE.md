# Interview Defense & Methodological FAQ Guide
## FinTech Retention Strategy (UCI Credit Card Analytics Engine)

This guide provides precise, interview-hardened responses to technical, business, and quantitative questions regarding the model architecture, unit economics, and capital allocation algorithms.

---

### Resume Pointers Reference (What is on your CV)
> **FinTech Retention Strategy**  
> * Built XGBoost model on 30k UCI accounts giving 0.779 test ROC-AUC; top 20% captures 51.7% defaulters.  
> * Modelled credit card unit economics (interchange, fees, interest, rewards, servicing) to compute 3-year capped LTV.  
> * Optimised $100k retention budget by knapsack sort for 752 accounts, netting 247% ROI on conservative estimates.  
> * Designed an executive decision deck featuring TreeSHAP root-cause diagnostics to guide portfolio retention strategy.

---

### Key Review Items & Interview-Safe Answers

#### 1. Target Formulation: Default vs. Voluntary Churn
* **Question / Challenge:** *"The UCI target is `default.payment.next.month`. Default is financial distress and delinquency, not voluntary customer churn. Why use retention offers instead of collections?"*
* **Interview-Safe Answer:**
  > *"That is an important distinction. The raw UCI dataset records next-month payment default (30+ DPD), which represents financial distress rather than contractual account closure. In this project, I used default as an attrition/impairment risk proxy within a simulated proactive retention and delinquency mitigation framework.*
  > 
  > *In a retail bank, high-risk delinquent accounts typically go to collections or credit-line reductions. However, for transactors and borderline revolvers, proactive interventions—like temporary APR relief or fee waivers—can cure early-stage delinquency and prevent full charge-offs. In a live banking environment with core transaction ledgers, I would separate the decision engine into two models: voluntary churn (attrition) and involuntary default (credit risk)."*

---

#### 2. Assumed Intervention Effectiveness vs. Observed Causal Uplift
* **Question / Challenge:** *"Your ROI calculations assume that retention offers reduce default risk by set multipliers. Did you measure causal uplift or run an A/B test?"*
* **Interview-Safe Answer:**
  > *"No, because the UCI dataset is historical observational data without historical marketing campaigns or randomized trials. The intervention effectiveness multipliers in `config.py` are scenario assumptions based on industry benchmarks.*
  > 
  > *Because of this, I explicitly labeled the 593% base return as an optimistic scenario and applied a 50% sensitivity haircut to customer response rates. Even under this severe haircut, the campaign nets 247% ROI ($246k net profit). In my executive deck and recommendations, Phase 2 is specifically an A/B pilot with a randomized holdout control group using CausalML / uplift modeling (e.g. T-Learner) to calibrate true incremental lift before enterprise rollout."*

---

#### 3. Scoring the Full Portfolio vs. Out-of-Sample Evaluation
* **Question / Challenge:** *"Did you score the 30,000 accounts using a model that trained on 75% of them? Isn't that in-sample scoring for the campaign?"*
* **Interview-Safe Answer:**
  > *"The machine learning model's predictive power was evaluated strictly out-of-sample on the held-out 25% test set (7,500 accounts), which yielded 0.7789 test ROC-AUC and 51.66% top-decile capture.*
  > 
  > *However, for the enterprise campaign simulation, I scored the full 30,000-account book to simulate how a bank would allocate a $100k budget across its entire existing customer base. Because the simulation includes training accounts, the portfolio return figures are directional scenario estimates. In production, I would generate cross-validated out-of-fold (OOF) scores for existing accounts or deploy the model prospectively on an out-of-time (OOT) cohort."*

---

#### 4. Greedy Knapsack Sorting vs. Global Multi-Choice Optimizer
* **Question / Challenge:** *"Why use a greedy density ranking instead of an exact 0/1 knapsack integer linear program (ILP)?"*
* **Interview-Safe Answer:**
  > *"I implemented Dantzig’s greedy Knapsack heuristic, which ranks candidate items by ROI density ($\text{Net ROI Value} / \text{Offer Cost}$) and fills the budget greedily. This provides a bounded near-optimal approximation (typically within 1–2% of optimal) and scales to millions of accounts in $O(N \log N)$ time.*
  > 
  > *The algorithm operates in two stages: each account pre-selects its highest net-value offer first, and accounts are then ranked by density. In an enterprise system with complex multi-offer eligibility or cross-product constraints, I would formulate this as a Multi-Choice Knapsack Problem (MCKP) using an integer linear solver like PuLP or SCIP."*

---

#### 5. Synthetic Behavioral Proxies (`days_since_last_txn`)
* **Question / Challenge:** *"UCI doesn't have transaction timestamps. How did you compute days since last transaction, and why does it rank high in TreeSHAP?"*
* **Interview-Safe Answer:**
  > *"The UCI dataset contains 6-month monthly billing and payment aggregations, not line-item point-of-sale timestamps. To demonstrate an RFM segmentation framework, I constructed `days_since_last_txn` as a behavioral proxy tied to payment timeliness (`PAY_0`) with controlled jitter for on-time payers.*
  > 
  > *Because `days_since_last_txn` incorporates `PAY_0`, its high TreeSHAP attribution directly reflects the predictive primacy of repayment delay. In a production environment with access to authorization feeds, this proxy would be replaced with real point-of-sale swipe recency from the merchant clearing ledger."*

---

#### 6. Billing Statement Balance as Spend Proxy
* **Question / Challenge:** *"Why did you use average billing balance for spend? Statement balances include carryover debt, interest, and fees."*
* **Interview-Safe Answer:**
  > *"In the UCI dataset, monthly bill amounts (`BILL_AMT1-6`) are the only recorded expenditure proxies available. While a monthly statement balance combines new purchase activity with revolving balances and finance charges, it serves as an account-level balance and activity proxy.*
  > 
  > *In production banking, interchange fees (1.5%–2.5%) and rewards expenses (1.5%) are calculated strictly on cleared gross purchase authorization volume from transaction logs, while finance charges are calculated separately on revolving balances."*

---

#### 7. Capitalized Retention Multiplier vs. Multi-Period DCF
* **Question / Challenge:** *"Your LTV formula is $\text{Margin} \times \min(3, 1/(p+d))$. Why not a multi-period survival discounted cash flow (DCF) model?"*
* **Interview-Safe Answer:**
  > *"I used a capitalized retention multiplier, which is the standard customer equity formula: $\text{LTV} = \text{Margin} / (\text{Attrition} + \text{Discount})$. Because that formula can blow up for low-risk customers ($p < 0.05$), I introduced a strict 3.0-year ceiling ($\min(3.0, 1/(p+d))$) to prevent over-crediting long-tenured accounts.*
  > 
  > *This provides a realistic, conservative annualized LTV proxy. In a production quantitative credit risk setting, I would implement a discrete 36-month survival DCF model: $\text{NPV} = \sum_{t=1}^{36} \frac{\text{Margin}_t \cdot S(t)}{(1+d/12)^t}$, where $S(t)$ is estimated via Cox Proportional Hazards or discrete hazard models."*
