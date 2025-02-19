# Predicting Divorce Likelihood: Gottman’s Framework and Machine Learning

## 📌 Project Purpose
This project aims to analyze divorce prediction using the **Divorce Predictors Scale (DPS)** based on **Gottman’s framework**. By leveraging **Regression Analysis, Decision Trees, and Recursive Feature Elimination (RFE)**, we identify key factors influencing marital outcomes. Our analysis provides insights into divorce likelihood based on behavioral attributes.

---

## 📊 Dataset Description
We used the **Divorce Predictors Dataset** from the **UCI Machine Learning Repository**, which consists of **170 participants** (86 married, 84 divorced). The dataset includes **54 behavioral attributes** measured on a **Likert scale (0-5)**, covering aspects of marital stability such as emotional intimacy, conflict resolution, and communication quality.

### **📖 Data Dictionary**
| Column Name | Description |
|-------------|-------------|
| `Divorce_Status` | 1 = Divorced, 0 = Married |
| `Enhancing_Positive_Interactions` | Measures positive behaviors in a marriage (e.g., quality time, mutual respect) |
| `Four_Horsemen_Behavior_Patterns` | Negative behaviors (criticism, contempt, defensiveness, stonewalling) |
| `Love_Maps` | Understanding partner’s social connections and personal interests |
| `Positive_Perspective` | Maintaining an optimistic view of the partner and relationship |
| `Conflict_Resolution_Style` | Approach to handling marital disputes (avoidance, compromise, aggression) |

---

## 🏷 Methodology
### 1️⃣ Data Preprocessing
- Categorized **54 attributes** into **5 behavioral categories** based on **Gottman’s framework**.
- Applied **weighted averaging** to quantify each category into independent variables.
- Split the dataset into **training (80%)** and **test (20%)** sets.

### 2️⃣ Analysis Techniques
- **Logistic Regression**: Assessed the relationship between attributes and divorce likelihood.
- **Decision Tree Model**: Identified key factors with the highest predictive power.
- **Recursive Feature Elimination (RFE)**: Ranked feature importance and eliminated the least impactful variables.

### 3️⃣ Data Visualization
- **Odds Ratios from Logistic Regression** – Quantified the impact of behavioral traits.
- **Decision Tree Visualization** – Showed classification thresholds for divorce likelihood.
- **Feature Importance Ranking from RFE** – Identified the most predictive attributes.

---

## 🔍 Key Findings & Insights
- **Four Horsemen Behavior Patterns** had the highest impact on divorce prediction:
  - **Odds Ratio = 4.03** in Logistic Regression
  - **30% of top predictors in RFE analysis**
- **Enhancing Positive Interactions** ranked highest in Decision Tree model (**Feature Importance = 0.90**).
- **Criticism, Defensiveness, and Stonewalling** were the strongest individual predictors.
- **Positive Perspective and Love Maps** contributed to marriage stability but had lower predictive power compared to negative behaviors.

---

## 🚀 Business Recommendations
Based on our findings, we propose the following strategies:
- **Relationship Counseling Programs** → Focus on reducing Four Horsemen behaviors.
- **Marital Therapy Interventions** → Prioritize conflict resolution techniques.
- **Predictive Divorce Risk Models** → Use machine learning to assess relationship health.

---

## 📌 Conclusion
This project successfully predicted divorce likelihood using **machine learning techniques** and **behavioral psychology theories**. Our results confirmed that **negative behaviors (criticism, contempt, defensiveness, stonewalling) are the strongest predictors of divorce**. Future work could integrate additional psychological datasets and explore real-time prediction models.

---

## 🛠 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Logistic Regression, Decision Trees, RFE**
- **Data Preprocessing & Feature Engineering**

---

## 📂 Repository Files
- **`divorce2.xlsx`** – Processed dataset with behavioral attributes.
- **`divorce_prediction.py`** – Python script for regression, decision tree, and feature selection.
- **`README.md`** – Project documentation.



