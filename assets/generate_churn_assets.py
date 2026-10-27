import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

sys.stdout.reconfigure(encoding='utf-8')

# Paths
BASE_DIR = r"C:\Users\admin\Desktop\Xuân Thu\Da_Portfolio\04_ML\churn_prediction"
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "churn_prediction.xlsx")
TRAIN_PROC_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
TEST_PROC_PATH = os.path.join(BASE_DIR, "data", "processed", "test_processed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Styling Configuration
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
BG_COLOR = "#0D1117"
CARD_BG = "#161B22"
TEXT_COLOR = "#F0F6FC"
MUTED_TEXT = "#8B949E"
CYAN = "#58A6FF"
CORAL = "#F85149"
GREEN = "#3FB950"
GOLD = "#D29922"
PURPLE = "#BC8CFF"

def apply_dark_style(fig, axes):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#30363D")
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        ax.grid(color="#21262D", linestyle="--", linewidth=0.7, alpha=0.7)

print("Starting asset generation...")

# ==============================================================================
# FIGURE 1: STEP 1 - DATA UNDERSTANDING & PROFILING
# ==============================================================================
print("Generating Step 1 figure...")
df_raw = pd.read_excel(RAW_DATA_PATH)

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7))
apply_dark_style(fig1, axes1)

# Subplot 1: Target Imbalance (Donut Chart)
churn_counts = df_raw['Churn'].value_counts()
labels = [f"Retained (0)\n{churn_counts[0]:,} ({churn_counts[0]/len(df_raw)*100:.1f}%)", 
          f"Churned (1)\n{churn_counts[1]:,} ({churn_counts[1]/len(df_raw)*100:.1f}%)"]
colors = [CYAN, CORAL]
wedges, texts = axes1[0].pie(churn_counts, labels=labels, colors=colors, startangle=140,
                             wedgeprops=dict(width=0.45, edgecolor=BG_COLOR, linewidth=2),
                             textprops=dict(color=TEXT_COLOR, fontsize=12, fontweight='bold'))
axes1[0].set_title("Target Class Imbalance Distribution\n(Total: 5,630 Customers | Ratio: ~4.94:1)", 
                   fontsize=14, fontweight='bold', pad=15)
axes1[0].text(0, 0, "16.84%\nChurn", ha='center', va='center', color=CORAL, fontsize=16, fontweight='heavy')

# Subplot 2: Missing Values Profile
missing_s = df_raw.isnull().sum()
missing_df = pd.DataFrame({
    'Column': missing_s[missing_s > 0].index,
    'MissingCount': missing_s[missing_s > 0].values,
    'Pct': (missing_s[missing_s > 0].values / len(df_raw)) * 100
}).sort_values('Pct', ascending=True)

bars = axes1[1].barh(missing_df['Column'], missing_df['Pct'], color=GOLD, height=0.55, edgecolor="#F0883E", alpha=0.9)
axes1[1].set_title("Missing Values Audit (7 Operational Features < 6%)\n(Requires Missing Indicator + Leakage-Free Imputation)", 
                   fontsize=14, fontweight='bold', pad=15)
axes1[1].set_xlabel("Missing Percentage (%)", fontsize=11, fontweight='bold')
axes1[1].set_xlim(0, 7)
for bar, count in zip(bars, missing_df['MissingCount']):
    axes1[1].text(bar.get_width() + 0.15, bar.get_y() + bar.get_height()/2, 
                  f"{bar.get_width():.2f}% ({count} rows)", 
                  va='center', color=TEXT_COLOR, fontsize=10, fontweight='bold')

plt.suptitle("STEP 01: DATA UNDERSTANDING & DATASET INTEGRITY PROFILING (01_data_understanding.ipynb)", 
             color=TEXT_COLOR, fontsize=16, fontweight='heavy', y=0.98)
plt.tight_layout()
fig1.savefig(os.path.join(ASSETS_DIR, "step_1_data_understanding.png"), dpi=150, bbox_inches='tight')
plt.close(fig1)

# ==============================================================================
# FIGURE 2: STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("Generating Step 2 figure...")
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6.5))
apply_dark_style(fig2, axes2)

# Subplot 1: Correlation with Churn
num_cols = df_raw.select_dtypes(include='number').columns.drop(['CustomerID', 'Churn'])
corr_s = df_raw[list(num_cols) + ['Churn']].corr()['Churn'].drop('Churn').sort_values(ascending=True)
bar_colors = [CYAN if x < 0 else CORAL for x in corr_s.values]
axes2[0].barh(corr_s.index, corr_s.values, color=bar_colors, height=0.6, edgecolor="#30363D")
axes2[0].axvline(0, color=MUTED_TEXT, linestyle="--", linewidth=1)
axes2[0].set_title("Feature Correlation with Churn\n(Tenure -0.35 vs Complain +0.25)", fontsize=13, fontweight='bold')
axes2[0].set_xlabel("Pearson Correlation Coefficient", fontsize=10, fontweight='bold')
axes2[0].set_xlim(-0.45, 0.35)
for i, v in enumerate(corr_s.values):
    offset = 0.015 if v >= 0 else -0.06
    axes2[0].text(v + offset, i, f"{v:+.3f}", va='center', color=TEXT_COLOR, fontsize=9, fontweight='bold')

# Subplot 2: Tenure Boxplot / Distribution
sns.boxplot(data=df_raw, x='Churn', y='Tenure', ax=axes2[1], palette=[CYAN, CORAL], width=0.45,
            boxprops=dict(edgecolor=TEXT_COLOR, linewidth=1.2), medianprops=dict(color=GOLD, linewidth=2))
axes2[1].set_xticklabels(["Retained (0)", "Churned (1)"], fontsize=11, fontweight='bold')
axes2[1].set_title("Tenure Cliff: Customer Relationship Lifespan\n(Median Churned: 2.0 mos vs Retained: 10.0 mos)", 
                   fontsize=13, fontweight='bold')
axes2[1].set_ylabel("Tenure (Months)", fontsize=11, fontweight='bold')
axes2[1].set_xlabel("")

# Subplot 3: Churn Rate by Complaint Status
complain_churn = df_raw.groupby('Complain')['Churn'].agg(['mean', 'count'])
bars_c = axes2[2].bar(["No Complaint (0)\n(4,026 Cust)", "Has Complaint (1)\n(1,604 Cust)"], 
                      complain_churn['mean'] * 100, color=[CYAN, CORAL], width=0.45, edgecolor=TEXT_COLOR)
axes2[2].set_title("Customer Complaint Impact on Churn Rate\n(31.7% Churn with Complaint vs 10.9% Without)", 
                   fontsize=13, fontweight='bold')
axes2[2].set_ylabel("Churn Rate (%)", fontsize=11, fontweight='bold')
axes2[2].set_ylim(0, 40)
for bar in bars_c:
    axes2[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                  f"{bar.get_height():.1f}%", ha='center', color=TEXT_COLOR, fontsize=12, fontweight='bold')

plt.suptitle("STEP 02: EXPLORATORY DATA ANALYSIS & CHURN BEHAVIOR PATTERNS (02_eda.ipynb)", 
             color=TEXT_COLOR, fontsize=16, fontweight='heavy', y=0.98)
plt.tight_layout()
fig2.savefig(os.path.join(ASSETS_DIR, "step_2_eda_insights.png"), dpi=150, bbox_inches='tight')
plt.close(fig2)

# ==============================================================================
# FIGURE 3: STEP 3 - FEATURE ENGINEERING & PREPROCESSING
# ==============================================================================
print("Generating Step 3 figure...")
fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7))
apply_dark_style(fig3, axes3)

# Subplot 1: Churn Rate by Engineered Feature: TenureGroup
tenure_rates = pd.Series({'Senior (>36 mos)': 0.0, 'Mid (12-36 mos)': 5.18, 'New (≤12 mos)': 22.93})
bars_tg = axes3[0].bar(tenure_rates.index, tenure_rates.values, color=[GREEN, CYAN, CORAL], width=0.45, edgecolor=TEXT_COLOR)
axes3[0].set_title("Engineered Feature: Churn Rate across Tenure Groups\n(Extreme Vulnerability in New Onboarding Cohort)", 
                   fontsize=13, fontweight='bold')
axes3[0].set_ylabel("Churn Rate (%)", fontsize=11, fontweight='bold')
axes3[0].set_ylim(0, 28)
for b in bars_tg:
    axes3[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, 
                  f"{b.get_height():.2f}%", ha='center', color=TEXT_COLOR, fontsize=11, fontweight='bold')

# Subplot 2: Churn Rate by IsRecentBuyer & Architecture Infographic Text
buyer_rates = pd.Series({'Dormant (>7 days)': 9.58, 'Recent Buyer (≤7 days)': 18.99})
bars_rb = axes3[1].bar(buyer_rates.index, buyer_rates.values, color=[CYAN, CORAL], width=0.4, edgecolor=TEXT_COLOR)
axes3[1].set_title("Engineered Feature: IsRecentBuyer Recency Interaction\n(Higher Churn in Recent Buyers Seeking Support)", 
                   fontsize=13, fontweight='bold')
axes3[1].set_ylabel("Churn Rate (%)", fontsize=11, fontweight='bold')
axes3[1].set_ylim(0, 25)
for b in bars_rb:
    axes3[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.6, 
                  f"{b.get_height():.2f}%", ha='center', color=TEXT_COLOR, fontsize=11, fontweight='bold')

plt.suptitle("STEP 03: LEAKAGE-FREE FEATURE ENGINEERING & PREPROCESSING (03_feature_engineering.ipynb)", 
             color=TEXT_COLOR, fontsize=16, fontweight='heavy', y=0.98)
plt.tight_layout()
fig3.savefig(os.path.join(ASSETS_DIR, "step_3_feature_engineering.png"), dpi=150, bbox_inches='tight')
plt.close(fig3)

# ==============================================================================
# FIGURE 4: STEP 4 - MODEL BENCHMARKING & TUNING
# ==============================================================================
print("Generating Step 4 figure...")
models_data = [
    {'Model': 'Logistic Regression', 'Accuracy': 0.8242, 'Precision': 0.4879, 'Recall': 0.8474, 'F1': 0.6192, 'ROC-AUC': 0.9043},
    {'Model': 'Random Forest', 'Accuracy': 0.9751, 'Precision': 0.9091, 'Recall': 0.9474, 'F1': 0.9278, 'ROC-AUC': 0.9928},
    {'Model': 'XGBoost', 'Accuracy': 0.9671, 'Precision': 0.8844, 'Recall': 0.9263, 'F1': 0.9049, 'ROC-AUC': 0.9862},
    {'Model': 'LightGBM (Champion)', 'Accuracy': 0.9778, 'Precision': 0.9231, 'Recall': 0.9474, 'F1': 0.9351, 'ROC-AUC': 0.9930},
    {'Model': 'LightGBM-Tuned', 'Accuracy': 0.9796, 'Precision': 0.9326, 'Recall': 0.9474, 'F1': 0.9399, 'ROC-AUC': 0.9929}
]
df_models = pd.DataFrame(models_data)

fig4, ax4 = plt.subplots(figsize=(16, 7))
apply_dark_style(fig4, [ax4])

x = np.arange(len(df_models))
width = 0.16

b1 = ax4.bar(x - 2*width, df_models['Recall'] * 100, width, label='Recall (Churn - Class 1) [PRIORITY]', color=CORAL, edgecolor=TEXT_COLOR)
b2 = ax4.bar(x - width, df_models['F1'] * 100, width, label='F1-Score', color=GOLD, edgecolor=TEXT_COLOR)
b3 = ax4.bar(x, df_models['Precision'] * 100, width, label='Precision (Churn - Class 1)', color=CYAN, edgecolor=TEXT_COLOR)
b4 = ax4.bar(x + width, df_models['Accuracy'] * 100, width, label='Accuracy', color=GREEN, edgecolor=TEXT_COLOR)
b5 = ax4.bar(x + 2*width, df_models['ROC-AUC'] * 100, width, label='ROC-AUC', color=PURPLE, edgecolor=TEXT_COLOR)

ax4.set_title("Machine Learning Model Benchmark & Selection (Test Set: 1,126 Customers)\nPrioritizing Churn Recall (Capturing Defectors) with High Precision & F1", 
              fontsize=14, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(df_models['Model'], fontsize=11, fontweight='bold')
ax4.set_ylabel("Score (%)", fontsize=11, fontweight='bold')
ax4.set_ylim(40, 105)
ax4.legend(loc='lower right', facecolor=CARD_BG, edgecolor="#30363D", labelcolor=TEXT_COLOR, fontsize=10)

# Annotate Champion
ax4.annotate("CHAMPION MODEL\n94.74% Recall | 92.31% Precision", 
             xy=(3 - 2*width, 94.74), xytext=(2.3, 101),
             arrowprops=dict(facecolor=GOLD, shrink=0.08, width=2, headwidth=8),
             color=GOLD, fontsize=11, fontweight='heavy', ha='center',
             bbox=dict(boxstyle="round,pad=0.4", fc=CARD_BG, ec=GOLD, lw=1.5))

plt.suptitle("STEP 04: MODEL TRAINING, BENCHMARKING & TUNING (04_model_training.ipynb)", 
             color=TEXT_COLOR, fontsize=16, fontweight='heavy', y=0.98)
plt.tight_layout()
fig4.savefig(os.path.join(ASSETS_DIR, "step_4_model_benchmark.png"), dpi=150, bbox_inches='tight')
plt.close(fig4)

# ==============================================================================
# FIGURE 5: STEP 5 - MODEL EVALUATION & EXPLAINABILITY
# ==============================================================================
print("Generating Step 5 figure...")
test_df = pd.read_csv(TEST_PROC_PATH)
X_test = test_df.drop(columns=['Churn'])
y_test = test_df['Churn']
best_model = joblib.load(MODEL_PATH)
preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

fig5, axes5 = plt.subplots(2, 2, figsize=(18, 14))
axes5_flat = axes5.flatten()
apply_dark_style(fig5, axes5_flat)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=False, cmap='Blues', ax=axes5[0, 0], cbar=False)
# Custom annotations
axes5[0, 0].text(0.5, 0.5, f"True Negative (TN)\n{cm[0,0]} accounts\n(98.4% Retained correctly)", 
                 ha='center', va='center', color=TEXT_COLOR, fontsize=12, fontweight='bold')
axes5[0, 0].text(1.5, 0.5, f"False Positive (FP)\n{cm[0,1]} accounts\n(False alarms: only 1.6%)", 
                 ha='center', va='center', color=GOLD, fontsize=12, fontweight='bold')
axes5[0, 0].text(0.5, 1.5, f"False Negative (FN)\n{cm[1,0]} accounts\n(Missed churners: only 0.88%!)", 
                 ha='center', va='center', color=CORAL, fontsize=12, fontweight='heavy')
axes5[0, 0].text(1.5, 1.5, f"True Positive (TP)\n{cm[1,1]} accounts\n(94.74% Churners Caught!)", 
                 ha='center', va='center', color=GREEN, fontsize=12, fontweight='heavy')
axes5[0, 0].set_xlabel("Predicted Label (0: Retained, 1: Churned)", fontsize=11, fontweight='bold')
axes5[0, 0].set_ylabel("Actual Label (0: Retained, 1: Churned)", fontsize=11, fontweight='bold')
axes5[0, 0].set_title("Test Confusion Matrix (Total: 1,126 Accounts)\nFN = 10 (Critical Defectors Missed) vs TP = 180", 
                      fontsize=13, fontweight='bold')

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)
axes5[0, 1].plot(fpr, tpr, color=CYAN, linewidth=3, label=f"LightGBM (AUC = {roc_auc:.4f})")
axes5[0, 1].plot([0, 1], [0, 1], color=MUTED_TEXT, linestyle="--", label="Random Baseline (AUC = 0.50)")
axes5[0, 1].set_title(f"Receiver Operating Characteristic (ROC Curve)\nAUC = {roc_auc:.4f} (Near-Perfect Class Separation)", 
                      fontsize=13, fontweight='bold')
axes5[0, 1].set_xlabel("False Positive Rate", fontsize=11, fontweight='bold')
axes5[0, 1].set_ylabel("True Positive Rate", fontsize=11, fontweight='bold')
axes5[0, 1].legend(loc="lower right", facecolor=CARD_BG, edgecolor="#30363D", labelcolor=TEXT_COLOR)

# 3. Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, probs)
pr_auc = auc(rec, prec)
axes5[1, 0].plot(rec, prec, color=GREEN, linewidth=3, label=f"LightGBM (PR-AUC = {pr_auc:.4f})")
axes5[1, 0].axhline(y=y_test.mean(), color=MUTED_TEXT, linestyle="--", label=f"Baseline Ratio ({y_test.mean():.4f})")
axes5[1, 0].set_title(f"Precision-Recall Curve (Imbalanced Focus)\nPR-AUC = {pr_auc:.4f} (Superior Minority Class Accuracy)", 
                      fontsize=13, fontweight='bold')
axes5[1, 0].set_xlabel("Recall", fontsize=11, fontweight='bold')
axes5[1, 0].set_ylabel("Precision", fontsize=11, fontweight='bold')
axes5[1, 0].legend(loc="lower left", facecolor=CARD_BG, edgecolor="#30363D", labelcolor=TEXT_COLOR)

# 4. Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=X_test.columns).sort_values(ascending=False).head(15).sort_values(ascending=True)
bars_imp = axes5[1, 1].barh(feat_imp.index, feat_imp.values, color=PURPLE, height=0.6, edgecolor="#30363D")
axes5[1, 1].set_title("Top 15 Feature Importances (Tree Splits)\nTenure, Recency, Distance & Complaints Dominate", 
                      fontsize=13, fontweight='bold')
axes5[1, 1].set_xlabel("Importance (Split Count)", fontsize=11, fontweight='bold')
for b in bars_imp:
    axes5[1, 1].text(b.get_width() + 5, b.get_y() + b.get_height()/2, 
                     f"{int(b.get_width())}", va='center', color=TEXT_COLOR, fontsize=9, fontweight='bold')

plt.suptitle("STEP 05: MODEL EVALUATION, CONFUSION MATRIX & FEATURE ATTRIBUTION (05_model_evaluation.ipynb)", 
             color=TEXT_COLOR, fontsize=16, fontweight='heavy', y=0.98)
plt.tight_layout()
fig5.savefig(os.path.join(ASSETS_DIR, "step_5_model_evaluation.png"), dpi=150, bbox_inches='tight')
plt.close(fig5)

print("All 5 step assets generated successfully in assets/!")
