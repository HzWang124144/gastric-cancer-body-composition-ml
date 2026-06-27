Code for Gastric Cancer Body Composition Machine-Learning Study

This repository contains the machine-learning analysis code for a retrospective study developing interpretable prediction models that integrate CT-derived body composition features and clinicopathological variables to predict postoperative recurrence and 3-year survival in patients with gastric cancer after curative resection and adjuvant chemotherapy.

The code is provided to improve transparency and reproducibility of the machine-learning workflow, including data preprocessing, missing-value imputation, feature selection, hyperparameter tuning, model training, internal validation, threshold selection, calibration assessment, decision curve analysis, SHAP interpretation, SHAP stability analysis, full-versus-reduced model comparison, and recurrence sensitivity analysis.

Study overview

The institutional cohort was used for model development and internal validation. CT-derived body composition features, including visceral adipose tissue (VAT), subcutaneous adipose tissue (SAT), and their longitudinal changes, were quantified from L3-level abdominal CT images and integrated with clinicopathological variables.

The main prediction tasks were:

Postoperative recurrence prediction.
Three-year survival prediction.

Multiple candidate models were evaluated, including Logistic Regression, Support Vector Classifier, Multilayer Perceptron, Random Forest, XGBoost, LightGBM, and CatBoost. CatBoost was selected as the final model based on its overall performance profile, including discrimination, recall-prioritized classification performance, calibration, decision curve analysis, and interpretability.

Scope of this repository

This repository includes code related to the machine-learning analyses performed in the institutional cohort. The main analytical components include:

Data preprocessing and missing-value imputation.
Train-test splitting.
Feature selection using RFECV.
Multicollinearity assessment using VIF and Spearman correlation analysis.
Hyperparameter tuning using grid search or randomized search.
Training and evaluation of candidate machine-learning models.
ROC-AUC and PR-AUC analysis.
Classification metrics including accuracy, precision, recall, F1 score, and specificity.
Decision-threshold exploration and recall-prioritized threshold selection.
Calibration analysis using calibration curves, Brier score, expected calibration error, calibration intercept, and calibration slope.
Decision curve analysis.
Paired DeLong tests for AUC comparison.
Full-versus-reduced model comparison to evaluate the contribution of body composition variables.
Sensitivity analysis for recurrence prediction.
SHAP-based model interpretation.
Bootstrap-based SHAP feature-importance stability analysis.

Data availability

Individual-level clinical data and imaging-derived body composition data are not publicly shared because of patient privacy, institutional ethical restrictions, and data-use limitations.

This repository provides analysis code and model-development workflow information to improve transparency. The original patient-level dataset is required to reproduce the numerical results reported in the manuscript. De-identified data may be made available from the corresponding author upon reasonable request and subject to institutional approval.

Software environment

The machine-learning analyses were performed using Python.

Main Python packages include:

numpy
pandas
scikit-learn
catboost
xgboost
lightgbm
shap
matplotlib
scipy
statsmodels
openpyxl

Exact package versions should be recorded in requirements.txt or another environment file if available.

Reproducibility notes

To improve reproducibility, the analysis code includes or records the following information where applicable:

Candidate predictor variables.
Full and reduced feature sets.
Train-test split settings.
Random seeds.
RFECV settings.
Hyperparameter search spaces.
Cross-validation strategy.
Final model evaluation metrics.
Threshold-selection procedure.
Calibration metrics.
Decision curve analysis settings.
SHAP interpretation workflow.
Bootstrap settings for SHAP stability analysis.

Because the original clinical dataset is not publicly available, the code is intended to document the analytical workflow and allow reproduction by authorized users with access to the required institutional data.

Important notes
The institutional cohort was used for model development and internal validation.
No true external validation cohort is included in this repository.
The operating thresholds reported in the manuscript are exploratory and require external validation and recalibration before clinical implementation.
SHAP values should be interpreted as model-level feature attribution rather than evidence of independent causal effects.
The code is provided for academic transparency and reproducibility, not for direct clinical deployment.
Citation

If you use or adapt this code, please cite the associated manuscript:

Wang H, et al. Interpretable machine-learning prediction of postoperative recurrence and 3-year survival in gastric cancer using CT-derived body composition features. 
License

This repository is released under the MIT License.
