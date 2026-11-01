# UCS Prediction — Streamlit

Streamlit deployment of the UCS XGBoost model with category-specific SHAP violin and local explanations.

## Included model assets

`ModelsGUI/Best_XGBoost_Regressor_UCS.joblib`, `PolyTransformer_UCS.joblib`, `Scaler_UCS.joblib`, and `Data_Split_UCS.xlsx` are included. Keep these files together. The saved polynomial transform requires this **exact** order: `Category`, `VP (m/s)`, `PLS (MPa)`, `SH`, `SH type`.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Use Python **3.11** (recommended). The polynomial transformer and scaler were serialized with scikit-learn **1.6.1**; using a different version can be unsafe.

## Deploy

1. Create a GitHub repository and upload **the contents of this folder** (not the ZIP itself).
2. Visit https://share.streamlit.io/ and connect the repository.
3. Set the main file path to `app.py`, choose Python 3.11 in advanced settings if available, and deploy.
4. Keep the model artifacts in the `ModelsGUI` folder and do not change their names.

## Important limitations

**Model applicability is limited to the multivariate training domain, even when individual inputs fall within their respective ranges.**

Input limits in the UI are displayed ranges, not proof of in-domain validity. The optional nearest-neighbor distance is descriptive, without a validated threshold. No prediction interval is supplied.

SHAP values are calculated on scaled polynomial features, then **allocated equally across the original inputs present in each polynomial term** for readability. These allocated attributions are an interpretation heuristic, not exact raw-input SHAP values. The `Category` attribution is intentionally omitted from figures but remains in the model. SHAP reference observations are sampled within the selected rock category, using the saved training reference sheet.

**Model compatibility:** The included XGBoost model was saved with an earlier XGBoost release; if loading fails after deployment, re-export the trained model with the deployment version and verify prediction parity before publication. Confirm deployed predictions against a set of trusted notebook examples before sharing publicly.

© 2026 IA Tijani, IA Lawal & collaborators. All rights reserved.
