# Web Deploy

This folder contains a production-oriented Streamlit app package.

## Files

- `app.py`: web entrypoint
- `web_support.py`: artifact discovery + feature spec helpers
- `model_compat.py`: sklearn/joblib compatibility loader
- `requirements.txt`: Python dependencies

## Expected layout

The app looks for model artifacts under `WEB_BASE_DIR` (or `web_deploy` by default):

- `outputs_*/models/best_model.joblib`
- `outputs_*/models/best_model_metadata.json`

## Data file policy (production)

- `data.xlsx` is optional.
- If present, it is used for richer input defaults and SHAP background sampling.
- If missing, prediction still works by inferring feature specs from the fitted pipeline.
- When `data.xlsx` is missing, SHAP force plot is skipped.
- Override the data file path with `WEB_DATA_FILE` (supports absolute path).

## Run locally

```bash
cd web_deploy
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with custom base path

```bash
set WEB_BASE_DIR=F:\咸鱼代写\900机器学习
streamlit run web_deploy\app.py
```
