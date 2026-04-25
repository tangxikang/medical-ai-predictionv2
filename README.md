# Web Deploy

This folder contains a production-oriented Streamlit app package.

## Files

- `app.py`: web entrypoint
- `web_support.py`: model artifact discovery + feature spec helper
- `requirements.txt`: python dependencies

## Expected project layout

The app looks for artifacts under `WEB_BASE_DIR` (or project root by default):

- `outputs_*/models/best_model.joblib`
- `outputs_*/models/best_model_metadata.json`
- `data.xlsx`

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

