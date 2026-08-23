# SmartAnalytics

AI-driven analytics dashboard for CSV and Excel uploads. Uses DuckDB for in-memory querying and Groq for LLM-generated insights and chat Q&A.

## Local setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY
streamlit run app.py
```

Open http://localhost:8501

## Deploy to Streamlit Community Cloud

Repo: [github.com/MadhuGannabathula/Smart-Analytics](https://github.com/MadhuGannabathula/Smart-Analytics)

### 1. Push code to GitHub

```bash
git init
git add .
git commit -m "Initial SmartAnalytics app"
git branch -M main
git remote add origin https://github.com/MadhuGannabathula/Smart-Analytics.git
git push -u origin main
```

### 2. Create the app on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select repository **MadhuGannabathula/Smart-Analytics**.
4. Set **Main file path** to `app.py`.
5. Click **Advanced settings** → **Secrets** and paste:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = "0"
```

6. Click **Deploy**.

Your app will be live at a URL like `https://smart-analytics-xxxxx.streamlit.app`.

## Usage

1. Upload one or more **CSV** or **Excel** files in the sidebar.
2. The main dashboard auto-generates insights from your data.
3. Ask follow-up questions in the sidebar chat.
4. Use **➕ Add to dashboard** to pin chat charts to the main view.

## Architecture

| Module | Role |
|--------|------|
| `data_layer.py` | Parse uploads, validate, register DuckDB tables, build schema profiles |
| `llm.py` | Groq client, insight suggestions, chat Q&A, SQL validation & retry |
| `charts.py` | Plotly rendering with the sky-blue theme |
| `config.py` | Reads settings from `.env` (local) or Streamlit secrets (cloud) |
| `app.py` | Streamlit UI — sidebar upload/chat, main dashboard |
