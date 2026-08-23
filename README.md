# SmartAnalytics

Upload your spreadsheets and get an AI-generated dashboard plus a chat assistant that answers questions about your data.

---

## How to run from GitHub

### Prerequisites

- Python 3.10+
- A [Groq](https://console.groq.com/) API key

### 1. Clone and install

```bash
git clone https://github.com/MadhuGannabathula/Smart-Analytics.git
cd Smart-Analytics
pip install -r requirements.txt
```

### 2. Add your API key

Copy the example env file and add your Groq key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Optional overrides:

```env
GROQ_MODEL=openai/gpt-oss-20b
LLM_TEMPERATURE=0
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set **Main file path** to `app.py`.
4. Under **Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "openai/gpt-oss-20b"
```

See `.streamlit/secrets.toml.example` for the full template.

### Quick start in the app

1. In the sidebar, upload one or more **CSV** or **Excel** (`.xlsx`) files.
2. Wait a few seconds — the dashboard populates automatically.
3. Use the chat panel to ask questions about your data.
4. Click **➕ Add to dashboard** on any chat chart you want to keep.
5. Click **＋ New Analysis** to clear everything and start over.

---

## What the app does

SmartAnalytics is a Streamlit app that turns uploaded spreadsheets into an AI-generated dashboard and lets you ask questions about your data in a sidebar chat.

### Upload and load data

- Accepts **CSV** and **Excel** (`.xlsx`, `.xls`) files
- Supports **multiple files** and **multiple Excel sheets** (each sheet becomes its own table)
- Stores everything in an **in-memory DuckDB** database for fast querying
- Profiles each table (columns, types, row counts, sample rows) for the AI

### Auto-generated dashboard

After upload, the AI builds a dashboard in this priority order:

1. **KPI** — key totals or averages
2. **Trend** — line charts over time
3. **Comparison** — bar or grouped bar across categories
4. **Composition** — pie charts for category breakdowns
5. **Cross-file** — joins across multiple uploaded tables

Use **Generate more insights** to add more charts without repeating existing ones.

### Chat Q&A

For each question, the app:

1. Sends schema, chat history, and current dashboard context to the LLM
2. LLM plans **new SQL** for your question
3. SQL runs against your DuckDB data
4. A second LLM call **summarizes real query results** (not guesses)
5. Optionally returns a chart in chat (pin it with **➕ Add to dashboard**)

The AI can also **remove outdated dashboard charts on its own** when a correction or refinement makes them redundant.

### Raw data browser

The **Raw Data** tab shows up to 500 rows per table directly from DuckDB.

### Architecture

| Module | Role |
|---|---|
| `app.py` | Streamlit UI, session state, upload/chat/dashboard wiring |
| `data_layer.py` | Parse files, validate data, DuckDB tables, schema profiles |
| `llm.py` | Groq API, insight generation, chat Q&A, SQL validation and retry |
| `charts.py` | Plotly charts (bar, line, pie, KPI, table, etc.) |

### What it handles

| Area | Details |
|---|---|
| File types | CSV, Excel (`.xlsx`, `.xls`) |
| Multi-file | Multiple uploads; duplicate filenames are skipped |
| Multi-sheet Excel | Each sheet becomes a separate table |
| Chart types | `kpi`, `bar`, `line`, `pie`, `grouped_bar`, `scatter`, `table` |
| Chat topics | Data exploration, calculations, chart requests |
| Follow-ups | Corrections, refinements, clarifications (last 5 turns) |
| Dashboard | Add charts from chat; AI can remove obsolete ones |
| Model | Groq (`openai/gpt-oss-20b` default, with automatic fallbacks) |

### Checks and validations

**Data upload**

- Empty files or missing columns → warning
- Duplicate column names → warning
- Inconsistent types within a column → warning
- Unsupported file types → error
- Table names sanitized for safe SQL use

**SQL safety**

- **SELECT-only** queries allowed (`WITH ... SELECT` CTEs are OK)
- Blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, and similar
- Failed SQL → LLM retries with schema-aware fixes (up to 2 retries)

**Chat input**

- Max **500 characters** per question
- Prompt injection patterns blocked
- Off-topic questions rejected (general knowledge, coding, chit-chat)
- Relevance guard — questions must be about your data, calculations, or dashboards

**LLM output**

- JSON parsing with retry for malformed responses
- Insight SQL is validated by running it before showing on the dashboard
- Query results capped at **30 rows** for LLM context

### Session state

During a session, the app keeps:

| Key | Holds |
|---|---|
| `conn` | DuckDB connection with your data |
| `file_registry` | Uploaded file metadata and warnings |
| `schema_profiles` | Table and column profiles for the AI |
| `insights` | Dashboard charts (title, SQL, data, chart spec) |
| `messages` | Chat history |

Chat does **not** regenerate dashboard insights from scratch — it runs fresh SQL per question using the same session data.

### Example chat questions

- *"What was total revenue last quarter?"* (calculation)
- *"What columns are in the sales table?"* (data)
- *"Show sign-ups by region as a bar chart"* (dashboard)

### Tips

- You can upload multiple files at once. Excel files with multiple sheets are supported.
- If a file has issues (empty rows, duplicate columns), a warning badge appears next to the file name — the app will still try to use the data.
- Keep questions focused on your data, calculations, or dashboard charts.
