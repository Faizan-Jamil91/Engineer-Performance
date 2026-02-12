# Engineer Performance 360 Analytics

Professional Streamlit dashboard for field engineer performance analysis.

## Overview
- Visualize and analyze engineer tasks, time, accounts and performance.
- AI-powered insights (optional, requires Gemini API key).
- ML analytics: clustering, anomaly detection, simple prediction.

## Quick Start

1. Create and activate a Python virtual environment.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Add a `.env` file with your Gemini API key to enable AI insights:

```
GEMINI_API_KEY=your_key_here
```

4. Run the app with Streamlit:

```bash
streamlit run app.py
```

## Project Structure

- `app.py` — main Streamlit app
- `utils/` — data processing, analytics, ML, visualizations, AI helpers
- `config/` — configuration constants
- `requirements.txt` — pinned dependencies
- `Engineer_Performance.csv` — sample/default data file (not included in package)

## Tests

Run the basic smoke tests with `pytest`:

```bash
pytest -q
```

## Notes & Recommendations
- The AI insights require `google-generativeai` and a valid `GEMINI_API_KEY` in `.env`.
- Review `requirements.txt` for exact versions; consider upgrading dependencies periodically.
- For production deployments, containerize the app and secure secrets using environment variables or a secret manager.

---
Maintainer: FaizanJamil
