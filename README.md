# Sanger AB1 Analysis

A single-page Streamlit interface. Scroll from OF/OR upload through read-quality plots, the full OF and OR_rc alignment, methods A–D, candidate ranking, and reference comparison. The consensus candidate includes a nucleotide-by-nucleotide audit showing both read calls, Q values, the selected base, and its combined Phred likelihood.

The AB1 Evidence Index is a transparent per-candidate ranking aid, not calibrated whole-sequence accuracy. The reference rank is calculated separately after candidates are generated.

## Run locally

```bash
cd sanger_streamlit
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app processes uploaded AB1 files in its session and does not write them to disk.
