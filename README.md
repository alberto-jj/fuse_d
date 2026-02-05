# FUSE-D — Clinician-Friendly Feature Selection

FUSE-D is a Streamlit web app for clinician-friendly feature selection and model evaluation. It guides you through:
1. Uploading a wide-format dataset (one row per subject)
2. Running a baseline classifier
3. Estimating feature importance (PFI/CFI/LOCO via hidimstat)
4. Re-running the baseline model using selected features
5. Comparing models visually

This app is designed for clinicians and researchers who can install Anaconda and run a local Streamlit app.

## Quick Start

1. **Install Anaconda**
2. **Create an environment**
   ```bash
   conda create -n fuse_d python=3.10 -y
   conda activate fuse_d
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Dataset Format

Your CSV should be **wide-format**:
- One row per subject
- Columns = features + metadata
- You will specify:
  - Subject ID column
  - Target (class label) column
  - Columns to exclude from features

## Dependencies

Core:
- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib (for some charts)

Feature importance:
- hidimstat
- statsmodels

## Notes

- Feature importance methods are based on **hidimstat** (Mind-Inria).  
  Reference: `https://github.com/mind-inria/hidimstat`

## Troubleshooting

If you run into missing package errors:
```bash
pip install -r requirements.txt
```

If the app looks stale after changes:
```bash
streamlit run app.py --server.runOnSave true
```
