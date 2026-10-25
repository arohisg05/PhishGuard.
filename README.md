# PhishGuard 🛡️

PhishGuard is a machine learning project that explores phishing website detection using URL patterns and domain reputation.

## What Is Phishing?

Phishing is an online fraud technique where attackers use fake websites or links to trick people into sharing sensitive information, such as passwords or banking details.

## Project Features

### 1. Check a URL
Uses the production Random Forest model to analyze URL features and predict whether a URL looks phishing or legitimate.

### 2. Experimental Hybrid Preview
A separate research prototype compares character-pattern analysis with a domain-aware hybrid approach that considers domain reputation. It is experimental and should not be treated as a guarantee of website safety.

## Dataset

The project began with the PhiUSIIL Phishing URL Dataset from the UCI Machine Learning Repository.

Project label mapping:
- 0 = Phishing
- 1 = Legitimate

Additional experimental evaluation uses domain-separated data. Results should only be compared when their evaluation datasets and methods are clearly stated.

## How It Works

1. Extract URL-related features or character patterns.
2. Apply a trained machine learning model.
3. Compare the model's score with its decision threshold.
4. Display a phishing or legitimate prediction.
5. Evaluate model behavior using precision, recall, F1-score, and false-positive and false-negative counts.

## Technology Stack

- Python
- Machine learning and feature engineering
- Pure-Python inference
- Streamlit
- JSON model artifacts
- Git and GitHub

## Run Locally

Create and activate a Python virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

Run the commands from the project root. Confirm that the required model files and local reputation data are present.

## Project Structure

- `data/` — Dataset documentation and supporting data
- `docs/` — Project progress and learning notes
- `notebooks/` — Data analysis notebooks
- `src/` — Feature extraction and inference code
- `models/` — Model artifacts and benchmark files
- `tests/` — Automated tests
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies

## Limitations

- HTTPS does not prove a website is legitimate; phishing websites can also use HTTPS.
- Domain reputation can become outdated as websites change.
- New domains may not appear in a static reputation list.
- Experimental results do not guarantee real-world detection performance.
- A legitimate website may be flagged incorrectly, and a phishing website may be missed.
- Model scores are estimates, not guarantees of safety.

## Project Status

PhishGuard includes a production URL-classification path and a separate experimental hybrid preview. Model evaluation and real-world testing are ongoing.

This project is for educational and research purposes. Always verify suspicious websites independently before entering sensitive information.
