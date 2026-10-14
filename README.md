# PhishGuard 🛡️

PhishGuard is a machine learning project focused on identifying phishing websites using features from the PhiUSIIL Phishing URL Dataset.

## What Is Phishing?

Phishing is an online fraud technique where attackers use fake websites or links to trick people into sharing sensitive information, such as passwords or banking details.

## Project Goals

- Understand and analyze a phishing dataset.
- Clean data and study useful patterns.
- Create features for machine learning.
- Train and compare classification models.
- Build a simple demo using a pure-Python inference engine.
- Explore direct URL analysis as a future improvement.

## Dataset

This project uses the PhiUSIIL Phishing URL Dataset from the UCI Machine Learning Repository.

The target labels used in this project are:

- 0 = Phishing
- 1 = Legitimate

## Machine Learning Workflow

1. Data understanding: inspect dataset shape, columns, data types, missing values, duplicates, and label distribution.
2. Exploratory data analysis: examine URL length, HTTPS usage, and other feature patterns.
3. Feature engineering: create additional features, including LongURL.
4. Model training and comparison: evaluate Logistic Regression, Decision Tree, and Random Forest.
5. Model export: export Random Forest decision trees to JSON for pure-Python inference.
6. Demo development: create a Streamlit interface connected to the inference engine.

## Current Demo

The current demo accepts 49 manually entered numerical features and returns a prediction with probabilities.

**Important limitation:** The demo does not analyze raw URLs directly. The 49 feature values must be provided separately. Some features describe webpage content and structure and cannot be obtained from a URL string alone.

## Technology Stack

- Python
- Pandas and scikit-learn for data analysis and model development
- Streamlit for the demo interface
- Pure Python for model inference
- Git and GitHub for version control

## Project Structure

- data/ — Dataset documentation
- docs/ — Daily progress and learning notes
- notebooks/ — Data analysis notebook
- src/ — Pure-Python inference engine
- models/ — Exported model trees and benchmark samples
- tests/ — Automated engine tests
- app.py — Streamlit demo interface
- requirements.txt — Python dependencies

## Running the Demo Locally

1. Set up a Python virtual environment.
2. Install dependencies from requirements.txt.
3. Run this command from the project root:

    python -m streamlit run app.py

The file models/phishguard_trees.json must be available in the expected location.

## Future Improvements

- Develop a reliable URL and webpage feature-extraction pipeline.
- Explore a separate model using only features available directly from a URL.
- Test with more representative samples.
- Add clearer explanations of predictions.

## Project Status

The repository includes data analysis, feature engineering, model inference code, benchmark samples, automated tests, and a basic demo interface. The demo still needs local testing and validation.

**Note:** Performance on a dataset does not guarantee the same performance on real-world websites.
