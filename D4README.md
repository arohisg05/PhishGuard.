
# Day 4 — Exploratory Data Analysis (EDA)

## Project
PhishGuard — Phishing URL Detection

## Today's Goal
Explore the dataset using charts and summary statistics to identify patterns in phishing and legitimate URLs.

## Tools Used
- Python
- Google Colab
- Pandas
- Matplotlib

## Analysis 1: Label Distribution
- Phishing (label 0): 100,945 records
- Legitimate (label 1): 134,850 records

Observation:
The dataset contains more legitimate records than phishing records.

## Analysis 2: URL Length
- Average phishing URL length: approximately 45.72 characters
- Average legitimate URL length: approximately 26.23 characters

Observation:
Phishing URLs are longer on average in this dataset. However, URL length alone cannot reliably identify every phishing URL.

I plotted URL-length histograms and limited the displayed range to 0–200 characters to make the main distributions easier to inspect. This changed only the chart view, not the dataset.

## Analysis 3: HTTPS Usage
The average IsHTTPS values were:
- Phishing (label 0): 0.492238, approximately 49.22%
- Legitimate (label 1): 1.000000, or 100%

Observation:
HTTPS is present in about 49.22% of phishing records and all legitimate records in this dataset. HTTPS alone does not guarantee that a website is trustworthy.

The perfect value for the legitimate class should be investigated further for possible dataset bias or other patterns.

## What I Completed Today
- Created a bar chart of phishing versus legitimate records.
- Compared URL-length distributions.
- Compared average URL lengths between the two classes.
- Analyzed HTTPS usage by class.
- Practiced interpreting charts and summary statistics.

## Key Takeaway
EDA helps us discover patterns in a dataset before training a model. These patterns need careful interpretation and should not be treated as proof that a single feature can detect phishing.

## What Comes Next?
Day 5 — Feature Engineering: prepare appropriate features for machine learning.
