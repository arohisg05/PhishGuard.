Day 5 — Feature Engineering

Project: PhishGuard — Phishing URL Detection

Objective: Create a new feature from an existing dataset column to help investigate URL patterns.

Work completed:

Studied URLLength, IsHTTPS, and NoOfSubDomain.

Compared their average values for phishing and legitimate URLs.

Investigated unusually long URLs.

Created a new feature called LongURL using the condition URLLength > 100.

Validated the new feature's data type, missing values, and unique values.

Feature definition:

LongURL = 1: URL length is greater than 100 characters.

LongURL = 0: URL length is 100 characters or less.

Results:

Total dataset records: 235,795

URLs of length 100 or less: 230,932

URLs longer than 100 characters: 4,863

Missing values in LongURL: 0

Unique values: 0 and 1

Observation: In this dataset, all 4,863 URLs longer than 100 characters have the phishing label. However, many phishing URLs are shorter, so this feature alone is insufficient for reliable classification.

Learning: Feature engineering means creating useful new input features from existing data. The LongURL feature is an initial experiment; its usefulness must be evaluated during model development.

Status: Day 5 feature engineering experiment completed. Model training has not started.