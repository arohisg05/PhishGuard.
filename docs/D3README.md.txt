
# Day 3 — Data Inspection and Cleaning

## Project
PhishGuard — Phishing URL Detection

## Today's Goal
Load the dataset using Python and check its quality before using it for machine learning.

## Tools Used
- Python
- Google Colab
- Pandas

## Dataset Summary
- Dataset: PhiUSIIL Phishing URL Dataset
- Total records: 235,795
- Total columns: 56
- Numeric columns: 51 (including the label)
- Text columns: 5

## Data Quality Checks

### 1. Missing Values
Result: 0 missing values.

I used `df.isnull().sum()` to check for missing values in each column.

### 2. Duplicate Rows
Result: 0 duplicate rows.

I used `df.duplicated().sum()` to count rows that exactly repeat earlier rows across all columns.

### 3. Data Types
I inspected the data types of the columns.

Examples:
- `URL` — text
- `Domain` — text
- `URLLength` — integer
- `URLSimilarityIndex` — decimal number
- `label` — integer

### 4. Target Label Distribution
- Label 0 — Phishing: 100,945 records
- Label 1 — Legitimate: 134,850 records

The dataset contains both phishing and legitimate examples, with more legitimate records.

## Important Observations
- The CSV loaded successfully in Google Colab.
- The dataset contains 235,795 rows and 56 columns.
- No missing values or fully duplicated rows were detected.
- The `label` column is the target we want the model to predict.
- Numeric columns still need to be checked for suitability before model training.

## What I Completed Today
- Loaded the CSV file using Pandas.
- Checked the dataset dimensions.
- Checked missing values and duplicate rows.
- Inspected column data types.
- Identified numeric and text columns.
- Counted the examples for each target label.

## What Comes Next?
Day 4 — Exploratory Data Analysis (EDA), using charts to understand patterns in the dataset.

## Key Takeaway
Data inspection helps us understand the dataset and identify possible quality issues before training a machine learning model.