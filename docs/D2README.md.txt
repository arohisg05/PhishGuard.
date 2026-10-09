
# Day 2 — Dataset Collection and Understanding

## Project
PhishGuard — Phishing URL Detection

## Today's Goal
To collect a dataset and understand its rows, columns, and labels before building a machine learning model.

## Dataset Details
- Dataset name: PhiUSIIL Phishing URL Dataset
- Source: UCI Machine Learning Repository
- Total records: 235,795
- Total features/columns: 54
- File format: CSV

Dataset source: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

## What I Learned

### 1. What is a dataset?
A dataset is a collection of examples used to study data and train a machine learning model.

### 2. What is a row?
Each row represents one URL or website example.

### 3. What is a column?
Each column contains information about a URL or website. Examples include URL length and domain-related features.

### 4. What is a label?
The label is the answer the model learns to predict.

In this dataset:
- Label 1 = Legitimate URL
- Label 0 = Phishing URL

### 5. What is the target column?
The `label` column is the target because PhishGuard will learn to predict whether a URL is legitimate or phishing.

## Important Observation
The dataset contains examples with both label values, 0 and 1. I located the `label` column in Excel and inspected sample records.

## What I Completed Today
- Downloaded the dataset from the UCI Machine Learning Repository.
- Extracted the dataset and opened the CSV file in Excel.
- Identified the `label` column.
- Learned the meaning of the target labels.
- Understood the basic difference between rows, features, and labels.

## What Comes Next?
Day 3 — Inspect and prepare the data for machine learning.

## Key Takeaway
Before training a machine learning model, I must understand the dataset and identify which information the model will use and what it must predict.