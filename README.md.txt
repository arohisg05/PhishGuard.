# PhishGuard 🛡️

PhishGuard is a project that checks whether a website URL is genuine
or fake.

The main purpose of this project is to detect phishing URLs.

## What is Phishing?

Phishing is a type of online fraud where a fake website or link is
used to trick people and steal their information.

For example, a fake login website may look similar to a real website
and try to collect a user's password.

## What does PhishGuard do?

PhishGuard takes a URL as input and checks whether it is:

- 0 → Genuine / Safe
- 1 → Fake / Phishing

So, the final answer is basically **Yes or No**:

- Is this URL phishing? → Yes
- Is this URL phishing? → No

## Machine Learning

For the Machine Learning part, I am using **Logistic Regression**.

I selected Logistic Regression because our problem has two possible
answers:

- 0 → Genuine
- 1 → Phishing

This makes it a **binary classification problem**.

## Project Goal

The goal is to build a system that can identify phishing URLs and
later explain why a URL is considered safe or suspicious.

## Current Progress

### Day 1

- Understood the basic idea of phishing
- Defined the problem as a binary classification problem
- Selected Logistic Regression for the Machine Learning part
- Created the GitHub repository