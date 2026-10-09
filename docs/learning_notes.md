# Learning Notes

## Day 1 — Understanding PhishGuard

### What is phishing?

Phishing is an online trick where someone creates a fake website
or link to fool people and steal their information.

### What is our project?

PhishGuard checks a URL and predicts whether it is:

- 0 → Genuine / Safe
- 1 → Fake / Phishing

### Why 0 and 1?

A machine learning model works with numbers.

So we represent:

0 = Genuine

1 = Phishing

The model then learns the patterns that are different between
genuine and phishing URLs.

### Why Logistic Regression?

Our problem has only two possible answers:

Yes → Phishing

No → Not Phishing

Therefore, this is a binary classification problem.

Logistic Regression is suitable for this type of problem.

### What I learned today

- What phishing means
- What a phishing URL is
- What our PhishGuard project will do
- What binary classification means
- Why we use 0 and 1
- Why Logistic Regression is suitable
