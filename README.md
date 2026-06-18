# Clickbait Detector Web App

## Overview

Web application for detecting Indonesian clickbait news headlines using a Stacking Ensemble model combining Naive Bayes, Support Vector Machine (SVM), and XGBoost with TF-IDF and linguistic features.

This application was developed as part of the Artificial Intelligence course project.

---

## Author

**Name:** Humaida Athifah
**NIM:** 241730040
**Program:** Informatika
**University:** UIN Sultan Maulana Hasanuddin Banten

---

## Model Information

### Best Model

* Stacking Ensemble
* Accuracy: 78.23%
* F1-Score: 71.91%
* ROC-AUC: 84.07%

### Features Used

* TF-IDF (10,000 features)
* Linguistic Features:

  * Word Count
  * Exclamation Count
  * Question Count
  * Ellipsis Count
  * Capital Letter Count
  * Number Count
  * Trigger Word Count
  * Punctuation Ratio

---

## Project Structure

```text
clickbait-detector/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
│
└── README.md
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/USERNAME/clickbait-detector.git
cd clickbait-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit application:

```bash
streamlit run app.py
```

---

## Usage

1. Open the application in a web browser.
2. Enter an Indonesian news headline.
3. Click the **Predict** button.
4. The system will classify the headline as:

   * Clickbait
   * Non-Clickbait

---

## Deployment

This application is deployed using Streamlit Cloud.

Public URL:

```text
https://clickbait-detector-byhumai.streamlit.app
```

---

## Example

### Input

```text
Terungkap! Alasan Sebenarnya Mengapa Harga Beras Naik Drastis
```

### Output

```text
Clickbait
```

### Input

```text
Pemerintah Menetapkan Harga BBM Baru Mulai 1 Juli 2026
```

### Output

```text
Non-Clickbait
```

---

## Requirements

Main libraries:

```text
streamlit
scikit-learn
xgboost
numpy
pandas
joblib
```

See `requirements.txt` for complete dependencies.

---

## License

This project was developed for educational and academic purposes.
