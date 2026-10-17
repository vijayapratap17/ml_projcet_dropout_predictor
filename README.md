# Student Dropout Risk Predictor (Machine Learning)

A beginner-friendly ML project that estimates whether a student may be at risk of dropping out, using academic and engagement indicators. It uses a scikit-learn classification pipeline and a small **synthetic demonstration dataset** included in `data/student_data.csv`.

> **Important:** This is an educational demo, not a validated tool for making decisions about real students. The included data is synthetic. Predictions can be biased or inaccurate and must never be used to penalize students. Real-world use requires consent, privacy safeguards, representative data, validation, and human review.

## Features
- Data loading and basic validation
- Train/test split and evaluation metrics
- Random Forest classifier with preprocessing
- Saved model pipeline (`models/dropout_model.joblib`)
- CLI prediction for one student's indicators

## Project structure
```text
dropout-predictor-ml/
├── data/student_data.csv
├── models/                 # model saved here after training
├── src/train.py
├── src/predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model
Run from this project folder:

```bash
python src/train.py
```

The script prints accuracy, a classification report, and a confusion matrix, then saves the trained pipeline to `models/dropout_model.joblib`.

## Predict for a student
```bash
python src/predict.py --attendance 65 --study-hours 4 --previous-failures 2 --grade 48 --missed-assignments 5
```

The output is a predicted class and a probability-based risk estimate. Treat it only as a learning demonstration.

## Dataset columns
- `attendance_pct`: attendance percentage (0–100)
- `study_hours_week`: estimated weekly study hours
- `previous_failures`: number of previous course failures
- `average_grade`: average grade (0–100)
- `missed_assignments`: number of missed assignments
- `dropout`: target label (1 = synthetic dropout-risk outcome, 0 = synthetic non-dropout outcome)

The target was generated using a simple artificial rule with randomness so you can practice the ML workflow. It does not represent real-world dropout rates or causal relationships.
