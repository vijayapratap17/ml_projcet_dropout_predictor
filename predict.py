from pathlib import Path
import argparse
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "dropout_model.joblib"

def main():
    parser = argparse.ArgumentParser(description="Estimate student dropout risk (educational demo).")
    parser.add_argument("--attendance", type=float, required=True, help="Attendance percentage, 0-100")
    parser.add_argument("--study-hours", type=float, required=True, help="Study hours per week")
    parser.add_argument("--previous-failures", type=int, required=True, help="Previous course failures")
    parser.add_argument("--grade", type=float, required=True, help="Average grade, 0-100")
    parser.add_argument("--missed-assignments", type=int, required=True, help="Missed assignments")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run `python src/train.py` first.")
    if not 0 <= args.attendance <= 100 or not 0 <= args.grade <= 100:
        raise SystemExit("Attendance and grade must be between 0 and 100.")
    if min(args.study_hours, args.previous_failures, args.missed_assignments) < 0:
        raise SystemExit("Study hours, previous failures, and missed assignments cannot be negative.")

    row = pd.DataFrame([{
        "attendance_pct": args.attendance,
        "study_hours_week": args.study_hours,
        "previous_failures": args.previous_failures,
        "average_grade": args.grade,
        "missed_assignments": args.missed_assignments
    }])
    model = joblib.load(MODEL_PATH)
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])
    label = "Higher estimated risk" if prediction == 1 else "Lower estimated risk"
    print(f"Prediction: {label}")
    print(f"Model risk score: {probability:.1%}")
    print("Reminder: synthetic-data demo only; not suitable for real student decisions.")

if __name__ == "__main__":
    main()
