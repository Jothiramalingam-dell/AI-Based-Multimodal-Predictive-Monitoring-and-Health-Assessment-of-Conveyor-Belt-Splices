import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_health_index(
    camera_defect,
    vibration_rms,
    motor_current
):
    """
    Calculate Conveyor Belt Splice Health Index.

    Health Index:
        100 = Healthy
        0   = Critical degradation
    """

    # Normalize sensor values
    camera_risk = np.clip(camera_defect, 0, 1)

    vibration_risk = np.clip(
        vibration_rms / 1.2,
        0,
        1
    )

    current_risk = np.clip(
        motor_current / 8.0,
        0,
        1
    )

    # Multimodal weighted fusion
    degradation_score = (
        0.40 * camera_risk +
        0.35 * vibration_risk +
        0.25 * current_risk
    )

    # Convert degradation into health index
    health_index = 100 * (1 - degradation_score)

    return round(health_index, 2)


def classify_degradation(health_index):
    """
    Classify splice condition based on Health Index.
    """

    if health_index >= 80:
        return "Healthy"

    elif health_index >= 60:
        return "Mild Degradation"

    elif health_index >= 40:
        return "Moderate Degradation"

    elif health_index >= 20:
        return "Severe Degradation"

    else:
        return "Critical"


# --------------------------------------------------
# Example Health Index Calculation
# --------------------------------------------------

data = pd.read_csv(
    "data/sample_sensor_data.csv"
)

health_results = []

for _, row in data.iterrows():

    health_index = calculate_health_index(
        row["camera_defect_score"],
        row["vibration_rms"],
        row["motor_current"]
    )

    degradation = classify_degradation(
        health_index
    )

    health_results.append({
        "Sample ID": int(row["sample_id"]),
        "Health Index": health_index,
        "Degradation Level": degradation
    })


results = pd.DataFrame(health_results)

print("\nHealth Index & Degradation Classification")
print("------------------------------------------")

print(results.to_string(index=False))


# --------------------------------------------------
# Classification Metrics
# --------------------------------------------------

actual_labels = data["label"].values

predicted_labels = (
    data["camera_defect_score"] >= 0.30
).astype(int)


accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

recall = recall_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

f1 = f1_score(
    actual_labels,
    predicted_labels,
    zero_division=0
)

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)


print("\nClassification Metrics")
print("----------------------")

print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")

print("\nConfusion Matrix:")
print(cm)
