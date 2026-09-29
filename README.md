# AI-Based Multimodal Conveyor Belt Splice Failure Prediction

## Project Overview

This project uses Artificial Intelligence and multimodal sensor fusion
to predict possible conveyor belt splice failure before major damage occurs.

## Input Modalities

### 1. Camera
The camera captures images of the conveyor belt splice.

Possible visual defects:
- Cracks
- Tears
- Surface damage
- Splice deformation
- Misalignment

### 2. Vibration Sensor

The vibration sensor monitors abnormal vibration patterns.

Example features:
- RMS vibration
- Peak vibration
- Frequency
- Variance
- Kurtosis
- Standard deviation

### 3. Current / Load Sensor

The system monitors conveyor motor operating conditions.

Example features:
- Motor current
- Conveyor load
- Power consumption

## Multimodal Fusion

The three data sources are processed separately and their extracted
features are combined using feature-level fusion.

Camera Features
        +
Vibration Features
        +
Current/Load Features
        ↓
Multimodal Feature Fusion
        ↓
AI Neural Network
        ↓
Failure Risk Prediction

## Output

The model produces a failure-risk value between 0 and 1.

Example:

0.10 - Low Risk
0.40 - Moderate Risk
0.75 - High Risk
0.90 - Critical Risk

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- OpenCV
- Scikit-learn

## Future Implementation

The trained model can be connected to an ESP32/edge controller and
a monitoring dashboard for real-time conveyor belt condition monitoring.
