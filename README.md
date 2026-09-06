# Delhi House Price Prediction

## How to run it

### 1. Install dependencies
pip install -r requirements.txt

### 2. Generate the dataset
The CSV is already included, but you can regenerate it (with different
random noise) anytime:

python3 generate_data.py

### 3. Run the main project
python3 house_price_prediction.ipynb

## Understanding the results

- **MAE (Mean Absolute Error)**: On average, how far off (in ₹) the
  predictions are from the true price. Lower is better.
- **RMSE (Root Mean Squared Error)**: Similar to MAE, but penalizes large
  mistakes more heavily. Lower is better.
- **R² Score**: How much of the variation in price the model explains, from
  0 to 1. Closer to 1 is better.

Open `outputs/predicted_vs_actual.png` — the closer the dots hug the diagonal
dashed line, the better the model is predicting.

## Reusing the trained model later

import joblib
import pandas as pd

model = joblib.load("outputs/best_model.joblib")