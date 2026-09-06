# Delhi House Price Prediction (Beginner Project)

A complete, beginner-friendly machine learning project that predicts house
prices in Delhi based on features like area, locality, BHK, furnishing, and
distance from the nearest metro station.

This is a great first ML project because it covers the **entire real-world
workflow** — from raw data to a working prediction — using one of the most
intuitive problems there is: "how much should this house cost?"

## What's inside

```
delhi-house-price-prediction/
├── generate_data.py              # Creates the synthetic dataset
├── house_price_prediction.py     # Main project: EDA + model training + evaluation
├── requirements.txt              # Python packages needed
├── data/
│   └── delhi_house_data.csv      # 2,000 rows of realistic Delhi housing data
└── outputs/                      # Created when you run the script
    ├── price_distribution.png
    ├── price_vs_area.png
    ├── avg_price_by_locality.png
    ├── correlation_heatmap.png
    ├── predicted_vs_actual.png
    ├── feature_importance.png
    └── best_model.joblib         # The trained model, saved to disk
```

## About the dataset

Real scraped housing listings are hard to share freely, so this project uses
a **realistic synthetic dataset**: 2,000 fake-but-sensible house listings
generated with real Delhi localities (Vasant Vihar, Dwarka, Rohini, Karol
Bagh, etc.) and realistic pricing logic — bigger homes cost more, posh
localities cost more, older houses are worth less, and so on, with random
noise added so it behaves like real market data.

**Want to use real data instead?** Once you're comfortable with the code,
swap in a real dataset (e.g. from Kaggle or 99acres/MagicBricks exports) —
as long as it has similar columns, the script will work with minimal changes.

Columns in the dataset:

| Column | Meaning |
|---|---|
| Locality | Neighbourhood in Delhi |
| Property_Type | Apartment / Independent House / Builder Floor / Villa |
| BHK | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Balconies | Number of balconies |
| Area_SqFt | Built-up area in square feet |
| Age_Years | Age of the property |
| Floor_No | Which floor the unit is on |
| Total_Floors | Total floors in the building |
| Furnishing | Unfurnished / Semi-Furnished / Fully-Furnished |
| Parking_Spaces | Number of parking spots |
| Distance_From_Metro_KM | Distance to nearest metro station |
| Price_INR | Target variable — the price we want to predict |

## How to run it

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset
The CSV is already included, but you can regenerate it (with different
random noise) anytime:
```bash
python3 generate_data.py
```

### 3. Run the main project
```bash
python3 house_price_prediction.py
```

This single command will:
1. Load and inspect the data
2. Create exploratory charts (saved to `outputs/`)
3. Encode categorical columns (Locality, Property_Type, Furnishing)
4. Split data into training (80%) and testing (20%) sets
5. Train a **Linear Regression** model (simple baseline)
6. Train a **Random Forest** model (usually stronger)
7. Compare both models using MAE, RMSE, and R² score
8. Show which features matter most for price
9. Save the best model to `outputs/best_model.joblib`
10. Predict the price of one brand-new example house

## Understanding the results

- **MAE (Mean Absolute Error)**: On average, how far off (in ₹) the
  predictions are from the true price. Lower is better.
- **RMSE (Root Mean Squared Error)**: Similar to MAE, but penalizes large
  mistakes more heavily. Lower is better.
- **R² Score**: How much of the variation in price the model explains, from
  0 (no better than guessing the average) to 1 (perfect predictions). Closer
  to 1 is better. In this project both models typically score around **0.89–0.90**,
  which is quite strong for a first model.

Open `outputs/predicted_vs_actual.png` — the closer the dots hug the diagonal
dashed line, the better the model is predicting.

## Reusing the trained model later

```python
import joblib
import pandas as pd

model = joblib.load("outputs/best_model.joblib")

new_house = pd.DataFrame([{
    "Locality": "Rohini",
    "Property_Type": "Apartment",
    "BHK": 2,
    "Bathrooms": 2,
    "Balconies": 1,
    "Area_SqFt": 950,
    "Age_Years": 8,
    "Floor_No": 2,
    "Total_Floors": 5,
    "Furnishing": "Unfurnished",
    "Parking_Spaces": 1,
    "Distance_From_Metro_KM": 2.5,
}])

predicted_price = model.predict(new_house)[0]
print(f"Predicted price: Rs {predicted_price:,.0f}")
```

## Ideas to extend this project (great next steps)

1. **Try more models**: Gradient Boosting (`XGBoost`, `LightGBM`) often beats
   Random Forest on tabular data like this.
2. **Hyperparameter tuning**: Use `GridSearchCV` to find the best settings
   for Random Forest (e.g. `n_estimators`, `max_depth`).
3. **Feature engineering**: Add a "price per sq ft" column, or bucket
   localities into tiers (premium / mid / budget).
4. **Cross-validation**: Instead of one train/test split, use `KFold`
   cross-validation for a more reliable performance estimate.
5. **Build a simple web app**: Wrap the trained model in a small Streamlit or
   Flask app so you can enter house details in a form and get an instant
   price prediction.
6. **Use real data**: Replace the synthetic CSV with a real Delhi housing
   dataset for a genuine, deployable model.

## Key ML concepts you'll practice here

- Exploratory Data Analysis (EDA)
- Handling categorical data with One-Hot Encoding
- Train/test splitting to avoid overfitting
- Building `scikit-learn` Pipelines (preprocessing + model in one object)
- Comparing multiple models with proper metrics
- Interpreting feature importance
- Saving and reloading a trained model
