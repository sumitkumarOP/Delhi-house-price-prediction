"""
generate_data.py
-----------------
Creates a realistic SYNTHETIC dataset of Delhi house listings.

Why synthetic? Real scraped housing data usually can't be redistributed freely,
so this script builds a fake-but-realistic dataset using sensible pricing rules
(bigger area = more expensive, posh localities cost more, older houses are
cheaper, etc.) plus random noise - just like real-world data behaves.

Run this once to create data/delhi_house_data.csv
"""

import numpy as np
import pandas as pd

# Setting a "seed" makes the random numbers reproducible -
# you'll get the exact same dataset every time you run this script.
np.random.seed(42)

N_SAMPLES = 2000

# A mix of well-known Delhi localities across different price tiers.
# The number next to each locality is a rough "desirability multiplier"
# (higher = pricier area, purely for simulation purposes).
localities = {
    "Vasant Vihar": 2.6,
    "Golf Links": 2.8,
    "Greater Kailash": 2.2,
    "Hauz Khas": 2.0,
    "Defence Colony": 2.3,
    "Saket": 1.7,
    "Dwarka": 1.1,
    "Rohini": 0.9,
    "Pitampura": 1.0,
    "Karol Bagh": 1.3,
    "Lajpat Nagar": 1.4,
    "Mayur Vihar": 0.95,
    "Janakpuri": 1.15,
    "Vikaspuri": 1.05,
    "Uttam Nagar": 0.75,
    "Najafgarh": 0.6,
    "Shahdara": 0.8,
    "Paschim Vihar": 1.2,
    "Model Town": 1.5,
    "Chattarpur": 1.0,
}

furnishing_types = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
property_types = ["Apartment", "Independent House", "Builder Floor", "Villa"]

rows = []
locality_names = list(localities.keys())

for _ in range(N_SAMPLES):
    locality = np.random.choice(locality_names)
    locality_factor = localities[locality]
    property_type = np.random.choice(property_types, p=[0.55, 0.15, 0.25, 0.05])

    # Area in square feet - varies with property type
    if property_type == "Villa":
        area = np.random.normal(3200, 500)
    elif property_type == "Independent House":
        area = np.random.normal(2000, 400)
    elif property_type == "Builder Floor":
        area = np.random.normal(1400, 300)
    else:  # Apartment
        area = np.random.normal(1100, 300)
    area = max(300, area)  # no absurdly tiny homes

    bhk = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.35, 0.35, 0.15, 0.05])
    bathrooms = min(bhk, np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]) + 1)
    balconies = np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])

    age_years = np.random.randint(0, 40)  # age of the property
    floor_no = np.random.randint(0, 15)
    total_floors = floor_no + np.random.randint(1, 10)

    furnishing = np.random.choice(furnishing_types, p=[0.45, 0.35, 0.2])
    parking = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])

    distance_from_metro_km = round(np.random.exponential(2.0), 2)  # closer = more common
    distance_from_metro_km = min(distance_from_metro_km, 15)

    # ---- Price formula (this is the "ground truth" logic the model must learn) ----
    base_price_per_sqft = 6500  # base rate in INR
    price = area * base_price_per_sqft * locality_factor

    price *= (1 + 0.08 * (bhk - 2))                  # more BHK -> more value
    price *= (1 - 0.006 * age_years)                 # older -> cheaper
    price *= (1 - 0.015 * distance_from_metro_km)     # farther from metro -> cheaper
    price *= (1 + 0.03 * balconies)
    price *= (1 + 0.02 * parking)

    if furnishing == "Semi-Furnished":
        price *= 1.05
    elif furnishing == "Fully-Furnished":
        price *= 1.12

    if property_type == "Villa":
        price *= 1.25
    elif property_type == "Independent House":
        price *= 1.1
    elif property_type == "Builder Floor":
        price *= 1.02

    # Add random market noise (+/- ~10%) since real prices are never perfectly formulaic
    price *= np.random.normal(1.0, 0.10)
    price = max(price, 800000)  # floor price so nothing is unrealistically cheap

    rows.append({
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Bathrooms": int(bathrooms),
        "Balconies": int(balconies),
        "Area_SqFt": round(area, 1),
        "Age_Years": age_years,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Furnishing": furnishing,
        "Parking_Spaces": int(parking),
        "Distance_From_Metro_KM": distance_from_metro_km,
        "Price_INR": round(price, -3),  # round to nearest thousand
    })

df = pd.DataFrame(rows)
df.to_csv("data/delhi_house_data.csv", index=False)

print(f"Saved {len(df)} rows to data/delhi_house_data.csv")
print(df.head())
