import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("yield_df.csv")

df = df.dropna(subset=[
    "Area",
    "District",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield"
])

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["average_rain_fall_mm_per_year"] = pd.to_numeric(df["average_rain_fall_mm_per_year"], errors="coerce")
df["pesticides_tonnes"] = pd.to_numeric(df["pesticides_tonnes"], errors="coerce")
df["avg_temp"] = pd.to_numeric(df["avg_temp"], errors="coerce")
df["hg/ha_yield"] = pd.to_numeric(df["hg/ha_yield"], errors="coerce")

df = df.dropna()

X = df[[
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "Area",
    "District",
    "Item"
]]

y = df["hg/ha_yield"]

numeric_features = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp"
]

categorical_features = [
    "Area",
    "District",
    "Item"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_processed, y)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("dtr.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model retrained successfully with District column.")
print("Target summary:")
print(y.describe())