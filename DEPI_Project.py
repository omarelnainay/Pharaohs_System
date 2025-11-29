import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from math import sqrt
import matplotlib.pyplot as plt


path = "C:/Users/Omare/Downloads/archive (16)/Retail-Supply-Chain-Sales-Dataset.xlsx"

df = pd.read_excel(path, sheet_name="Retails Order Full Dataset", parse_dates=["Order Date"])

df = df[['Order Date', 'Sales']].copy()
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df = df.dropna()

df['YearMonth'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
monthly = df.groupby('YearMonth')['Sales'].sum().reset_index()
monthly = monthly.rename(columns={'YearMonth': 'ds', 'Sales': 'y'})

monthly = monthly.set_index('ds').asfreq('MS').fillna(0).reset_index()


def create_features(df, lag=12):
    df = df.copy()
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df["rolling_3"] = df['y'].shift(1).rolling(3).mean()
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    return df

monthly_feat = create_features(monthly)
monthly_feat = monthly_feat.dropna().reset_index(drop=True)

test_periods = 12
train = monthly_feat[:-test_periods].copy()
test = monthly_feat[-test_periods:].copy()

feature_cols = [c for c in monthly_feat.columns if c not in ["ds", "y"]]

X_train, y_train = train[feature_cols], train["y"]
X_test, y_test = test[feature_cols], test["y"]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "sales_model.pkl")


pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
rmse = sqrt(mean_squared_error(y_test, pred_test))

print("MAE:", mae)
print("RMSE:", rmse)

future_periods = 12
last_data = monthly_feat.copy()

future_rows = []

for i in range(future_periods):
    next_month = last_data["ds"].iloc[-1] + pd.DateOffset(months=1)
    
    row = {"ds": next_month}
    for lag in range(1, 13):
        row[f"lag_{lag}"] = last_data["y"].iloc[-lag]
    row["rolling_3"] = last_data["y"].iloc[-3:].mean()
    row["month"] = next_month.month
    row["year"] = next_month.year

    Xf = pd.DataFrame([row])[feature_cols]
    pred = model.predict(Xf)[0]
    row["y"] = pred

    future_rows.append(row)
    last_data = pd.concat([last_data, pd.DataFrame([row])], ignore_index=True)

forecast_df = pd.DataFrame(future_rows)[["ds", "y"]]
forecast_df.to_csv("future_sales_forecast.csv", index=False)

print("Forecast saved to future_sales_forecast.csv")

plt.figure(figsize=(10,5))
plt.plot(monthly['ds'], monthly['y'], label="Actual")
plt.plot(forecast_df['ds'], forecast_df['y'], label="Forecast")
plt.title("Sales Forecast - Next 12 Months")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()
