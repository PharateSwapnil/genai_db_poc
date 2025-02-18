import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go
from chronos import ChronosPipeline

# Initialize Chronos
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", torch_dtype=torch.bfloat16)


def generate_forecast(df, week_start, horizon):
    """
    Generates a forecast for a given validation week.
    """
    train_df = df[df["Timestamp"] < week_start]
    val_df = df[(df["Timestamp"] >= week_start) & (df["Timestamp"] < week_start + pd.Timedelta(hours=horizon))]
    print(week_start)
    print(horizon)
    context = torch.tensor(train_df["actual_kwh"].values, dtype=torch.float32)
    prediction_length = len(val_df)
    print(train_df)
    print(val_df)
    print(f"Prediction length: {prediction_length}")
    forecast = pipeline.predict(context, prediction_length)
    
    median_forecast = np.median(forecast[0].cpu().numpy(), axis=0)
    actuals = val_df["actual_kwh"].values
    
    mape = np.mean(np.abs((actuals - median_forecast) / actuals)) * 100

    return {
        "timestamps": val_df["Timestamp"].tolist(),
        "actuals": actuals.tolist(),
        "forecasts": median_forecast.tolist(),
        "mape": mape
    }

def plot_forecasts(all_results, start_date, end_date, mean_mape):
    """
    Plots actual vs. forecasted load.
    """
    fig = go.Figure()

    all_timestamps = []
    all_actuals = []
    all_forecasts = []

    for result in all_results:
        all_timestamps.extend(result["timestamps"])
        all_actuals.extend(result["actuals"])
        all_forecasts.extend(result["forecasts"])

    fig.add_trace(go.Scatter(x=all_timestamps, y=all_actuals, mode='lines', name='Actual Load', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=all_timestamps, y=all_forecasts, mode='lines', name='Forecast', line=dict(color='red')))

    fig.update_layout(
        title=f"Chronos Load Forecast from {start_date} to {end_date} Mean MAPE: {mean_mape:.2f}%",
        xaxis_title="Timestamp",
        yaxis_title="Load (kWh)",
        legend_title="Legend",
        template="plotly_dark"
    )

    fig.show()


import sqlite3

if __name__ == "__main__":
    # Load data
    csv_path = "/home/richhiey/Desktop/code/genai/data/time_series/load_temperature_data.csv"
    df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_localize(None)
    # Define validation weeks and forecast horizon
    val_weeks = pd.date_range("2013-11-01", periods=5, freq="1D")
    horizon = 1 * 24  # Forecasting 7 days (hourly data)

    all_results = []
    mape_scores = []  # Initialize before the loop

    for i, week_start in enumerate(val_weeks):
        result = generate_forecast(df, pipeline, week_start, horizon)
        all_results.append(result)
        mape_scores.append(result["mape"])
        print(f"Week {i+1} ({week_start.strftime('%Y-%m-%d')}): MAPE = {result['mape']:.2f}%")
    # Compute and print mean MAPE
    mean_mape = np.mean(mape_scores)
    print(f"\nMean MAPE: {mean_mape:.2f}%")

    # Plot results
    plot_forecasts(all_results, val_weeks[0].strftime("%Y-%m-%d"), (val_weeks[-1] + pd.Timedelta(days=6)).strftime("%Y-%m-%d"), mean_mape)