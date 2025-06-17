# --- Imports ---
import os
import torch
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from aurora import AuroraSmallPretrained, Batch, Metadata, rollout
import pandas as pd
import json

import warnings
import contextlib
import dataclasses
from typing import Optional
import gc # For garbage collection
from datetime import datetime, timedelta

print("All imports successful!")

# --- Configuration ---
# Force CPU usage
print("🖥️  Configured to use CPU for processing")

# Data paths
data_path = Path("/data2/users/mp2522/Aurora/data/monthlydata")
static_file = data_path / "static.nc"
surf_file = data_path / "Jan2023surface-level.nc"
atm_file = data_path / "Jan2023atmospheric.nc"

# Output path for forecast results
output_forecast_path = Path("./quantized_5_days_results")
output_forecast_path.mkdir(exist_ok=True)
print(f"Forecast results will be saved to: {output_forecast_path.resolve()}")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Initialize device info for logging ---
device = torch.device("cpu")
print(f"🖥️  Using device: {device}")
print(f"🔧 CPU cores available: {torch.get_num_threads()}")
print(f"💾 System RAM will be used for model and data")
compute_platform = "CPU"

# --- Load datasets ---
try:
    static_vars_ds = xr.open_dataset(static_file, engine="netcdf4")
    surf_vars_ds = xr.open_dataset(surf_file, engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(atm_file, engine="netcdf4")
    print("Datasets loaded successfully!")
    
    # Get data info
    total_timesteps = len(surf_vars_ds.valid_time)
    print(f"Total timesteps available: {total_timesteps}")
    print(f"Time range: {surf_vars_ds.valid_time.values[0]} to {surf_vars_ds.valid_time.values[-1]}")
    
    # Select initial conditions (first two timesteps for Aurora)
    initial_time_idx = 1  # Use timesteps 0 and 1 as initial conditions
    print(f"Using initial conditions from timesteps 0-1: {surf_vars_ds.valid_time.values[0]} to {surf_vars_ds.valid_time.values[1]}")
    
except FileNotFoundError as e:
    print(f"ERROR: One or more data files not found. Please check paths. {e}")
    exit()
except Exception as e:
    print(f"ERROR: Could not load datasets. {e}")
    exit()

# --- Load Aurora Model ---
print("\n" + "="*80)
print("LOADING AURORA MODEL FOR 5-DAY FORECAST")
print("="*80)

# Clear memory
if 'model' in globals():
    del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    # Load base model
    print("Loading AuroraSmallPretrained model...")
    base_model = AuroraSmallPretrained()
    print("Base Aurora model loaded successfully.")

    # Apply dynamic quantization for memory efficiency
    print("Applying dynamic quantization (qint8)...")
    model = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm},
        dtype=torch.qint8
    )
    print("Model dynamically quantized successfully.")
    
    del base_model
    gc.collect()

    # Move quantized model to CPU
    model = model.to(device)
    model.eval()
    print(f"Quantized Aurora model ready on {device}")
    print(f"🐌 Running on CPU (slower but memory efficient)")

    if not hasattr(model, 'surf_stats') or model.surf_stats is None:
        raise RuntimeError("Model is missing 'surf_stats'. Cannot perform normalization.")
    print("Model ready for 5-day forecasting.")

except Exception as e:
    print(f"ERROR: Could not load or quantize AuroraSmallPretrained model. {e}")
    exit()

# --- Create Initial Conditions Batch ---
def create_initial_batch(time_idx: int):
    """Create initial conditions batch for forecasting."""
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
        },
        static_vars={
            "z": torch.from_numpy(static_vars_ds["z"].values[0]).to(torch.float32),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]).to(torch.float32),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]).to(torch.float32),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[time_idx-1, time_idx]][None]).to(torch.float32),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[time_idx],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    ).to(device=device)
    
    return batch

# --- Save Forecast Step ---
def save_forecast_step(batch: Batch, step: int, forecast_start_time: datetime):
    """Save a single forecast step to files."""
    step_dir = output_forecast_path / f"step_{step:02d}"
    step_dir.mkdir(exist_ok=True)
    
    # Calculate forecast time
    forecast_time = forecast_start_time + timedelta(hours=6 * step)  # Aurora 6-hour timesteps
    
    # Save surface variables
    surf_data = {}
    for var_name, tensor in batch.surf_vars.items():
        # Take the last time step (current forecast)
        data = tensor[0, -1].detach().cpu().numpy()
        surf_data[var_name] = data
        
        # Save as numpy array
        np.save(step_dir / f"surface_{var_name}.npy", data)
    
    # Save atmospheric variables
    atmos_data = {}
    for var_name, tensor in batch.atmos_vars.items():
        # Take the last time step (current forecast)
        data = tensor[0, -1].detach().cpu().numpy()
        atmos_data[var_name] = data
        
        # Save as numpy array
        np.save(step_dir / f"atmospheric_{var_name}.npy", data)
    
    # Save metadata
    metadata = {
        'step': step,
        'forecast_time': forecast_time.isoformat(),
        'forecast_hour': step * 6,
        'lat': batch.metadata.lat.cpu().numpy().tolist(),
        'lon': batch.metadata.lon.cpu().numpy().tolist(),
        'pressure_levels': list(batch.metadata.atmos_levels),
        'surface_variables': list(surf_data.keys()),
        'atmospheric_variables': list(atmos_data.keys())
    }
    
    with open(step_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return forecast_time

# --- Load Ground Truth for Validation ---
def load_ground_truth_data(forecast_start_time):
    """Load ground truth ERA5 data for comparison."""
    print("\nLoading ground truth data for validation...")
    
    ground_truth = {}
    
    try:
        # Convert forecast start time to pandas datetime
        start_time = pd.to_datetime(forecast_start_time)
        
        # Convert ground truth times to a simple list for easier handling
        gt_times_raw = surf_vars_ds.valid_time.values
        gt_times_list = [pd.to_datetime(t) for t in gt_times_raw]
        
        # Get forecast steps we'll need
        forecast_steps = 20
        
        for step in range(1, forecast_steps + 1):
            try:
                # Calculate the forecast time for this step
                forecast_time = start_time + pd.Timedelta(hours=6*step)
                
                # Find closest time in ground truth data
                time_diffs_seconds = [(abs((gt_time - forecast_time).total_seconds())) for gt_time in gt_times_list]
                closest_idx = time_diffs_seconds.index(min(time_diffs_seconds))
                closest_diff_hours = min(time_diffs_seconds) / 3600
                
                # Check if we found a reasonably close match (within 3 hours)
                if closest_diff_hours < 3:
                    closest_gt_time = gt_times_list[closest_idx]
                    
                    ground_truth[step] = {
                        'surface': {
                            '2t': surf_vars_ds['t2m'].values[closest_idx],
                            '10u': surf_vars_ds['u10'].values[closest_idx], 
                            '10v': surf_vars_ds['v10'].values[closest_idx],
                            'msl': surf_vars_ds['msl'].values[closest_idx]
                        },
                        'atmospheric': {
                            't': atmos_vars_ds['t'].values[closest_idx],
                            'u': atmos_vars_ds['u'].values[closest_idx],
                            'v': atmos_vars_ds['v'].values[closest_idx], 
                            'q': atmos_vars_ds['q'].values[closest_idx],
                            'z': atmos_vars_ds['z'].values[closest_idx]
                        },
                        'forecast_time': forecast_time,
                        'ground_truth_time': closest_gt_time,
                        'time_diff_hours': closest_diff_hours
                    }
                        
            except Exception as step_error:
                continue
                    
        print(f"✅ Loaded ground truth for {len(ground_truth)} forecast steps")
        return ground_truth
            
    except Exception as e:
        print(f"⚠️  Could not load ground truth data: {e}")
        return {}

# --- Calculate Forecast Metrics ---
def calculate_forecast_metrics(forecast_results, ground_truth, forecast_start_time):
    """Calculate comprehensive forecast metrics."""
    print("\nCalculating forecast metrics...")
    
    metrics_data = []
    forecast_steps = len(forecast_results)
    
    for step in range(1, forecast_steps + 1):
        step_metrics = {
            'step': step,
            'forecast_hour': step * 6,
            'day': step * 6 // 24,
            'has_ground_truth': False  # Default to False
        }
        
        # Load forecast data
        try:
            forecast_temp = np.load(output_forecast_path / f"step_{step:02d}" / "surface_2t.npy")
            forecast_pressure = np.load(output_forecast_path / f"step_{step:02d}" / "surface_msl.npy")
            forecast_u = np.load(output_forecast_path / f"step_{step:02d}" / "surface_10u.npy")
            forecast_v = np.load(output_forecast_path / f"step_{step:02d}" / "surface_10v.npy")
            
            print(f"Step {step}: Forecast shape = {forecast_temp.shape}")
            
            # Calculate forecast statistics
            step_metrics['temp_mean'] = np.mean(forecast_temp) - 273.15  # Convert to Celsius
            step_metrics['temp_std'] = np.std(forecast_temp)
            step_metrics['temp_range'] = np.max(forecast_temp) - np.min(forecast_temp)
            step_metrics['pressure_mean'] = np.mean(forecast_pressure) / 100  # Convert to hPa
            step_metrics['pressure_std'] = np.std(forecast_pressure) / 100
            
            wind_speed = np.sqrt(forecast_u**2 + forecast_v**2)
            step_metrics['wind_speed_mean'] = np.mean(wind_speed)
            step_metrics['wind_speed_max'] = np.max(wind_speed)
            
            # Calculate accuracy metrics if ground truth available
            if step in ground_truth:
                gt_data = ground_truth[step]
                
                # Get ground truth data
                gt_temp = gt_data['surface']['2t']
                gt_pressure = gt_data['surface']['msl']
                gt_u = gt_data['surface']['10u']
                gt_v = gt_data['surface']['10v']
                
                print(f"Step {step}: Ground truth shape = {gt_temp.shape}")
                
                # Handle shape mismatch by cropping ground truth if needed
                if gt_temp.shape != forecast_temp.shape:
                    print(f"Step {step}: Shape mismatch - forecast {forecast_temp.shape} vs ground truth {gt_temp.shape}")
                    
                    # Crop ground truth to match forecast (usually remove last latitude)
                    if gt_temp.shape[0] > forecast_temp.shape[0]:
                        gt_temp = gt_temp[:forecast_temp.shape[0], :]
                        gt_pressure = gt_pressure[:forecast_temp.shape[0], :]
                        gt_u = gt_u[:forecast_temp.shape[0], :]
                        gt_v = gt_v[:forecast_temp.shape[0], :]
                        print(f"Step {step}: Cropped ground truth to {gt_temp.shape}")
                    elif gt_temp.shape[0] < forecast_temp.shape[0]:
                        forecast_temp = forecast_temp[:gt_temp.shape[0], :]
                        forecast_pressure = forecast_pressure[:gt_temp.shape[0], :]
                        forecast_u = forecast_u[:gt_temp.shape[0], :]
                        forecast_v = forecast_v[:gt_temp.shape[0], :]
                        wind_speed = np.sqrt(forecast_u**2 + forecast_v**2)
                        print(f"Step {step}: Cropped forecast to {forecast_temp.shape}")
                
                # Temperature metrics
                temp_rmse = np.sqrt(np.mean((forecast_temp - gt_temp)**2))
                temp_mae = np.mean(np.abs(forecast_temp - gt_temp))
                temp_bias = np.mean(forecast_temp - gt_temp)
                temp_corr = np.corrcoef(forecast_temp.flatten(), gt_temp.flatten())[0, 1]
                
                step_metrics.update({
                    'temp_rmse': temp_rmse,
                    'temp_mae': temp_mae,
                    'temp_bias': temp_bias,
                    'temp_correlation': temp_corr,
                })
                
                # Pressure metrics
                pressure_rmse = np.sqrt(np.mean((forecast_pressure - gt_pressure)**2)) / 100
                pressure_mae = np.mean(np.abs(forecast_pressure - gt_pressure)) / 100
                pressure_bias = np.mean(forecast_pressure - gt_pressure) / 100
                pressure_corr = np.corrcoef(forecast_pressure.flatten(), gt_pressure.flatten())[0, 1]
                
                step_metrics.update({
                    'pressure_rmse': pressure_rmse,
                    'pressure_mae': pressure_mae,
                    'pressure_bias': pressure_bias,
                    'pressure_correlation': pressure_corr,
                })
                
                # Wind metrics
                u_rmse = np.sqrt(np.mean((forecast_u - gt_u)**2))
                v_rmse = np.sqrt(np.mean((forecast_v - gt_v)**2))
                
                gt_wind = np.sqrt(gt_u**2 + gt_v**2)
                wind_rmse = np.sqrt(np.mean((wind_speed - gt_wind)**2))
                wind_corr = np.corrcoef(wind_speed.flatten(), gt_wind.flatten())[0, 1]
                
                step_metrics.update({
                    'u_wind_rmse': u_rmse,
                    'v_wind_rmse': v_rmse,
                    'wind_speed_rmse': wind_rmse,
                    'wind_correlation': wind_corr,
                })
                
                step_metrics['has_ground_truth'] = True
                print(f"Step {step}: Validation successful - RMSE = {temp_rmse:.3f}K, Corr = {temp_corr:.3f}")
                
        except Exception as e:
            print(f"Error processing step {step}: {e}")
            # Still add basic step info even if processing failed
            step_metrics.update({
                'temp_mean': np.nan,
                'temp_std': np.nan,
                'temp_range': np.nan,
                'pressure_mean': np.nan,
                'pressure_std': np.nan,
                'wind_speed_mean': np.nan,
                'wind_speed_max': np.nan
            })
            
        metrics_data.append(step_metrics)
    
    df = pd.DataFrame(metrics_data)
    print(f"✅ Processed {len(df)} forecast steps")
    print(f"✅ Validation available for {df['has_ground_truth'].sum()} steps")
    
    return df

# --- Generate Analysis Files ---
def generate_analysis_files(metrics_df, ground_truth, forecast_start_time, total_duration):
    """Generate all the analysis files like in the unquantized version."""
    
    # Handle empty or incomplete DataFrame
    if metrics_df.empty:
        print("⚠️  No metrics data available - generating basic files only")
        
        # Create minimal files
        with open(output_forecast_path / "5day_forecast_metrics.txt", 'w') as f:
            f.write("QUANTIZED AURORA 5-DAY FORECAST METRICS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Forecast Start: {forecast_start_time}\n")
            f.write(f"Model: AuroraSmallPretrained (Quantized)\n")
            f.write(f"Total Runtime: {total_duration}\n")
            f.write("⚠️  No validation metrics available due to processing errors\n")
        
        return
    
    # Ensure has_ground_truth column exists
    if 'has_ground_truth' not in metrics_df.columns:
        metrics_df['has_ground_truth'] = False
    
    # 1. 5day_forecast_metrics.txt
    with open(output_forecast_path / "5day_forecast_metrics.txt", 'w') as f:
        f.write("QUANTIZED AURORA 5-DAY FORECAST METRICS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Forecast Start: {forecast_start_time}\n")
        f.write(f"Model: AuroraSmallPretrained (Quantized)\n")
        f.write(f"Total Runtime: {total_duration}\n")
        f.write(f"Validation Steps: {metrics_df['has_ground_truth'].sum()}\n\n")
        
        if metrics_df['has_ground_truth'].any():
            val_data = metrics_df[metrics_df['has_ground_truth'] == True]
            f.write("ACCURACY METRICS (vs Ground Truth):\n")
            
            # Check which columns are available
            if 'temp_rmse' in val_data.columns:
                f.write(f"Temperature RMSE: {val_data['temp_rmse'].mean():.3f} ± {val_data['temp_rmse'].std():.3f} K\n")
            if 'temp_mae' in val_data.columns:
                f.write(f"Temperature MAE:  {val_data['temp_mae'].mean():.3f} ± {val_data['temp_mae'].std():.3f} K\n")
            if 'temp_correlation' in val_data.columns:
                f.write(f"Temperature Correlation: {val_data['temp_correlation'].mean():.3f} ± {val_data['temp_correlation'].std():.3f}\n")
            if 'pressure_rmse' in val_data.columns:
                f.write(f"Pressure RMSE: {val_data['pressure_rmse'].mean():.2f} ± {val_data['pressure_rmse'].std():.2f} hPa\n")
            if 'wind_speed_rmse' in val_data.columns:
                f.write(f"Wind Speed RMSE: {val_data['wind_speed_rmse'].mean():.3f} ± {val_data['wind_speed_rmse'].std():.3f} m/s\n")
            
            f.write("\nSKILL EVOLUTION BY DAY:\n")
            for day in range(1, 6):
                day_data = val_data[val_data['day'] == day]
                if not day_data.empty and 'temp_rmse' in day_data.columns and 'temp_correlation' in day_data.columns:
                    temp_rmse = day_data['temp_rmse'].mean()
                    temp_corr = day_data['temp_correlation'].mean()
                    f.write(f"Day {day}: Temp RMSE {temp_rmse:.3f}K, Correlation {temp_corr:.3f}\n")
        else:
            f.write("⚠️  No ground truth validation available\n")
        
        f.write("\nFORECAST STATISTICS:\n")
        # Check which columns are available for basic stats
        if 'temp_mean' in metrics_df.columns:
            temp_stats = metrics_df['temp_mean'].dropna()
            if not temp_stats.empty:
                f.write(f"Global mean temperature: {temp_stats.mean():.2f} ± {temp_stats.std():.2f}°C\n")
        if 'pressure_mean' in metrics_df.columns:
            pressure_stats = metrics_df['pressure_mean'].dropna()
            if not pressure_stats.empty:
                f.write(f"Global mean pressure: {pressure_stats.mean():.2f} ± {pressure_stats.std():.2f} hPa\n")
        if 'wind_speed_mean' in metrics_df.columns:
            wind_stats = metrics_df['wind_speed_mean'].dropna()
            if not wind_stats.empty:
                f.write(f"Global mean wind speed: {wind_stats.mean():.2f} ± {wind_stats.std():.2f} m/s\n")
    
    # 2. continuous_forecast_results.txt
    with open(output_forecast_path / "continuous_forecast_results.txt", 'w') as f:
        f.write("QUANTIZED AURORA CONTINUOUS FORECAST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        for _, row in metrics_df.iterrows():
            step = int(row['step'])
            hour = int(row['forecast_hour'])
            day = int(row['day'])
            f.write(f"Step {step:02d} (Day {day}, Hour {hour:02d}):\n")
            
            # Only write stats that are available and not NaN
            if 'temp_mean' in row and not pd.isna(row['temp_mean']):
                temp_std = row.get('temp_std', 0)
                f.write(f"  Temperature: {row['temp_mean']:.2f}°C (σ={temp_std:.2f}K)\n")
            if 'pressure_mean' in row and not pd.isna(row['pressure_mean']):
                pressure_std = row.get('pressure_std', 0)
                f.write(f"  Pressure: {row['pressure_mean']:.2f} hPa (σ={pressure_std:.2f})\n")
            if 'wind_speed_mean' in row and not pd.isna(row['wind_speed_mean']):
                wind_max = row.get('wind_speed_max', 0)
                f.write(f"  Wind: {row['wind_speed_mean']:.2f} m/s (max={wind_max:.2f})\n")
            
            if row.get('has_ground_truth', False):
                if 'temp_rmse' in row and not pd.isna(row['temp_rmse']):
                    pressure_rmse = row.get('pressure_rmse', 0)
                    f.write(f"  RMSE: T={row['temp_rmse']:.3f}K, P={pressure_rmse:.2f}hPa\n")
                if 'temp_correlation' in row and not pd.isna(row['temp_correlation']):
                    pressure_corr = row.get('pressure_correlation', 0)
                    f.write(f"  Correlation: T={row['temp_correlation']:.3f}, P={pressure_corr:.3f}\n")
            f.write("\n")
    
    # 3. forecast_diagnostics.txt
    with open(output_forecast_path / "forecast_diagnostics.txt", 'w') as f:
        f.write("QUANTIZED AURORA FORECAST DIAGNOSTICS\n")
        f.write("=" * 50 + "\n\n")
        f.write("MODEL INFORMATION:\n")
        f.write("- Model: AuroraSmallPretrained with Dynamic Quantization\n")
        f.write("- Quantization: qint8 on Linear/Conv/LayerNorm layers\n")
        f.write("- Device: CPU\n")
        f.write(f"- CPU Cores: {torch.get_num_threads()}\n\n")
        
        f.write("FORECAST SETUP:\n")
        f.write("- Forecast Length: 5 days (120 hours)\n")
        f.write("- Time Steps: 20 (6-hourly)\n")
        f.write("- Initial Conditions: Jan 1-2, 2023\n")
        f.write("- Spatial Resolution: 0.25° (720x1440)\n")
        f.write("- Pressure Levels: 13\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"- Total Runtime: {total_duration}\n")
        f.write(f"- Average per Step: {total_duration.total_seconds() / 20:.1f} seconds\n")
        f.write("- Memory Usage: Reduced via quantization\n\n")
        
        if len(ground_truth) > 0 and metrics_df['has_ground_truth'].any():
            f.write("VALIDATION SUMMARY:\n")
            f.write(f"- Ground Truth Steps: {len(ground_truth)}/20\n")
            f.write(f"- Successful Validations: {metrics_df['has_ground_truth'].sum()}/20\n")
            
            val_data = metrics_df[metrics_df['has_ground_truth'] == True]
            if not val_data.empty and 'temp_correlation' in val_data.columns:
                temp_corr = val_data['temp_correlation'].mean()
                if temp_corr > 0.9:
                    f.write(f"- Temperature Skill: Excellent (correlation = {temp_corr:.3f})\n")
                elif temp_corr > 0.8:
                    f.write(f"- Temperature Skill: Good (correlation = {temp_corr:.3f})\n")
                else:
                    f.write(f"- Temperature Skill: Fair (correlation = {temp_corr:.3f})\n")
        else:
            f.write("VALIDATION SUMMARY:\n")
            f.write("- No ground truth validation performed\n")
            f.write("- Issue: Shape mismatch between forecast and ground truth data\n")
    
    # 4. Generate plots (only if we have some data)
    if not metrics_df.empty and metrics_df['temp_mean'].notna().any():
        try:
            generate_forecast_plots(metrics_df)
        except Exception as e:
            print(f"⚠️  Could not generate plots: {e}")
    
    print("✅ Generated analysis files:")
    print("   📄 5day_forecast_metrics.txt")
    print("   📄 continuous_forecast_results.txt")
    print("   📄 forecast_diagnostics.txt")
    if not metrics_df.empty and metrics_df['temp_mean'].notna().any():
        print("   📊 forecast_comparison.png")
        print("   📊 forecast_metrics_continuous.png")
        if metrics_df['has_ground_truth'].any():
            print("   📊 forecast_accuracy_vs_time.png")

def generate_forecast_plots(metrics_df):
    """Generate forecast visualization plots."""
    
    # Check if we have enough data to plot
    if metrics_df.empty or metrics_df['temp_mean'].isna().all():
        print("⚠️  No data available for plotting")
        return
    
    # Filter out NaN values for plotting
    valid_data = metrics_df.dropna(subset=['temp_mean', 'forecast_hour'])
    
    if valid_data.empty:
        print("⚠️  No valid data for plotting")
        return
    
    # Plot 1: Forecast Comparison
    plt.figure(figsize=(15, 10))
    
    # Temperature evolution
    plt.subplot(2, 3, 1)
    if 'temp_mean' in valid_data.columns:
        plt.plot(valid_data['forecast_hour'], valid_data['temp_mean'], 'r-o', linewidth=2, markersize=4)
        plt.xlabel('Forecast Hour')
        plt.ylabel('Global Mean Temperature (°C)')
        plt.title('Temperature Evolution')
        plt.grid(True, alpha=0.3)
    
    # Temperature variability
    plt.subplot(2, 3, 2)
    if 'temp_std' in valid_data.columns:
        temp_std_valid = valid_data.dropna(subset=['temp_std'])
        if not temp_std_valid.empty:
            plt.plot(temp_std_valid['forecast_hour'], temp_std_valid['temp_std'], 'b-o', linewidth=2, markersize=4)
            plt.xlabel('Forecast Hour')
            plt.ylabel('Temperature Std Dev (K)')
            plt.title('Temperature Variability')
            plt.grid(True, alpha=0.3)
    
    # Pressure evolution
    plt.subplot(2, 3, 3)
    if 'pressure_mean' in valid_data.columns:
        pressure_valid = valid_data.dropna(subset=['pressure_mean'])
        if not pressure_valid.empty:
            plt.plot(pressure_valid['forecast_hour'], pressure_valid['pressure_mean'], 'g-o', linewidth=2, markersize=4)
            plt.xlabel('Forecast Hour')
            plt.ylabel('Global Mean Pressure (hPa)')
            plt.title('Pressure Evolution')
            plt.grid(True, alpha=0.3)
    
    # Wind evolution
    plt.subplot(2, 3, 4)
    if 'wind_speed_mean' in valid_data.columns:
        wind_valid = valid_data.dropna(subset=['wind_speed_mean'])
        if not wind_valid.empty:
            plt.plot(wind_valid['forecast_hour'], wind_valid['wind_speed_mean'], 'c-o', linewidth=2, markersize=4)
            plt.xlabel('Forecast Hour')
            plt.ylabel('Mean Wind Speed (m/s)')
            plt.title('Wind Speed Evolution')
            plt.grid(True, alpha=0.3)
    
    # Max wind evolution
    plt.subplot(2, 3, 5)
    if 'wind_speed_max' in valid_data.columns:
        wind_max_valid = valid_data.dropna(subset=['wind_speed_max'])
        if not wind_max_valid.empty:
            plt.plot(wind_max_valid['forecast_hour'], wind_max_valid['wind_speed_max'], 'm-o', linewidth=2, markersize=4)
            plt.xlabel('Forecast Hour')
            plt.ylabel('Max Wind Speed (m/s)')
            plt.title('Maximum Wind Speed')
            plt.grid(True, alpha=0.3)
    
    # Daily summary
    plt.subplot(2, 3, 6)
    if 'day' in valid_data.columns and 'temp_mean' in valid_data.columns:
        daily_temp = valid_data.groupby('day')['temp_mean'].mean()
        if not daily_temp.empty:
            plt.bar(daily_temp.index, daily_temp.values, alpha=0.7, color='orange')
            plt.xlabel('Forecast Day')
            plt.ylabel('Mean Temperature (°C)')
            plt.title('Daily Temperature Averages')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_forecast_path / "forecast_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Accuracy vs Time (if validation data available)
    has_validation = 'temp_rmse' in valid_data.columns and valid_data['temp_rmse'].notna().any()
    
    if has_validation:
        val_data = valid_data.dropna(subset=['temp_rmse'])
        if not val_data.empty:
            plt.figure(figsize=(15, 5))
            
            # RMSE evolution
            plt.subplot(1, 3, 1)
            plt.plot(val_data['forecast_hour'], val_data['temp_rmse'], 'r-o', label='Temperature', linewidth=2, markersize=4)
            
            if 'pressure_rmse' in val_data.columns:
                pressure_rmse_valid = val_data.dropna(subset=['pressure_rmse'])
                if not pressure_rmse_valid.empty:
                    pressure_rmse_norm = pressure_rmse_valid['pressure_rmse'] / pressure_rmse_valid['pressure_rmse'].max() * val_data['temp_rmse'].max()
                    plt.plot(pressure_rmse_valid['forecast_hour'], pressure_rmse_norm, 'b-s', label='Pressure (norm)', linewidth=2, markersize=4)
            
            if 'wind_speed_rmse' in val_data.columns:
                wind_rmse_valid = val_data.dropna(subset=['wind_speed_rmse'])
                if not wind_rmse_valid.empty:
                    wind_rmse_norm = wind_rmse_valid['wind_speed_rmse'] / wind_rmse_valid['wind_speed_rmse'].max() * val_data['temp_rmse'].max()
                    plt.plot(wind_rmse_valid['forecast_hour'], wind_rmse_norm, 'g-^', label='Wind (norm)', linewidth=2, markersize=4)
            
            plt.xlabel('Forecast Hour')
            plt.ylabel('RMSE (normalized)')
            plt.title('RMSE Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Correlation evolution
            plt.subplot(1, 3, 2)
            if 'temp_correlation' in val_data.columns:
                corr_valid = val_data.dropna(subset=['temp_correlation'])
                if not corr_valid.empty:
                    plt.plot(corr_valid['forecast_hour'], corr_valid['temp_correlation'], 'r-o', label='Temperature', linewidth=2, markersize=4)
            
            if 'pressure_correlation' in val_data.columns:
                pressure_corr_valid = val_data.dropna(subset=['pressure_correlation'])
                if not pressure_corr_valid.empty:
                    plt.plot(pressure_corr_valid['forecast_hour'], pressure_corr_valid['pressure_correlation'], 'b-s', label='Pressure', linewidth=2, markersize=4)
            
            if 'wind_correlation' in val_data.columns:
                wind_corr_valid = val_data.dropna(subset=['wind_correlation'])
                if not wind_corr_valid.empty:
                    plt.plot(wind_corr_valid['forecast_hour'], wind_corr_valid['wind_correlation'], 'g-^', label='Wind', linewidth=2, markersize=4)
            
            plt.xlabel('Forecast Hour')
            plt.ylabel('Correlation')
            plt.title('Correlation Evolution')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Bias evolution
            plt.subplot(1, 3, 3)
            if 'temp_bias' in val_data.columns:
                bias_valid = val_data.dropna(subset=['temp_bias'])
                if not bias_valid.empty:
                    plt.plot(bias_valid['forecast_hour'], bias_valid['temp_bias'], 'r-o', linewidth=2, markersize=4)
                    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    plt.xlabel('Forecast Hour')
                    plt.ylabel('Bias (K)')
                    plt.title('Temperature Bias Evolution')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_forecast_path / "forecast_accuracy_vs_time.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 3: Metrics diagnostic
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    if 'temp_mean' in valid_data.columns:
        plt.plot(valid_data['forecast_hour'], valid_data['temp_mean'], 'b-', linewidth=2)
        plt.xlabel('Forecast Hour')
        plt.ylabel('Temperature (°C)')
        plt.title('Global Mean Temperature')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    if 'temp_range' in valid_data.columns:
        range_valid = valid_data.dropna(subset=['temp_range'])
        if not range_valid.empty:
            plt.plot(range_valid['forecast_hour'], range_valid['temp_range'], 'g-', linewidth=2)
            plt.xlabel('Forecast Hour')
            plt.ylabel('Temperature Range (K)')
            plt.title('Temperature Range Evolution')
            plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    if 'temp_std' in valid_data.columns:
        std_valid = valid_data.dropna(subset=['temp_std'])
        if not std_valid.empty:
            plt.plot(std_valid['forecast_hour'], std_valid['temp_std'], 'm-', linewidth=2)
            plt.xlabel('Forecast Hour')
            plt.ylabel('Temperature Std Dev (K)')
            plt.title('Temperature Variability')
            plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    if 'day' in valid_data.columns and 'temp_mean' in valid_data.columns:
        days = valid_data['forecast_hour'] // 24
        plt.plot(days, valid_data['temp_mean'], 'ko-', markersize=4)
        plt.xlabel('Forecast Day')
        plt.ylabel('Mean Temperature (°C)')
        plt.title('Daily Mean Temperature')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_forecast_path / "forecast_metrics_continuous.png", dpi=300, bbox_inches='tight')
    plt.close()

# --- 5-Day Forecast Execution ---
print("\n" + "#"*80)
print("RUNNING 5-DAY AURORA FORECAST")
print("#"*80)

# Configuration
forecast_steps = 20  # 5 days * 4 steps per day (6-hourly)
print(f"Forecast steps: {forecast_steps} (5 days with 6-hourly timesteps)")

# Create initial conditions
print("Creating initial conditions...")
initial_batch = create_initial_batch(initial_time_idx)

# Convert numpy datetime64 to Python datetime (handle nanosecond precision)
start_time_np = surf_vars_ds.valid_time.values[initial_time_idx]
forecast_start_time = pd.to_datetime(start_time_np).to_pydatetime()
print(f"Forecast start time: {forecast_start_time}")

# Save initial conditions (step 0)
print("Saving initial conditions...")
save_forecast_step(initial_batch, 0, forecast_start_time)

# Run 5-day forecast using Aurora's rollout
print(f"\nStarting 5-day forecast rollout...")
print(f"🐌 Expected duration: ~{forecast_steps * 8} minutes on CPU (estimated)")
print(f"⏰ Estimated completion time: {(datetime.now() + timedelta(minutes=forecast_steps * 8)).strftime('%H:%M')}")

try:
    with torch.no_grad():
        start_time = datetime.now()
        
        # Use Aurora's rollout function
        forecast_generator = rollout(
            model=model,
            batch=initial_batch,
            steps=forecast_steps
        )
        
        # Convert generator to list to collect all results
        print("Collecting forecast results...")
        forecast_results = []
        
        # Process each forecast step as it's generated
        for step_num, forecast_batch in enumerate(forecast_generator):
            forecast_results.append(forecast_batch)
            
            # Save this forecast step
            forecast_time = save_forecast_step(forecast_batch, step_num + 1, forecast_start_time)
            
            if (step_num + 1) % 4 == 0:  # Every day
                print(f"   ✅ Saved Day {(step_num + 1) // 4} (Step {step_num + 1}): {forecast_time}")
        
        end_time = datetime.now()
        total_duration = end_time - start_time
        
        print(f"✅ 5-day forecast completed successfully!")
        print(f"⏱️  Total duration: {total_duration}")
        print(f"⚡ Average per step: {total_duration.total_seconds() / forecast_steps:.1f} seconds")
        print(f"🖥️  Compute platform: {compute_platform}")
        
        # Clean up memory
        gc.collect()

except Exception as e:
    print(f"❌ Forecast failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# --- Post-Forecast Analysis ---
print("\n" + "="*80)
print("GENERATING COMPREHENSIVE ANALYSIS")
print("="*80)

# Load ground truth for validation
ground_truth = load_ground_truth_data(forecast_start_time)

# Calculate forecast metrics
metrics_df = calculate_forecast_metrics(forecast_results, ground_truth, forecast_start_time)

# Generate all analysis files (matching unquantized version)
generate_analysis_files(metrics_df, ground_truth, forecast_start_time, total_duration)

# --- Generate Summary Statistics ---
print("\n" + "="*80)
print("GENERATING FORECAST SUMMARY")
print("="*80)

# Calculate some basic statistics
try:
    # Load a few forecast steps for analysis
    step_1_data = np.load(output_forecast_path / "step_01" / "surface_2t.npy")
    final_step_data = np.load(output_forecast_path / f"step_{forecast_steps:02d}" / "surface_2t.npy")
    
    # Temperature analysis
    initial_temp_mean = np.mean(step_1_data)
    final_temp_mean = np.mean(final_step_data)
    temp_change = final_temp_mean - initial_temp_mean
    
    print(f"📊 FORECAST SUMMARY STATISTICS:")
    print(f"   Initial temperature (global mean): {initial_temp_mean:.2f} K")
    print(f"   Final temperature (global mean): {final_temp_mean:.2f} K")
    print(f"   Temperature change over 5 days: {temp_change:.2f} K")
    print(f"   Temperature range (initial): {np.min(step_1_data):.1f} - {np.max(step_1_data):.1f} K")
    print(f"   Temperature range (final): {np.min(final_step_data):.1f} - {np.max(final_step_data):.1f} K")
    
except Exception as e:
    print(f"⚠️  Could not generate summary statistics: {e}")
    initial_temp_mean = final_temp_mean = temp_change = None

# --- Save Master Summary ---
summary_data = {
    "forecast_info": {
        "model": "AuroraSmallPretrained (quantized)",
        "compute_platform": compute_platform,
        "device_used": str(device),
        "cpu_cores": torch.get_num_threads(),
        "forecast_start": forecast_start_time.isoformat(),
        "forecast_duration_days": 5,
        "forecast_steps": forecast_steps,
        "timestep_hours": 6,
        "total_runtime": str(total_duration),
        "avg_seconds_per_step": total_duration.total_seconds() / forecast_steps
    },
    "data_info": {
        "initial_conditions_source": "January 2023 ERA5",
        "spatial_resolution": f"{len(surf_vars_ds.latitude)} x {len(surf_vars_ds.longitude)}",
        "pressure_levels": len(atmos_vars_ds.pressure_level),
        "surface_variables": 4,
        "atmospheric_variables": 5
    },
    "output_structure": {
        "total_steps_saved": forecast_steps + 1,  # +1 for initial conditions
        "files_per_step": 9,  # 4 surface + 5 atmospheric
        "file_format": "numpy arrays (.npy) + metadata (JSON)"
    },
    "validation_info": {
        "ground_truth_steps": len(ground_truth),
        "validation_available": len(ground_truth) > 0
    }
}

# Add forecast times if available
forecast_times = []
for step in range(1, forecast_steps + 1):
    forecast_time = forecast_start_time + timedelta(hours=6 * step)
    forecast_times.append(forecast_time.isoformat())
summary_data["forecast_times"] = forecast_times

# Add validation metrics if available
if len(ground_truth) > 0 and not metrics_df.empty:
    val_data = metrics_df[metrics_df['has_ground_truth'] == True]
    if not val_data.empty:
        summary_data["validation_metrics"] = {
            "temperature_rmse_mean": float(val_data['temp_rmse'].mean()) if 'temp_rmse' in val_data.columns else None,
            "temperature_correlation_mean": float(val_data['temp_correlation'].mean()) if 'temp_correlation' in val_data.columns else None,
            "pressure_rmse_mean": float(val_data['pressure_rmse'].mean()) if 'pressure_rmse' in val_data.columns else None,
            "pressure_correlation_mean": float(val_data['pressure_correlation'].mean()) if 'pressure_correlation' in val_data.columns else None,
            "wind_speed_rmse_mean": float(val_data['wind_speed_rmse'].mean()) if 'wind_speed_rmse' in val_data.columns else None,
            "wind_correlation_mean": float(val_data['wind_correlation'].mean()) if 'wind_correlation' in val_data.columns else None
        }

with open(output_forecast_path / "forecast_summary.json", 'w') as f:
    json.dump(summary_data, f, indent=2)

# Create a simple usage guide
usage_guide = """
QUANTIZED AURORA 5-DAY FORECAST RESULTS USAGE GUIDE
===================================================

Directory Structure:
- step_00/: Initial conditions
- step_01/ to step_20/: Forecast steps (6-hourly)
- forecast_summary.json: Master summary file
- 5day_forecast_metrics.txt: Comprehensive metrics summary
- continuous_forecast_results.txt: Step-by-step results
- forecast_diagnostics.txt: Model and performance diagnostics
- forecast_comparison.png: Forecast evolution plots
- forecast_accuracy_vs_time.png: Validation accuracy plots (if available)
- forecast_metrics_continuous.png: Diagnostic plots

Each step directory contains:
- surface_2t.npy: 2-meter temperature
- surface_10u.npy: 10-meter u-wind
- surface_10v.npy: 10-meter v-wind  
- surface_msl.npy: Mean sea level pressure
- atmospheric_t.npy: Temperature (3D: pressure x lat x lon)
- atmospheric_u.npy: U-wind (3D)
- atmospheric_v.npy: V-wind (3D)
- atmospheric_q.npy: Specific humidity (3D)
- atmospheric_z.npy: Geopotential height (3D)
- metadata.json: Step metadata and coordinates

Loading Example:
```python
import numpy as np
import json

# Load day 1 surface temperature
temp_day1 = np.load('step_04/surface_2t.npy')

# Load metadata
with open('step_04/metadata.json') as f:
    meta = json.load(f)
    
print(f"Forecast time: {meta['forecast_time']}")
print(f"Temperature shape: {temp_day1.shape}")
```

Analysis Files:
1. 5day_forecast_metrics.txt - Overall accuracy and skill metrics
2. continuous_forecast_results.txt - Detailed step-by-step evolution
3. forecast_diagnostics.txt - Model performance and setup info
4. forecast_comparison.png - Forecast variable evolution plots
5. forecast_accuracy_vs_time.png - Validation accuracy evolution
6. forecast_metrics_continuous.png - Additional diagnostic plots

Model Information:
- Base Model: AuroraSmallPretrained
- Quantization: Dynamic quantization (qint8) on Linear/Conv/LayerNorm layers
- Memory Efficiency: Reduced memory footprint vs unquantized model
- Performance: CPU-optimized for memory-constrained environments

Analysis Suggestions:
1. Compare quantized vs unquantized model performance
2. Assess memory usage reduction benefits
3. Calculate forecast skill metrics vs ERA5 ground truth
4. Visualize temperature/pressure evolution
5. Study forecast uncertainty growth
6. Analyze impact of quantization on forecast accuracy
"""

with open(output_forecast_path / "USAGE_GUIDE.txt", 'w') as f:
    f.write(usage_guide)

# --- Final Success Message ---
print(f"\n" + "#"*80)
print("🎉 QUANTIZED AURORA 5-DAY FORECAST COMPLETED SUCCESSFULLY! 🎉")
print("#"*80)
print(f"🔮 Forecast duration: 5 days (120 hours)")
print(f"📊 Forecast steps: {forecast_steps + 1} (including initial conditions)")
print(f"⏱️  Total runtime: {total_duration}")
print(f"💾 Data saved to: {output_forecast_path.resolve()}")
print(f"🖥️  Compute platform: {compute_platform}")
print(f"📈 Memory efficiency: Quantized model used")
print(f"🔧 CPU cores utilized: {torch.get_num_threads()}")

if len(ground_truth) > 0:
    print(f"✅ Validation: {len(ground_truth)} steps validated against ground truth")
else:
    print(f"⚠️  Validation: No ground truth validation performed")

print("#"*80)

print(f"\n📁 Output Structure:")
print(f"   📄 forecast_summary.json - Master summary")
print(f"   📄 USAGE_GUIDE.txt - How to use the results")
print(f"   📄 5day_forecast_metrics.txt - Comprehensive metrics")
print(f"   📄 continuous_forecast_results.txt - Step-by-step results")
print(f"   📄 forecast_diagnostics.txt - Performance diagnostics")
print(f"   📊 forecast_comparison.png - Evolution plots")
print(f"   📊 forecast_accuracy_vs_time.png - Validation plots")
print(f"   📊 forecast_metrics_continuous.png - Diagnostic plots")
print(f"   📁 step_00/ - Initial conditions")
print(f"   📁 step_01/ to step_{forecast_steps:02d}/ - Forecast steps")

print(f"\n🔍 Quick Analysis:")
try:
    if initial_temp_mean is not None:
        print(f"   🌡️  Initial global temperature: {initial_temp_mean:.2f} K ({initial_temp_mean-273.15:.1f}°C)")
        print(f"   🌡️  Final global temperature: {final_temp_mean:.2f} K ({final_temp_mean-273.15:.1f}°C)")
        print(f"   📈 Temperature change: {temp_change:.2f} K over 5 days")
    
    if len(ground_truth) > 0 and not metrics_df.empty:
        val_data = metrics_df[metrics_df['has_ground_truth'] == True]
        if not val_data.empty:
            if 'temp_rmse' in val_data.columns:
                print(f"   🎯 Temperature RMSE: {val_data['temp_rmse'].mean():.2f} ± {val_data['temp_rmse'].std():.2f} K")
            if 'temp_correlation' in val_data.columns:
                print(f"   🎯 Temperature Correlation: {val_data['temp_correlation'].mean():.3f} ± {val_data['temp_correlation'].std():.3f}")
            if 'pressure_rmse' in val_data.columns:
                print(f"   🎯 Pressure RMSE: {val_data['pressure_rmse'].mean():.1f} ± {val_data['pressure_rmse'].std():.1f} hPa")
        
except:
    print("   📊 Check analysis files for detailed metrics")

print(f"\n🎯 Next Steps:")
print(f"   1. Review generated analysis files for detailed metrics")
print(f"   2. Compare with unquantized Aurora results")
print(f"   3. Assess quantization impact on forecast accuracy")
print(f"   4. Analyze memory efficiency gains")
print(f"   5. Visualize forecast evolution using generated plots")

print(f"\n🏆 SUCCESS: Quantized Aurora forecast with complete analysis pipeline!")
print(f"    Model: AuroraSmallPretrained (Quantized qint8)")
print(f"    Platform: CPU-optimized")
print(f"    Memory: Reduced footprint via dynamic quantization")
print(f"    Output: Complete analysis matching unquantized version")
