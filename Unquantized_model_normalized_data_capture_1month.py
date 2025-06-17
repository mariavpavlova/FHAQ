# --- Imports ---
import os
import torch
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error
from aurora import AuroraSmallPretrained, Batch, Metadata, rollout
import pandas as pd
import json

import warnings
import contextlib
import dataclasses
from typing import Optional
import gc # For garbage collection
from datetime import datetime

print("All imports successful!")

# --- Configuration ---
# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:256'

# Data paths
data_path = Path("/data2/users/mp2522/Aurora/data/monthlydata")
static_file = data_path / "static.nc"
surf_file = data_path / "Jan2023surface-level.nc"
atm_file = data_path / "Jan2023atmospheric.nc"

# Output path for captured normalized data
output_capture_path = Path("./captured_normalized_aurora_full_month")
output_capture_path.mkdir(exist_ok=True)
print(f"Captured normalized data will be saved to: {output_capture_path.resolve()}")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Initialize device info for logging ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Will map to GPU 2 due to CUDA_VISIBLE_DEVICES=2
    print(f"Using device: {device} (Physical GPU 2)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Memory Available: ~49.1 GB (GPU 2 is completely free!)")
    # Clear any existing GPU memory
    torch.cuda.empty_cache()
else:
    print("❌ CUDA not available! This script requires GPU.")
    exit(1)

# --- Load datasets ---
try:
    static_vars_ds = xr.open_dataset(static_file, engine="netcdf4")
    surf_vars_ds = xr.open_dataset(surf_file, engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(atm_file, engine="netcdf4")
    print("Datasets loaded successfully!")
    
    # Get full month info
    total_timesteps = len(surf_vars_ds.valid_time)
    print(f"Total timesteps available: {total_timesteps}")
    print(f"Time range: {surf_vars_ds.valid_time.values[0]} to {surf_vars_ds.valid_time.values[-1]}")
    print(f"Will process full month: {total_timesteps - 1} forward passes")
    
except FileNotFoundError as e:
    print(f"ERROR: One or more data files not found. Please check paths. {e}")
    exit()
except Exception as e:
    print(f"ERROR: Could not load datasets. {e}")
    exit()

# --- Full Month Aurora Normalized Data Capture System ---

class FullMonthAuroraNormalizedDataCapture:
    """
    Captures Aurora's normalized data for the entire month of January 2023.
    Records all variables at each timestep for comprehensive analysis.
    """
    def __init__(self, save_tensors: bool = True, output_dir: Path = Path("./captured_normalized_aurora_full_month"), device: torch.device = torch.device("cuda:0")):
        # Store all normalized data for the full month
        self.monthly_normalized_inputs = []     # All normalized inputs by timestep
        self.monthly_encoder_outputs = []       # All 3D latent representations 
        self.monthly_decoder_inputs = []        # All decoder inputs
        self.monthly_pre_unnorm_outputs = []    # All pre-unnormalize predictions
        
        # Statistics tracking
        self.monthly_stats = {
            'surface_vars': {},     # Statistics by variable name
            'atmospheric_vars': {}, # Statistics by variable name and level
            'latent_representations': [], # 3D latent stats
            'prediction_quality': []      # Pre-unnorm prediction stats
        }
        
        self.hooks = []                    # PyTorch hook handles
        self.current_timestep = 0          # Current processing timestep
        self.total_timesteps = 0           # Total timesteps to process
        self.original_forward = None       # Original forward method
        self.save_tensors = save_tensors   # Whether to save tensor data
        self.output_dir = output_dir       # Output directory
        self.device = device               # GPU device
        
        # Create subdirectories for organized storage
        (self.output_dir / "surface_normalized").mkdir(exist_ok=True)
        (self.output_dir / "atmospheric_normalized").mkdir(exist_ok=True)
        (self.output_dir / "latent_representations").mkdir(exist_ok=True)
        (self.output_dir / "pre_unnormalized_predictions").mkdir(exist_ok=True)
        (self.output_dir / "monthly_statistics").mkdir(exist_ok=True)

    def setup_aurora_hooks(self, model: torch.nn.Module) -> bool:
        """Sets up hooks for full month data capture."""
        print("\n" + "="*70)
        print("SETTING UP AURORA HOOKS FOR FULL MONTH NORMALIZED DATA CAPTURE")
        print("="*70)

        hooks_registered = 0

        # Monkey patch the model's forward method
        if hasattr(model, 'forward'):
            self.original_forward = model.forward
            model.forward = self._create_full_month_capturing_forward(model)
            hooks_registered += 1
            print(f"   Successfully patched model.forward for full month capture.")
        else:
            print("   ERROR: Model does not have a 'forward' method.")
            return False

        # Hook encoder output
        if hasattr(model, 'encoder') and isinstance(model.encoder, torch.nn.Module):
            hook = model.encoder.register_forward_hook(self._encoder_output_hook)
            self.hooks.append(hook)
            hooks_registered += 1
            print(f"   Hooked encoder output for 3D latent capture.")

        # Hook decoder input
        if hasattr(model, 'decoder') and isinstance(model.decoder, torch.nn.Module):
            hook = model.decoder.register_forward_hook(self._decoder_input_hook)
            self.hooks.append(hook)
            hooks_registered += 1
            print(f"   Hooked decoder input for 3D latent capture.")

        print(f"Total hooks registered: {hooks_registered}")
        return hooks_registered > 0

    def _create_full_month_capturing_forward(self, model: torch.nn.Module):
        """Creates a forward wrapper that captures data for every timestep."""
        def full_month_capturing_forward(batch: Batch) -> Batch:
            print(f"\n--- Processing timestep {self.current_timestep + 1}/{self.total_timesteps} ---")
            
            # Store current timestep info
            current_time = batch.metadata.time[0] if batch.metadata.time else f"timestep_{self.current_timestep}"
            print(f"   Current time: {current_time}")

            # --- Replicate Aurora's forward pass exactly ---
            batch = model.batch_transform_hook(batch)
            p = next(model.parameters())

            # CAPTURE POINT 1: After batch.normalise()
            print("   Normalizing batch...")
            normalized_batch = batch.normalise(surf_stats=model.surf_stats)
            self._capture_full_normalized_batch(normalized_batch, current_time)

            # Continue with Aurora's forward pass
            batch_processed = normalized_batch.crop(patch_size=model.patch_size)
            batch_processed = batch_processed.to(self.device)

            H, W = batch_processed.spatial_shape
            patch_res = (
                model.encoder.latent_levels,
                H // model.encoder.patch_size,
                W // model.encoder.patch_size,
            )

            B, T = next(iter(batch_processed.surf_vars.values())).shape[:2]
            batch_processed = dataclasses.replace(
                batch_processed,
                static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch_processed.static_vars.items()},
            )

            transformed_batch = batch_processed

            # Clamp positive variables
            if model.positive_surf_vars:
                transformed_batch = dataclasses.replace(
                    transformed_batch,
                    surf_vars={
                        k: v.clamp(min=0) if k in model.positive_surf_vars else v
                        for k, v in batch_processed.surf_vars.items()
                    },
                )
            if model.positive_atmos_vars:
                transformed_batch = dataclasses.replace(
                    transformed_batch,
                    atmos_vars={
                        k: v.clamp(min=0) if k in model.positive_atmos_vars else v
                        for k, v in batch_processed.atmos_vars.items()
                    },
                )

            transformed_batch = model._pre_encoder_hook(transformed_batch)

            # Encoder processing
            print(f"   Processing with encoder...")
            x = model.encoder(transformed_batch, lead_time=model.timestep)

            # Backbone processing
            print(f"   Processing with backbone...")
            device_type = self.device.type
            with torch.autocast(device_type=device_type) if model.autocast else contextlib.nullcontext():
                x = model.backbone(
                    x,
                    lead_time=model.timestep,
                    patch_res=patch_res,
                    rollout_step=batch_processed.metadata.rollout_step,
                )

            # Decoder processing
            print(f"   Processing with decoder...")
            pred = model.decoder(x, batch_processed, lead_time=model.timestep, patch_res=patch_res)

            pred = dataclasses.replace(
                pred,
                static_vars={k: v[0, 0] for k, v in batch_processed.static_vars.items()},
            )

            pred = dataclasses.replace(
                pred,
                surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
                atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
            )

            pred = model._post_decoder_hook(batch_processed, pred)

            # Clamp positive variables (only for rollout_step > 1)
            if model.positive_surf_vars and pred.metadata.rollout_step > 1:
                pred = dataclasses.replace(
                    pred,
                    surf_vars={
                        k: v.clamp(min=0) if k in model.positive_surf_vars else v
                        for k, v in pred.surf_vars.items()
                    },
                )
            if model.positive_atmos_vars and pred.metadata.rollout_step > 1:
                pred = dataclasses.replace(
                    pred,
                    atmos_vars={
                        k: v.clamp(min=0) if k in model.positive_atmos_vars else v
                        for k, v in pred.atmos_vars.items()
                    },
                )

            # CAPTURE POINT 2: Before unnormalise()
            self._capture_full_pre_unnormalize(pred, current_time)

            # Final unnormalization
            print(f"   Running unnormalise()...")
            pred = pred.unnormalise(surf_stats=model.surf_stats)

            print(f"   Timestep {self.current_timestep + 1} completed.")
            print(f"   GPU Memory Used: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            self.current_timestep += 1
            return pred

        return full_month_capturing_forward

    def _capture_full_normalized_batch(self, normalized_batch: Batch, current_time):
        """Captures all normalized variables for current timestep."""
        print(f"     Capturing ALL normalized variables for timestep {self.current_timestep + 1}")
        
        timestep_data = {
            'timestep': self.current_timestep,
            'time': current_time,
            'surface_vars': {},
            'atmospheric_vars': {},
            'static_vars': {}
        }

        # Capture ALL surface variables
        for var_name, tensor in normalized_batch.surf_vars.items():
            # Take first batch, first time step for analysis - move to CPU for analysis
            values = tensor[0, 0].detach().cpu().numpy()
            flat_values = values.flatten()
            
            var_stats = {
                'mean': float(np.mean(flat_values)),
                'std': float(np.std(flat_values)),
                'min': float(np.min(flat_values)),
                'max': float(np.max(flat_values)),
                'shape': values.shape,
                'var_type': 'surface'
            }
            
            timestep_data['surface_vars'][var_name] = var_stats
            
            # Update monthly statistics
            if var_name not in self.monthly_stats['surface_vars']:
                self.monthly_stats['surface_vars'][var_name] = {
                    'timestep_stats': [],
                    'overall_mean': [],
                    'overall_std': [],
                    'overall_range': []
                }
            
            self.monthly_stats['surface_vars'][var_name]['timestep_stats'].append(var_stats)
            self.monthly_stats['surface_vars'][var_name]['overall_mean'].append(var_stats['mean'])
            self.monthly_stats['surface_vars'][var_name]['overall_std'].append(var_stats['std'])
            self.monthly_stats['surface_vars'][var_name]['overall_range'].append([var_stats['min'], var_stats['max']])
            
            print(f"       Surface '{var_name}': mean={var_stats['mean']:.4f}, std={var_stats['std']:.4f}")
            
            # Save tensor data if requested
            if self.save_tensors:
                self._save_timestep_tensor(values, f'surface_normalized/{var_name}_timestep_{self.current_timestep:03d}')

        # Capture ALL atmospheric variables at ALL levels
        for var_name, tensor in normalized_batch.atmos_vars.items():
            timestep_data['atmospheric_vars'][var_name] = {}
            
            if var_name not in self.monthly_stats['atmospheric_vars']:
                self.monthly_stats['atmospheric_vars'][var_name] = {}
            
            # Process each pressure level
            for level_idx, pressure_level in enumerate(normalized_batch.metadata.atmos_levels):
                # Take first batch, first time step, specific level - move to CPU for analysis
                values = tensor[0, 0, level_idx].detach().cpu().numpy()
                flat_values = values.flatten()
                
                level_stats = {
                    'mean': float(np.mean(flat_values)),
                    'std': float(np.std(flat_values)),
                    'min': float(np.min(flat_values)),
                    'max': float(np.max(flat_values)),
                    'shape': values.shape,
                    'pressure_level': int(pressure_level),
                    'var_type': 'atmospheric'
                }
                
                timestep_data['atmospheric_vars'][var_name][f'level_{pressure_level}'] = level_stats
                
                # Update monthly statistics
                if pressure_level not in self.monthly_stats['atmospheric_vars'][var_name]:
                    self.monthly_stats['atmospheric_vars'][var_name][pressure_level] = {
                        'timestep_stats': [],
                        'overall_mean': [],
                        'overall_std': [],
                        'overall_range': []
                    }
                
                level_data = self.monthly_stats['atmospheric_vars'][var_name][pressure_level]
                level_data['timestep_stats'].append(level_stats)
                level_data['overall_mean'].append(level_stats['mean'])
                level_data['overall_std'].append(level_stats['std'])
                level_data['overall_range'].append([level_stats['min'], level_stats['max']])
                
                print(f"       Atmospheric '{var_name}' @ {pressure_level}hPa: mean={level_stats['mean']:.4f}, std={level_stats['std']:.4f}")
                
                # Save tensor data if requested
                if self.save_tensors:
                    self._save_timestep_tensor(values, f'atmospheric_normalized/{var_name}_level_{pressure_level}_timestep_{self.current_timestep:03d}')

        # Store timestep data
        self.monthly_normalized_inputs.append(timestep_data)

    def _encoder_output_hook(self, module: torch.nn.Module, input_tensor: tuple, output_tensor: torch.Tensor):
        """Captures 3D latent representations from encoder."""
        if isinstance(output_tensor, torch.Tensor):
            # Move to CPU for analysis
            values = output_tensor.detach().cpu().numpy()
            flat_values = values.flatten()
            
            latent_stats = {
                'timestep': self.current_timestep,
                'mean': float(np.mean(flat_values)),
                'std': float(np.std(flat_values)),
                'min': float(np.min(flat_values)),
                'max': float(np.max(flat_values)),
                'shape': list(values.shape),
                'source': 'encoder_output'
            }
            
            self.monthly_encoder_outputs.append(latent_stats)
            self.monthly_stats['latent_representations'].append(latent_stats)
            
            print(f"     3D Latent (Encoder): shape={values.shape}, mean={latent_stats['mean']:.4f}, std={latent_stats['std']:.4f}")
            
            if self.save_tensors:
                self._save_timestep_tensor(values, f'latent_representations/encoder_output_timestep_{self.current_timestep:03d}')

    def _decoder_input_hook(self, module: torch.nn.Module, input_tensor: tuple, output_tensor: torch.Tensor):
        """Captures 3D latent inputs to decoder."""
        if input_tensor and len(input_tensor) > 0 and isinstance(input_tensor[0], torch.Tensor):
            decoder_input_latent = input_tensor[0]
            # Move to CPU for analysis
            values = decoder_input_latent.detach().cpu().numpy()
            flat_values = values.flatten()
            
            decoder_stats = {
                'timestep': self.current_timestep,
                'mean': float(np.mean(flat_values)),
                'std': float(np.std(flat_values)),
                'min': float(np.min(flat_values)),
                'max': float(np.max(flat_values)),
                'shape': list(values.shape),
                'source': 'decoder_input'
            }
            
            self.monthly_decoder_inputs.append(decoder_stats)
            print(f"     3D Latent (Decoder Input): shape={values.shape}, mean={decoder_stats['mean']:.4f}, std={decoder_stats['std']:.4f}")
            
            if self.save_tensors:
                self._save_timestep_tensor(values, f'latent_representations/decoder_input_timestep_{self.current_timestep:03d}')

    def _capture_full_pre_unnormalize(self, pred_batch: Batch, current_time):
        """Captures all pre-unnormalized predictions."""
        print(f"     Capturing ALL pre-unnormalized predictions for timestep {self.current_timestep + 1}")
        
        prediction_data = {
            'timestep': self.current_timestep,
            'time': current_time,
            'surface_predictions': {},
            'atmospheric_predictions': {}
        }

        # Capture ALL surface predictions
        for var_name, tensor in pred_batch.surf_vars.items():
            # Move to CPU for analysis
            values = tensor[0, 0].detach().cpu().numpy()
            flat_values = values.flatten()
            
            pred_stats = {
                'mean': float(np.mean(flat_values)),
                'std': float(np.std(flat_values)),
                'min': float(np.min(flat_values)),
                'max': float(np.max(flat_values)),
                'shape': values.shape,
                'var_type': 'surface_prediction'
            }
            
            prediction_data['surface_predictions'][var_name] = pred_stats
            print(f"       Pre-unnorm Surface '{var_name}': mean={pred_stats['mean']:.4f}, std={pred_stats['std']:.4f}")
            
            if self.save_tensors:
                self._save_timestep_tensor(values, f'pre_unnormalized_predictions/surface_{var_name}_timestep_{self.current_timestep:03d}')

        # Capture ALL atmospheric predictions at ALL levels
        for var_name, tensor in pred_batch.atmos_vars.items():
            prediction_data['atmospheric_predictions'][var_name] = {}
            
            for level_idx, pressure_level in enumerate(pred_batch.metadata.atmos_levels):
                # Move to CPU for analysis
                values = tensor[0, 0, level_idx].detach().cpu().numpy()
                flat_values = values.flatten()
                
                pred_stats = {
                    'mean': float(np.mean(flat_values)),
                    'std': float(np.std(flat_values)),
                    'min': float(np.min(flat_values)),
                    'max': float(np.max(flat_values)),
                    'shape': values.shape,
                    'pressure_level': int(pressure_level),
                    'var_type': 'atmospheric_prediction'
                }
                
                prediction_data['atmospheric_predictions'][var_name][f'level_{pressure_level}'] = pred_stats
                print(f"       Pre-unnorm Atmospheric '{var_name}' @ {pressure_level}hPa: mean={pred_stats['mean']:.4f}, std={pred_stats['std']:.4f}")
                
                if self.save_tensors:
                    self._save_timestep_tensor(values, f'pre_unnormalized_predictions/atmospheric_{var_name}_level_{pressure_level}_timestep_{self.current_timestep:03d}')

        self.monthly_pre_unnorm_outputs.append(prediction_data)
        self.monthly_stats['prediction_quality'].append(prediction_data)

    def _save_timestep_tensor(self, tensor_data: np.ndarray, filename_suffix: str):
        """Saves tensor data to organized subdirectories."""
        try:
            save_path = self.output_dir / f"{filename_suffix}.npy"
            np.save(save_path, tensor_data)
        except Exception as e:
            print(f"         WARNING: Failed to save {filename_suffix}: {e}")

    def save_monthly_statistics(self):
        """Saves comprehensive monthly statistics to JSON and CSV files."""
        print("\n" + "="*70)
        print("SAVING COMPREHENSIVE MONTHLY STATISTICS")
        print("="*70)
        
        # Save detailed JSON statistics
        json_path = self.output_dir / "monthly_statistics" / "full_month_normalized_stats.json"
        with open(json_path, 'w') as f:
            json.dump(self.monthly_stats, f, indent=2, default=str)
        print(f"   Detailed statistics saved to: {json_path.name}")
        
        # Create CSV summaries for easy analysis
        self._create_csv_summaries()
        
        # Save timestep-by-timestep data
        timestep_data_path = self.output_dir / "monthly_statistics" / "timestep_normalized_data.json"
        with open(timestep_data_path, 'w') as f:
            json.dump(self.monthly_normalized_inputs, f, indent=2, default=str)
        print(f"   Timestep data saved to: {timestep_data_path.name}")
        
        # Save prediction data
        prediction_data_path = self.output_dir / "monthly_statistics" / "timestep_prediction_data.json"
        with open(prediction_data_path, 'w') as f:
            json.dump(self.monthly_pre_unnorm_outputs, f, indent=2, default=str)
        print(f"   Prediction data saved to: {prediction_data_path.name}")

    def _create_csv_summaries(self):
        """Creates CSV files for easy statistical analysis."""
        stats_dir = self.output_dir / "monthly_statistics"
        
        # Surface variables summary
        surface_data = []
        for var_name, var_data in self.monthly_stats['surface_vars'].items():
            for timestep, stats in enumerate(var_data['timestep_stats']):
                surface_data.append({
                    'timestep': timestep,
                    'variable': var_name,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'normalization_quality': 'excellent' if abs(stats['mean']) < 0.1 and 0.9 < stats['std'] < 1.1 else 'good' if abs(stats['mean']) < 0.3 and 0.7 < stats['std'] < 1.3 else 'poor'
                })
        
        surface_df = pd.DataFrame(surface_data)
        surface_csv = stats_dir / "surface_variables_monthly_stats.csv"
        surface_df.to_csv(surface_csv, index=False)
        print(f"   Surface variables CSV: {surface_csv.name}")
        
        # Atmospheric variables summary (sample key levels)
        key_levels = [1000, 850, 500, 250, 100]  # Sample key pressure levels
        atm_data = []
        for var_name, level_data in self.monthly_stats['atmospheric_vars'].items():
            for pressure_level, level_stats in level_data.items():
                if pressure_level in key_levels:
                    for timestep, stats in enumerate(level_stats['timestep_stats']):
                        atm_data.append({
                            'timestep': timestep,
                            'variable': var_name,
                            'pressure_level': pressure_level,
                            'mean': stats['mean'],
                            'std': stats['std'],
                            'min': stats['min'],
                            'max': stats['max'],
                            'normalization_quality': 'excellent' if abs(stats['mean']) < 0.1 and 0.9 < stats['std'] < 1.1 else 'good' if abs(stats['mean']) < 0.3 and 0.7 < stats['std'] < 1.3 else 'poor'
                        })
        
        atm_df = pd.DataFrame(atm_data)
        atm_csv = stats_dir / "atmospheric_variables_key_levels_monthly_stats.csv"
        atm_df.to_csv(atm_csv, index=False)
        print(f"   Atmospheric variables CSV: {atm_csv.name}")
        
        # Latent representations summary
        latent_df = pd.DataFrame(self.monthly_stats['latent_representations'])
        latent_csv = stats_dir / "latent_representations_monthly_stats.csv"
        latent_df.to_csv(latent_csv, index=False)
        print(f"   Latent representations CSV: {latent_csv.name}")

    def analyze_full_month_data(self):
        """Provides comprehensive analysis of the full month's normalized data."""
        print("\n" + "="*70)
        print("FULL MONTH NORMALIZED DATA ANALYSIS")
        print("="*70)
        
        print(f"\nDATASET OVERVIEW:")
        print(f"   Total timesteps processed: {len(self.monthly_normalized_inputs)}")
        print(f"   Surface variables captured: {len(self.monthly_stats['surface_vars'])}")
        print(f"   Atmospheric variables captured: {len(self.monthly_stats['atmospheric_vars'])}")
        print(f"   Total 3D latent representations: {len(self.monthly_stats['latent_representations'])}")
        
        # Analyze normalization quality across the month
        print(f"\nNORMALIZATION QUALITY ANALYSIS:")
        excellent_count = 0
        good_count = 0
        poor_count = 0
        
        for var_name, var_data in self.monthly_stats['surface_vars'].items():
            for stats in var_data['timestep_stats']:
                if abs(stats['mean']) < 0.1 and 0.9 < stats['std'] < 1.1:
                    excellent_count += 1
                elif abs(stats['mean']) < 0.3 and 0.7 < stats['std'] < 1.3:
                    good_count += 1
                else:
                    poor_count += 1
        
        total_surface_samples = excellent_count + good_count + poor_count
        print(f"   Surface variables normalization quality:")
        print(f"     Excellent: {excellent_count}/{total_surface_samples} ({excellent_count/total_surface_samples*100:.1f}%)")
        print(f"     Good: {good_count}/{total_surface_samples} ({good_count/total_surface_samples*100:.1f}%)")
        print(f"     Poor: {poor_count}/{total_surface_samples} ({poor_count/total_surface_samples*100:.1f}%)")
        
        # Show temporal evolution of key variables
        print(f"\nTEMPORAL EVOLUTION OF KEY VARIABLES:")
        key_surface_vars = ['2t', '10u', '10v', 'msl']
        for var_name in key_surface_vars:
            if var_name in self.monthly_stats['surface_vars']:
                means = self.monthly_stats['surface_vars'][var_name]['overall_mean']
                stds = self.monthly_stats['surface_vars'][var_name]['overall_std']
                print(f"   {var_name}: Mean range [{min(means):.3f}, {max(means):.3f}], Std range [{min(stds):.3f}, {max(stds):.3f}]")
        
        # Memory usage estimate
        if self.save_tensors:
            saved_files = list(self.output_dir.rglob("*.npy"))
            total_size_mb = sum(f.stat().st_size for f in saved_files) / (1024 * 1024)
            print(f"\nSTORAGE SUMMARY:")
            print(f"   Total .npy files saved: {len(saved_files)}")
            print(f"   Total storage used: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")

    def cleanup_hooks(self, model: torch.nn.Module):
        """Cleans up all hooks and saves final statistics."""
        print("\n" + "="*70)
        print("CLEANING UP FULL MONTH CAPTURE HOOKS AND SAVING FINAL DATA")
        print("="*70)
        
        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

        # Restore original forward method
        if self.original_forward is not None and hasattr(model, 'forward'):
            model.forward = self.original_forward
            print("   Model's original forward method restored.")
        
        # Save all collected statistics
        self.save_monthly_statistics()
        
        # Generate final analysis
        self.analyze_full_month_data()
        
        print("Full month data capture cleanup complete.")

# --- Load model ---
print("\n" + "="*80)
print("LOADING AURORA MODEL FOR FULL MONTH PROCESSING")
print("="*80)

# Clear memory
if 'model' in globals():
    del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

try:
    # Load base model directly to GPU (no quantization for GPU)
    print("Loading AuroraSmallPretrained model directly to GPU...")
    model = AuroraSmallPretrained().to(device)
    model.eval()
    print(f"Aurora model loaded successfully on {device}")

    if not hasattr(model, 'surf_stats') or model.surf_stats is None:
        raise RuntimeError("Model is missing 'surf_stats'. Cannot perform normalization.")
    print("Model ready for full month normalized data capture.")

except Exception as e:
    print(f"ERROR: Could not load AuroraSmallPretrained model. {e}")
    exit()

# --- Full Month Processing ---
print("\n" + "#"*80)
print("INITIATING FULL MONTH NORMALIZED DATA CAPTURE")
print("Processing all timesteps in January 2023")
print("#"*80)

# Initialize the full month capture system
full_month_capture = FullMonthAuroraNormalizedDataCapture(
    save_tensors=True, 
    output_dir=output_capture_path,
    device=device
)

# Set up hooks
full_month_capture.total_timesteps = total_timesteps - 1  # -1 because we need pairs for Aurora
full_month_capture.setup_aurora_hooks(model)

print(f"\nProcessing {full_month_capture.total_timesteps} timesteps...")
print(f"This will capture normalized data for the entire month of January 2023")

# Process each timestep in the month
processed_count = 0
failed_count = 0

for i in range(1, total_timesteps):  # Start from 1 because we need i-1 and i
    try:
        print(f"\n{'='*50}")
        print(f"PROCESSING TIMESTEP {i}/{total_timesteps-1}")
        print(f"Time: {surf_vars_ds.valid_time.values[i]}")
        print(f"{'='*50}")
        
        # Create batch for current timestep
        batch = Batch(
            surf_vars={
                "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i-1, i]][None]).to(torch.float32),
                "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i-1, i]][None]).to(torch.float32),
                "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i-1, i]][None]).to(torch.float32),
                "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i-1, i]][None]).to(torch.float32),
            },
            static_vars={
                "z": torch.from_numpy(static_vars_ds["z"].values[0]).to(torch.float32),
                "slt": torch.from_numpy(static_vars_ds["slt"].values[0]).to(torch.float32),
                "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]).to(torch.float32),
            },
            atmos_vars={
                "t": torch.from_numpy(atmos_vars_ds["t"].values[[i-1, i]][None]).to(torch.float32),
                "u": torch.from_numpy(atmos_vars_ds["u"].values[[i-1, i]][None]).to(torch.float32),
                "v": torch.from_numpy(atmos_vars_ds["v"].values[[i-1, i]][None]).to(torch.float32),
                "q": torch.from_numpy(atmos_vars_ds["q"].values[[i-1, i]][None]).to(torch.float32),
                "z": torch.from_numpy(atmos_vars_ds["z"].values[[i-1, i]][None]).to(torch.float32),
            },
            metadata=Metadata(
                lat=torch.from_numpy(surf_vars_ds.latitude.values),
                lon=torch.from_numpy(surf_vars_ds.longitude.values),
                time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
                atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
            ),
        ).to(device=device)

        # Run forward pass to capture normalized data
        with torch.no_grad():
            output_batch = model(batch)
        
        processed_count += 1
        print(f"   ✅ Successfully processed timestep {i}")
        
        # Clean up memory periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"   🧹 Memory cleanup performed (every 10 timesteps)")
            
    except Exception as e:
        failed_count += 1
        print(f"   ❌ Failed to process timestep {i}: {e}")
        # Continue with next timestep instead of stopping
        continue

print(f"\n" + "="*80)
print("FULL MONTH PROCESSING COMPLETE")
print("="*80)
print(f"Successfully processed: {processed_count}/{total_timesteps-1} timesteps")
print(f"Failed: {failed_count} timesteps")
print(f"Success rate: {processed_count/(total_timesteps-1)*100:.1f}%")

# Generate comprehensive analysis and cleanup
full_month_capture.cleanup_hooks(model)

# --- Create Monthly Summary Report ---
print("\n" + "#"*80)
print("GENERATING COMPREHENSIVE MONTHLY REPORT")
print("#"*80)

summary_report_path = output_capture_path / "FULL_MONTH_SUMMARY_REPORT.txt"
with open(summary_report_path, 'w') as f:
    f.write("AURORA FULL MONTH NORMALIZED DATA CAPTURE REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Processing completed: {datetime.now()}\n")
    f.write(f"Dataset: January 2023 ERA5 data\n")
    f.write(f"Model: AuroraSmallPretrained (GPU accelerated)\n")
    f.write(f"GPU Used: {torch.cuda.get_device_name(0)} (GPU 2)\n\n")
    
    f.write("PROCESSING STATISTICS:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total timesteps available: {total_timesteps}\n")
    f.write(f"Timesteps processed: {processed_count}/{total_timesteps-1}\n")
    f.write(f"Success rate: {processed_count/(total_timesteps-1)*100:.1f}%\n")
    f.write(f"Failed timesteps: {failed_count}\n\n")
    
    f.write("DATA CAPTURED:\n")
    f.write("-" * 20 + "\n")
    f.write(f"Surface variables: {len(full_month_capture.monthly_stats['surface_vars'])} variables\n")
    f.write(f"Atmospheric variables: {len(full_month_capture.monthly_stats['atmospheric_vars'])} variables\n")
    f.write(f"Pressure levels per atmospheric variable: {len(atmos_vars_ds.pressure_level.values)}\n")
    f.write(f"3D latent representations: {len(full_month_capture.monthly_stats['latent_representations'])}\n")
    f.write(f"Pre-unnormalized predictions: {len(full_month_capture.monthly_stats['prediction_quality'])}\n\n")
    
    # List all captured variables
    f.write("CAPTURED VARIABLES:\n")
    f.write("-" * 25 + "\n")
    f.write("Surface variables:\n")
    for var_name in full_month_capture.monthly_stats['surface_vars'].keys():
        f.write(f"  - {var_name}\n")
    
    f.write("\nAtmospheric variables:\n")
    for var_name in full_month_capture.monthly_stats['atmospheric_vars'].keys():
        f.write(f"  - {var_name} (at {len(atmos_vars_ds.pressure_level.values)} pressure levels)\n")
    
    f.write(f"\nPressure levels: {list(atmos_vars_ds.pressure_level.values)} hPa\n\n")
    
    # Storage information
    if full_month_capture.save_tensors:
        saved_files = list(output_capture_path.rglob("*.npy"))
        total_size_mb = sum(f.stat().st_size for f in saved_files) / (1024 * 1024)
        f.write("STORAGE INFORMATION:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Total .npy files saved: {len(saved_files)}\n")
        f.write(f"Total storage used: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)\n")
        f.write(f"Average per timestep: {total_size_mb/max(processed_count,1):.1f} MB\n\n")
    
    f.write("OUTPUT FILES:\n")
    f.write("-" * 20 + "\n")
    f.write("📁 surface_normalized/ - Normalized surface variable tensors\n")
    f.write("📁 atmospheric_normalized/ - Normalized atmospheric variable tensors\n")
    f.write("📁 latent_representations/ - 3D latent representations from encoder/decoder\n")
    f.write("📁 pre_unnormalized_predictions/ - Predictions before unnormalization\n")
    f.write("📁 monthly_statistics/ - JSON and CSV statistical summaries\n\n")
    
    f.write("KEY FILES FOR ANALYSIS:\n")
    f.write("-" * 30 + "\n")
    f.write("📄 full_month_normalized_stats.json - Complete statistical data\n")
    f.write("📄 surface_variables_monthly_stats.csv - Surface variable statistics\n")
    f.write("📄 atmospheric_variables_key_levels_monthly_stats.csv - Atmospheric statistics\n")
    f.write("📄 latent_representations_monthly_stats.csv - 3D latent statistics\n")
    f.write("📄 timestep_normalized_data.json - Complete timestep-by-timestep data\n")
    f.write("📄 timestep_prediction_data.json - Complete prediction data\n\n")
    
    f.write("USAGE RECOMMENDATIONS:\n")
    f.write("-" * 30 + "\n")
    f.write("1. Load CSV files in pandas/Excel for time series analysis\n")
    f.write("2. Use .npy files for detailed tensor analysis in Python\n")
    f.write("3. Check normalization quality statistics for model validation\n")
    f.write("4. Analyze 3D latent representations for model interpretability\n")
    f.write("5. Compare pre-unnormalized vs normalized data for debugging\n")

print(f"📋 Comprehensive report saved to: {summary_report_path.name}")

# --- Final Success Message ---
print(f"\n" + "#"*80)
print("🎉 FULL MONTH AURORA NORMALIZED DATA CAPTURE COMPLETED SUCCESSFULLY! 🎉")
print("#"*80)
print(f"📊 Processed {processed_count} timesteps covering January 2023")
print(f"📁 All data saved to: {output_capture_path.resolve()}")
print(f"📈 {len(full_month_capture.monthly_stats['surface_vars'])} surface variables captured")
print(f"🌡️  {len(full_month_capture.monthly_stats['atmospheric_vars'])} atmospheric variables captured")
print(f"🧠 {len(full_month_capture.monthly_stats['latent_representations'])} latent representations captured")
print(f"💾 Complete dataset ready for analysis and model understanding")
print(f"🚀 GPU 2 Performance: Excellent!")
print("#"*80)

print("\nNext steps:")
print("1. 📈 Analyze the CSV files for temporal patterns")
print("2. 🔍 Examine normalization quality across variables")
print("3. 🧠 Study 3D latent representations for model insights") 
print("4. 📊 Compare normalized vs unnormalized data distributions")
print("5. 🎯 Use this data for Aurora model validation and improvement")
