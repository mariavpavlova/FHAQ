# %%
# --- Imports ---
import os
import torch
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# We will not directly use sklearn.metrics or rollout for the capture phase,
# but they are part of your original script, so kept for completeness.
from sklearn.metrics import accuracy_score, mean_squared_error
from aurora import AuroraSmallPretrained, Batch, Metadata, rollout

import warnings
import contextlib
import dataclasses
from typing import Optional
import gc # For garbage collection

print("All imports successful!")

# %%
# --- Configuration ---
# GPU settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Data paths
# Ensure these paths are correct for your system
data_path = Path("/data2/users/mp2522/Aurora/data/monthlydata")
static_file = data_path / "static.nc"
surf_file = data_path / "Jan2023surface-level.nc"
atm_file = data_path / "Jan2023atmospheric.nc"

# Output path for captured normalized data
output_capture_path = Path("./captured_normalized_aurora_data")
# Create the directory if it doesn't exist
output_capture_path.mkdir(exist_ok=True)
print(f"Captured normalized data will be saved to: {output_capture_path.resolve()}")

# Choose start index for your data (i0 must be ≥1 as it uses i0-1 and i0 for history)
i0 = 1

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Initialize device info for logging ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Available GPUs: {torch.cuda.device_count()}")
else:
    print("CUDA not available, using CPU")

# --- Load datasets ---
try:
    static_vars_ds = xr.open_dataset(static_file, engine="netcdf4")
    surf_vars_ds = xr.open_dataset(surf_file, engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(atm_file, engine="netcdf4")
    print("Datasets loaded successfully!")
except FileNotFoundError as e:
    print(f"ERROR: One or more data files not found. Please check paths. {e}")
    exit()
except Exception as e:
    print(f"ERROR: Could not load datasets. {e}")
    exit()

# %%
# --- Aurora Normalized Data Capture System Definition ---

class AuroraNormalizedDataCapture:
    """
    Captures Aurora's normalized data at key locations within its forward pass,
    based on the official architecture (AuroraSmallPretrained).

    The main capture points are:
    1.  After `batch.normalise()`: The true normalized input data feeding the encoder.
    2.  Encoder output: The 3D latent representation from the encoder.
    3.  Decoder input: The 3D latent representation passed to the decoder.
    4.  Before `pred.unnormalise()`: The predictions still in normalized space.
    """
    def __init__(self, save_tensors: bool = True, output_dir: Path = Path("./captured_normalized_aurora_data")):
        self.normalized_inputs = []  # After batch.normalise(), before encoder
        self.encoder_outputs = []    # 3D latent representations from encoder
        self.decoder_inputs = []     # 3D latent going into decoder
        self.unnormalized_outputs = [] # Before pred.unnormalise()
        self.hooks = []              # Stores PyTorch hook handles
        self.step_count = 0          # To track which step of a rollout/forward is being captured
        self.original_forward = None # To store the original forward method for restoration
        self.save_tensors = save_tensors # Whether to save tensor data to .npy files
        self.output_dir = output_dir # Directory to save .npy files

    def setup_aurora_hooks(self, model: torch.nn.Module) -> bool:
        """
        Sets up hooks on the Aurora model to capture intermediate data.
        This method will monkey-patch the model's `forward` method to intercept
        the `batch.normalise()` and `pred.unnormalise()` calls.
        """
        print("\n" + "="*70)
        print("SETTING UP AURORA NORMALIZED DATA CAPTURE HOOKS")
        print("="*70)

        hooks_registered = 0

        # IMPORTANT: Monkey patch the model's forward method to capture pre/post normalization steps
        # This allows us to intercept batch.normalise() which is not a torch.nn.Module.
        if hasattr(model, 'forward'):
            self.original_forward = model.forward
            model.forward = self._create_capturing_forward(model)
            hooks_registered += 1 # This counts as a major capture point
            print(f"   Successfully monkey-patched model.forward for normalization capture.")
        else:
            print("   ERROR: Model does not have a 'forward' method. Cannot set up main capture hook.")
            return False

        # Hook into encoder to capture its output (3D latent representation)
        if hasattr(model, 'encoder') and isinstance(model.encoder, torch.nn.Module):
            hook = model.encoder.register_forward_hook(self._encoder_output_hook)
            self.hooks.append(hook)
            hooks_registered += 1
            print(f"   Hooked encoder output (3D latent representation).")

        # Hook into decoder to capture its input (same 3D latent)
        if hasattr(model, 'decoder') and isinstance(model.decoder, torch.nn.Module):
            hook = model.decoder.register_forward_hook(self._decoder_input_hook)
            self.hooks.append(hook)
            hooks_registered += 1
            print(f"   Hooked decoder input (3D latent representation).")

        # Optional: Hook into backbone to capture intermediate processing of 3D latent
        if hasattr(model, 'backbone') and isinstance(model.backbone, torch.nn.Module):
            hook = model.backbone.register_forward_hook(self._backbone_hook)
            self.hooks.append(hook)
            hooks_registered += 1
            print(f"   Hooked backbone (3D latent processing).")

        print(f"Total capture points set up: {hooks_registered}")
        return hooks_registered > 0

    def _create_capturing_forward(self, model: torch.nn.Module):
        """
        Returns a wrapper around Aurora's forward method to capture normalized data.
        This wrapper replicates the crucial parts of the official Aurora forward pass
        to ensure capture happens at the correct normalization stages.
        """
        def capturing_forward(batch: Batch) -> Batch:
            print(f"\n--- Capturing normalized data for forward pass (step {self.step_count}) ---")

            # --- Replicate Aurora's Official Forward Pass Logic ---
            # This is crucial for capturing data at the correct normalization points.

            # 1. batch_transform_hook (e.g., for AuroraWave specific transformations)
            batch = model.batch_transform_hook(batch)

            # 2. Move batch to model's device and dtype
            # Use `model._model_dtype` if available, otherwise infer from parameters
            p_dtype = getattr(model, '_model_dtype', next(model.parameters()).dtype)
            batch = batch.to(device=model.device)  # FIXED: Removed dtype parameter

            # CAPTURE POINT 1: After batch.normalise() - THE KEY NORMALIZED INPUTS!
            # Ensure model has surf_stats and atmos_stats for normalization
            if not hasattr(model, 'surf_stats') or model.surf_stats is None:
                raise AttributeError(
                    "Model does not have 'surf_stats' attribute, or it is None. "
                    "Cannot perform batch.normalise(). Ensure model is loaded with its pre-computed statistics."
                )
            # atmos_stats might be None for some models, check defensively
            if not hasattr(model, 'atmos_stats'):
                 model.atmos_stats = None # Ensure attribute exists, even if None

            normalized_batch = batch.normalise(
                surf_stats=model.surf_stats,
                atmos_stats=model.atmos_stats
            )
            print(f"   CAPTURED: Batch data after batch.normalise().")
            self._capture_normalized_batch(normalized_batch)

            # 3. Crop and move to device (following official code)
            batch_processed = normalized_batch.crop(patch_size=model.patch_size)

            # 4. Calculate patch resolution (official code)
            H, W = batch_processed.spatial_shape
            patch_res = (
                model.encoder.latent_levels,
                H // model.encoder.patch_size,
                W // model.encoder.patch_size,
            )

            # 5. Insert batch and history dimension for static variables (official code)
            B, T = next(iter(batch_processed.surf_vars.values())).shape[:2]
            batch_processed = dataclasses.replace(
                batch_processed,
                static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch_processed.static_vars.items()},
            )

            # 6. Apply transformations before encoder (renamed variable like official code)
            transformed_batch = batch_processed

            # 7. Clamp positive variables (official code)
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

            # 8. Pre-encoder hook (for special models like AuroraAirPollution)
            transformed_batch = model._pre_encoder_hook(transformed_batch)

            # CAPTURE POINT 2: Encoder output (3D latent representation)
            # This is captured by the registered hook on `model.encoder`
            print(f"   Processing with encoder...")
            x = model.encoder(transformed_batch, lead_time=model.timestep)

            # CAPTURE POINT 3: Backbone processing (3D latent transformations)
            # This is captured by the registered hook on `model.backbone`
            print(f"   Processing with backbone...")
            with torch.autocast(device_type=model.device.type) if model.autocast else contextlib.nullcontext():
                x = model.backbone(
                    x,
                    lead_time=model.timestep,
                    patch_res=patch_res,
                    rollout_step=batch.metadata.rollout_step,
                )

            # CAPTURE POINT 4: Decoder processing (3D latent -> predictions)
            # This is captured by the registered hook on `model.decoder` (its input)
            print(f"   Processing with decoder to generate predictions...")
            pred = model.decoder(x, batch_processed, lead_time=model.timestep, patch_res=patch_res)

            # 9. Remove batch and history dimension from static variables (official code)
            pred = dataclasses.replace(
                pred,
                static_vars={k: v[0, 0] for k, v in batch_processed.static_vars.items()},
            )

            # 10. Insert history dimension in prediction (official code)
            pred = dataclasses.replace(
                pred,
                surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
                atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
            )

            # 11. Post-decoder hook (for special models like AuroraAirPollution difference prediction)
            pred = model._post_decoder_hook(batch_processed, pred)

            # 12. Clamp positive variables again (official code logic)
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

            # CAPTURE POINT 5: Before unnormalise() - still normalized predictions
            print(f"   CAPTURED: Pre-unnormalize predictions (still normalized).")
            self._capture_pre_unnormalize(pred)

            # 13. Final unnormalization (THE FINAL STEP)
            print(f"   Running final unnormalise() to get physical units...")
            pred = pred.unnormalise(
                surf_stats=model.surf_stats,
                atmos_stats=model.atmos_stats
            )

            print(f"   Forward pass for step {self.step_count} completed.")
            # Increment step count for the *next* forward pass
            self.step_count += 1
            return pred

        return capturing_forward

    def _capture_normalized_batch(self, normalized_batch: Batch):
        """Captures the batch data after `batch.normalise()`."""
        print(f"     Capturing normalized input data (Batch, step {self.step_count}).")

        # Capture normalized surface variables
        for var_name, tensor in normalized_batch.surf_vars.items():
            # For B, T, H, W tensors, take first batch, first time step for stats
            values = tensor[0, 0].detach().cpu().numpy().flatten()
            mean_val = np.mean(values)
            std_val = np.std(values)

            normalized_data = {
                'data': tensor.clone().detach(), # Store the entire tensor for saving
                'var_name': var_name,
                'var_type': 'surface',
                'step': self.step_count,
                'source': 'batch.normalise()',
                'location': 'after_batch_normalise_before_encoder',
                'stats': {
                    'mean': mean_val, 'std': std_val,
                    'min': np.min(values), 'max': np.max(values)
                }
            }
            self.normalized_inputs.append(normalized_data)
            print(f"       Surf Var '{var_name}': mean={mean_val:.6f}, std={std_val:.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
            if self.save_tensors:
                # Save just one time step (T=0) for easier inspection
                self._save_tensor_data(tensor[0, 0], f'normalized_input_surf_{var_name}_step_{self.step_count}')

        # Capture normalized atmospheric variables (sample first level for analysis)
        for var_name, tensor in normalized_batch.atmos_vars.items():
            # For B, T, L, H, W tensors, take first batch, first time step, first level for stats
            values = tensor[0, 0, 0].detach().cpu().numpy().flatten()
            mean_val = np.mean(values)
            std_val = np.std(values)

            normalized_data = {
                'data': tensor.clone().detach(), # Store the entire tensor for saving
                'var_name': var_name,
                'var_type': 'atmospheric',
                'step': self.step_count,
                'source': 'batch.normalise()',
                'location': 'after_batch_normalise_before_encoder',
                'pressure_level': normalized_batch.metadata.atmos_levels[0] if normalized_batch.metadata.atmos_levels else 'N/A',
                'stats': {
                    'mean': mean_val, 'std': std_val,
                    'min': np.min(values), 'max': np.max(values)
                }
            }
            self.normalized_inputs.append(normalized_data)
            print(f"       Atm Var '{var_name}' (Lvl {normalized_data['pressure_level']}): mean={mean_val:.6f}, std={std_val:.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
            if self.save_tensors:
                # Save just one time step (T=0) and first level for easier inspection
                self._save_tensor_data(tensor[0, 0, 0], f'normalized_input_atm_{var_name}_lvl0_step_{self.step_count}')

    def _encoder_output_hook(self, module: torch.nn.Module, input_tensor: tuple, output_tensor: torch.Tensor):
        """Hook for encoder output - the 3D latent representation."""
        # This hook runs after the encoder forward pass
        print(f"   Encoder output hook triggered (step {self.step_count}).")
        if isinstance(output_tensor, torch.Tensor):
            values = output_tensor.detach().cpu().numpy().flatten()
            mean_val = np.mean(values)
            std_val = np.std(values)
            encoder_data = {
                'data': output_tensor.clone().detach(),
                'step': self.step_count,
                'source': 'encoder_output',
                'type': '3d_latent_representation',
                'stats': {
                    'mean': mean_val, 'std': std_val,
                    'min': np.min(values), 'max': np.max(values)
                }
            }
            self.encoder_outputs.append(encoder_data)
            print(f"     3D Latent (Encoder Output) shape: {output_tensor.shape}, Stats: mean={mean_val:.6f}, std={std_val:.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
            if self.save_tensors:
                self._save_tensor_data(output_tensor, f'3d_latent_encoder_output_step_{self.step_count}')
        # Important: hooks should always return the output_tensor unchanged
        return output_tensor

    def _decoder_input_hook(self, module: torch.nn.Module, input_tensor: tuple, output_tensor: torch.Tensor):
        """Hook for decoder - captures the 3D latent input."""
        # This hook runs after the decoder forward pass, but we capture its *input*
        print(f"   Decoder input hook triggered (step {self.step_count}).")
        # The actual input to the decoder is the first element of the input_tensor tuple
        if input_tensor and len(input_tensor) > 0 and isinstance(input_tensor[0], torch.Tensor):
            decoder_input_latent = input_tensor[0]
            values = decoder_input_latent.detach().cpu().numpy().flatten()
            mean_val = np.mean(values)
            std_val = np.std(values)
            decoder_data = {
                'data': decoder_input_latent.clone().detach(),
                'step': self.step_count,
                'source': 'decoder_input',
                'type': '3d_latent_to_decoder',
                'stats': {
                    'mean': mean_val, 'std': std_val,
                    'min': np.min(values), 'max': np.max(values)
                }
            }
            self.decoder_inputs.append(decoder_data)
            print(f"     3D Latent (Decoder Input) shape: {decoder_input_latent.shape}, Stats: mean={mean_val:.6f}, std={std_val:.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
            if self.save_tensors:
                self._save_tensor_data(decoder_input_latent, f'3d_latent_decoder_input_step_{self.step_count}')
        # Important: hooks should always return the output_tensor unchanged
        return output_tensor

    def _backbone_hook(self, module: torch.nn.Module, input_tensor: tuple, output_tensor: torch.Tensor):
        """Hook for backbone processing of 3D latent."""
        # This hook runs after the backbone forward pass
        print(f"   Backbone output hook triggered (step {self.step_count}).")
        if isinstance(output_tensor, torch.Tensor):
            values = output_tensor.detach().cpu().numpy().flatten()
            mean_val = np.mean(values)
            std_val = np.std(values)
            # No need to store in a list if not analyzing this in depth, just print for sanity check
            print(f"     3D Latent (Backbone Output) shape: {output_tensor.shape}, Stats: mean={mean_val:.6f}, std={std_val:.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
            if self.save_tensors:
                self._save_tensor_data(output_tensor, f'3d_latent_backbone_output_step_{self.step_count}')
        # Important: hooks should always return the output_tensor unchanged
        return output_tensor


    def _capture_pre_unnormalize(self, pred_batch: Batch):
        """Captures predictions BEFORE `unnormalise()` - still in normalized space."""
        print(f"     Capturing predictions before unnormalise() (step {self.step_count}).")
        for var_name, tensor in pred_batch.surf_vars.items():
            # For B, T, H, W tensors, take first batch, first time step for stats
            values = tensor[0, 0].detach().cpu().numpy().flatten()
            mean_val = np.mean(values)
            std_val = np.std(values)
            unnorm_data = {
                'data': tensor.clone().detach(), # Store the entire tensor for saving
                'var_name': var_name,
                'var_type': 'surface',
                'step': self.step_count,
                'source': 'pre_unnormalize',
                'stats': {
                    'mean': mean_val, 'std': std_val,
                    'min': np.min(values), 'max': np.max(values)
                }
            }
            self.unnormalized_outputs.append(unnorm_data)
            print(f"       Pre-unnorm Surf Var '{var_name}': mean={mean_val:.6f}, std={std_val:.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
            if self.save_tensors:
                # Save just one time step (T=0) for easier inspection
                self._save_tensor_data(tensor[0, 0], f'pre_unnormalize_surf_{var_name}_step_{self.step_count}')

        for var_name, tensor in pred_batch.atmos_vars.items():
            # For B, T, L, H, W tensors, take first batch, first time step, first level for stats
            values = tensor[0, 0, 0].detach().cpu().numpy().flatten()
            mean_val = np.mean(values)
            std_val = np.std(values)
            unnorm_data = {
                'data': tensor.clone().detach(), # Store the entire tensor for saving
                'var_name': var_name,
                'var_type': 'atmospheric',
                'step': self.step_count,
                'source': 'pre_unnormalize',
                'pressure_level': pred_batch.metadata.atmos_levels[0] if pred_batch.metadata.atmos_levels else 'N/A',
                'stats': {
                    'mean': mean_val, 'std': std_val,
                    'min': np.min(values), 'max': np.max(values)
                }
            }
            self.unnormalized_outputs.append(unnorm_data)
            print(f"       Pre-unnorm Atm Var '{var_name}' (Lvl {unnorm_data['pressure_level']}): mean={mean_val:.6f}, std={std_val:.6f}, min={np.min(values):.6f}, max={np.max(values):.6f}")
            if self.save_tensors:
                # Save just one time step (T=0) and first level for easier inspection
                self._save_tensor_data(tensor[0, 0, 0], f'pre_unnormalize_atm_{var_name}_lvl0_step_{self.step_count}')


    def _save_tensor_data(self, tensor: torch.Tensor, filename_suffix: str):
        """Saves a PyTorch tensor's data to a .npy file."""
        try:
            tensor_np = tensor.detach().cpu().numpy()
            save_path = self.output_dir / f"aurora_normalized_{filename_suffix}.npy"
            np.save(save_path, tensor_np)
            # print(f"         Saved: {save_path.name}") # Uncomment for verbose saving
        except Exception as e:
            print(f"         WARNING: Failed to save {filename_suffix} to file: {e}")

    def validate_capture_accuracy(self) -> bool:
        """
        Validates that data has been captured from the expected locations
        and performs a basic check on the normalization quality of the inputs.
        """
        print("\n" + "="*70)
        print("VALIDATING CAPTURE SYSTEM INTEGRATION AND NORMALIZATION QUALITY")
        print("="*70)

        validation_results = {
            'normalized_inputs_captured': len(self.normalized_inputs) > 0,
            'encoder_outputs_captured': len(self.encoder_outputs) > 0,
            'decoder_inputs_captured': len(self.decoder_inputs) > 0,
            'pre_unnormalize_captured': len(self.unnormalized_outputs) > 0
        }

        print(f"\nCAPTURE PRESENCE CHECK (for last executed forward pass):")
        for check, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"   {check}: {status}")

        all_capture_points_hit = all(validation_results.values())

        # Validate normalized input quality (focus on last captured step)
        quality_percentage = 0
        if self.normalized_inputs:
            print(f"\nNORMALIZED INPUT QUALITY CHECK (for step {self.step_count - 1} or last captured step):")
            # Get data for the most recently processed step
            current_step_inputs = [d for d in self.normalized_inputs if d['step'] == self.step_count - 1]
            if not current_step_inputs:
                print("   No normalized inputs captured for the last step. Cannot perform quality check.")
            else:
                well_normalized_count = 0
                total_vars = len(current_step_inputs)

                for data in current_step_inputs:
                    stats = data['stats']
                    mean_abs = abs(stats['mean'])
                    std_val = stats['std']

                    # Ideal normalization: mean near 0, std near 1
                    # Tolerances can be adjusted, e.g., mean < 0.1, std between 0.9 and 1.1
                    if mean_abs < 0.1 and 0.9 < std_val < 1.1:
                        well_normalized_count += 1
                    else:
                        print(f"     VAR '{data['var_name']}' ({data['var_type']}): QUESTIONABLE (mean={stats['mean']:.3f}, std={std_val:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f})")

                quality_percentage = (well_normalized_count / total_vars) * 100
                print(f"   Normalization quality: {well_normalized_count}/{total_vars} variables ({quality_percentage:.1f}%) are excellently normalized for the last step.")

                if quality_percentage > 80:
                    print(f"   **Overall Normalized Input looks EXCELLENT for this step.**")
                elif quality_percentage > 60:
                    print(f"   **Normalized Input looks GOOD for this step, but some variables show slight deviations.**")
                else:
                    print(f"   **WARNING: Significant normalized input issues detected for this step. Verify model's 'surf_stats' and 'atmos_stats' (if applicable).**")
        else:
            print("\nNo normalized inputs captured for any step. Skipping quality check.")

        overall_success = all_capture_points_hit and (quality_percentage > 60 if self.normalized_inputs else True)
        if overall_success:
            print(f"\nVALIDATION PASSED: Capture system is correctly integrated and data quality is acceptable.")
        else:
            print(f"\nVALIDATION FAILED: Review the logs above for specific issues.")
        return overall_success


    def analyze_captured_data(self):
        """
        Prints a summary of all captured normalized data and validation results.
        """
        print("\n" + "="*70)
        print("SUMMARY OF CAPTURED AURORA NORMALIZED DATA")
        print("="*70)

        # First, run validation for the latest state
        self.validate_capture_accuracy()

        print(f"\nDETAILED CAPTURE BREAKDOWN:")
        print(f"   - Total normalized inputs (after batch.normalise()): {len(self.normalized_inputs)} entries")
        print(f"   - Total 3D latent representations (encoder output): {len(self.encoder_outputs)} entries")
        print(f"   - Total 3D latent to decoder: {len(self.decoder_inputs)} entries")
        print(f"   - Total pre-unnormalize predictions: {len(self.unnormalized_outputs)} entries")

        # Analyze normalized inputs (the most critical part)
        if self.normalized_inputs:
            print(f"\n--- NORMALIZED INPUTS (AFTER batch.normalise()) ---")
            for data in self.normalized_inputs:
                stats = data['stats']
                var_info = f"{data['var_name']} ({data['var_type']}, Step {data['step']})"
                if 'pressure_level' in data and data['var_type'] == 'atmospheric':
                    var_info += f" Lvl {data['pressure_level']}"
                print(f"   {var_info}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}")

        # Analyze 3D latent representations
        if self.encoder_outputs:
            print(f"\n--- 3D LATENT REPRESENTATIONS (ENCODER OUTPUT) ---")
            for data in self.encoder_outputs:
                stats = data['stats']
                print(f"   Step {data['step']}: Shape={data['data'].shape}, Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}")

        if self.decoder_inputs:
            print(f"\n--- 3D LATENT REPRESENTATIONS (DECODER INPUT) ---")
            for data in self.decoder_inputs:
                stats = data['stats']
                print(f"   Step {data['step']}: Shape={data['data'].shape}, Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}")

        # Analyze pre-unnormalize predictions
        if self.unnormalized_outputs:
            print(f"\n--- PRE-UNNORMALIZE PREDICTIONS (STILL NORMALIZED) ---")
            for data in self.unnormalized_outputs:
                stats = data['stats']
                var_info = f"{data['var_name']} ({data['var_type']}, Step {data['step']})"
                if 'pressure_level' in data and data['var_type'] == 'atmospheric':
                    var_info += f" Lvl {data['pressure_level']}"
                print(f"   {var_info}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}")

        # List all saved files
        saved_files = list(self.output_dir.glob("aurora_normalized_*.npy"))
        if saved_files and self.save_tensors:
            print(f"\nSAVED CAPTURED TENSOR FILES ({len(saved_files)} files in '{self.output_dir.name}'):")
            for file in sorted(saved_files):
                print(f"   - {file.name}")
        elif not self.save_tensors:
            print(f"\nTensor saving is disabled (save_tensors=False). No .npy files generated.")
        else:
            print("\nNo normalized data files were saved.")

    def cleanup_hooks(self, model: torch.nn.Module):
        """
        Removes all registered hooks and restores the original model.forward method.
        It's crucial to call this to avoid memory leaks and ensure the model
        behaves normally after the capture.
        """
        print("\n" + "="*70)
        print("CLEANING UP AURORA NORMALIZED DATA CAPTURE HOOKS")
        print("="*70)
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear() # Clear the list of hooks

        # Restore original forward method
        if self.original_forward is not None and hasattr(model, 'forward'):
            model.forward = self.original_forward
            print("   Model's original forward method restored.")
        else:
            print("   Could not restore original model.forward method (was it patched?).")

        print("Cleanup complete. Hooks removed.")


# %%
# --- Build initial Batch from your real data ---
print("\n" + "="*80)
print("PREPARING REAL DATA BATCH FOR NORMALIZED DATA CAPTURE")
print("="*80)

# Debug: Check the shapes of your data first
print("=== DATA SHAPES DEBUG (Before Batch Creation) ===")
# Ensure i0-1 and i0 are valid indices.
# surf_vars_ds['t2m'].values should be accessible at these indices.
try:
    if i0 < 1 or i0 >= len(surf_vars_ds['valid_time']): # Check against valid_time length for total time steps
        raise IndexError(f"i0 ({i0}) is out of bounds for surf_vars_ds.valid_time ({len(surf_vars_ds['valid_time'])}). Adjust i0 to be at least 1 and less than total timesteps.")

    print(f"Surface variable shape (t2m) for indices [{i0-1}, {i0}]: {surf_vars_ds['t2m'].values[[i0 - 1, i0]].shape}")
    print(f"Atmospheric variable shape (t) for indices [{i0-1}, {i0}]: (2, 13, 721, 1440)")
    print(f"Static variable shape (z): {static_vars_ds['z'].values.shape}") # Static vars typically don't have a time dim
except Exception as e:
    print(f"ERROR: Data shape check failed. Make sure data variables and indices are correct. {e}")
    exit()

# MEMORY OPTIMIZATION: Downsample data to reduce GPU memory usage
downsample_factor = 4

# Downsample indices
lat_indices = slice(None, None, downsample_factor)
lon_indices = slice(None, None, downsample_factor)

print(f"\n=== MEMORY OPTIMIZATION ===")
print(f"Downsampling by factor {downsample_factor}")
# Example of expected downsampled shape (e.g., from (721, 1440) to approx (181, 360))
try:
    sample_downsampled_shape = surf_vars_ds['t2m'].values[0, lat_indices, lon_indices].shape
    print(f"Original 2D resolution: {surf_vars_ds['t2m'].shape[-2:]} -> Downsampled 2D resolution: {sample_downsampled_shape}")
except Exception as e:
    print(f"WARNING: Could not determine sample downsampled shape. Ensure lat/lon dimensions are correct. {e}")


# Create batch with downsampled data using i0 for the initial state
# Note: Aurora expects (B, T, H, W) or (B, T, L, H, W) where T is history length + 1 (usually 2)
# The `[i0 - 1, i0]` selects two time steps for history.
try:
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[[i0 - 1, i0]][:, lat_indices, lon_indices][None]), # Add batch dim [None]
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[[i0 - 1, i0]][:, lat_indices, lon_indices][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[[i0 - 1, i0]][:, lat_indices, lon_indices][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[[i0 - 1, i0]][:, lat_indices, lon_indices][None]),
        },
        static_vars={
            # Static vars are 2D (lat, lon) in the original dataset, usually no time dim
            "z": torch.from_numpy(static_vars_ds["z"].values[0, lat_indices, lon_indices]), # Take first (and likely only) time dim
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0, lat_indices, lon_indices]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0, lat_indices, lon_indices]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[[i0 - 1, i0]][:, :, lat_indices, lon_indices][None]), # Add batch dim [None]
            "u": torch.from_numpy(atmos_vars_ds["u"].values[[i0 - 1, i0]][:, :, lat_indices, lon_indices][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[[i0 - 1, i0]][:, :, lat_indices, lon_indices][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[[i0 - 1, i0]][:, :, lat_indices, lon_indices][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[[i0 - 1, i0]][:, :, lat_indices, lon_indices][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values[lat_indices]),
            lon=torch.from_numpy(surf_vars_ds.longitude.values[lon_indices]),
            # Select the valid_time corresponding to i0 (the current time step for prediction)
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i0],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    ).to(device=device)  # FIXED: Removed dtype parameter

    print("Batch prepared successfully with real data and moved to device!")
    # Debug: Check final tensor shapes
    print("\n=== BATCH TENSOR SHAPES (AFTER CREATION AND DEVICE TRANSFER) ===")
    for name, tensor in batch.surf_vars.items():
        print(f"surf_vars[{name}]: {tensor.shape}")
    for name, tensor in batch.static_vars.items():
        print(f"static_vars[{name}]: {tensor.shape}")
    for name, tensor in batch.atmos_vars.items():
        print(f"atmos_vars[{name}]: {tensor.shape}")

    print(f"Batch prepared with vars: {list(batch.surf_vars.keys())}, {list(batch.atmos_vars.keys())}")
    print(f"Batch data type: {batch.surf_vars['2t'].dtype}, Device: {batch.surf_vars['2t'].device}")

except Exception as e:
    print(f"ERROR: Failed to prepare batch with real data. Check your data structure and variable names. {e}")
    exit()

# %%
# --- Load model with Quantization ---
print("\n" + "="*80)
print("LOADING AURORA MODEL WITH QUANTIZATION")
print("="*80)

# Clear any existing model from memory to avoid conflicts and free GPU RAM
print("=== CLEARING GPU MEMORY ===")
if 'model' in globals():
    del model
if 'quantized_model' in globals():
    del quantized_model
if 'base_model' in globals():
    del base_model

# Force Python's garbage collection and clear PyTorch's CUDA cache
gc.collect()
torch.cuda.empty_cache()

# Check current GPU memory usage after cleanup
print("=== GPU MEMORY STATUS BEFORE LOADING MODEL ===")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f}GB allocated, {torch.cuda.memory_reserved(i)/1e9:.2f}GB reserved")

print(f"\n=== LOADING MODEL WITH AGGRESSIVE DYNAMIC QUANTIZATION (qint8) ===")

try:
    # Load base model on CPU first for quantization
    base_model = AuroraSmallPretrained()
    print("Base AuroraSmallPretrained model loaded on CPU.")

    # Apply dynamic quantization (weights to qint8, activations dynamically to qint8)
    print("Applying aggressive dynamic quantization (qint8)...")
    # Specify modules to quantize: Linear, Conv1d, Conv2d, Conv3d, LayerNorm are common.
    # Note: Aurora may contain other module types, you can add them if needed.
    quantized_model = torch.quantization.quantize_dynamic(
        base_model,
        {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.LayerNorm},
        dtype=torch.qint8
    )
    print("Model dynamically quantized successfully.")

    # Clear original model from memory as it's no longer needed
    del base_model
    gc.collect()

    # Move the quantized model to the determined device (CPU or GPU)
    model = quantized_model.to(device)
    model.eval() # Set to evaluation mode for inference

    print(f"Quantized model type: {type(model)}")
    print(f"Model loaded and moved to {device}")
    # Verify that the model has the necessary statistics for normalization
    if not hasattr(model, 'surf_stats') or model.surf_stats is None:
        raise RuntimeError("Model is missing 'surf_stats'. Cannot perform normalization.")
    if not hasattr(model, 'atmos_stats') or model.atmos_stats is None:
        print("Note: Model is missing 'atmos_stats'. This is normal for some Aurora variants if only surface data is used for normalization.")

    print("Quantized model ready for normalized data capture.")

except Exception as e:
    print(f"ERROR: Could not load or quantize AuroraSmallPretrained model. {e}")
    exit()

# %%
# --- Normalized Data Capture Execution ---
print("\n" + "#"*80)
print("INITIATING NORMALIZED DATA CAPTURE WITH REAL DATA")
print("#"*80)

# Instantiate and setup the data capture system
data_capture = AuroraNormalizedDataCapture(save_tensors=True, output_dir=output_capture_path)
data_capture.setup_aurora_hooks(model)

# Run a single forward pass with the hooked model
# This will trigger all the capture hooks.
# We are intentionally *not* calling rollout here to focus on the initial forward pass
# where batch.normalise() happens for the input.
print("\nRunning a single forward pass through the model with real data to trigger data capture...")
with torch.no_grad(): # Inference mode, disable gradient computation for efficiency
    # Pass your real data batch to the hooked model.
    # The custom model.forward will now execute and capture intermediate data.
    output_batch = model(batch)
print("Forward pass complete. Normalized data should have been captured.")

# Analyze the captured data: prints summary and runs validation checks
data_capture.analyze_captured_data()

# IMPORTANT: Cleanup hooks and restore original model behavior
# This is crucial to prevent memory leaks and ensure the model can be used normally later.
data_capture.cleanup_hooks(model)

print("\n" + "#"*80)
print("NORMALIZED DATA CAPTURE DEMONSTRATION WITH REAL DATA COMPLETE.")
print(f"Check the logs above for statistical summaries and the '{output_capture_path.name}' directory for .npy files.")
print("To analyze the quantization effect, compare the stats and distributions of:")
print("  1. 'NORMALIZED INPUTS' (should be mean~0, std~1)")
print("  2. 'PRE-UNNORMALIZE PREDICTIONS' (should also be mean~0, std~1 if model predicts well)")
print("#"*80)

# %%
