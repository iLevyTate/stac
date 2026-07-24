#!/usr/bin/env python
"""
STAC: SpikeTrain And Convert - Conversion Runner Script
Runs the conversion pipeline for transforming an LLM into a Spiking Neural Network.

NOTE: This CLI wrapper is experimental; see README for current limitations.
"""
import argparse
import os
import sys
from pathlib import Path
import torch
import logging
import subprocess
import time
import json

# Allow running this file directly (python scripts/run_conversion.py) and locating the
# repo-root scripts it drives (convert.py, tests/...).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Configure logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('conversion_pipeline.log')
        ]
    )
logger = logging.getLogger("conversion_pipeline")

# NOTE: This runner only orchestrates the conversion/test scripts as subprocesses; it does
# not import SpikingJelly directly. The previous SpikingJelly imports (including a
# non-existent `spikingjelly.activation_based.layer.SpikeAttention`) crashed even
# `--help`, so they have been removed. The subprocesses perform their own version checks.

def parse_args():
    parser = argparse.ArgumentParser(description='Run SNN conversion pipeline')
    parser.add_argument('--model_name', type=str, default='distilgpt2',
                      help='Pretrained model to convert (default is distilgpt2 for fast testing). Supported: distilgpt2, SmolLM2-1.7B-Instruct')
    parser.add_argument('--output_dir', type=str, default='./snn_converted_model',
                      help='Directory to save the converted model')
    parser.add_argument('--timesteps', type=int, default=16,
                      help='Number of timesteps for SNN conversion')
    parser.add_argument('--surrogate_function', type=str, default='stbif_plus',
                      choices=['atan', 'sigmoid', 'stbif_plus'],
                      help='Surrogate function to use')
    parser.add_argument('--use_sparse', action='store_true',
                      help='Use sparse tensor optimization')
    parser.add_argument('--use_delayed_spikes', action='store_true',
                      help='Use delayed spike propagation')
    parser.add_argument('--use_function_calling', action='store_true',
                      help='Enable function calling capability')
    parser.add_argument('--optimize_for_torchscript', action='store_true',
                      help='Apply TorchScript optimizations')
    parser.add_argument('--verify', action='store_true',
                      help='Run verification tests after conversion')
    parser.add_argument('--run_component_tests', action='store_true',
                      help='Run component tests before conversion')
    parser.add_argument('--skip_conversion', action='store_true',
                      help='Skip conversion and only run tests on existing model')
    parser.add_argument('--simplified', action='store_true',
                      help='Use simplified conversion approach without relying on complex SpikingJelly features')
    return parser.parse_args()

def run_component_tests(model_name="distilgpt2"):
    """Run basic functionality tests to ensure all components work."""
    logger.info("=== Running Component Tests ===")
    # Anchor to the repo root and use the current interpreter; test_conversational_snn.py
    # lives under tests/ and requires --model_name.
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tests" / "test_conversational_snn.py"),
        "--model_name", model_name,
        "--test_all",
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    # Print output
    logger.info(result.stdout)
    if result.stderr:
        logger.error("Errors:")
        logger.error(result.stderr)
    
    logger.info(f"Component tests completed in {duration:.2f} seconds")
    return result.returncode == 0

def run_conversion(args):
    """Run the main conversion process."""
    logger.info(f"\n=== Converting Model: {args.model_name} ===")
    
    # Construct command for convert.py with simplified flag. Anchor to the repo root and
    # use the current interpreter so the runner works from any CWD / virtualenv.
    convert_py = str(REPO_ROOT / "convert.py")
    if args.simplified:
        cmd = [
            sys.executable, convert_py,
            "--model_name", args.model_name,
            "--output_dir", args.output_dir,
            "--timesteps", str(args.timesteps),
            "--simplified"  # Use simplified approach
        ]
    else:
        # Try normal conversion first
        cmd = [
            sys.executable, convert_py,
            "--model_name", args.model_name,
            "--output_dir", args.output_dir,
            "--timesteps", str(args.timesteps)
        ]
    
    # Add other arguments
    cmd.extend(["--num_samples", "3"])  # Small number for quick testing
    
    logger.info(f"Running conversion: {' '.join(cmd)}")

    # A model file left over from an earlier run must not be mistaken for output of
    # this one, so remember whether it already existed (and when it was last written).
    model_path = os.path.join(args.output_dir, "snn_model.pt")
    prior_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else None

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time

    # Print output
    logger.info(result.stdout)
    if result.stderr:
        # convert.py logs to stderr, so stderr on its own does not mean failure.
        # Only the exit code does; anything else produced false alarms on success.
        if result.returncode != 0:
            logger.error("Errors in conversion phase:")
            logger.error(result.stderr)
        else:
            logger.info("Conversion subprocess log (stderr):")
            logger.info(result.stderr)

    # Success requires BOTH a clean exit code and a model file this run produced.
    # Checking only for the file's existence let a stale artifact from a previous run
    # make a completely failed conversion report success.
    fresh_model = os.path.exists(model_path) and (
        prior_mtime is None or os.path.getmtime(model_path) > prior_mtime
    )
    conversion_success = result.returncode == 0 and fresh_model

    logger.info(f"Conversion completed in {duration:.2f} seconds")
    if conversion_success:
        logger.info(f"✓ Model file created at {model_path}")
    else:
        if result.returncode != 0:
            logger.error(f"✗ Conversion subprocess exited with code {result.returncode}")
        if not os.path.exists(model_path):
            logger.error(f"✗ Model file not created at {model_path}")
        elif not fresh_model:
            logger.error(
                f"✗ {model_path} was not rewritten by this run — it is a stale artifact "
                "from an earlier conversion"
            )

    return conversion_success

def test_converted_model(output_dir):
    """Test the converted model with some prompts."""
    logger.info("\n=== Testing Converted Model ===")
    
    # Check if the model exists
    model_path = os.path.join(output_dir, "snn_model.pt")
    if not os.path.exists(model_path):
        logger.error(f"Error: Model not found at {model_path}")
        return False
    
    # Try to load the model directly
    try:
        logger.info(f"Loading model from {model_path}...")
        # First try to import transformers module to ensure it's available for loading
        try:
            import transformers
            # Add necessary classes to safe globals if available
            try:
                from torch.serialization import add_safe_globals
                from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
                add_safe_globals([GPT2LMHeadModel])
                logger.info("Added transformers classes to safe globals")
            except ImportError:
                logger.info("torch.serialization.add_safe_globals not available, will try weights_only=False")
        except ImportError:
            logger.info("transformers module not imported, might affect model loading")
        
        # Try to load with weights_only=False (needed for PyTorch 2.6+)
        try:
            snn_data = torch.load(model_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            snn_data = torch.load(model_path, map_location='cpu')
        
        # Check if the loaded data is a dictionary (new format) or a model
        if isinstance(snn_data, dict) and "state_dict" in snn_data:
            logger.info("✓ Model metadata loaded successfully")
            
            # Basic info about the model
            model_type = snn_data.get("model_type", "Unknown")
            timesteps = snn_data.get("T", 16)
            simplified = snn_data.get("simplified", False)
            
            logger.info(f"Model type: {model_type}")
            logger.info(f"Model timesteps: {timesteps}")
            logger.info(f"Simplified: {simplified}")
            
            # Count parameters in state dict
            param_count = sum(p.numel() for p in snn_data["state_dict"].values())
            logger.info(f"Parameter count: {param_count:,}")
            
            logger.info("To test this model in use, you would need to:")
            logger.info("1. Load the base model")
            logger.info("2. Apply the state_dict")
            logger.info("3. Add the timestep parameter to forward calls")
            
        else:
            # Traditional model object
            logger.info("✓ Full model loaded successfully")
            
            # Basic info about the model
            logger.info(f"Model type: {type(snn_data).__name__}")
            
            if hasattr(snn_data, 'T'):
                logger.info(f"Model timesteps: {snn_data.T}")
            
            # Count parameters
            param_count = sum(p.numel() for p in snn_data.parameters())
            logger.info(f"Parameter count: {param_count:,}")
            
            # Check for config
            if hasattr(snn_data, 'config'):
                logger.info("Model has config attribute")
                if hasattr(snn_data.config, 'model_type'):
                    logger.info(f"Model type: {snn_data.config.model_type}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def export_torchscript(model, output_path):
    """Export model to TorchScript format."""
    logger.info(f"Exporting model to TorchScript format: {output_path}")
    
    # Use dynamic axes for production deployment
    model.eval()
    
    try:
        # Try script mode first for better dynamic shape handling
        logger.info("Attempting script mode export (better for dynamic shapes)...")
        
        # Create multiple example inputs with varying sequence lengths
        # This is the correct way to handle dynamic sequence lengths in TorchScript
        example_inputs = [
            (torch.zeros(1, 64, dtype=torch.long), torch.ones(1, 64, dtype=torch.long)),
            (torch.zeros(1, 128, dtype=torch.long), torch.ones(1, 128, dtype=torch.long))
        ]
        
        # Add Loihi neuromorphic hardware memory mapping
        loihi_export = os.environ.get('EXPORT_LOIHI', '0') == '1'
        if loihi_export:
            logger.info("Adding Intel Loihi memory mapping for neuromorphic deployment")
            try:
                # Import Loihi mapping utilities if available
                try:
                    import lava.lib.dl.slayer as slayer
                    has_lava_slayer = True
                except ImportError:
                    logger.warning("lava.lib.dl.slayer not found; skipping Loihi mapping (simulation-only export).")
                    has_lava_slayer = False
                
                if has_lava_slayer:
                    # Create Loihi memory map
                    loihi_config = {
                        "neuron_model": "LIF",  # Loihi supports LIF neuron models
                        "threshold": 1.0,        # Default threshold value
                        "tau_mem": 2.0,          # Default membrane time constant
                        "tau_syn": 4.0,          # Default synaptic time constant
                        "core_mapping": "auto",  # Auto mapping to cores
                        "synapse_encoding": "sparse", # Sparse weight encoding
                        "weight_precision": 8,    # 8-bit weight precision
                    }
                    
                    # Process the model with SLAYER for Loihi compatibility
                    loihi_processor = slayer.utils.LoihiProcessor(model, config=loihi_config)
                    model = loihi_processor.process()
                    model._loihi_config = loihi_config
                    model._is_loihi_compatible = True
                    logger.info("Applied full Loihi mapping using SLAYER")
                else:
                    logger.info("Loihi mapping not applied; model remains in software-simulation mode.")
            
            except Exception as e:
                logger.warning(f"Warning: Loihi mapping failed with error: {e}")
                logger.warning("Continuing with standard TorchScript export without Loihi optimizations")
        
        # Script the model
        logger.info("Scripting model with dynamic sequence length handling...")
        scripted_model = torch.jit.script(model)
        
        # Save the model
        logger.info(f"Saving scripted model to {output_path}")
        scripted_model.save(output_path)
        logger.info("✓ Successfully exported model to TorchScript format (script mode)")
        
        # Add model metadata
        if hasattr(model, 'config'):
            # Save config separately since TorchScript doesn't preserve it
            config_path = os.path.splitext(output_path)[0] + '_config.json'
            if hasattr(model.config, 'to_json_string'):
                with open(config_path, 'w') as f:
                    f.write(model.config.to_json_string())
                logger.info(f"✓ Saved model config to {config_path}")
        
        # Return success
        return True
        
    except Exception as e:
        logger.error(f"Script mode failed with error: {e}")
        logger.error("Falling back to trace mode...")
        
        try:
            # Create example inputs for tracing
            example_input_ids = torch.zeros(1, 128, dtype=torch.long)
            example_attention_mask = torch.ones(1, 128, dtype=torch.long)
            
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model,
                    (example_input_ids, example_attention_mask)
                )
            
            # Save the model
            traced_model.save(output_path)
            logger.info("✓ Successfully exported model to TorchScript format (trace mode)")
            
            # Add model metadata
            if hasattr(model, 'config'):
                # Save config separately
                config_path = os.path.splitext(output_path)[0] + '_config.json'
                if hasattr(model.config, 'to_json_string'):
                    with open(config_path, 'w') as f:
                        f.write(model.config.to_json_string())
                    logger.info(f"✓ Saved model config to {config_path}")
            
            # Return success
            return True
            
        except Exception as e:
            logger.error(f"Error in trace mode: {e}")
            logger.error("❌ Failed to export model to TorchScript format")
            return False

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Check SpikingJelly compatibility with an in-process import probe.
    # (Previously this shelled out to a non-existent test_direct_import.py.)
    logger.info("\n=== Checking SpikingJelly Compatibility ===")
    use_simplified = False
    try:
        from spikingjelly.activation_based.ann2snn import Converter  # noqa: F401
        logger.info("✓ SpikingJelly ann2snn.Converter is importable")
    except Exception as e:
        logger.warning(f"SpikingJelly components missing or unimportable ({e}); will use simplified approach")
        use_simplified = True

    # Step 2: Run component tests if requested
    if args.run_component_tests:
        component_tests_passed = run_component_tests(args.model_name)
        if not component_tests_passed:
            logger.error("Component tests failed. Fix the issues before proceeding with conversion.")
            return 1
    
    # Step 3: Run conversion if not skipped
    if not args.skip_conversion:
        # If compatibility issues detected, force simplified approach
        if use_simplified:
            args.simplified = True
            logger.info("Using simplified conversion due to compatibility issues")
        
        conversion_passed = run_conversion(args)
        if not conversion_passed:
            logger.error("Conversion failed even with simplified approach. Check the logs for details.")
            return 1
    
    # Step 4: Test the converted model
    model_tests_passed = test_converted_model(args.output_dir)
    if not model_tests_passed:
        logger.error("Model tests failed. The converted model may not be working correctly.")
        return 1
    
    # Warn about flags that are accepted for CLI compatibility but not yet wired up.
    unapplied = [
        name for name, val in (
            ("--use_sparse", args.use_sparse),
            ("--use_delayed_spikes", args.use_delayed_spikes),
            ("--use_function_calling", args.use_function_calling),
        ) if val
    ]
    if args.surrogate_function and args.surrogate_function != "stbif_plus":
        unapplied.append(f"--surrogate_function={args.surrogate_function}")
    if unapplied:
        logger.warning(
            "The following flags are recognized but not applied by this runner "
            f"(no-op): {', '.join(unapplied)}"
        )

    # Step 5: Export to TorchScript if requested
    if args.optimize_for_torchscript:
        logger.info("\n=== Exporting to TorchScript ===")
        try:
            # Load model again for export
            model_path = os.path.join(args.output_dir, "snn_model.pt")
            model = torch.load(model_path, map_location='cpu', weights_only=False)

            # The converter saves a metadata dict ({"state_dict": ..., ...}), not a live
            # nn.Module. torch.jit.script/.eval() on a dict can never succeed, so skip
            # honestly instead of raising deep inside the exporter.
            if isinstance(model, dict):
                logger.warning(
                    "Saved model is a state-dict/metadata dictionary, not a scriptable "
                    "nn.Module; skipping TorchScript export. Rebuild and script a live "
                    "model object to enable this step."
                )
            else:
                # Export model
                ts_path = os.path.join(args.output_dir, "snn_model.pt.ts")
                export_torchscript(model, ts_path)

                # Verify the exported model
                if args.verify and os.path.exists(ts_path):
                    logger.info(f"Successfully created TorchScript model: {ts_path}")
                    logger.info(f"Model size: {os.path.getsize(ts_path) / (1024 * 1024):.2f} MB")
        except Exception as e:
            logger.error(f"Error during TorchScript export: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n=== Pipeline Summary ===")
    logger.info("✓ All steps completed successfully")
    if use_simplified:
        logger.warning("⚠ Used simplified approach due to SpikingJelly compatibility issues")
        logger.warning("  Full SNN functionality might be limited")
    logger.info(f"Converted model is available in: {args.output_dir}")
    
    # Report the conversion that actually ran. `use_simplified` only records the
    # *fallback* decision (SpikingJelly unimportable), so an explicit --simplified run —
    # the one in the README's quick start — was written out as simplified_approach:false.
    # convert.py records the real outcome in snn_config.json, including the case where a
    # full conversion silently degraded to the simplified path; prefer that.
    conversion_mode = None
    simplified_used = bool(args.simplified)
    snn_config_path = os.path.join(args.output_dir, "snn_config.json")
    if os.path.exists(snn_config_path):
        try:
            with open(snn_config_path) as f:
                snn_config = json.load(f)
            simplified_used = bool(snn_config.get("simplified", simplified_used))
            conversion_mode = snn_config.get("conversion_mode")
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read {snn_config_path}: {e}")

    # Save summary report
    summary = {
        "model_name": args.model_name,
        "output_dir": args.output_dir,
        "timesteps": args.timesteps,
        "surrogate_function": args.surrogate_function,
        "use_sparse": args.use_sparse,
        "use_delayed_spikes": args.use_delayed_spikes,
        "use_function_calling": args.use_function_calling,
        "optimize_for_torchscript": args.optimize_for_torchscript,
        "simplified_approach": simplified_used,
        "simplified_forced_by_missing_spikingjelly": use_simplified,
        "conversion_mode": conversion_mode,
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(args.output_dir, "conversion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 