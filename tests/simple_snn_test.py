#!/usr/bin/env python
"""
Simple SNN Test - Test basic functionality of SNN conversion
"""
import os
import torch
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from smollm2_converter import replace_gelu_with_relu, simplified_conversion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simple_snn_test.log')
    ]
)
logger = logging.getLogger("simple_snn_test")

def main():
    # Parameters
    model_name = "distilgpt2"
    timesteps = 16
    test_prompt = "Artificial intelligence is"
    output_dir = "simple_test_output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Step 1: Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Fix for tokenizer which doesn't have a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer pad_token to eos_token")
        
        # Step 2: Run baseline inference
        logger.info("Running baseline inference with original model")
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        model = model.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        predicted_token = tokenizer.decode([next_token_id])
        
        logger.info(f"Original model predicted: '{predicted_token}' after '{test_prompt}'")
        
        # Step 3: Replace GeLU with ReLU
        logger.info("Replacing GeLU activations with ReLU")
        model_relu = replace_gelu_with_relu(model)
        
        # Run inference with ReLU model
        with torch.no_grad():
            outputs_relu = model_relu(**inputs)
        
        next_token_logits_relu = outputs_relu.logits[0, -1, :]
        next_token_id_relu = torch.argmax(next_token_logits_relu).item()
        predicted_token_relu = tokenizer.decode([next_token_id_relu])
        
        logger.info(f"ReLU model predicted: '{predicted_token_relu}' after '{test_prompt}'")
        
        # Step 4: Convert to SNN
        logger.info(f"Converting to SNN with T={timesteps}")
        try:
            model_relu.T = timesteps
            snn_model = simplified_conversion(model_relu, timesteps)
            snn_model = snn_model.to(device)
            logger.info("SNN conversion successful")
            
            # Step 5: Run inference with SNN model
            logger.info("Running inference with SNN model")
            with torch.no_grad():
                outputs_snn = snn_model(**inputs)
            
            next_token_logits_snn = outputs_snn.logits[0, -1, :] if hasattr(outputs_snn, 'logits') else outputs_snn[0, -1, :]
            next_token_id_snn = torch.argmax(next_token_logits_snn).item()
            predicted_token_snn = tokenizer.decode([next_token_id_snn])
            
            logger.info(f"SNN model predicted: '{predicted_token_snn}' after '{test_prompt}'")
            
            # Step 6: Compare results
            logger.info("\nPrediction comparison:")
            logger.info(f"Original model: '{predicted_token}'")
            logger.info(f"ReLU model:     '{predicted_token_relu}'")
            logger.info(f"SNN model:      '{predicted_token_snn}'")
            
            if predicted_token_snn == predicted_token_relu:
                logger.info("✅ SNN model prediction matches ReLU model!")
            else:
                logger.info("⚠️ SNN model prediction differs from ReLU model")
            
            # Save the SNN model (optional)
            # torch.save(snn_model.state_dict(), os.path.join(output_dir, "snn_model.pt"))
            # logger.info(f"SNN model saved to {output_dir}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Error during SNN conversion: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
    except Exception as e:
        logger.error(f"Error during model loading or inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 