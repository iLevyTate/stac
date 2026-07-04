import os
import sys
import tempfile
from pathlib import Path

# Allow running this file directly (python tests/test_v1.py) by putting the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer

from stac_v1.model import AdExParams, DLPFCAdExNeuron, DLPFCLayer, DLPFCTransformer, HyperdimensionalMemoryModule, surrogate_spike
from stac_v1.pipeline import STACV1Config, build_dataloader_from_texts, build_model_and_tokenizer, freeze_for_hybrid_finetune, set_seed, train_steps


print("--- Setting up STAC V1 Tests (imported implementation) ---")

TEST_CFG = STACV1Config(
    model_name="sshleifer/tiny-gpt2",
    seq_length=16,
    dlpfc_output_size=8,
    num_recurrent_layers=1,
    dropout_prob=0.1,
    hdm_dim=16,
    warmup_steps=2,
    output_dir=os.path.join(tempfile.gettempdir(), "test_stac_v1_output"),
)

set_seed(TEST_CFG.seed)
test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(TEST_CFG.output_dir, exist_ok=True)

# --- Test Functions ---

def test_surrogate_spike():
    print("Testing SurrogateSpikeFunction...")
    # Ensure this is a *leaf* tensor (avoid autograd warning on .grad access).
    input_tensor = (torch.randn(5, device=test_device) * 0.5).requires_grad_()
    spikes = surrogate_spike(input_tensor)  # Non-leaf tensor
    assert spikes.shape == input_tensor.shape, "Forward shape mismatch"
    assert spikes.dtype == torch.float32, "Forward output dtype mismatch"
    assert torch.all((spikes == 0) | (spikes == 1)), "Forward output not 0 or 1"
    print("  Forward pass OK.")
    dummy_grad = torch.ones_like(spikes)
    try:
        spikes.backward(dummy_grad)
        print("  Backward pass executed without error.")
    except Exception as e:
        raise AssertionError(f"Backward pass failed with error: {e}")
    # Check gradient properties on the leaf tensor AFTER backward pass
    assert input_tensor.grad is not None, "Expected gradient on leaf tensor after backward()"
    assert input_tensor.grad.shape == input_tensor.shape, f"Backward grad shape mismatch: {input_tensor.grad.shape}"
    assert input_tensor.grad.dtype == input_tensor.dtype, f"Backward grad dtype mismatch: {input_tensor.grad.dtype}"
    print("  Gradient shape and type on leaf tensor OK.")
    print("SurrogateSpikeFunction Test PASSED.")

def test_adex_neuron():
    print("Testing DLPFCAdExNeuron...")
    batch_size = 2
    output_size = TEST_CFG.dlpfc_output_size
    neuron = DLPFCAdExNeuron(AdExParams()).to(test_device)
    input_current = torch.randn(batch_size, output_size, device=test_device) * 10
    V_init = torch.full((batch_size, output_size), neuron.V_reset.item(), device=test_device)
    w_init = torch.zeros(batch_size, output_size, device=test_device)
    spike, V_next, w_next = neuron(input_current, V_init, w_init)
    assert spike.shape == (batch_size, output_size), f"Spike shape: {spike.shape}"
    assert V_next.shape == (batch_size, output_size), f"V_next shape: {V_next.shape}"
    assert w_next.shape == (batch_size, output_size), f"w_next shape: {w_next.shape}"
    assert spike.dtype == torch.float32
    assert V_next.dtype == torch.float32
    assert w_next.dtype == torch.float32
    print("  Output shapes and dtypes OK.")
    params = list(neuron.parameters())
    assert len(params) > 0, "No parameters registered"
    print(f"  Expected device type: {test_device.type}, index: {test_device.index}")
    all_on_device = True
    for name, p in neuron.named_parameters():
        param_device = p.device
        print(f"  Param '{name}' device: {param_device}")
        if param_device.type != test_device.type:
            all_on_device = False
            print(f"  !!! Type mismatch for '{name}': {param_device.type} != {test_device.type}")
            break
        if test_device.type == 'cuda':
            expected_index = test_device.index if test_device.index is not None else 0
            actual_index = p.device.index if p.device.index is not None else 0
            if expected_index != actual_index:
                all_on_device = False
                print(f"  !!! Index mismatch for '{name}': {actual_index} != {expected_index}")
                break
    assert all_on_device, "One or more parameters were not moved to the correct device"
    print("  Parameters registered and on correct device.")
    print("DLPFCAdExNeuron Test PASSED.")

def test_dlpfc_layer():
    print("Testing DLPFCLayer...")
    batch_size = 2
    seq_len = TEST_CFG.seq_length
    model, _tok = build_model_and_tokenizer(TEST_CFG)
    input_size = int(model.gpt2.config.hidden_size)
    output_size = TEST_CFG.dlpfc_output_size
    layer = DLPFCLayer(
        input_size,
        output_size,
        num_recurrent_layers=TEST_CFG.num_recurrent_layers,
        adex_params=AdExParams(),
        dropout_prob=TEST_CFG.dropout_prob,
    ).to(test_device)
    layer.eval()
    hidden_states = torch.randn(batch_size, seq_len, input_size, device=test_device)
    with torch.no_grad():
        spk_trains = layer(hidden_states)
    expected_shape = (batch_size, seq_len, output_size)
    assert spk_trains.shape == expected_shape, f"Output shape mismatch: {spk_trains.shape} vs {expected_shape}"
    assert spk_trains.dtype == torch.float32, f"Output dtype mismatch: {spk_trains.dtype}"
    print("  Output shape and dtype OK.")
    print("DLPFCLayer Test PASSED.")

def test_memory_module():
    print("Testing HyperdimensionalMemoryModule...")
    batch_size = 2
    seq_len = TEST_CFG.seq_length
    input_dim = TEST_CFG.dlpfc_output_size
    hdm_dim = TEST_CFG.hdm_dim
    output_dim = TEST_CFG.dlpfc_output_size
    module = HyperdimensionalMemoryModule(input_dim, hdm_dim, output_dim).to(test_device)
    module.eval()
    spike_train = torch.randint(0, 2, (batch_size, seq_len, input_dim), dtype=torch.float, device=test_device)
    with torch.no_grad():
        memory_bias = module(spike_train)
    expected_shape = (batch_size, output_dim)
    assert memory_bias.shape == expected_shape, f"Output shape mismatch: {memory_bias.shape} vs {expected_shape}"
    assert memory_bias.dtype == torch.float32, f"Output dtype mismatch: {memory_bias.dtype}"
    print("  Output shape and dtype OK.")
    print("HyperdimensionalMemoryModule Test PASSED.")

def test_dlpfc_transformer():
    print("Testing DLPFCTransformer (Full Model Forward Pass)...")
    batch_size = 2
    seq_len = TEST_CFG.seq_length
    model, _tok = build_model_and_tokenizer(TEST_CFG)
    model = model.to(test_device)
    model.eval()
    vocab_size = int(model.gpt2.config.vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=test_device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=test_device)
    with torch.no_grad():
        logits, spk_trains = model(input_ids, attention_mask=attention_mask)
    expected_logits_shape = (batch_size, seq_len, vocab_size)
    expected_spk_trains_shape = (batch_size, seq_len, TEST_CFG.dlpfc_output_size)
    assert logits.shape == expected_logits_shape, f"Logits shape mismatch: {logits.shape} vs {expected_logits_shape}"
    assert spk_trains.shape == expected_spk_trains_shape, f"Spike trains shape mismatch: {spk_trains.shape} vs {expected_spk_trains_shape}"
    assert logits.dtype == torch.float32
    assert spk_trains.dtype == torch.float32
    print("  Output shapes and dtypes OK.")
    try:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        criterion = nn.CrossEntropyLoss()
        loss_xent = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_l1 = TEST_CFG.l1_lambda * torch.mean(torch.abs(spk_trains))
        total_loss = loss_xent + loss_l1
        print("  Loss calculation structure compatible with output shapes.")
    except Exception as e:
        raise AssertionError(f"Failed during simulated loss calculation: {e}")
    print("DLPFCTransformer Test PASSED.")

def test_data_pipeline():
    print("Testing Data Pipeline (Tokenization and DataLoader)...")
    dummy_texts = ["Sentence one.", "Sentence two is longer.", "Short.", "=Title="]
    dummy_texts_filtered = [text for text in dummy_texts if len(text.strip()) > 0]
    class DummyTextDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            return {"text": self.texts[idx]}
    dummy_dataset = DummyTextDataset(dummy_texts_filtered)
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(TEST_CFG.model_name)
    except Exception as e:
        raise AssertionError(f"Failed to load tokenizer for test: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function_test(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=TEST_CFG.seq_length)
    tokenized_data = [tokenize_function_test({"text": t}) for t in dummy_dataset.texts]
    for item in tokenized_data:
        item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long)
        item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.long)
    test_loader = DataLoader(tokenized_data, batch_size=2)
    try:
        batch = next(iter(test_loader))
    except Exception as e:
        raise AssertionError(f"Failed to get batch from DataLoader: {e}")
    assert 'input_ids' in batch and 'attention_mask' in batch
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    expected_batch_size = min(2, len(tokenized_data))
    expected_shape = (expected_batch_size, TEST_CFG.seq_length)
    assert input_ids.shape == expected_shape, f"Batch input_ids shape: {input_ids.shape} vs {expected_shape}"
    assert attention_mask.shape == expected_shape, f"Batch attention_mask shape: {attention_mask.shape} vs {expected_shape}"
    assert input_ids.dtype == torch.long and attention_mask.dtype == torch.long
    print("  Tokenization and DataLoader batch structure OK.")
    print("Data Pipeline Test PASSED.")

def test_hybrid_finetune_freeze():
    print("Testing hybrid fine-tuning freeze behavior...")
    model, _tok = build_model_and_tokenizer(TEST_CFG)
    freeze_for_hybrid_finetune(model, freeze_backbone=True, train_lm_head=True)

    gpt2_trainable = sum(1 for p in model.gpt2.parameters() if p.requires_grad)
    dlpfc_trainable = sum(1 for p in model.dlpfc.parameters() if p.requires_grad)
    assert gpt2_trainable == 0, "GPT-2 backbone should be frozen in hybrid fine-tuning mode"
    assert dlpfc_trainable > 0, "DLPFC should remain trainable in hybrid fine-tuning mode"
    print("Hybrid fine-tuning freeze Test PASSED.")


def test_pipeline_smoke():
    print("Testing end-to-end pipeline smoke run (few steps)...")
    model, tok = build_model_and_tokenizer(TEST_CFG)
    loader = build_dataloader_from_texts(
        tok,
        ["Hello world.", "Spiking transformers can be trained with surrogate gradients."],
        seq_length=TEST_CFG.seq_length,
        batch_size=2,
        shuffle=True,
    )
    metrics = train_steps(
        model,
        loader,
        cfg=TEST_CFG,
        device=test_device,
        max_steps=2,
        hybrid_finetune=True,
        train_lm_head=True,
        write_loihi_report=True,
    )
    assert metrics["steps"] > 0
    assert "train_loss" in metrics
    assert "loihi_export_ready" in metrics  # may be 0 due to dense attention; presence is what we care about
    print("Pipeline smoke Test PASSED.")

# --- Test Runner ---
def run_all_tests():
    print("\n--- Running All Tests ---")
    tests_passed = 0
    tests_failed = 0
    test_functions = [
        test_surrogate_spike,
        test_adex_neuron,
        test_dlpfc_layer,
        test_memory_module,
        test_dlpfc_transformer,
        test_data_pipeline,
        test_hybrid_finetune_freeze,
        test_pipeline_smoke,
    ]
    for test_func in test_functions:
        try:
            test_func()
            tests_passed += 1
        except AssertionError as e:
            print(f"!!! Test Failed: {test_func.__name__} !!!\n  Error: {e}")
            tests_failed += 1
        except Exception as e:
            import traceback
            print(f"!!! Test Errored: {test_func.__name__} !!!\n  Unexpected Error: {e}")
            traceback.print_exc()
            tests_failed += 1
        print("-" * 30)
    print("\n--- Test Summary ---")
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print("--- End of Tests ---")
    if tests_failed == 0:
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Please review the errors above.")
    return tests_failed


# --- Execute Tests ---
# When run as a script, run the whole suite and set the exit code from the failure count.
# Under pytest, the individual `test_*` functions above are collected and run directly.
if __name__ == "__main__":
    sys.exit(1 if run_all_tests() else 0)

