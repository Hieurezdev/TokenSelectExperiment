"""
Capture và phân tích attention weights.

Usage:
    python capture_attention_example.py --mode local    # Chạy inference và capture attention
    python capture_attention_example.py --mode analyze  # Phân tích attention đã save
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_local():
    """Chạy inference và capture attention."""
    print("Loading model...")
    model_name = "Qwen/Qwen2-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",  # Required to get attention weights
    )
    
    # Prompt
    key = "Quang cười haha"
    n_repeat = 50
    prompt = (
        "The grass is green. The sky is blue. " * n_repeat +
        f"The pass key is {key}. Remember it. " +
        "The grass is green. The sky is blue. " * n_repeat +
        "What is the pass key?"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input tokens: {inputs.input_ids.shape[1]}")
    
    # Generate with attention capture
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    
    if outputs.attentions:
        attentions = outputs.attentions[0]  # Prefill step
        print(f"Layers: {len(attentions)}, Shape: {attentions[0].shape}")
        
        torch.save(outputs.attentions, "attention_weights.pt")
        print("✓ Saved to attention_weights.pt")
        
        visualize(attentions, layer_idx=0, head_idx=0)
    
    print(f"\nOutput: {tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)}")


def visualize(attentions, layer_idx=0, head_idx=0):
    """Visualize attention heatmap."""
    try:
        import matplotlib.pyplot as plt
        attn = attentions[layer_idx][0, head_idx].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(attn, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(f'Layer {layer_idx}, Head {head_idx}')
        plt.savefig('attention_heatmap.png', dpi=150)
        plt.close()
        print("✓ Saved attention_heatmap.png")
    except ImportError:
        print("pip install matplotlib to visualize")


def analyze():
    """Phân tích attention đã save."""
    attentions = torch.load("attention_weights.pt")
    prefill = attentions[0]
    
    print(f"Steps: {len(attentions)}, Layers: {len(prefill)}")
    
    for i in [0, len(prefill)//2, len(prefill)-1]:
        attn = prefill[i]
        entropy = -(attn * torch.log(attn + 1e-10)).sum(-1).mean()
        sparsity = (attn < 0.01).float().mean() * 100
        print(f"Layer {i}: shape={attn.shape}, entropy={entropy:.2f}, sparsity={sparsity:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "analyze"], default="local")
    args = parser.parse_args()
    
    if args.mode == "local":
        run_local()
    else:
        analyze()
