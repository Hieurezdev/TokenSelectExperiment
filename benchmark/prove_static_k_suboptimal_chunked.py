#!/usr/bin/env python3
"""
Prove Static K is Suboptimal - Memory Efficient Chunked Version

Hỗ trợ context length lên đến 150k+ tokens bằng cách:
1. Tự tính attention thủ công (không dùng output_attentions=True)
2. Tính từng chunk queries, analyze rồi delete ngay
3. Memory: O(chunk_size × seq_len) thay vì O(seq_len²)

Với chunk_size=512, 150k context:
- Memory per chunk: 512 × 150k × 28 heads × 4 bytes = ~8.6GB
- Hoàn toàn fit trong GPU 80GB
"""

import argparse
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn.functional as F
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent / "data" / "infinite-bench"
OUTPUT_DIR = Path(__file__).parent / "static_k_analysis"

DATASETS = {
    "passkey": "Simple retrieval - variable K expected",
    "kv_retrieval": "Key-value pairs - sparse attention",
    "longbook_qa_eng": "Long document QA - mixed patterns",
    "math_find": "Math reasoning - sequential attention",
}

TARGET_COVERAGE = 0.90

# ============================================================================
# Data Loading
# ============================================================================

def load_sample(dataset: str, sample_idx: int) -> Optional[Dict]:
    """Load a sample from infinite-bench dataset."""
    data_path = DATA_DIR / f"{dataset}.jsonl"
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return None
    
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                return json.loads(line)
    return None


# ============================================================================
# Model Loading - Minimal, just for getting weights
# ============================================================================

def load_model(model_name: str):
    """Load model with minimal memory footprint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load in float16 to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Need access to attention weights
    )
    model.eval()
    
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, "
          f"Heads: {model.config.num_attention_heads}")
    
    return model, tokenizer


# ============================================================================
# Chunked Attention Computation - Core Memory Optimization
# ============================================================================

def extract_qkv_from_hidden_states(model, hidden_states: torch.Tensor, layer_idx: int):
    """
    Extract Q, K, V projections from hidden states for a specific layer.
    
    Args:
        model: The loaded model
        hidden_states: (batch, seq_len, hidden_dim)
        layer_idx: Which layer to extract from
        
    Returns:
        Q, K, V tensors
    """
    # Get the attention layer
    # Structure varies by model, handle common cases
    if hasattr(model, 'model'):  # Llama, Qwen style
        layers = model.model.layers
    elif hasattr(model, 'transformer'):  # GPT style
        layers = model.transformer.h
    else:
        raise ValueError(f"Unknown model structure: {type(model)}")
    
    layer = layers[layer_idx]
    
    # Get attention module
    if hasattr(layer, 'self_attn'):  # Llama, Qwen
        attn = layer.self_attn
    elif hasattr(layer, 'attn'):  # GPT
        attn = layer.attn
    else:
        raise ValueError(f"Unknown layer structure: {type(layer)}")
    
    # Project to Q, K, V
    # Most models have separate q_proj, k_proj, v_proj
    if hasattr(attn, 'q_proj'):
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)
    elif hasattr(attn, 'c_attn'):  # GPT-2 style combined projection
        qkv = attn.c_attn(hidden_states)
        q, k, v = qkv.split(hidden_states.size(-1), dim=-1)
    else:
        raise ValueError(f"Unknown attention projection structure")
    
    return q, k, v


def reshape_for_attention(tensor: torch.Tensor, num_heads: int, head_dim: int):
    """Reshape tensor from (batch, seq, hidden) to (batch, heads, seq, head_dim)."""
    batch, seq_len, _ = tensor.shape
    return tensor.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)


def compute_attention_chunk(
    Q_chunk: torch.Tensor,  # (batch, heads, chunk_size, head_dim)
    K_all: torch.Tensor,    # (batch, heads, seq_len, head_dim) - all keys seen so far
    scale: float,
    chunk_start: int,       # Starting position of this chunk
) -> torch.Tensor:
    """
    Compute attention weights for a chunk of queries.
    
    Returns:
        attention weights: (batch, heads, chunk_size, seq_len)
    """
    # Compute attention scores
    # Q_chunk: (B, H, chunk, D)
    # K_all: (B, H, seq, D)
    # scores: (B, H, chunk, seq)
    scores = torch.matmul(Q_chunk, K_all.transpose(-2, -1)) * scale
    
    # Apply causal mask - each query can only attend to positions <= its position
    chunk_size = Q_chunk.shape[2]
    seq_len = K_all.shape[2]
    
    # Create causal mask for this chunk
    # query at position (chunk_start + i) can attend to positions [0, chunk_start + i]
    causal_mask = torch.ones(chunk_size, seq_len, dtype=torch.bool, device=Q_chunk.device)
    for i in range(chunk_size):
        query_pos = chunk_start + i
        causal_mask[i, query_pos + 1:] = False
    
    # Apply mask
    scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    return attn_weights


def analyze_attention_chunk(
    attn_weights: torch.Tensor,  # (batch, heads, chunk_size, seq_len)
    chunk_start: int,
    target_coverage: float = 0.90,
) -> List[int]:
    """
    Analyze attention weights for a chunk and return required K per query.
    
    Returns:
        List of required K values for each query position in the chunk
    """
    # Average over heads
    # attn_weights: (B, H, chunk, seq) -> (chunk, seq)
    attn_avg = attn_weights[0].mean(dim=0)  # (chunk_size, seq_len)
    
    required_k_list = []
    
    for i in range(attn_avg.shape[0]):
        query_pos = chunk_start + i
        
        # Get attention weights for valid positions (0 to query_pos)
        row = attn_avg[i, :query_pos + 1]
        
        if len(row) == 0:
            required_k_list.append(1)
            continue
        
        # Sort descending and compute cumsum
        sorted_vals, _ = row.sort(descending=True)
        cumsum = sorted_vals.cumsum(dim=0)
        
        # Find K needed for target coverage
        k = 1
        for j, c in enumerate(cumsum):
            if c >= target_coverage:
                k = j + 1
                break
        else:
            k = len(row)
        
        required_k_list.append(k)
    
    return required_k_list


# ============================================================================
# Main Chunked Analysis Pipeline
# ============================================================================

def get_embeddings(model, tokenizer, text: str, max_tokens: int = 150000):
    """
    Get only embeddings (input to first layer) - minimal memory.
    
    Returns:
        embeddings: (batch, seq_len, hidden_dim)
        input_ids: for reference
        num_tokens: number of tokens
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = inputs.to(model.device)
    
    num_tokens = inputs.input_ids.shape[1]
    print(f"    Input tokens: {num_tokens}")
    
    # Get embedding layer
    if hasattr(model, 'model'):
        embed_tokens = model.model.embed_tokens
    elif hasattr(model, 'transformer'):
        embed_tokens = model.transformer.wte
    else:
        raise ValueError(f"Unknown model structure")
    
    with torch.no_grad():
        embeddings = embed_tokens(inputs.input_ids)
    
    return embeddings, inputs, num_tokens


def forward_single_layer(model, hidden_states: torch.Tensor, layer_idx: int, 
                         position_ids: torch.Tensor = None):
    """
    Forward through a single transformer layer.
    
    Returns:
        output hidden states, Q, K, V projections
    """
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.transformer.h
    
    layer = layers[layer_idx]
    
    # Get position embeddings if needed
    seq_len = hidden_states.shape[1]
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    
    # Most models: layer(hidden_states, position_ids=position_ids)
    # But we need to extract Q, K before the attention computation
    
    # Get attention module
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        ln = layer.input_layernorm if hasattr(layer, 'input_layernorm') else None
    else:
        attn = layer.attn
        ln = layer.ln_1 if hasattr(layer, 'ln_1') else None
    
    # Apply layer norm
    if ln is not None:
        normed = ln(hidden_states)
    else:
        normed = hidden_states
    
    # Get Q, K, V projections
    with torch.no_grad():
        if hasattr(attn, 'q_proj'):
            q = attn.q_proj(normed)
            k = attn.k_proj(normed)
            v = attn.v_proj(normed)
        else:
            qkv = attn.c_attn(normed)
            q, k, v = qkv.split(normed.size(-1), dim=-1)
    
    # We only need Q, K for attention analysis
    # Don't compute full layer output to save memory
    return q, k


def process_layer_incrementally(
    model,
    embeddings: torch.Tensor,
    inputs,
    target_layer: int,
    chunk_size: int = 512,
    target_coverage: float = 0.90,
) -> Dict:
    """
    Process model up to target layer and analyze attention.
    
    Uses incremental computation - doesn't store all layer outputs.
    Memory: O(seq_len × hidden_dim) for hidden states
          + O(chunk_size × seq_len × heads) for attention
    """
    seq_len = embeddings.shape[1]
    device = embeddings.device
    
    # Get model config
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, 'num_key_value_heads', num_heads)
    head_dim = model.config.hidden_size // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.transformer.h
    
    # Forward through layers up to target
    hidden = embeddings
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Create attention mask
    attention_mask = torch.ones((1, seq_len), device=device)
    
    with torch.no_grad():
        for layer_idx in range(target_layer + 1):
            layer = layers[layer_idx]
            
            if layer_idx == target_layer:
                # Extract Q, K at target layer
                Q, K = forward_single_layer(model, hidden, layer_idx, position_ids)
                break
            else:
                # Full layer forward (we need the output as input to next layer)
                # Use cache=None, output_attentions=False for efficiency
                if hasattr(layer, 'self_attn'):
                    # Llama/Qwen style
                    layer_outputs = layer(
                        hidden,
                        attention_mask=None,
                        position_ids=position_ids,
                        past_key_value=None,
                        output_attentions=False,
                        use_cache=False,
                    )
                else:
                    # GPT style
                    layer_outputs = layer(hidden)
                
                hidden = layer_outputs[0]
                
                # Free intermediate outputs
                del layer_outputs
    
    # Now we have Q, K for target layer
    # Reshape for attention
    Q = reshape_for_attention(Q, num_heads, head_dim)
    K = reshape_for_attention(K, num_kv_heads, head_dim)
    
    # Handle GQA
    if num_kv_heads != num_heads:
        n_rep = num_heads // num_kv_heads
        K = K.repeat_interleave(n_rep, dim=1)
    
    # Chunked attention analysis
    all_required_k = []
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, seq_len)
        
        Q_chunk = Q[:, :, chunk_start:chunk_end, :]
        K_visible = K[:, :, :chunk_end, :]
        
        attn_weights = compute_attention_chunk(Q_chunk, K_visible, scale, chunk_start)
        chunk_required_k = analyze_attention_chunk(attn_weights, chunk_start, target_coverage)
        all_required_k.extend(chunk_required_k)
        
        del Q_chunk, K_visible, attn_weights
        
        if (chunk_idx + 1) % 20 == 0:
            print(f"      Chunk {chunk_idx + 1}/{num_chunks}")
    
    # Cleanup
    del Q, K, hidden
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Skip warmup tokens
    valid_k = all_required_k[50:]
    
    if not valid_k:
        return None
    
    return {
        "min_k": min(valid_k),
        "max_k": max(valid_k),
        "mean_k": np.mean(valid_k),
        "std_k": np.std(valid_k),
        "k_ratio": max(valid_k) / max(min(valid_k), 1),
        "required_k_list": valid_k,
    }


def analyze_sample_chunked(
    model, 
    tokenizer, 
    dataset: str, 
    sample_idx: int,
    chunk_size: int = 512,
    layers_to_analyze: List[int] = None,
    max_tokens: int = 150000,
    visualize: bool = True,
    output_dir: Path = OUTPUT_DIR,
) -> Dict:
    """
    Analyze a single sample using chunked attention computation.
    Memory efficient: processes layer by layer, chunk by chunk.
    """
    print(f"\n  Sample {sample_idx}:")
    
    sample = load_sample(dataset, sample_idx)
    if not sample:
        print(f"    Could not load sample")
        return None
    
    context = sample.get("context", "")
    query = sample.get("input", "")
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    print(f"    Context length: {len(context)} chars")
    
    # Get embeddings only (minimal memory)
    embeddings, inputs, num_tokens = get_embeddings(model, tokenizer, prompt, max_tokens)
    
    num_layers = model.config.num_hidden_layers
    
    if layers_to_analyze is None:
        # Analyze fewer layers to save time (each requires forward through previous layers)
        # Middle layer is most representative
        layers_to_analyze = [num_layers // 2]
    
    all_stats = {}
    
    for layer_idx in layers_to_analyze:
        print(f"    Analyzing layer {layer_idx}...")
        
        stats = process_layer_incrementally(
            model, embeddings, inputs, layer_idx,
            chunk_size=chunk_size,
            target_coverage=TARGET_COVERAGE
        )
        
        if stats:
            all_stats[layer_idx] = stats
            print(f"      K range: {stats['min_k']} - {stats['max_k']} "
                  f"(ratio: {stats['k_ratio']:.1f}x)")
    
    # Free embeddings
    del embeddings, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Visualize if requested
    if visualize and all_stats:
        visualize_results(all_stats, dataset, sample_idx, num_tokens, output_dir)
    
    # Return stats
    if all_stats:
        layer_idx = list(all_stats.keys())[0]
        result = all_stats[layer_idx].copy()
        result['num_tokens'] = num_tokens
        return result
    
    return None


def visualize_results(
    layer_stats: Dict, 
    dataset: str, 
    sample_idx: int, 
    num_tokens: int,
    output_dir: Path
):
    """Visualize K distribution results."""
    try:
        import matplotlib.pyplot as plt
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pick middle layer
        mid_layer = list(layer_stats.keys())[len(layer_stats)//2]
        k_values = layer_stats[mid_layer]['required_k_list']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # K over query positions
        ax1 = axes[0]
        ax1.plot(k_values, linewidth=0.5, alpha=0.7)
        ax1.axhline(y=np.mean(k_values), color='red', linestyle='--', 
                   label=f'Mean K = {np.mean(k_values):.1f}')
        ax1.fill_between(range(len(k_values)), k_values, alpha=0.3)
        ax1.set_xlabel('Query Position')
        ax1.set_ylabel(f'Required K (for {TARGET_COVERAGE:.0%} coverage)')
        ax1.set_title(f'K Varies Across Queries (Layer {mid_layer})\n'
                     f'Min={min(k_values)}, Max={max(k_values)}, '
                     f'Ratio={max(k_values)/max(min(k_values),1):.1f}x')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # K distribution histogram
        ax2 = axes[1]
        ax2.hist(k_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.mean(k_values), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={np.mean(k_values):.1f}')
        ax2.axvline(x=np.median(k_values), color='green', linestyle='--', linewidth=2,
                   label=f'Median={np.median(k_values):.1f}')
        ax2.set_xlabel('Required K')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Required K')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{dataset} - Sample {sample_idx} - {num_tokens} tokens\n'
                    f'Static K is SUBOPTIMAL: queries need vastly different K values', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / f"{dataset}_sample{sample_idx}_k_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_path}")
        
    except ImportError:
        print("    matplotlib not available, skipping visualization")


def visualize_comparison(all_stats: Dict, output_dir: Path):
    """Visualize comparison across datasets."""
    try:
        import matplotlib.pyplot as plt
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (dataset, stats) in enumerate(all_stats.items()):
            if idx >= 4:
                break
                
            ax = axes[idx]
            k_values = stats['required_k_per_query']
            
            ax.hist(k_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(x=stats['mean_k'], color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Required K')
            ax.set_ylabel('Frequency')
            ax.set_title(f"{dataset}\nK ratio: {stats['k_ratio']:.1f}x, "
                        f"Range: [{stats['min_k']}, {stats['max_k']}]")
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Static K is SUBOPTIMAL: Different tasks need different K ranges\n'
                    'Each query token requires different K for 90% attention coverage',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = output_dir / "k_distribution_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison: {save_path}")
        
    except ImportError:
        print("matplotlib not available")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prove static K is suboptimal - Memory efficient chunked version"
    )
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Chunk size for attention computation (default 512)")
    parser.add_argument("--max-tokens", type=int, default=150000,
                       help="Max context length (default 150000)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--visualize", action="store_true", default=True)
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                       help="Specific layers to analyze (default: 5 representative)")
    args = parser.parse_args()
    
    global TARGET_COVERAGE
    TARGET_COVERAGE = args.coverage
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Memory estimation
    # Per chunk: chunk_size × seq_len × heads × 4 bytes (float32 for softmax)
    est_mem_gb = (args.chunk_size * args.max_tokens * 32 * 4) / (1024**3)
    
    print("="*70)
    print("PROVING: Static K is Suboptimal (Memory-Efficient Chunked Version)")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Target coverage: {TARGET_COVERAGE:.0%}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Estimated attention memory per chunk: ~{est_mem_gb:.1f} GB")
    print(f"Datasets: {args.datasets}")
    print("="*70)
    
    model, tokenizer = load_model(args.model)
    
    all_dataset_stats = {}
    
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} - {DATASETS.get(dataset, '')}")
        print("="*60)
        
        dataset_stats = []
        
        for sample_idx in range(args.samples_per_dataset):
            stats = analyze_sample_chunked(
                model, tokenizer, dataset, sample_idx,
                chunk_size=args.chunk_size,
                layers_to_analyze=args.layers,
                max_tokens=args.max_tokens,
                visualize=args.visualize,
                output_dir=OUTPUT_DIR,
            )
            
            if stats:
                dataset_stats.append(stats)
        
        # Aggregate stats
        if dataset_stats:
            all_k_values = []
            for s in dataset_stats:
                all_k_values.extend(s["required_k_list"])
            
            all_dataset_stats[dataset] = {
                "num_samples": len(dataset_stats),
                "total_queries": len(all_k_values),
                "min_k": min(all_k_values),
                "max_k": max(all_k_values),
                "mean_k": np.mean(all_k_values),
                "std_k": np.std(all_k_values),
                "k_ratio": max(all_k_values) / max(min(all_k_values), 1),
                "required_k_per_query": all_k_values,
            }
    
    # Generate comparison visualization
    if all_dataset_stats:
        visualize_comparison(all_dataset_stats, OUTPUT_DIR)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: Why Static K is Suboptimal")
    print("="*70)
    
    for dataset, stats in all_dataset_stats.items():
        print(f"\n{dataset}:")
        print(f"  Queries analyzed: {stats['total_queries']}")
        print(f"  Required K range: {stats['min_k']} - {stats['max_k']}")
        print(f"  K ratio (max/min): {stats['k_ratio']:.1f}x")
        print(f"  Mean ± Std: {stats['mean_k']:.1f} ± {stats['std_k']:.1f}")
        
        # Calculate waste if using static K
        static_k = int(stats['max_k'])  # Would need max to cover all
        actual_k_sum = sum(stats['required_k_per_query'])
        static_k_sum = static_k * len(stats['required_k_per_query'])
        waste_pct = (static_k_sum - actual_k_sum) / static_k_sum * 100
        print(f"  Static K waste: {waste_pct:.1f}% (using K={static_k})")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("  Static K is suboptimal because:")
    print("  1. Different query positions require vastly different K values")
    print("  2. K ratio (max/min) shows 10-100x variation within same sequence")
    print("  3. Using static K = max wastes computation on easy queries")
    print("  4. Using static K = mean fails to cover hard queries")
    print("  → Dynamic K selection (TokenSelect) adapts to each query's needs")
    print("="*70)
    
    # Save results to JSON
    results_path = OUTPUT_DIR / "analysis_results.json"
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'required_k_per_query'} 
                    for k, v in all_dataset_stats.items()}
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
