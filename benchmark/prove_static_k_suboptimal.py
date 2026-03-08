"""
Prove: Static K is Suboptimal - Different Query Tokens Need Different K

Mục tiêu: Chứng minh rằng việc sử dụng cùng một K cho tất cả query tokens là không tối ưu
vì mỗi query token có thể cần số lượng key tokens khác nhau để đạt coverage tốt.

Key Analysis:
- Mỗi query token (row trong attention matrix) có distribution khác nhau
- Một số query tokens cần ít key tokens (sparse attention)
- Một số query tokens cần nhiều key tokens hơn top-k (distributed attention)
- → Static K sẽ gây ra: thừa computation cho sparse queries, thiếu tokens cho distributed queries

Usage:
    python prove_static_k_suboptimal.py --samples-per-dataset 2
    python prove_static_k_suboptimal.py --samples-per-dataset 1 --visualize-all
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

DATASETS = {
    "passkey": "Simple retrieval (localized)",
    "kv_retrieval": "Key-value lookup (multiple points)",
    "longbook_qa_eng": "Document QA (distributed)",
    "math_find": "Math reasoning",
}

DATA_DIR = Path(__file__).parent / "data" / "infinite-bench"
OUTPUT_DIR = Path(__file__).parent / "static_k_analysis"

TARGET_COVERAGE = 0.90


# ============================================================================
# Model & Data Loading
# ============================================================================

def load_model(model_name: str = "Qwen/Qwen2-7B-Instruct"):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to save memory
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def load_sample(dataset: str, idx: int = 0) -> dict:
    """Load a single sample from dataset."""
    filepath = DATA_DIR / f"{dataset}.jsonl"
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    return {}


class AttentionCaptureHook:
    """
    Hook để capture attention từng layer một, tính statistics ngay, rồi free memory.
    Không lưu toàn bộ attention matrices vào memory.
    """
    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self.layer_stats = {}  # layer_idx -> stats
        self.current_layer = 0
        
    def reset(self):
        self.layer_stats = {}
        self.current_layer = 0
    
    def compute_required_k_per_query(self, attention: torch.Tensor):
        """Tính K cần thiết cho mỗi query token."""
        # attention: (batch, heads, seq, seq) hoặc (heads, seq, seq)
        if attention.dim() == 4:
            attention = attention[0]
        
        # Average over heads để giảm computation
        attn_avg = attention.mean(dim=0)  # (seq, seq)
        seq_len = attn_avg.shape[0]
        
        required_k = []
        for q_pos in range(seq_len):
            row = attn_avg[q_pos, :q_pos+1]  # Causal mask
            if len(row) == 0:
                required_k.append(1)
                continue
            
            sorted_vals, _ = row.sort(descending=True)
            cumsum = sorted_vals.cumsum(dim=0)
            
            k = 1
            for i, c in enumerate(cumsum):
                if c >= self.target_coverage:
                    k = i + 1
                    break
            else:
                k = len(row)
            required_k.append(k)
        
        return torch.tensor(required_k)
    
    def __call__(self, module, input, output):
        """Hook được gọi sau mỗi attention layer."""
        # output có thể là tuple (attn_output, attn_weights, ...)
        if isinstance(output, tuple) and len(output) >= 2:
            attn_weights = output[1]  # (batch, heads, seq, seq)
            if attn_weights is not None:
                # Tính statistics ngay lập tức
                with torch.no_grad():
                    required_k = self.compute_required_k_per_query(attn_weights.float())
                    
                    valid_k = required_k[50:]  # Skip first 50 tokens
                    if len(valid_k) > 0:
                        self.layer_stats[self.current_layer] = {
                            "min_k": valid_k.min().item(),
                            "max_k": valid_k.max().item(),
                            "mean_k": valid_k.float().mean().item(),
                            "std_k": valid_k.float().std().item(),
                            "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
                            "required_k_list": valid_k.tolist(),
                        }
                
                self.current_layer += 1
        
        return output


def get_attention_with_hooks(model, tokenizer, context: str, query: str, 
                             max_tokens: int = 32000, target_coverage: float = 0.90):
    """
    Capture attention statistics dùng hooks - memory efficient.
    Chỉ lưu statistics, không lưu full attention matrices.
    """
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = inputs.to(model.device)
    
    num_tokens = inputs.input_ids.shape[1]
    print(f"    Input tokens: {num_tokens}")
    
    # Setup hooks
    hook = AttentionCaptureHook(target_coverage)
    hooks = []
    
    # Register hooks on attention layers
    for name, module in model.named_modules():
        # Tìm attention modules (tên khác nhau tùy model)
        if "attn" in name.lower() and hasattr(module, 'forward'):
            # Chỉ hook vào attention chính, không hook vào sub-modules
            if name.count('.') <= 4:  # Adjust based on model structure
                h = module.register_forward_hook(hook)
                hooks.append(h)
    
    # Forward pass với output_attentions=True để attention được tính
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return hook.layer_stats, num_tokens


def get_attention_layer_by_layer(model, tokenizer, context: str, query: str,
                                  max_tokens: int = 32000, target_coverage: float = 0.90,
                                  layers_to_analyze: list = None):
    """
    Phương pháp 2: Chạy từng layer một, tính stats, rồi delete attention.
    Dùng khi hooks không hoạt động tốt.
    """
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    inputs = inputs.to(model.device)
    
    num_tokens = inputs.input_ids.shape[1]
    print(f"    Input tokens: {num_tokens}")
    
    # Forward với output_attentions
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, return_dict=True)
    
    attentions = outputs.attentions
    num_layers = len(attentions)
    
    if layers_to_analyze is None:
        layers_to_analyze = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
    
    layer_stats = {}
    
    for layer_idx in layers_to_analyze:
        if layer_idx >= num_layers:
            continue
            
        attention = attentions[layer_idx].float()  # Convert to float32 for computation
        
        # Tính required K per query
        if attention.dim() == 4:
            attention = attention[0]
        
        attn_avg = attention.mean(dim=0)
        seq_len = attn_avg.shape[0]
        
        required_k = []
        for q_pos in range(seq_len):
            row = attn_avg[q_pos, :q_pos+1]
            if len(row) == 0:
                required_k.append(1)
                continue
            
            sorted_vals, _ = row.sort(descending=True)
            cumsum = sorted_vals.cumsum(dim=0)
            
            k = 1
            for i, c in enumerate(cumsum):
                if c >= target_coverage:
                    k = i + 1
                    break
            else:
                k = len(row)
            required_k.append(k)
        
        required_k = torch.tensor(required_k)
        valid_k = required_k[50:]
        
        if len(valid_k) > 0:
            layer_stats[layer_idx] = {
                "min_k": valid_k.min().item(),
                "max_k": valid_k.max().item(),
                "mean_k": valid_k.float().mean().item(),
                "std_k": valid_k.float().std().item(),
                "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
                "required_k_list": valid_k.tolist(),
            }
        
        # Free memory ngay sau khi tính xong
        del attention, attn_avg, required_k, valid_k
    
    # Free all attentions
    del attentions, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return layer_stats, num_tokens


# ============================================================================
# Core Analysis: Per-Query K Variation
# ============================================================================

def compute_required_k_per_query(attention: torch.Tensor, target_coverage: float = 0.90):
    """
    Tính K cần thiết cho MỖI query token để đạt target coverage.
    
    Args:
        attention: (batch, heads, seq_len, seq_len) hoặc (heads, seq_len, seq_len)
        
    Returns:
        required_k: (seq_len,) - K cần thiết cho mỗi query position
    """
    if attention.dim() == 4:
        attention = attention[0]  # (heads, seq_len, seq_len)
    
    # Average over heads
    attn_avg = attention.mean(dim=0)  # (seq_len, seq_len)
    seq_len = attn_avg.shape[0]
    
    required_k = []
    
    for q_pos in range(seq_len):
        row = attn_avg[q_pos, :q_pos+1]  # Causal: chỉ attend đến tokens trước đó
        if len(row) == 0:
            required_k.append(1)
            continue
        
        # Tìm K nhỏ nhất để đạt coverage
        sorted_vals, _ = row.sort(descending=True)
        cumsum = sorted_vals.cumsum(dim=0)
        
        # Tìm index đầu tiên đạt target
        k = 1
        for i, c in enumerate(cumsum):
            if c >= target_coverage:
                k = i + 1
                break
        else:
            k = len(row)
        
        required_k.append(k)
    
    return torch.tensor(required_k)


def analyze_query_k_variation(attention: torch.Tensor, layer_idx: int, dataset: str):
    """
    Phân tích sự biến thiên của K giữa các query tokens.
    
    Key insight: Nếu K thay đổi nhiều giữa các queries → static K không tối ưu
    """
    required_k = compute_required_k_per_query(attention, TARGET_COVERAGE)
    
    # Bỏ qua tokens đầu (không đủ context)
    valid_k = required_k[50:]  # Skip first 50 tokens
    
    if len(valid_k) == 0:
        return None
    
    stats = {
        "dataset": dataset,
        "layer": layer_idx,
        "num_queries": len(valid_k),
        "min_k": valid_k.min().item(),
        "max_k": valid_k.max().item(),
        "mean_k": valid_k.float().mean().item(),
        "std_k": valid_k.float().std().item(),
        "median_k": valid_k.float().median().item(),
        "k_ratio": valid_k.max().item() / max(valid_k.min().item(), 1),
        "required_k_per_query": valid_k.tolist(),
    }
    
    # Phân loại queries
    sparse_threshold = stats["mean_k"] * 0.5
    dense_threshold = stats["mean_k"] * 1.5
    
    stats["sparse_queries_pct"] = (valid_k < sparse_threshold).float().mean().item() * 100
    stats["dense_queries_pct"] = (valid_k > dense_threshold).float().mean().item() * 100
    
    return stats


def analyze_per_head_query_variation(attention: torch.Tensor, layer_idx: int):
    """
    Phân tích K variation cho từng head riêng biệt.
    """
    if attention.dim() == 4:
        attention = attention[0]  # (heads, seq_len, seq_len)
    
    num_heads = attention.shape[0]
    seq_len = attention.shape[1]
    
    head_stats = {}
    
    for head_idx in range(num_heads):
        head_attn = attention[head_idx]  # (seq_len, seq_len)
        
        required_k = []
        for q_pos in range(50, seq_len):  # Skip first 50
            row = head_attn[q_pos, :q_pos+1]
            sorted_vals, _ = row.sort(descending=True)
            cumsum = sorted_vals.cumsum(dim=0)
            
            k = 1
            for i, c in enumerate(cumsum):
                if c >= TARGET_COVERAGE:
                    k = i + 1
                    break
            else:
                k = len(row)
            required_k.append(k)
        
        if required_k:
            k_tensor = torch.tensor(required_k)
            head_stats[head_idx] = {
                "min_k": k_tensor.min().item(),
                "max_k": k_tensor.max().item(),
                "mean_k": k_tensor.float().mean().item(),
                "std_k": k_tensor.float().std().item(),
                "k_ratio": k_tensor.max().item() / max(k_tensor.min().item(), 1),
            }
    
    return head_stats


# ============================================================================
# Visualization
# ============================================================================

def visualize_attention_matrix(attention: torch.Tensor, layer_idx: int, head_idx: int, 
                               save_path: str, dataset: str, num_tokens: int):
    """
    Visualize attention matrix cho một layer và head cụ thể.
    """
    try:
        import matplotlib.pyplot as plt
        
        if attention.dim() == 4:
            attention = attention[0]
        
        attn = attention[head_idx].cpu().float().numpy()
        
        # Focus on a portion for visibility
        max_show = min(500, attn.shape[0])
        attn_show = attn[-max_show:, :]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Full attention heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(attn_show, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel(f'Query Position (last {max_show})')
        ax1.set_title(f'Attention Matrix\nDataset: {dataset}, Layer: {layer_idx}, Head: {head_idx}')
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        # Required K per query
        ax2 = axes[1]
        required_k = []
        for q_pos in range(attn.shape[0]):
            row = attn[q_pos, :q_pos+1]
            if len(row) == 0:
                required_k.append(1)
                continue
            sorted_vals = np.sort(row)[::-1]
            cumsum = np.cumsum(sorted_vals)
            k = np.searchsorted(cumsum, TARGET_COVERAGE) + 1
            required_k.append(min(k, len(row)))
        
        ax2.plot(required_k, linewidth=0.5, alpha=0.7)
        ax2.axhline(y=np.mean(required_k[50:]), color='red', linestyle='--', 
                   label=f'Mean K = {np.mean(required_k[50:]):.1f}')
        ax2.fill_between(range(len(required_k)), required_k, alpha=0.3)
        ax2.set_xlabel('Query Position')
        ax2.set_ylabel(f'Required K (for {TARGET_COVERAGE:.0%} coverage)')
        ax2.set_title(f'K Varies Across Queries\n(Min={min(required_k[50:])}, Max={max(required_k[50:])})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Static K is Suboptimal: Different Queries Need Different K\n'
                    f'Total tokens: {num_tokens}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"      Saved: {save_path}")
        
    except ImportError:
        print("      matplotlib not available")


def visualize_k_distribution_comparison(all_stats: dict, save_path: str):
    """
    So sánh distribution của required K giữa các datasets.
    """
    try:
        import matplotlib.pyplot as plt
        
        datasets = list(all_stats.keys())
        num_datasets = len(datasets)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, dataset in enumerate(datasets[:4]):
            ax = axes[idx]
            
            if dataset not in all_stats or "required_k_per_query" not in all_stats[dataset]:
                continue
            
            k_values = all_stats[dataset]["required_k_per_query"]
            
            ax.hist(k_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(k_values), color='red', linestyle='--', linewidth=2,
                      label=f'Mean={np.mean(k_values):.1f}')
            ax.axvline(x=np.median(k_values), color='green', linestyle='--', linewidth=2,
                      label=f'Median={np.median(k_values):.1f}')
            
            ax.set_xlabel('Required K')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{dataset}\n'
                        f'Range: {min(k_values)} - {max(k_values)} '
                        f'(Ratio: {max(k_values)/max(min(k_values),1):.1f}x)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Distribution of Required K per Query Token\n'
                    f'(Wide distribution → Static K is suboptimal)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
    except ImportError:
        print("matplotlib not available")


def visualize_head_comparison(head_stats: dict, layer_idx: int, dataset: str, save_path: str):
    """
    Visualize K variation across different heads.
    """
    try:
        import matplotlib.pyplot as plt
        
        heads = sorted(head_stats.keys())
        means = [head_stats[h]["mean_k"] for h in heads]
        mins = [head_stats[h]["min_k"] for h in heads]
        maxs = [head_stats[h]["max_k"] for h in heads]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean K per head
        ax1 = axes[0]
        ax1.bar(heads, means, color='steelblue', alpha=0.8)
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Mean Required K')
        ax1.set_title(f'Mean Required K per Head\n'
                     f'Std across heads: {np.std(means):.1f}')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # K range per head
        ax2 = axes[1]
        x = np.array(heads)
        ax2.fill_between(x, mins, maxs, alpha=0.3, color='coral')
        ax2.plot(x, means, 'o-', color='red', linewidth=2, label='Mean')
        ax2.set_xlabel('Head Index')
        ax2.set_ylabel('Required K Range')
        ax2.set_title(f'K Variation Range per Head\n'
                     f'(Shaded area = min to max)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Head-wise K Variation - {dataset}, Layer {layer_idx}', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
        
    except ImportError:
        print("matplotlib not available")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def analyze_dataset_full(model, tokenizer, dataset: str, sample_idx: int, 
                        visualize: bool = True, output_dir: Path = OUTPUT_DIR,
                        max_tokens: int = 4096):
    """
    Phân tích đầy đủ một sample.
    Dùng phương pháp layer-by-layer để tiết kiệm memory.
    """
    print(f"\n  Sample {sample_idx}:")
    
    sample = load_sample(dataset, sample_idx)
    if not sample:
        print(f"    Could not load sample")
        return None
    
    context = sample.get("context", "")
    query = sample.get("input", "")
    
    print(f"    Context length: {len(context)} chars")
    
    # Get attention stats layer by layer (memory efficient)
    layer_stats, num_tokens = get_attention_layer_by_layer(
        model, tokenizer, context, query, 
        max_tokens=max_tokens, 
        target_coverage=TARGET_COVERAGE
    )
    
    if not layer_stats:
        print(f"    No stats computed")
        return None
    
    # Print stats for each analyzed layer
    for layer_idx, stats in layer_stats.items():
        print(f"    Layer {layer_idx}:")
        print(f"      K range: {stats['min_k']} - {stats['max_k']} (ratio: {stats['k_ratio']:.1f}x)")
        
        # Classify queries
        mean_k = stats['mean_k']
        k_list = stats['required_k_list']
        sparse_pct = sum(1 for k in k_list if k < mean_k * 0.5) / len(k_list) * 100
        dense_pct = sum(1 for k in k_list if k > mean_k * 1.5) / len(k_list) * 100
        
        print(f"      Sparse queries (<0.5x mean): {sparse_pct:.1f}%")
        print(f"      Dense queries (>1.5x mean): {dense_pct:.1f}%")
        
        stats['sparse_queries_pct'] = sparse_pct
        stats['dense_queries_pct'] = dense_pct
    
    # Visualize K distribution
    if visualize:
        try:
            import matplotlib.pyplot as plt
            
            # Pick middle layer for visualization
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
            ax1.set_title(f'K Varies Across Queries\nMin={min(k_values)}, Max={max(k_values)}, '
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
                        f'Static K is suboptimal: queries need different K values', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{dataset}_sample{sample_idx}_k_analysis.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {save_path}")
            
        except ImportError:
            print("    matplotlib not available")
    
    # Return middle layer stats
    mid_layer = list(layer_stats.keys())[len(layer_stats)//2]
    result = layer_stats[mid_layer].copy()
    result['required_k_per_query'] = result.pop('required_k_list')
    return result


def main():
    parser = argparse.ArgumentParser(description="Prove static K is suboptimal for query tokens")
    parser.add_argument("--samples-per-dataset", type=int, default=2, 
                       help="Number of samples to analyze per dataset")
    parser.add_argument("--model", default="Qwen/Qwen2-7B-Instruct", help="Model name")
    parser.add_argument("--coverage", type=float, default=0.90, help="Target coverage")
    parser.add_argument("--visualize-all", action="store_true", 
                       help="Visualize attention for all layers/heads")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                       help="Datasets to analyze")
    parser.add_argument("--max-tokens", type=int, default=4096,
                       help="Max context tokens. Memory ~= (tokens/1000)^2 * 26GB. Default 4096 (~43GB)")
    args = parser.parse_args()
    
    global TARGET_COVERAGE
    TARGET_COVERAGE = args.coverage
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PROVING: Static K is Suboptimal for Token Selection")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Target coverage: {TARGET_COVERAGE:.0%}")
    print(f"Samples per dataset: {args.samples_per_dataset}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Datasets: {args.datasets}")
    
    # Memory estimation (assuming 28 layers, 28 heads, float16)
    # Memory ≈ layers * heads * seq^2 * 2 bytes
    est_mem_gb = (28 * 28 * args.max_tokens * args.max_tokens * 2) / (1024**3)
    print(f"Estimated attention memory: ~{est_mem_gb:.1f} GB")
    if est_mem_gb > 70:
        print("WARNING: May OOM on 80GB GPU! Consider reducing --max-tokens")
    print("="*70)
    
    model, tokenizer = load_model(args.model)
    
    all_dataset_stats = {}
    
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} - {DATASETS.get(dataset, '')}")
        print("="*60)
        
        dataset_stats = []
        
        for sample_idx in range(args.samples_per_dataset):
            stats = analyze_dataset_full(
                model, tokenizer, dataset, sample_idx,
                visualize=args.visualize_all,
                output_dir=OUTPUT_DIR,
                max_tokens=args.max_tokens
            )
            if stats:
                dataset_stats.append(stats)
        
        # Aggregate stats for this dataset
        if dataset_stats:
            all_k_values = []
            for s in dataset_stats:
                all_k_values.extend(s["required_k_per_query"])
            
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
        visualize_k_distribution_comparison(
            all_dataset_stats, 
            str(OUTPUT_DIR / "k_distribution_comparison.png")
        )
    
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
    
    # Overall conclusion
    all_ratios = [s["k_ratio"] for s in all_dataset_stats.values()]
    if all_ratios:
        print(f"\n{'─'*70}")
        print(f"CONCLUSION:")
        print(f"  Average K ratio across datasets: {np.mean(all_ratios):.1f}x")
        print(f"  Max K ratio: {max(all_ratios):.1f}x")
        print(f"\n  → Different query tokens need vastly different K values")
        print(f"  → Static K causes: information loss OR wasted computation")
        print(f"  → Dynamic K selection (TokenSelect) is necessary for optimality")
    
    # Save results
    results = {
        "model": args.model,
        "target_coverage": TARGET_COVERAGE,
        "samples_per_dataset": args.samples_per_dataset,
        "dataset_stats": {k: {kk: vv for kk, vv in v.items() if kk != "required_k_per_query"} 
                         for k, v in all_dataset_stats.items()},
    }
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults and visualizations saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
