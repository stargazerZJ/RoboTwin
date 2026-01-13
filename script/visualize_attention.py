import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def visualize_attention_map(file_path, output_dir=None, head_agg='mean', layer_agg='mean'):
    """
    Visualizes attention map from a .npy file as a heatmap.
    
    Args:
        file_path: Path to the .npy file.
        output_dir: Directory to save the image. If None, saves in the same directory as the .npy file.
        head_agg: How to aggregate heads ('mean', 'max', 'first', or int index).
        layer_agg: How to aggregate layers if present ('mean', 'max', 'last', or int index).
    """
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    print(f"Loaded {file_path}. Shape: {data.shape}")
    
    # Handle Bfloat16 if present (though we casted to float32 in policy, numpy load handles it)
    if data.dtype == 'bfloat16':
        data = data.astype(np.float32)

    # Assume shape structure. 
    # Possible shapes based on gemma implementation: 
    # (Batch, Layers, Heads, Query, Key) or (Layers, Batch, Heads, Query, Key)
    # or (Batch, Heads, Query, Key) if layers are squeezed.
    
    # Squeeze batch dimension if it's 1
    if data.shape[0] == 1:
        data = np.squeeze(data, axis=0)
        print(f"Squeezed batch dim 0. New shape: {data.shape}")
    
    # Now we might have (Layers, Heads, Q, K) or (Heads, Q, K)
    
    # If 4D, assume (Layers, Heads, Q, K)
    if data.ndim == 4:
        # Aggregate layers
        if layer_agg == 'mean':
            data = np.mean(data, axis=0)
        elif layer_agg == 'max':
            data = np.max(data, axis=0)
        elif layer_agg == 'last':
            data = data[-1]
        elif isinstance(layer_agg, int):
            data = data[layer_agg]
        else: # try to parse int
            try:
                idx = int(layer_agg)
                data = data[idx]
            except:
                print(f"Unknown layer_agg method: {layer_agg}. Using mean.")
                data = np.mean(data, axis=0)
        print(f"Aggregated layers. New shape: {data.shape}")

    # Now we have (Heads, Q, K) assuming 3D
    if data.ndim == 3:
        # Aggregate heads
        if head_agg == 'mean':
            data = np.mean(data, axis=0)
        elif head_agg == 'max':
            data = np.max(data, axis=0)
        elif head_agg == 'first':
            data = data[0]
        elif isinstance(head_agg, int):
            data = data[head_agg]
        else:
             try:
                idx = int(head_agg)
                data = data[idx]
             except:
                print(f"Unknown head_agg method: {head_agg}. Using mean.")
                data = np.mean(data, axis=0)
        print(f"Aggregated heads. New shape: {data.shape}")

    # Now we should have 2D (Q, K)
    if data.ndim != 2:
        print(f"Error: Expected 2D data after aggregation, got {data.shape}")
        return

    # Create Heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Attention Probability')
    plt.title(f'Attention Map: {os.path.basename(file_path)}')
    plt.xlabel('Keys (Source Token)')
    plt.ylabel('Queries (Target Token)')
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_name = os.path.basename(file_path).replace('.npy', '.png')
        out_path = os.path.join(output_dir, out_name)
    else:
        out_path = file_path.replace('.npy', '.png')
    
    plt.savefig(out_path)
    plt.close()
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Attention Maps from .npy files")
    parser.add_argument("path", type=str, help="Path to .npy file or directory containing .npy files")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save images")
    parser.add_argument("--head_agg", type=str, default="mean", help="Head aggregation: mean, max, first, or index")
    parser.add_argument("--layer_agg", type=str, default="last", help="Layer aggregation: mean, max, last, or index")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        files = glob.glob(os.path.join(args.path, "*.npy"))
        files.sort()
        for f in files:
            visualize_attention_map(f, args.out_dir, args.head_agg, args.layer_agg)
    else:
        visualize_attention_map(args.path, args.out_dir, args.head_agg, args.layer_agg)
