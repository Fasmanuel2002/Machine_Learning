import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_batch

def plot_and_log_attention(model, val_data, sequence_length, device, writer, global_step, decode):
    """
    Extracts the attention matrix from a validation sample, decodes actual characters
    for axis labels, and logs the formatted plot to TensorBoard and disk.
    """
    model.eval()
    
    with torch.no_grad():
        # 1. Fetch a validation sequence
        xb, _ = get_batch('val', batch_size=1, sequence_length=sequence_length, 
                          train_data=val_data, val_data=val_data, device=device)
        
        # 2. Decode token IDs into human-readable characters
        char_tokens = []
        for token_id in xb[0]:
            char = decode([token_id.item()])
            if char == '\n':
                char_tokens.append('\\n')
            elif char == ' ':
                char_tokens.append('␣')  # Visible symbol for space character
            else:
                char_tokens.append(char)

        # 3. Model inference
        _, _, attn_weights = model(xb) 
        
        # Extract matrix from the last layer, head 0 -> Expected Shape: (T, T)
        if attn_weights.dim() == 5:  # (B, n_layers, n_heads, T, T)
            matrix = attn_weights[0, -1, 0].cpu().numpy() 
            layer_str = "Last Layer"
        else:  # (B, n_heads, T, T)
            matrix = attn_weights[0, 0].cpu().numpy()
            layer_str = "Single Layer"
        

        # 4. Plot formatting & layout
        fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
        
        # Display exact numerical probabilities if the sequence length is short
        show_annot = sequence_length <= 16
        
        sns.heatmap(
            matrix, 
            cmap='viridis', 
            ax=ax, 
            cbar=True,
            xticklabels=char_tokens,
            yticklabels=char_tokens,
            annot=show_annot,
            fmt=".2f" if show_annot else "",
            annot_kws={"size": 8},
            cbar_kws={'label': 'Attention Weight (Softmax)'},
            linewidths=0.5 if show_annot else 0
        )

        # 5. Axis labels and titles
        ax.set_title(f"Attention Matrix - {layer_str}, Head 0 (Iter {global_step})", fontsize=11, pad=12)
        ax.set_xlabel("Keys (Attended Characters)", fontsize=10, labelpad=8)
        ax.set_ylabel("Queries (Current Character)", fontsize=10, labelpad=8)
        
        plt.xticks(rotation=0, fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        # 6. Save to TensorBoard and disk
        writer.add_figure("Attention_Matrix/Head_0", fig, global_step=global_step)
        plt.savefig("best_attention_matrix.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    model.train()