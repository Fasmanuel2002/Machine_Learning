import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from models.seq2seq_baseline import Seq2Seq
from models.seq2seq_attention import Seq2SeqAttention



def plot_weights_initialization(model : Seq2Seq | Seq2SeqAttention):
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.cpu().flatten().detach().numpy())
    
    all_weights_concat = np.concatenate(all_weights)
    plt.hist(all_weights_concat.flatten(), bins=50)
    plt.title("Weight Distribution of the Attention Model")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_attention_matrix(prompt, predicted_text, tokenizer, attention_weights):
    
    input_ids = tokenizer.encode(prompt)
    x_labels = [tokenizer.token_ids_to_token.get(idx, "<unk>") for idx in input_ids]

    y_labels = predicted_text.split()
    

    attention_matrix = np.vstack(attention_weights)
    

    min_y = min(len(y_labels), attention_matrix.shape[0])
    min_x = min(len(x_labels), attention_matrix.shape[1])
    
    attention_matrix = attention_matrix[:min_y, :min_x]
    y_labels = y_labels[:min_y]
    x_labels = x_labels[:min_x]
    

    plt.figure(figsize=(12, 8))

    sns.heatmap(attention_matrix, xticklabels=x_labels, yticklabels=y_labels, 
                cmap='magma', annot=False)
    
    plt.title("What is the model looking", fontsize=14, pad=20)
    plt.xlabel("Original Prompt (Memory of Encoder)", fontsize=12)
    plt.ylabel("generated prompt (prediction of Decoder)", fontsize=12)
    
    
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.show()