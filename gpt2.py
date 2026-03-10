from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm

class GPT2Encoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()

    def get_embeddings(self, texts, batch_size=32):
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating GPT-2 Embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the last hidden state
                hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, 768)
                # Mean pooling over the sequence dimension, considering padding
                attention_mask = inputs['attention_mask'].unsqueeze(-1) # (batch, seq_len, 1)
                masked_hidden_states = hidden_states * attention_mask
                sum_embeddings = masked_hidden_states.sum(dim=1)
                sum_mask = attention_mask.sum(dim=1)
                # Avoid division by zero
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask # (batch, 768)
                
                all_embeddings.append(mean_embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0)

if __name__ == "__main__":
    # Example usage
    encoder = GPT2Encoder(device='cpu')
    text = ["Hello, how are you?", "I am fine, thank you."]
    embeddings = encoder.get_embeddings(text)
    print(f"Embeddings shape: {embeddings.shape}")
