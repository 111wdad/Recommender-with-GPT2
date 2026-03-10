import torch
import numpy as np

# Patch for h5py/numpy compatibility issue
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from trainer import Trainer
from recommender import NCFRecommender
from gpt2 import GPT2Encoder
import os

def main():
    # 1. Load Dataset
    data_path = 'data'
    print(f"Loading dataset from {data_path}...")
    dataset = ML1MDataset(data_path)
    
    # 2. Generate GPT-2 Embeddings for Items
    print("Generating GPT-2 embeddings for items...")
    items_df = dataset.items_df
    # Ensure we cover all possible item indices up to item_num
    max_item_id = dataset.get_item_num()
    
    # Initialize embedding matrix with zeros or random
    # GPT-2 embedding dim is 768
    gpt2_dim = 768
    item_embeddings = torch.zeros(max_item_id, gpt2_dim)
    
    # Get titles for existing items
    # items_df index is movie_id
    existing_ids = items_df.index.values
    titles = items_df['title'].tolist()
    
    # Use GPT2Encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = GPT2Encoder(device=device)
    
    print(f"Encoding {len(titles)} movie titles...")
    title_embeddings = encoder.get_embeddings(titles, batch_size=32)
    
    # Fill the embedding matrix
    # title_embeddings corresponds to existing_ids in order
    item_embeddings[torch.LongTensor(existing_ids)] = title_embeddings
    
    print("Item embeddings generated.")
    
    # 3. Initialize Model with Pretrained Embeddings
    print("Initializing NCF model with GPT-2 embeddings...")
    # We use embed_dim=768 to match GPT-2
    # We can customize MLP dimensions if needed, e.g., [256, 128, 64]
    model = NCFRecommender(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=gpt2_dim,
        mlp_dims=[256, 128, 64], # Custom MLP layers to handle large input
        pretrained_item_embeddings=item_embeddings
    )
    
    # 4. Prepare Dataloaders
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')

    train_loader = TrainDataLoader(train_data, batch_size=2048, shuffle=True)
    valid_loader = EvalDataLoader(valid_data, train_data, batch_size=2048)
    test_loader = EvalDataLoader(test_data, train_data, batch_size=2048)

    # 5. Train
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        epochs=50, # Reduced epochs for demonstration
        batch_size=2048,
        device=device
    )
    
    print("Starting training...")
    valid_result, test_result = trainer.fit(save_model=True, model_path='gpt2_ncf_checkpoint.pth')
    print(f"Best Validation Result: {valid_result}")
    print(f"Test Result: {test_result}")

if __name__ == "__main__":
    main()
