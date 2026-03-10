import torch
import numpy as np
import os

# Patch for h5py/numpy compatibility issue
if not hasattr(np, 'typeDict'):
    np.typeDict = np.sctypeDict

from dataset import ML1MDataset
from dataloader import TrainDataLoader, EvalDataLoader
from trainer import Trainer
from recommender import NCFRecommender

def main():
    # 1. Load Dataset
    data_path = 'data'
    print(f"Loading dataset from {data_path}...")
    dataset = ML1MDataset(data_path)
    
    # 2. Initialize Model
    # We need to match the configuration used in training
    print("Initializing NCF model...")
    gpt2_dim = 768
    model = NCFRecommender(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=gpt2_dim,
        mlp_dims=[256, 128, 64],
        pretrained_item_embeddings=None # We will load state_dict, so this is not needed
    )
    
    # 3. Load Checkpoint
    checkpoint_path = 'gpt2_ncf_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        print(f"Checkpoint {checkpoint_path} not found! Please run training first.")
        return

    # 4. Prepare Test Dataloader
    train_data = dataset.get_split_data('train')
    test_data = dataset.get_split_data('test')
    test_loader = EvalDataLoader(test_data, train_data, batch_size=2048)

    # 5. Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=model,
        train_data=None, # Not needed for evaluation
        eval_data=None,  # Not needed for evaluation
        test_data=test_loader,
        device=device
    )
    
    print("Starting evaluation...")
    test_result = trainer.evaluate(test_loader)
    print(f"Test Result: {test_result}")

if __name__ == "__main__":
    main()
