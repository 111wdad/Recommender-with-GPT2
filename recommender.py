import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod


class AbstractRecommender(nn.Module, ABC):
    """Abstract base class for recommender models"""

    def __init__(self, n_users, n_items, embed_dim, pretrained_item_embeddings=None):
        """
        Initialize base recommender

        Args:
            n_users (int): Number of users in the dataset
            n_items (int): Number of items in the dataset
            embed_dim (int): Dimension of embeddings
            pretrained_item_embeddings (torch.Tensor, optional): Pretrained item embeddings
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        
        if pretrained_item_embeddings is not None:
            self.item_embedding = nn.Embedding.from_pretrained(pretrained_item_embeddings, freeze=False)
            # Ensure the dimension matches
            assert self.item_embedding.embedding_dim == embed_dim, "Pretrained embedding dim must match embed_dim"
        else:
            self.item_embedding = nn.Embedding(n_items, embed_dim)

        # Initialize embeddings with Xavier uniform
        self._init_weights(pretrained_item_embeddings is not None)

    def _init_weights(self, has_pretrained_items=False):
        """Initialize weights using Xavier uniform"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        if not has_pretrained_items:
            nn.init.xavier_uniform_(self.item_embedding.weight)

    @abstractmethod
    def forward(self, batch_data):
        """
        Forward pass to compute prediction scores

        Args:
            batch_data (torch.Tensor): Batch data from dataloader containing [users, pos_items, neg_items]

        Returns:
            tuple: (pos_scores, neg_scores) predicted scores for positive and negative samples
        """
        pass

    @abstractmethod
    def calculate_loss(self, pos_scores, neg_scores):
        """
        Calculate loss for training

        Args:
            pos_scores (torch.FloatTensor): Predicted scores for positive samples
            neg_scores (torch.FloatTensor): Predicted scores for negative samples

        Returns:
            torch.FloatTensor: Computed loss value
        """
        pass

    @torch.no_grad()
    def recommend(self, user_id, k=None):
        """
        Generate item recommendations for a user

        Args:
            user_id (int): User ID to generate recommendations for
            k (int, optional): Number of items to recommend. If None, returns scores for all items

        Returns:
            torch.FloatTensor: Predicted scores for items (shape: n_items)
        """
        self.eval()
        user_tensor = torch.LongTensor([user_id]).to(self.device)
        all_items = torch.arange(self.n_items).to(self.device)
        # Get scores for all items
        scores = self.predict(user_tensor.repeat(len(all_items)), all_items)

        if k is not None:
            _, indices = torch.topk(scores, k)
            return all_items[indices]

        return scores

    def predict(self, user_ids, item_ids):
        """
        Predict scores for given user-item pairs

        Args:
            user_ids (torch.LongTensor): User IDs
            item_ids (torch.LongTensor): Item IDs

        Returns:
            torch.FloatTensor: Predicted scores
        """
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        return (user_embeds * item_embeds).sum(dim=1)


    def get_user_embedding(self, user_id):
        """Get embedding for a user"""
        return self.user_embedding(torch.LongTensor([user_id]).to(self.device))

    def get_item_embedding(self, item_id):
        """Get embedding for an item"""
        return self.item_embedding(torch.LongTensor([item_id]).to(self.device))

    @property
    def device(self):
        """Get device model is on"""
        return next(self.parameters()).device


class MF(AbstractRecommender):
    def forward(self, batch_data):
        users, pos_items, neg_items = batch_data
        user_embeds = self.user_embedding(users)
        pos_item_embeds = self.item_embedding(pos_items)
        neg_item_embeds = self.item_embedding(neg_items)
        
        pos_scores = (user_embeds * pos_item_embeds).sum(dim=1)
        neg_scores = (user_embeds * neg_item_embeds).sum(dim=1)
        
        return pos_scores, neg_scores
    
    def calculate_loss(self, pos_scores, neg_scores):
        loss = -(pos_scores - neg_scores).sigmoid().log().mean()
        return loss



class NCFRecommender(AbstractRecommender):
    def __init__(self, n_users, n_items, embed_dim, mlp_dims=None, pretrained_item_embeddings=None):
        super().__init__(n_users, n_items, embed_dim, pretrained_item_embeddings)
        
        if mlp_dims is None:
            # 默认保持原有的两层结构: [embed_dim, embed_dim]
            mlp_dims = [embed_dim, embed_dim]

        layers = []
        input_dim = embed_dim * 2
        
        # 动态构建 MLP 层
        for dim in mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
            
        # 最后一层输出预测分数
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, batch_data):
        user_ids, pos_item_ids, neg_item_ids = batch_data
        user_embeds = self.user_embedding(user_ids)
        pos_item_embeds = self.item_embedding(pos_item_ids)
        neg_item_embeds = self.item_embedding(neg_item_ids)

        concated_embeds = torch.cat([user_embeds, pos_item_embeds], dim=-1)
        concated_embeds_neg = torch.cat([user_embeds, neg_item_embeds], dim=-1)
        mlp_output = self.mlp_layers(concated_embeds)
        mlp_output_neg = self.mlp_layers(concated_embeds_neg)

        return mlp_output, mlp_output_neg

    def calculate_loss(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

    @torch.no_grad()
    def recommend(self, user_id, k=None, batch_size_eval=1024):

        self.eval()
        device = self.device

        # get user index as int
        if isinstance(user_id, torch.Tensor):
            user_idx = int(user_id.detach().cpu().item())
        else:
            user_idx = int(user_id)

        # accumulate scores in chunks
        scores_chunks = []
        start = 0
        while start < self.n_items:
            end = min(start + batch_size_eval, self.n_items)
            items = torch.arange(start, end, device=device, dtype=torch.long)
            users = torch.full((len(items),), user_idx, device=device, dtype=torch.long)

            user_embeds = self.user_embedding(users)  # (batch, dim)
            item_embeds = self.item_embedding(items)  # (batch, dim)
            concated_embeds = torch.cat([user_embeds, item_embeds], dim=-1)  # (batch, 2*dim)
            output = self.mlp_layers(concated_embeds).view(-1)  # (batch,)
            scores_chunks.append(output)
            start = end

        scores = torch.cat(scores_chunks, dim=0)  # (n_items,)

        if k is None:
            return scores
        else:
            _, idxs = torch.topk(scores, k)
            return idxs


if __name__ == '__main__':
    # Example usage
    from dataset import ML1MDataset
    from dataloader import TrainDataLoader, EvalDataLoader
    from trainer import Trainer

    # Load dataset
    dataset = ML1MDataset('ml-1m')

    # Create model
    # You can initialize the model here with custom parameters
    model = NCFRecommender(
        n_users=dataset.get_user_num(),
        n_items=dataset.get_item_num(),
        embed_dim=64,
        mlp_dims=[128, 64, 32]  # Optional: Customize MLP layers (e.g., 3 layers)
    )

    # Get split datasets
    train_data = dataset.get_split_data('train')
    valid_data = dataset.get_split_data('validation')
    test_data = dataset.get_split_data('test')

    # Create dataloaders
    train_loader = TrainDataLoader(train_data, batch_size=2048, shuffle=True)
    valid_loader = EvalDataLoader(valid_data, train_data, batch_size=2048)
    test_loader = EvalDataLoader(test_data, train_data, batch_size=2048)

    trainer = Trainer(
        model=model,
        train_data=train_loader,
        eval_data=valid_loader,
        test_data=test_loader,
        epochs=100,
    )

    valid_result, test_result = trainer.fit(save_model=False)
    print(f"Best Validation Result: {valid_result}")
    print(f"Test Result: {test_result}")

