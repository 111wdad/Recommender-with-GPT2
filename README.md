# Recommender System with GPT-2 Enhanced Embeddings

This project implements a Neural Collaborative Filtering (NCF) recommender system on the MovieLens-1M dataset. It features a unique integration with GPT-2 to utilize semantic information from movie titles as initial embeddings for the recommendation model.

## Project Structure

*   **`dataset.py`**: Preprocesses the MovieLens-1M dataset, handling user/item/rating data loading and splitting.
*   **`dataloader.py`**: PyTorch DataLoaders for efficient batching of training and evaluation data.
*   **`recommender.py`**: Contains model definitions:
    *   `AbstractRecommender`: Base class with common functionality.
    *   `MF`: Standard Matrix Factorization implementation.
    *   `NCFRecommender`: Neural Collaborative Filtering model with customizable MLP layers and support for pretrained embeddings.
*   **`trainer.py`**: Handles the training loop, validation, early stopping, and metric calculation.
*   **`gpt2.py`**: Contains the `GPT2Encoder` class to extract semantic embeddings from text using a pre-trained GPT-2 model.
*   **`run_gpt2_training.py`**: The main execution script. It generates GPT-2 embeddings for movie titles and uses them to initialize and train the NCF model.
*   **`evaluate_only.py`**: A standalone script to evaluate a saved model checkpoint.

## Key Features & Optimizations

### 1. GPT-2 Semantic Initialization
Instead of initializing item embeddings randomly, this project uses a pre-trained GPT-2 model to encode movie titles into 768-dimensional vectors.
*   **Implementation**: `GPT2Encoder` extracts the last hidden state of the GPT-2 model and applies mean pooling (considering attention masks) to generate a fixed-size vector for each movie title.
*   **Fine-tuning**: These embeddings are injected into the NCF model using `nn.Embedding.from_pretrained(..., freeze=False)`. This allows the model to start with semantic knowledge and fine-tune the weights based on collaborative filtering signals during training.

### 2. Flexible Model Architecture
*   **Customizable MLP**: The `NCFRecommender` allows dynamic configuration of the MLP layers via the `mlp_dims` parameter. This is crucial when working with high-dimensional inputs like GPT-2 embeddings (768 dim).
    *   *Example*: A "tower" structure `[256, 128, 64]` is used to compress the 1536-dimensional input (User+Item) down to a prediction score.
*   **Matrix Factorization**: A standard `MF` class is implemented for baseline comparisons.

### 3. Comprehensive Evaluation Metrics
The `Trainer` class calculates a rich set of metrics to evaluate recommendation performance:
*   **NDCG@K** (Normalized Discounted Cumulative Gain): Measures ranking quality.
*   **HR@K** (Hit Ratio): Measures if the target item is present in the top-K recommendations.
*   **Recall@K**: Measures the proportion of relevant items retrieved.
*   **Precision@K**: Measures the proportion of retrieved items that are relevant.

*(Default K values: 10, 20, 50)*

## Setup & Usage

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Transformers (Hugging Face)
*   Pandas, NumPy, Scipy

### 1. Train the Model
Run the training script. This process will:
1.  Load the MovieLens-1M dataset.
2.  Download/Load GPT-2 and generate embeddings for all 3,883 movie titles.
3.  Initialize the NCF model with these embeddings.
4.  Train the model and save the best checkpoint to `gpt2_ncf_checkpoint.pth`.

```bash
# Optional: Set HF mirror if needed for faster download in some regions
export HF_ENDPOINT=https://hf-mirror.com

python run_gpt2_training.py
```

### 2. Evaluate the Model
To evaluate a trained model (loads from `gpt2_ncf_checkpoint.pth`):

```bash
python evaluate_only.py
```

## Performance
The model is evaluated on a held-out test set. Typical metrics after training (example):
*   **NDCG@10**: ~0.062
*   **HR@10**: ~0.401
*   **Recall@10**: ~0.051
*   **Precision@10**: ~0.052

*Note: Metrics are based on a random split strategy.*
