import argparse
import torch
import pandas as pd 
from io import BytesIO
from huggingface_hub import HfApi, create_repo

from tabstar_paper.preprocessing.text_embeddings import E5EmbeddingModel
from tabstar_paper.preprocessing.feat_types import classify_semantic_features
from tabstar.preprocessing.feat_types import detect_numerical_features, transform_feature_types
from tabstar_paper.datasets.downloading import download_dataset, get_dataset_from_arg
from tabstar.preprocessing.dates import fit_date_encoders, transform_date_features
from tabstar.preprocessing.sparse import densify_objects
from config import REPO_ID, E5_EMBEDDING_MODEL

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text features in a dataset")
    parser.add_argument('--dataset_id', type=str, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='xgb') # Added model argument for compatibility with do_benchmark.py
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset_id}")
    dataset_enum = get_dataset_from_arg(args.dataset_id)
    dataset = download_dataset(dataset_enum)
    x = dataset.x 
    
    # First handle dates and objects (like in abstract_model.py)
    x, _ = densify_objects(x=x, y=None)
    date_transformers = fit_date_encoders(x)
    x = transform_date_features(x=x, date_transformers=date_transformers)
    
    # Now detect numerical features
    numerical_features = detect_numerical_features(x)
    print(f"🔢 Detected {len(numerical_features)} numerical features: {sorted(numerical_features)}")

    x = transform_feature_types(x=x, numerical_features=numerical_features)
    semantic_types = classify_semantic_features(x=x, numerical_features=numerical_features)
    
    text_features = semantic_types.text_features
    categorical_features = semantic_types.categorical_features
    
    print(f"📝 Detected {len(text_features)} text features: {sorted(text_features)}")
    print(f"🏷️  Detected {len(categorical_features)} categorical features: {sorted(categorical_features)}")
    
    if not text_features:
        print("No text features detected in the dataset!")
        return
    
    # Create the repo if it doesn't exist
    create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    api = HfApi()

    # Generate embeddings for each text column
    device = torch.device(args.device)
    for col in text_features:

        texts = x[col].astype(str).tolist()
        embeddings = E5EmbeddingModel().embed(texts=texts, device=device)
        df = pd.DataFrame(embeddings, columns=[f'dim_{i}' for i in range(embeddings.shape[1])])
        df.insert(0, 'text', texts)
        
        # Upload directly to Hugging Face using BytesIO
        buffer = BytesIO()
        df.to_parquet(buffer, engine="pyarrow", compression="zstd")
        buffer.seek(0)  # Reset buffer position to beginning
        
        data_source = dataset.dataset_id.__objclass__.__name__ 
        dataset_name = dataset.dataset_id._name_ 
        path_in_repo = f"{E5_EMBEDDING_MODEL}/{data_source}/{dataset_name}/{col}"
        api.upload_file(
            path_or_fileobj=buffer,
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="dataset")
        print(f"Uploaded {df.shape} directly to HF: {path_in_repo}")

if __name__ == "__main__":
    main()
