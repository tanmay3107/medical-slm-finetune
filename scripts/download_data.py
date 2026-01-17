from datasets import load_dataset
import pandas as pd
import os

def download_and_save():
    print("Downloading PubMedQA dataset...")
    # Load a subset for demonstration
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # Convert to Pandas for easy viewing
    df = pd.DataFrame(dataset)
    
    # Keep only relevant columns
    df = df[['question', 'long_answer', 'final_decision']]
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/train_subset.csv', index=False)
    print(f"Saved {len(df)} rows to data/train_subset.csv")

if __name__ == "__main__":
    download_and_save()