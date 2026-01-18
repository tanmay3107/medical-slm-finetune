import pandas as pd
import matplotlib.pyplot as plt

def analyze():
    df = pd.read_csv('data/train_subset.csv')
    
    # Check length of answers (important for token limits)
    df['answer_length'] = df['long_answer'].apply(lambda x: len(str(x).split()))
    
    print("Dataset Stats:")
    print(df.describe())
    
    # Visual check
    print("\nSample Entry:")
    print(f"Q: {df.iloc[0]['question']}")
    print(f"A: {df.iloc[0]['long_answer']}")

if __name__ == "__main__":
    analyze()