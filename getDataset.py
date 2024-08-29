def get_dataset(huggingface_dataset_name):
    """ Get the dataset, shuffle according to the seed and turn into csv"""

    from datasets import load_dataset

    dataset = load_dataset(huggingface_dataset_name, "sentences_allagree", split='train', trust_remote_code=True)
    shuffled_dataset = dataset.shuffle(seed=32)

    df = shuffled_dataset.to_pandas()

    # Map sentiment labels to text (optional)
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['label'] = df['label'].map(label_mapping)

    # Save the dataset to a CSV file
    df.to_csv('financial_phrasebank.csv', index=False)

    return df