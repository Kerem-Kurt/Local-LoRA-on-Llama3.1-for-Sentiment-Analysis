def divide_dataset(theDataset,train_size, eval_size):
    """the Dataset, Training size, Evaluation size (between 0-1), rest is test size"""
    import pandas as pd


    # Split the DataFrame
    train_size = train_size
    eval_size = eval_size

    # Calculate sizes
    train_end = int(train_size * len(theDataset))
    eval_end = train_end + int(eval_size * len(theDataset))

    # Split the data
    X_train = theDataset[:train_end]
    X_eval = theDataset[train_end:eval_end]
    X_test = theDataset[eval_end:]

    # Keep a copy of the test labels before generating prompts
    y_true = X_test['label'].values

    # Define the prompt generation functions
    def generate_prompt(data_point):
        return f"""
                Classify the financial text as Positive, Negative, or Neutral.
    text: {data_point['sentence']}
    label: {data_point['label']}""".strip()

    def generate_test_prompt(data_point):
        return f"""
                Classify the financial text as Positive, Negative, or Neutral.
    text: {data_point['sentence']}
    label: """.strip()

    # Generate prompts for training and evaluation
    X_train = X_train.copy()
    X_eval = X_eval.copy()

    X_train.loc[:, 'text'] = X_train.apply(generate_prompt, axis=1)
    X_eval.loc[:, 'text'] = X_eval.apply(generate_prompt, axis=1)

    # Generate test prompts (without labels)
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

    ## If you want to see all of the text -> Use these:
    pd.set_option('display.max_colwidth', None)  # Set max column width to None to avoid truncation
    pd.set_option('display.max_columns', None)   # Show all columns

    return X_train, X_test, X_eval, y_true