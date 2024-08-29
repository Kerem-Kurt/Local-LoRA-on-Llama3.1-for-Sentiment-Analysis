import numpy as np
from tqdm import tqdm
from transformers import pipeline
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)

def predict(test, model, tokenizer):
    y_pred = []
    categories = ["positive", "negative", "neutral"]

    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=2,
                        temperature=0.1)

        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()

        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("none")

    return y_pred


def evaluate(y_true, y_pred):
    labels = ["positive", "negative", "neutral"]
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, but should not occur with correct data

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Filter out any -1 values which represent missing or incorrect labels
    valid_indices = (y_true_mapped != -1) & (y_pred_mapped != -1)
    y_true_mapped = y_true_mapped[valid_indices]
    y_pred_mapped = y_pred_mapped[valid_indices]

    # Check if there are any valid labels left
    if len(y_true_mapped) == 0 or len(y_pred_mapped) == 0:
        print("No valid labels found in y_true or y_pred.")
        return
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true_mapped)

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped, target_names=labels,
                                         labels=list(range(len(labels))))
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix
    present_labels = list(unique_labels)
    if len(present_labels) == 0:
        print("No labels present for confusion matrix.")
    else:
        conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped, labels=present_labels)
        print('\nConfusion Matrix:')
        print(conf_matrix)
