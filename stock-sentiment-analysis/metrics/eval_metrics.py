import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import os

def evaluate_finbert(test_dataset, model_dir='./'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []
    pred_probs = []

    for batch in test_dataset:
        inputs = {k: v.clone().detach().unsqueeze(0).to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().item()

        true_labels.append(labels)
        pred_labels.append(pred)
        pred_probs.append(probs.cpu().numpy())

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.vstack(pred_probs)

    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    

    cm = confusion_matrix(true_labels, pred_labels)
    cm_path = os.path.join(model_dir, "confusion_matrix.png")

# Ensure directory exists
    os.makedirs(model_dir, exist_ok=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(cm_path)
    plt.show()
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix_path": cm_path
    }

    return metrics