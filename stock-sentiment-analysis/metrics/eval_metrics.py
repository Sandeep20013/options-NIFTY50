import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer
import torch

def evaluate_finbert(test_dataset, model_dir='./finbert-india'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    true_labels = []
    pred_labels = []
    pred_probs = []

    for batch in test_dataset:
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in batch.items() if k != 'labels'}
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
    # For multiclass roc_auc, use 'ovo' or 'ovr'
    try:
        auc = roc_auc_score(true_labels, pred_probs, multi_class='ovo')
    except Exception:
        auc = None

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()