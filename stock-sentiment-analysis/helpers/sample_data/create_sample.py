from models.finbert import predict_sentiment
def create_sample(df):
    sample_df = df.sample(n=200, random_state=42)

    # Predict sentiment
    preds = predict_sentiment(sample_df["Title"].tolist())

    # Add predictions to DataFrame
    sample_df["finbert_pred"] = [p["label"] for p in preds]
    sample_df["finbert_confidence"] = [p["score"] for p in preds]

    # Export for manual labeling
    sample_df.to_csv("manual_label_check.csv", index=False)
    print("200 predictions exported to 'manual_label_check.csv' — ready for manual review.")