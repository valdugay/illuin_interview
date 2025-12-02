import pandas as pd
import joblib

LABELS = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']


def predict(input_path, model_path="models/tfidf_lgbm_full_model.joblib"):
    """
    Predict tags for the problems in an input JSONL file.
    """

    df = pd.read_json(input_path, lines=True)
    X = df["full_description"] + "\n" + df["source_code"]
    model = joblib.load(model_path)

    
    vectors = model.predict(X)

    
    predictions = []
    for row in vectors:
        tags = [LABELS[i] for i, val in enumerate(row) if val == 1]
        predictions.append(tags)

    return predictions



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=False,
                        default="models/tfidf_lgbm_full_model.joblib")

    args = parser.parse_args()

    preds = predict(args.input, args.model)
    print(preds)
