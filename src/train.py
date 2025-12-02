
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import joblib


best_params_full = {'num_leaves': 150, 'max_depth': 27, 'learning_rate': 0.02341509850272208, 'n_estimators': 299, 'class_weight': 'balanced', 'verbose': -1}

pipeline_final = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', OneVsRestClassifier(LGBMClassifier(**best_params_full)))
])



def get_labels(df):
    """ Return 8-length binary vectors representing the labels """

    focus_tags = ['math', 'graphs', 'strings', 'number theory',
              'trees', 'geometry', 'games', 'probabilities']

    
    def encode_tags(tag_list):
        return [1 if t in tag_list else 0 for t in focus_tags]

    labels_vector = df["tags"].apply(encode_tags)

    return np.vstack(labels_vector.values)


def train(data_path, output_model_name):
    """ Train the final model on the full training data and save it. """
    df = pd.read_json(data_path, lines=True)

    X = df["full_description"] + "\n" + df["source_code"]
    Y = get_labels(df)

    pipeline_final.fit(X, Y)

    # Save the model
    joblib.dump(pipeline_final, "models/" + output_model_name)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_model_name", required=True)
    args = parser.parse_args()

    train(args.data_path, args.output_model_name)
