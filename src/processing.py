import os
import json
import pandas as pd
from langdetect import detect

def load_json_folder(folder_path) -> pd.DataFrame:
    '''
    Load all JSON files in a folder into a pandas DataFrame.'''
    rows = []

    # Get all files in the given folder
    for fname in os.listdir(folder_path):
        
        full_path = os.path.join(folder_path, fname)

        with open(full_path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except Exception as e:
                print("Error in :", fname)
                raise e

        obj["file_name"] = fname
        rows.append(obj)

    return pd.DataFrame(rows)


def build_full_description(row):
    """ Build the full problem description from its components. """
    parts = []

    def add_section(title, content):
        if content and str(content).strip() != "":
            parts.append(f"{title}:\n{content}")

    add_section("Problem Description", row.get("prob_desc_description"))
    add_section("Input Specification", row.get("prob_desc_input_spec"))
    add_section("Output Specification", row.get("prob_desc_output_spec"))
    add_section("Notes", row.get("prob_desc_notes"))
    add_section("Sample Inputs", row.get("prob_desc_sample_inputs"))
    add_section("Sample Outputs", row.get("prob_desc_sample_outputs"))

    add_section("Time Limit", row.get("prob_desc_time_limit"))
    add_section("Memory Limit", row.get("prob_desc_memory_limit"))

    io_info = []
    if row.get("prob_desc_input_from"):
        io_info.append(f"Input From: {row['prob_desc_input_from']}")
    if row.get("prob_desc_output_to"):
        io_info.append(f"Output To: {row['prob_desc_output_to']}")
    if io_info:
        parts.append("I/O Format:\n" + "\n".join(io_info))

    add_section("Difficulty", row.get("difficulty"))

    # final concatenation
    return "\n\n".join(parts)

def filter_tags(tags, relevant_tags):
    return [tag for tag in tags if tag in relevant_tags]


def detect_language(text):
    try:
        return detect(text)
    except:
        return "error"

def process(input_path = "../data/raw/code_classification_dataset/", output_path= "../data/processed/cleaned_code_classification_dataset.jsonl"):
    """ Complete pipeline to process the raw data and save the cleaned DataFrame. """
    df = load_json_folder(input_path)

    df = df.drop(columns=["prob_desc_created_at"])

    # We apply our function to build the full description
    df["full_description"] = df.apply(build_full_description, axis=1)

    # And we can delete the used columns
    df = df.drop(columns=[
        "prob_desc_description",
        "prob_desc_input_spec",
        "prob_desc_output_spec",
        "prob_desc_notes",
        "prob_desc_sample_inputs",
        "prob_desc_sample_outputs",
        "prob_desc_time_limit",
        "prob_desc_memory_limit",
        "prob_desc_input_from",
        "prob_desc_output_to",
        "difficulty"
    ])

    df = df.set_index("src_uid")


    df = df.drop(columns=[
        "code_uid",
        "file_name",
        "hidden_unit_tests"
    ])


    df = df.drop(columns=["lang", "lang_cluster"])
    df = df.drop(columns=["exec_outcome"])

    

    relevant_tags = ['math',  'graphs',  'strings',  'number theory', 'trees', 'geometry', 'games', 'probabilities']

    df['tags'] = df['tags'].apply(lambda tags: filter_tags(tags, relevant_tags))

    df = df[df['tags'].map(len) > 0]

    languages = df['full_description'].apply(detect_language)
    df = df[languages != 'ru']

    df = df.reset_index()

    df.to_json(output_path, orient="records", lines=True)



def process_nolabel(input_path = "../data/raw/code_classification_dataset/", output_path= "../data/processed/cleaned_code_classification_dataset.jsonl"):
    """ Complete pipeline to process the raw data and save the cleaned DataFrame for NON labeled data. """
    df = load_json_folder(input_path)

    df = df.drop(columns=["prob_desc_created_at"])

    # We apply our function to build the full description
    df["full_description"] = df.apply(build_full_description, axis=1)

    # And we can delete the used columns
    df = df.drop(columns=[
        "prob_desc_description",
        "prob_desc_input_spec",
        "prob_desc_output_spec",
        "prob_desc_notes",
        "prob_desc_sample_inputs",
        "prob_desc_sample_outputs",
        "prob_desc_time_limit",
        "prob_desc_memory_limit",
        "prob_desc_input_from",
        "prob_desc_output_to",
        "difficulty"
    ])

    df = df.set_index("src_uid")


    df = df.drop(columns=[
        "code_uid",
        "file_name",
        "hidden_unit_tests"
    ])


    df = df.drop(columns=["lang", "lang_cluster"])
    df = df.drop(columns=["exec_outcome"])

    languages = df['full_description'].apply(detect_language)
    df = df[languages != 'ru']

    df = df.reset_index()

    df.to_json(output_path, orient="records", lines=True)

def load_processed_data(file_path = "../data/processed/cleaned_code_classification_dataset.jsonl") -> pd.DataFrame:
    """ Load the processed data from a JSONL file. """
    df =  pd.read_json(file_path, lines=True)
    df = df.set_index("src_uid")
    return df.drop(columns=["index"])







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", action="store_true")

    args = parser.parse_args()

    if args.label:
        process(input_path=args.input, output_path=args.output)
    else:
        process_nolabel(input_path=args.input, output_path=args.output)