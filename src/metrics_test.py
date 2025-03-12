import os
import pandas as pd
import evaluate
import argparse


def compute_metrics(csv_file):
    # Load the evaluation metrics from HF
    bleu = evaluate.load("bleu")
    sacrebleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    chrf = evaluate.load("chrf")
    comet = evaluate.load("comet")

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Ensure required columns exist
    required_columns = ["Original", "Original Translation", "Translated Sentence"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")

    # Extract the necessary columns
    source_sentences = df["Original"].tolist()
    reference_sentences = df["Original Translation"].tolist()
    translated_sentences = df["Translated Sentence"].tolist()

    # Compute BLEU
    bleu_score = bleu.compute(
        predictions=translated_sentences,
        references=[[ref] for ref in reference_sentences]
    )

    # Compute SacreBLEU (sacrebleu expects the references as a list of reference lists)
    sacrebleu_score = sacrebleu.compute(
        predictions=translated_sentences,
        references=[[ref] for ref in reference_sentences]
    )

    # Compute METEOR
    meteor_score = meteor.compute(
        predictions=translated_sentences,
        references=reference_sentences
    )

    # Compute chrF
    chrf_score = chrf.compute(
        predictions=translated_sentences,
        references=reference_sentences
    )

    # Compute COMET (requires specific input format)
    comet_score = comet.compute(
        predictions=translated_sentences,
        references=reference_sentences,
        sources=source_sentences
    )

    # Print results
    print("BLEU Score:", bleu_score["bleu"])
    print("SacreBLEU Score:", sacrebleu_score["score"])
    print("METEOR Score:", meteor_score["meteor"])
    print("chrF Score:", chrf_score["score"])
    print("COMET Score:", comet_score["mean_score"])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    file_path = "../results/model_nllb/outputs/predictions_ar_en_finetune.csv"
    compute_metrics(file_path)
