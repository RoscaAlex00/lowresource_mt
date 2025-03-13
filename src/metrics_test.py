import os
import pandas as pd
import evaluate
import argparse
from sacrebleu import corpus_bleu, corpus_chrf, BLEU, CHRF  # sacreBLEU imports
from nltk.translate.bleu_score import sentence_bleu  # NLTK sentence-level BLEU


def compute_metrics(csv_file):
    # Load the evaluation metrics from the HF evaluate library
    bleu_evaluate = evaluate.load("bleu")
    sacrebleu_evaluate = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    chrf_evaluate = evaluate.load("chrf")
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

    # Compute metrics using the evaluate library implementations:
    bleu_score_evaluate = bleu_evaluate.compute(
        predictions=translated_sentences,
        references=[[ref] for ref in reference_sentences]
    )
    sacrebleu_score_evaluate = sacrebleu_evaluate.compute(
        predictions=translated_sentences,
        references=[[ref] for ref in reference_sentences]
    )
    meteor_score = meteor.compute(
        predictions=translated_sentences,
        references=reference_sentences
    )
    chrf_score_evaluate = chrf_evaluate.compute(
        predictions=translated_sentences,
        references=reference_sentences
    )
    comet_score = comet.compute(
        predictions=translated_sentences,
        references=reference_sentences,
        sources=source_sentences
    )

    # --------------------------------------------------------
    # Print results:
    # --------------------------------------------------------
    print("Using evaluate library:")
    print("BLEU Score (evaluate):", bleu_score_evaluate["bleu"])
    print("SacreBLEU Score (evaluate):", sacrebleu_score_evaluate["score"])
    print("METEOR Score (evaluate):", meteor_score["meteor"])
    print("chrF Score (evaluate):", chrf_score_evaluate["score"])
    print("COMET Score (evaluate):", comet_score["mean_score"])

    # Compute BLEU score
    bleu = corpus_bleu(translated_sentences, [reference_sentences])
    print(f"BLEU score: {bleu.score:.2f}")

    # Compute CHRF score
    chrf = corpus_chrf(translated_sentences, [reference_sentences])
    print(f"CHRF score: {chrf.score:.2f}")




if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    file_path = "../results/model_opus/outputs/predictions_ar_en_finetuned_random.csv"
    compute_metrics(file_path)
