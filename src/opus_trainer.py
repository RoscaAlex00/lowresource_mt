import csv
import json
import os

import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset
from evaluate import load
from nltk import word_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np
from sacrebleu import corpus_bleu, corpus_chrf, BLEU, CHRF
from nltk.translate.meteor_score import single_meteor_score, meteor_score
import nltk

from src import utils

nltk.download('punkt')
nltk.download('wordnet')


class ModelEvaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples['src'], padding="max_length", truncation=True, max_length=128)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples['tgt'], padding="max_length", truncation=True, max_length=128)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def translate_sentences(self, sentences):
        self.model.to('cuda')
        translated_sentences = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True).to('cuda')
            translated_tokens = self.model.generate(**inputs,
                                                    max_length=128)
            translated_sentence = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_sentences.append(translated_sentence)
        return translated_sentences

    def evaluate_model_new(self, dataset, output_file):
        src_sentences = dataset['src']
        tgt_sentences = dataset['tgt']
        self.model.to('cuda')

        # Initialize metrics
        bleu = BLEU()
        chrf = CHRF()
        comet_metric = load("comet")

        predicted_sentences = []

        with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Original', 'Original Translation', 'Translated Sentence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for src, tgt in zip(src_sentences, tgt_sentences):
                inputs = self.tokenizer(src, return_tensors="pt", max_length=128, truncation=True).to('cuda')
                translated_tokens = self.model.generate(**inputs,
                                                        max_length=128)
                translated_sentence = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                predicted_sentences.append(translated_sentence)

                # Write to CSV
                writer.writerow(
                    {'Original': src, 'Original Translation': tgt, 'Translated Sentence': translated_sentence})

        # Calculate BLEU
        tgt_sentences_nested = [[sentence] for sentence in tgt_sentences]
        bleu_score = bleu.corpus_score(predicted_sentences, tgt_sentences_nested).score

        # Calculate METEOR
        predicted_tokens = [word_tokenize(sent) for sent in predicted_sentences]
        tgt_tokens = [word_tokenize(sent) for sent in tgt_sentences]
        meteor_scores = [meteor_score([tgt], pred) for tgt, pred in zip(tgt_tokens, predicted_tokens)]
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        # Calculate chrF
        chrf_score = chrf.corpus_score(predicted_sentences, tgt_sentences_nested).score

        # Calculate COMET
        comet_inputs = {
            "sources": src_sentences,
            "predictions": predicted_sentences,
            "references": tgt_sentences
        }

        # Compute COMET scores
        comet_scores = comet_metric.compute(**comet_inputs)
        # print(comet_scores)
        avg_comet = comet_scores["mean_score"]

        return {
            'BLEU': bleu_score,
            'METEOR': avg_meteor,
            'chrF': chrf_score,
            'COMET': avg_comet
        }

    def evaluate_model(self, dataset, output_file):
        src_sentences = dataset['src']
        tgt_sentences = dataset['tgt']

        predicted_sentences = []
        i = 0
        model = self.model.to('cuda')
        with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Original', 'Original Translation', 'Translated Sentence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            i = 0
            for src, tgt in zip(src_sentences, tgt_sentences):
                inputs = self.tokenizer(src, return_tensors="pt", max_length=128, truncation=True).to('cuda')
                translated_tokens = model.generate(**inputs,
                                                   max_length=128)
                translated_sentence = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                predicted_sentences.append(translated_sentence)

                # Write to CSV
                writer.writerow(
                    {'Original': src, 'Original Translation': tgt, 'Translated Sentence': translated_sentence})
                i += 1
                print(i)

        # Calculate BLEU
        tgt_sentences = [[sentence] for sentence in dataset['tgt']]
        bleu_score = corpus_bleu(predicted_sentences, tgt_sentences).score

        # Calculate METEOR
        predicted_tokens = [word_tokenize(sent) for sent in predicted_sentences]
        tgt_tokens = [word_tokenize(sent) for sent in dataset['tgt']]

        # Calculate METEOR
        meteor_scores = [meteor_score([tgt], pred) for tgt, pred in zip(tgt_tokens, predicted_tokens)]
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        chrf_score = corpus_chrf(predicted_sentences, tgt_sentences).score
        # print(predicted_sentences)
        # print(tgt_sentences)
        return {
            'BLEU': bleu_score,
            'METEOR': avg_meteor,
            'chrF': chrf_score
        }

    def fine_tune_model(self, train_set, validation_set, output_dir='../results/model_opus/checkpoints'):
        tokenized_train = train_set.map(self.tokenize_function, batched=True, remove_columns=['src', 'tgt'])
        tokenized_validation = validation_set.map(self.tokenize_function, batched=True, remove_columns=['src', 'tgt'])
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            learning_rate=1e-6,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=0,
            num_train_epochs=3,
            predict_with_generate=True,
            load_best_model_at_end=True,  # Load the best model based on validation loss
            metric_for_best_model="eval_loss"  # Metric to determine the best model
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validation,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        trainer.train()

        # self.model.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)


# Example usage
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluator = ModelEvaluator(
        model_name='Helsinki-NLP/opus-mt-en-ar',
    )
    # torch.set_float32_matmul_precision('medium')
    dataset_path = '../data/sentences_new_reversed.csv'
    # Load regular data
    prepared_datasets = utils.load_and_prepare_data(dataset_path)

    eval_bible = utils.load_arabench_data('../data/AraBench/bible.dev.mgr.0.ma.ar',
                                          '../data/AraBench/bible.dev.mgr.0.ma.en')
    eval_madar = utils.load_arabench_data('../data/AraBench/madar.dev.mgr.0.ma.ar',
                                          '../data/AraBench/madar.dev.mgr.0.ma.en')
    # Load regular + back_translated data vb
    prepared_datasets['train'] = utils.load_backtranslation_data('../data/paraphrased_target_data.csv', prepared_datasets['train'])
    # print(len(prepared_datasets['train']))
    # print(prepared_datasets['train']['tgt'])
    # Load regular + AraBench data
    # regular_data = utils.load_and_prepare_data(dataset_path)
    # bible_en_path = '../data/AraBench/madar.dev.mgr.0.ma.en'
    # bible_ar_path = '../data/AraBench/madar.dev.mgr.0.ma.ar'
    #
    # bible_data = utils.load_arabench_data(bible_en_path, bible_ar_path)
    #
    # prepared_datasets = utils.merge_datasets(regular_data, bible_data)
    print("Evaluating model before fine-tuning...")
    pre_tune_results = evaluator.evaluate_model_new(prepared_datasets['test'],
                                                    '../results/model_opus/outputs/predictions_en_ar_para.csv')
    # print(pre_tune_results)

    pre_tune_bible = evaluator.evaluate_model_new(eval_bible,
                                                  '../results/model_opus/outputs/predictions_en_ar_para_bible.csv')
    pre_tune_madar = evaluator.evaluate_model_new(eval_madar,
                                                  '../results/model_opus/outputs/predictions_en_ar_para_madar.csv')
    print(pre_tune_results)
    print('BIBLE:')
    print(pre_tune_bible)
    print('MADAR:')
    print(pre_tune_madar)
    print("Fine-tuning the model")
    evaluator.fine_tune_model(prepared_datasets['train'], prepared_datasets['validation'])

    print("Evaluation after the fine-tuning...")
    after_tuning_results = evaluator.evaluate_model_new(prepared_datasets['test'],
                                                        '../results/model_opus/outputs/predictions_en_ar_para_finetuned.csv')
    print(after_tuning_results)
    print('BIBLE:')
    after_tune_bible = evaluator.evaluate_model_new(eval_bible,
                                                    '../results/model_opus/outputs/predictions_en_ar_para_finetuned_bible.csv')
    print(after_tune_bible)
    print('MADAR')
    after_tune_madar = evaluator.evaluate_model_new(eval_madar,
                                                    '../results/model_opus/outputs/predictions_en_ar_para_finetuned_madar.csv')
    print(after_tune_madar)

    # Load the previously translated English sentences
    # input_file = '../data/forward_translations_opus_en_ar.csv'
    # original_sentences = []
    # translated_english = []
    # with open(input_file, newline='', encoding='utf-8') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         original_sentences.append(row['Original English'])
    #         translated_english.append(row['Translated AR'])
    #
    # # Back-translate English to Moroccan Arabic
    # back_translations = evaluator.translate_sentences(translated_english)
    #
    # # Save the back-translations along with the original sentences
    # output_file = '../data/back_translations_opus_en_ar.csv'
    # with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Original English', 'Translated AR', 'Back Translated English'])
    #     for original, english, back_translated in zip(original_sentences, translated_english, back_translations):
    #         writer.writerow([original, english, back_translated])
    #
    # print(f"Back translation completed and saved to {output_file}.")
