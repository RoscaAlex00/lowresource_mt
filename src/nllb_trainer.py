import csv
import json
import os
import matplotlib.pyplot as plt
import gc

from datasets import load_dataset, DatasetDict, concatenate_datasets
from nltk import word_tokenize
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np
from evaluate import load
from sacrebleu import corpus_bleu, corpus_chrf, BLEU, CHRF
from nltk.translate.meteor_score import single_meteor_score, meteor_score
import nltk

from src import utils

nltk.download('punkt')
nltk.download('wordnet')


def plot_training_loss(trainer):
    training_loss = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
    validation_loss = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]

    plt.figure(figsize=(10, 5))
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('../model_nllb/plots/training_validation_loss.png')
    plt.show()


class ModelEvaluator:
    def __init__(self, model_name, src_lang, tgt_lang):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples['src'], padding="max_length", truncation=True, max_length=128)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples['tgt'], padding="max_length", truncation=True, max_length=128)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def translate_and_save(self, dataset, output_file):
        src_sentences = dataset['train']['src'] + dataset['test']['src']  # Adjust the column name if necessary
        translations = []

        # Open file to write translations
        with open(output_file, 'w+', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['darija_ar', 'translated_eng'])  # Header

            # Translate each sentence and write to file
            for sentence in src_sentences:
                translated = self.translate_text(sentence)
                writer.writerow([sentence, translated])
                translations.append(translated)

        return translations

    def translate_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
        outputs = self.model.generate(**inputs,
                                      forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                                      max_length=128)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_sentences(self, sentences, source_lang, target_lang):
        self.model.to('cuda')
        translated_sentences = []
        for sentence in sentences:
            inputs = self.tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True).to('cuda')
            translated_tokens = self.model.generate(**inputs,
                                                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
                                                    max_length=128)
            translated_sentence = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            translated_sentences.append(translated_sentence)
        return translated_sentences

    def evaluate_model(self, dataset, output_file):
        src_sentences = dataset['src']
        tgt_sentences = dataset['tgt']

        predicted_sentences = []

        with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Original', 'Original Translation', 'Translated Sentence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for src, tgt in zip(src_sentences, tgt_sentences):
                inputs = self.tokenizer(src, return_tensors="pt", max_length=128, truncation=True).to('cuda')
                model = self.model.to('cuda')
                translated_tokens = model.generate(**inputs,
                                                   forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                                                   max_length=128)
                translated_sentence = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                predicted_sentences.append(translated_sentence)

                # Write to CSV
                writer.writerow(
                    {'Original': src, 'Original Translation': tgt, 'Translated Sentence': translated_sentence})

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

    def evaluate_model_new(self, dataset, output_file):
        src_sentences = dataset['src']
        tgt_sentences = dataset['tgt']

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
                model = self.model.to('cuda')
                translated_tokens = model.generate(**inputs,
                                                   forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang),
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

    def fine_tune_model(self, train_set, test_set, validation_set, output_dir='../results/model_nllb/checkpoints'):
        tokenized_train = train_set.map(self.tokenize_function, batched=True, remove_columns=['src', 'tgt'])
        tokenized_validation = validation_set.map(self.tokenize_function, batched=True, remove_columns=['src', 'tgt'])

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_total_limit=3,
            learning_rate=1e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            num_train_epochs=3,
            predict_with_generate=True,
            load_best_model_at_end=True,  # Load the best model based on validation loss
            metric_for_best_model="eval_loss"  # Metric to determine the best model
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_validation,  # Use validation set for evaluation
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        self.trainer.train()


# Example usage
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluator = ModelEvaluator(
        model_name='facebook/nllb-200-distilled-600M',
        src_lang='eng_Latn',
        tgt_lang='ary_Arab'
    )

    dataset_path = '../data/sentences_new_reversed.csv'
    prepared_datasets = utils.load_and_prepare_data(dataset_path)
    prepared_datasets['train'] = utils.load_backtranslation_data('../data/paraphrased_target_data.csv', prepared_datasets['train'])
    eval_bible = utils.load_arabench_data('../data/AraBench/bible.dev.mgr.0.ma.ar',
                                          '../data/AraBench/bible.dev.mgr.0.ma.en')
    eval_madar = utils.load_arabench_data('../data/AraBench/madar.dev.mgr.0.ma.ar',
                                          '../data/AraBench/madar.dev.mgr.0.ma.en')
    # print(prepared_datasets['train']['src'])
    # print(prepared_datasets['test']['tgt'])
    print("Evaluating model before fine-tuning...")
    pre_tune_results = evaluator.evaluate_model_new(prepared_datasets['test'],
                                                    '../results/model_nllb/outputs/predictions_en_ar_para.csv')

    pre_tune_bible = evaluator.evaluate_model_new(eval_bible,
                                                  '../results/model_nllb/outputs/predictions_en_ar_bible_para.csv')
    pre_tune_madar = evaluator.evaluate_model_new(eval_madar,
                                                  '../results/model_nllb/outputs/predictions_en_ar_madar_para.csv')
    print(pre_tune_results)
    print('BIBLE:')
    print(pre_tune_bible)
    print('MADAR:')
    print(pre_tune_madar)

    torch.cuda.empty_cache()
    gc.collect()
    print("Fine-tuning the model")
    evaluator.fine_tune_model(prepared_datasets['train'], prepared_datasets['test'], prepared_datasets['validation'])
    # plot_training_loss(evaluator.trainer)
    print("Evaluation after the fine-tuning...")
    after_tuning_results = evaluator.evaluate_model_new(prepared_datasets['test'],
                                                        '../results/model_nllb/outputs/predictions_en_ar_finetune_para.csv')
    print(after_tuning_results)

    print('BIBLE:')
    after_tune_bible = evaluator.evaluate_model_new(eval_bible,
                                                    '../results/model_nllb/outputs/predictions_en_ar_finetune_bible_para.csv')
    print(after_tune_bible)
    print('MADAR')
    after_tune_madar = evaluator.evaluate_model_new(eval_madar,
                                                    '../results/model_nllb/outputs/predictions_en_ar_finetune_madar_para.csv')
    print(after_tune_madar)

    # translation_output_file = '../data/translated_sentences_en_ar.csv'
    # evaluator.translate_and_save(prepared_datasets, translation_output_file)
    # print("Translations completed and saved.")

    # Forward translate Moroccan Arabic to English
    # original_sentences = [row['src'] for row in prepared_datasets['train']]
    # translated_sentences = evaluator.translate_sentences(original_sentences, source_lang='eng_Latn',
    #                                                               target_lang='ary_Arab')

    # Save translated sentences to a file for back-translation
    # output_file = '../data/forward_translations_nllb.csv'
    # with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Original Eng', 'Translated AR'])
    #     for original, translation in zip(original_sentences, translated_sentences):
    #         writer.writerow([original, translation])
    #
    # print(f"Forward translation completed and saved to {output_file}.")

    # Load the previously translated English sentences
    # input_file = '../data/forward_translations_nllb.csv'
    # original_sentences = []
    # translated_english = []
    # with open(input_file, newline='', encoding='utf-8') as csvfile:
    #     reader = csv.DictReader(csvfile)
    #     for row in reader:
    #         original_sentences.append(row['Original Eng'])
    #         translated_english.append(row['Translated AR'])
    #
    # # Back-translate English to Moroccan Arabic
    # back_translations = evaluator.translate_sentences(translated_english, source_lang='ary_Arab',
    #                                                               target_lang='eng_Latn')
    #
    # # Save the back-translations along with the original sentences
    # output_file = '../data/back_translations_nllb_en_ar.csv'
    # with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Original Eng', 'Translated AR', 'Back Translated Eng'])
    #     for original, english, back_translated in zip(original_sentences, translated_english, back_translations):
    #         writer.writerow([original, english, back_translated])
    #
    # print(f"Back translation completed and saved to {output_file}.")
