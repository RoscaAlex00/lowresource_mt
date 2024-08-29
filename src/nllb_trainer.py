import csv
import json
import os
import matplotlib.pyplot as plt

from datasets import load_dataset, DatasetDict, concatenate_datasets
from nltk import word_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np
from sacrebleu import corpus_bleu, corpus_chrf
from nltk.translate.meteor_score import single_meteor_score, meteor_score
import nltk

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

    # def load_and_prepare_data(self, file_path):
    #     # Load the dataset from a CSV file
    #     raw_datasets = load_dataset('csv', data_files=file_path)
    #
    #     # Remove unnecessary columns
    #     raw_datasets = raw_datasets.remove_columns('darija')
    #
    #     # Split the dataset into train and test sets
    #     split_datasets = raw_datasets['train'].train_test_split(test_size=0.2, seed=559)
    #     train_dataset = split_datasets['train']
    #     test_dataset = split_datasets['test']
    #
    #     # Further split the train dataset into train and validation sets
    #     validation_split = test_dataset.train_test_split(test_size=0.25,
    #                                                       seed=552)  # 10% of the original train set for validation
    #     test_dataset = validation_split['train']
    #     validation_dataset = validation_split['test']
    #
    #     # Filter out examples without source or target from train, validation, and test sets
    #     train_dataset = train_dataset.filter(lambda example: example['src'] is not None and example['tgt'] is not None)
    #     validation_dataset = validation_dataset.filter(
    #         lambda example: example['src'] is not None and example['tgt'] is not None)
    #     test_dataset = test_dataset.filter(lambda example: example['src'] is not None and example['tgt'] is not None)
    #
    #     # Print dataset shapes
    #     print(
    #         f"Train set: {len(train_dataset)}, Validation set: {len(validation_dataset)}, Test set: {len(test_dataset)}")
    #
    #     # Return a dictionary of the datasets
    #     return {'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset}

    # def load_and_prepare_data(self, file_path):
    #     raw_datasets = load_dataset('csv', data_files=file_path)
    #     raw_datasets = raw_datasets.remove_columns('darija')
    #     raw_datasets = raw_datasets['train'].train_test_split(test_size=0.15, seed=552)
    #
    #     raw_datasets['train'] = raw_datasets['train'].filter(
    #         lambda example: example['src'] is not None and example['tgt'] is not None)
    #     raw_datasets['test'] = raw_datasets['test'].filter(
    #         lambda example: example['src'] is not None and example['tgt'] is not None)
    #
    #     print(raw_datasets)
    #     return raw_datasets

    def load_and_prepare_data(self, original_file_path, additional_file_path):
        original_datasets = load_dataset('csv', data_files=original_file_path)
        additional_datasets = load_dataset('csv', data_files=additional_file_path)
        print(additional_datasets['train'][0])

        # Remove unnecessary columns
        original_datasets = original_datasets.remove_columns('darija')

        # Split the original dataset
        split_datasets = original_datasets['train'].train_test_split(test_size=0.15, seed=552)
        train_dataset = split_datasets['train']
        test_dataset = split_datasets['test']

        # Concatenate the additional data to the training dataset
        additional_train_dataset = additional_datasets['train']
        combined_train_dataset = concatenate_datasets([train_dataset, additional_train_dataset])

        # Filter out examples without source or target from train and test sets
        combined_train_dataset = combined_train_dataset.filter(
            lambda example: example['src'] is not None and example['tgt'] is not None)
        test_dataset = test_dataset.filter(lambda example: example['src'] is not None and example['tgt'] is not None)

        print(f"Train set: {len(combined_train_dataset)}, Test set: {len(test_dataset)}")
        return {'train': combined_train_dataset, 'test': test_dataset}

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples['src'], padding="max_length", truncation=True, max_length=128)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples['tgt'], padding="max_length", truncation=True, max_length=128)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def translate_and_save(self, dataset, output_file):
        src_sentences = dataset['train']['src'] + dataset['test']['src'] # Adjust the column name if necessary
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
        print(predicted_sentences)
        print(tgt_sentences)
        return {
            'BLEU': bleu_score,
            'METEOR': avg_meteor,
            'chrF': chrf_score
        }

    def fine_tune_model(self, train_set, test_set, output_dir='../model_nllb/checkpoints'):
        tokenized_train = train_set.map(self.tokenize_function, batched=True, remove_columns=['src', 'tgt'])
        tokenized_test = test_set.map(self.tokenize_function, batched=True, remove_columns=['src', 'tgt'])
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=2,
            predict_with_generate=True
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        self.trainer.train()
        # self.model.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)


# Example usage
if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluator = ModelEvaluator(
        model_name='facebook/nllb-200-distilled-600M',
        src_lang='ary_Arab',
        tgt_lang='eng_Latn'
    )

    dataset_path = '../data/sentences_nllb.csv'
    prepared_datasets = evaluator.load_and_prepare_data(dataset_path, '../data/back_translated_sentences.csv')
    print("Evaluating model before fine-tuning...")
    pre_tune_results = evaluator.evaluate_model(prepared_datasets['test'], '../model_nllb/outputs/predictions_pre.csv')
    print(pre_tune_results)
    print("Fine-tuning the model")
    evaluator.fine_tune_model(prepared_datasets['train'], prepared_datasets['test'])
    plot_training_loss(evaluator.trainer)
    print("Evaluation after the fine-tuning...")
    after_tuning_results = evaluator.evaluate_model(prepared_datasets['test'],
                                                    '../model_nllb/outputs/predictions_epoch2.csv')
    print(after_tuning_results)

    # translation_output_file = '../data/translated_sentences.csv'
    # evaluator.translate_and_save(prepared_datasets, translation_output_file)
    # print("Translations completed and saved.")
