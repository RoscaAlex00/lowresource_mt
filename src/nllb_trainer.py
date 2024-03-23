import json
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
import nltk

nltk.download('punkt')
nltk.download('wordnet')


class ModelEvaluator:
    def __init__(self, model_name, src_lang, tgt_lang):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def load_and_prepare_data(self, file_path):
        raw_datasets = load_dataset('csv', data_files=file_path)
        raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1)
        return raw_datasets

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples['src'], padding="max_length", truncation=True, max_length=128)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples['tgt'], padding="max_length", truncation=True, max_length=128)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def evaluate_model(self, dataset):
        src_sentences = dataset['src']
        tgt_sentences = [[sentence] for sentence in dataset['tgt']]

        predicted_sentences = []
        for src in src_sentences:
            inputs = self.tokenizer(src, return_tensors="pt", max_length=128, truncation=True)
            translated_tokens = self.model.generate(**inputs,
                                                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
                                                    max_length=128)
            translated_sentence = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            predicted_sentences.append(translated_sentence)

        # Calculate BLEU
        bleu_score = corpus_bleu([[word_tokenize(sent) for sent in tgt] for tgt in tgt_sentences],
                                 [word_tokenize(sent) for sent in predicted_sentences])

        # Calculate METEOR
        meteor_scores = [single_meteor_score(ref, hyp) for ref_list, hyp in zip(tgt_sentences, predicted_sentences) for
                         ref in ref_list]
        avg_meteor = np.mean(meteor_scores)

        return {
            'BLEU': bleu_score,
            'METEOR': avg_meteor,
        }

    def fine_tune_model(self, dataset, output_dir='./model'):
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True, remove_columns=['src', 'tgt'])
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
            num_train_epochs=3,
            predict_with_generate=True
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


# Example usage
if __name__ == "__main__":
    evaluator = ModelEvaluator(
        model_name='facebook/nllb-200-distilled-600M',
        src_lang='eng',
        tgt_lang='ary_arab'
    )

    # Load your dataset, ensure it has 'src' and 'tgt' columns for source and target texts
    dataset_path = 'path/to/your/dataset.csv'  # Adjust this path
    prepared_datasets = evaluator.load_and_prepare_data(dataset_path)
    print("Evaluating model before fine-tuning...")
    pre_tune_results = evaluator.evaluate_model(prepared_datasets['test'])
    print(pre_tune_results)
    print("Fine-tuning the model")
    evaluator.fine_tune_model(prepared_datasets['train'])
    print("Evaluation after the fine-tuning...")
    after_tuning_results = evaluator.evaluate_model(prepared_datasets['test'])
    print(after_tuning_results)
