from pathlib import Path

from nltk import word_tokenize
from nltk.translate.meteor_score import meteor_score
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import (EncoderDecoderModel, BertTokenizer, MarianTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from transformers.trainer_utils import set_seed
from datasets import load_dataset
from evaluate import load
import os
import csv
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging
from utils import load_and_prepare_data, load_arabench_data, load_backtranslation_data

logger = logging.getLogger(__name__)

# Initialize COMET metric (will be computed separately)
comet_metric = load("comet")


@hydra.main(config_path='configs', config_name='default')
def main(cfg: DictConfig):
    # Set seed for reproducibility
    if cfg.seed != 0:
        set_seed(cfg.seed)

    # Log parameters
    logger.info(OmegaConf.to_yaml(cfg))

    full_path = Path(get_original_cwd()) / '../data/sentences_new.csv'
    dataset = load_and_prepare_data(str(full_path))

    # BACK-TRANSLATION DATA LOAD (if needed)
    full_path_bt = Path(get_original_cwd()) / '../data/bt_en_ar_nllb.csv'
    dataset['train'] = load_backtranslation_data(str(full_path_bt), dataset['train'])

    # Load DarijaBERT tokenizer and MarianMT tokenizer for English
    tokenizer_src = BertTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
    tokenizer_tgt = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")

    # Load DarijaBERT encoder and MarianMT decoder
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("SI2M-Lab/DarijaBERT", "Helsinki-NLP/opus-mt-ar-en")

    model.config.decoder_start_token_id = tokenizer_tgt.pad_token_id
    model.config.pad_token_id = tokenizer_src.pad_token_id
    model.config.eos_token_id = tokenizer_tgt.eos_token_id
    model.config.max_length = cfg.generate.max_length
    model.config.min_length = cfg.generate.min_length
    model.config.early_stopping = cfg.generate.early_stopping
    model.config.num_beams = cfg.generate.num_beams
    model.config.no_repeat_ngram_size = cfg.generate.no_repeat_ngram_size

    # Preprocessing function
    def preprocess_function(batch):
        # Tokenize Moroccan Arabic (Darija) as input and English as output
        inputs = tokenizer_src(batch['src'], padding="max_length", truncation=True,
                                 max_length=cfg.tokenizer.encoder_max_length)
        outputs = tokenizer_tgt(batch['tgt'], padding="max_length", truncation=True,
                                max_length=cfg.tokenizer.decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids

        # Replace pad token with -100 in labels to ignore it in loss calculation
        batch["labels"] = [
            [-100 if token == tokenizer_tgt.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]
        return batch

    tokenized_train_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=['src', 'tgt'])
    tokenized_eval_dataset = dataset['validation'].map(preprocess_function, batched=True, remove_columns=['src', 'tgt'])
    tokenized_test_dataset = dataset['test'].map(preprocess_function, batched=True, remove_columns=['src', 'tgt'])

    data_collator = DataCollatorForSeq2Seq(tokenizer_src, model=model)

    def write_predictions_to_csv(trainer, dataset, output_file, src_sentences, ref_sentences):
        predictions = trainer.predict(dataset)
        decoded_preds = tokenizer_tgt.batch_decode(predictions.predictions, skip_special_tokens=True)

        with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Original', 'Original Translation', 'Translated Sentence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for src, ref, pred in zip(src_sentences, ref_sentences, decoded_preds):
                writer.writerow({
                    'Original': src,
                    'Original Translation': ref,
                    'Translated Sentence': pred
                })

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Decode predictions and labels
        decoded_preds = tokenizer_tgt.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer_tgt.pad_token_id
        decoded_labels = tokenizer_tgt.batch_decode(labels_ids, skip_special_tokens=True)

        # Calculate BLEU
        bleu_score = corpus_bleu(decoded_preds, [decoded_labels]).score

        # Calculate METEOR
        predicted_tokens = [word_tokenize(sent) for sent in decoded_preds]
        tgt_tokens = [word_tokenize(sent) for sent in decoded_labels]
        meteor_scores = [meteor_score([tgt], pred) for tgt, pred in zip(tgt_tokens, predicted_tokens)]
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        # Calculate chrF
        chrf_score = corpus_chrf(decoded_preds, [decoded_labels]).score

        return {
            "bleu": bleu_score,
            "meteor": avg_meteor,
            "chrF": chrf_score,
        }

    # Seq2Seq Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f'{cfg.model.name.split("/")[-1]}',  # output directory
        num_train_epochs=cfg.trainer.num_epochs,         # number of training epochs
        per_device_train_batch_size=cfg.trainer.batch_size,
        per_device_eval_batch_size=cfg.trainer.batch_size,
        learning_rate=cfg.trainer.learning_rate,
        warmup_steps=cfg.trainer.warmup_steps,
        logging_strategy='epoch',
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        evaluation_strategy='epoch',
        predict_with_generate=True,
        overwrite_output_dir=True,
        save_total_limit=3,
        fp16=cfg.trainer.fp16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    orig_eval_bible = load_arabench_data(
        Path(get_original_cwd()) / '../data/AraBench/bible.dev.mgr.0.ma.en',
        Path(get_original_cwd()) / '../data/AraBench/bible.dev.mgr.0.ma.ar'
    )
    orig_eval_madar = load_arabench_data(
        Path(get_original_cwd()) / '../data/AraBench/madar.dev.mgr.0.ma.en',
        Path(get_original_cwd()) / '../data/AraBench/madar.dev.mgr.0.ma.ar'
    )
    eval_bible = orig_eval_bible.map(preprocess_function, batched=True, remove_columns=['src', 'tgt'])
    eval_madar = orig_eval_madar.map(preprocess_function, batched=True, remove_columns=['src', 'tgt'])

    # -------------------------------------------
    # Evaluate BEFORE training
    # -------------------------------------------
    # Test set evaluation
    # logger.info("Evaluating model on test set before training...")
    # initial_test_output = trainer.predict(tokenized_test_dataset)
    # initial_test_metrics = initial_test_output.metrics
    # initial_test_metrics['model_name'] = cfg.model.name
    #
    # # Compute COMET
    # decoded_test_preds = tokenizer_tgt.batch_decode(initial_test_output.predictions, skip_special_tokens=True)
    # decoded_test_labels = tokenizer_tgt.batch_decode(initial_test_output.label_ids, skip_special_tokens=True)
    # comet_inputs_test = {
    #     "sources": dataset['test']['src'],
    #     "predictions": decoded_test_preds,
    #     "references": decoded_test_labels
    # }
    # comet_scores_test = comet_metric.compute(**comet_inputs_test)
    # initial_test_metrics['comet'] = comet_scores_test["mean_score"]
    # logger.info(f"Initial test set performance: {initial_test_metrics}")
    #
    # output_file_path = Path(get_original_cwd()) / '../results/models/encoderdecoder/initial_predictions_test_bt.csv'
    # write_predictions_to_csv(trainer, tokenized_test_dataset, str(output_file_path),
    #                          dataset['test']['src'], dataset['test']['tgt'])
    #
    # # MADAR evaluation
    # logger.info("Evaluating on MADAR dataset...")
    # initial_madar_output = trainer.predict(eval_madar)
    # initial_madar_metrics = initial_madar_output.metrics
    # initial_madar_metrics['model_name'] = cfg.model.name
    # decoded_madar_preds = tokenizer_tgt.batch_decode(initial_madar_output.predictions, skip_special_tokens=True)
    # decoded_madar_labels = tokenizer_tgt.batch_decode(initial_madar_output.label_ids, skip_special_tokens=True)
    # comet_inputs_madar = {
    #     "sources": orig_eval_madar['src'],
    #     "predictions": decoded_madar_preds,
    #     "references": decoded_madar_labels
    # }
    # comet_scores_madar = comet_metric.compute(**comet_inputs_madar)
    # initial_madar_metrics['comet'] = comet_scores_madar["mean_score"]
    # logger.info(f"Initial MADAR performance: {initial_madar_metrics}")
    #
    # output_file_path = Path(get_original_cwd()) / '../results/models/encoderdecoder/initial_predictions_madar_bt.csv'
    # write_predictions_to_csv(trainer, eval_madar, str(output_file_path),
    #                          orig_eval_madar['src'], orig_eval_madar['tgt'])
    #
    # # BIBLE evaluation
    # logger.info("Evaluating on BIBLE dataset...")
    # initial_bible_output = trainer.predict(eval_bible)
    # initial_bible_metrics = initial_bible_output.metrics
    # initial_bible_metrics['model_name'] = cfg.model.name
    # decoded_bible_preds = tokenizer_tgt.batch_decode(initial_bible_output.predictions, skip_special_tokens=True)
    # decoded_bible_labels = tokenizer_tgt.batch_decode(initial_bible_output.label_ids, skip_special_tokens=True)
    # comet_inputs_bible = {
    #     "sources": orig_eval_bible['src'],
    #     "predictions": decoded_bible_preds,
    #     "references": decoded_bible_labels
    # }
    # comet_scores_bible = comet_metric.compute(**comet_inputs_bible)
    # initial_bible_metrics['comet'] = comet_scores_bible["mean_score"]
    # logger.info(f"Initial BIBLE performance: {initial_bible_metrics}")
    #
    # output_file_path = Path(get_original_cwd()) / '../results/models/encoderdecoder/initial_predictions_bible_bt.csv'
    # write_predictions_to_csv(trainer, eval_bible, str(output_file_path),
    #                          orig_eval_bible['src'], orig_eval_bible['tgt'])

    # -------------------------------------------
    # Train the model
    # -------------------------------------------
    trainer.train()


    if cfg.trainer.predict:
        # Test set evaluation after training
        logger.info("Evaluating model on test set after training...")
        final_test_output = trainer.predict(tokenized_test_dataset)
        final_test_metrics = final_test_output.metrics
        final_test_metrics['model_name'] = cfg.model.name
        decoded_final_test_preds = tokenizer_tgt.batch_decode(final_test_output.predictions, skip_special_tokens=True)
        decoded_final_test_labels = tokenizer_tgt.batch_decode(final_test_output.label_ids, skip_special_tokens=True)
        comet_inputs_final_test = {
            "sources": dataset['test']['src'],
            "predictions": decoded_final_test_preds,
            "references": decoded_final_test_labels
        }
        comet_scores_final_test = comet_metric.compute(**comet_inputs_final_test)
        final_test_metrics['comet'] = comet_scores_final_test["mean_score"]
        logger.info(f"Final test set performance: {final_test_metrics}")

        output_file_path = Path(get_original_cwd()) / '../results/models/encoderdecoder/final_predictions_test_bt.csv'
        write_predictions_to_csv(trainer, tokenized_test_dataset, str(output_file_path),
                                 dataset['test']['src'], dataset['test']['tgt'])

        # MADAR evaluation after training
        logger.info("Evaluating on MADAR dataset...")
        final_madar_output = trainer.predict(eval_madar)
        final_madar_metrics = final_madar_output.metrics
        final_madar_metrics['model_name'] = cfg.model.name
        decoded_final_madar_preds = tokenizer_tgt.batch_decode(final_madar_output.predictions, skip_special_tokens=True)
        decoded_final_madar_labels = tokenizer_tgt.batch_decode(final_madar_output.label_ids, skip_special_tokens=True)
        comet_inputs_final_madar = {
            "sources": orig_eval_madar['src'],
            "predictions": decoded_final_madar_preds,
            "references": decoded_final_madar_labels
        }
        comet_scores_final_madar = comet_metric.compute(**comet_inputs_final_madar)
        final_madar_metrics['comet'] = comet_scores_final_madar["mean_score"]
        logger.info(f"Final MADAR performance: {final_madar_metrics}")

        output_file_path = Path(get_original_cwd()) / '../results/models/encoderdecoder/final_predictions_madar_bt.csv'
        write_predictions_to_csv(trainer, eval_madar, str(output_file_path),
                                 orig_eval_madar['src'], orig_eval_madar['tgt'])

        # BIBLE evaluation after training
        logger.info("Evaluating on BIBLE dataset...")
        final_bible_output = trainer.predict(eval_bible)
        final_bible_metrics = final_bible_output.metrics
        final_bible_metrics['model_name'] = cfg.model.name
        decoded_final_bible_preds = tokenizer_tgt.batch_decode(final_bible_output.predictions, skip_special_tokens=True)
        decoded_final_bible_labels = tokenizer_tgt.batch_decode(final_bible_output.label_ids, skip_special_tokens=True)
        comet_inputs_final_bible = {
            "sources": orig_eval_bible['src'],
            "predictions": decoded_final_bible_preds,
            "references": decoded_final_bible_labels
        }
        comet_scores_final_bible = comet_metric.compute(**comet_inputs_final_bible)
        final_bible_metrics['comet'] = comet_scores_final_bible["mean_score"]
        logger.info(f"Final BIBLE performance: {final_bible_metrics}")

        output_file_path = Path(get_original_cwd()) / '../results/models/encoderdecoder/final_predictions_bible_bt.csv'
        write_predictions_to_csv(trainer, eval_bible, str(output_file_path),
                                 orig_eval_bible['src'], orig_eval_bible['tgt'])

    # Save model if enabled
    if cfg.trainer.save_model:
        model_path = Path(get_original_cwd()) / Path(f'models/{cfg.model.name}_{cfg.trainer.num_epochs}')
        model_path.mkdir(exist_ok=True, parents=True)
        trainer.save_model(model_path)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
