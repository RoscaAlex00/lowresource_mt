from pathlib import Path

from sacrebleu import corpus_bleu
from transformers import (EncoderDecoderModel, BertTokenizer, MarianTokenizer,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from transformers.trainer_utils import set_seed
from datasets import load_dataset
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path='configs', config_name='default')
def main(cfg: DictConfig):
    # Set seed for reproducibility
    if cfg.seed != 0:
        set_seed(cfg.seed)

    # Log parameters
    logger.info(OmegaConf.to_yaml(cfg))

    # Load your dataset
    dataset = load_dataset('path_to_your_dataset')

    # Load DarijaBERT tokenizer and MarianMT tokenizer for English
    tokenizer_src = BertTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
    tokenizer_tgt = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en")

    # Load DarijaBERT encoder and MarianMT decoder
    model = EncoderDecoderModel.from_encoder_decoder_pretrained("SI2M-Lab/DarijaBERT", "Helsinki-NLP/opus-mt-en")

    # Set the special tokens for the decoder
    model.config.decoder_start_token_id = tokenizer_tgt.pad_token_id
    model.config.pad_token_id = tokenizer_src.pad_token_id
    model.config.eos_token_id = tokenizer_tgt.eos_token_id

    # Set text generation parameters (these can be adjusted)
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
        batch["labels"] = [[-100 if token == tokenizer_tgt.pad_token_id else token for token in labels] for labels in
                           batch["labels"]]

        return batch

    # Create tokenized datasets
    tokenized_train_dataset = dataset['train'].map(preprocess_function, batched=True, remove_columns=['src', 'tgt'])
    tokenized_eval_dataset = dataset['validation'].map(preprocess_function, batched=True, remove_columns=['src', 'tgt'])
    tokenized_test_dataset = dataset['test'].map(preprocess_function, batched=True, remove_columns=['src', 'tgt'])

    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer_src, model=model)

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Decode predictions and labels
        decoded_preds = tokenizer_tgt.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer_tgt.pad_token_id
        decoded_labels = tokenizer_tgt.batch_decode(labels_ids, skip_special_tokens=True)

        # Compute BLEU/ROUGE/other metrics (adapt based on the task)
        # Example using BLEU
        bleu_score = corpus_bleu(decoded_preds, [decoded_labels]).score
        return {"bleu": bleu_score}

    # Seq2Seq Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f'{cfg.model.name.split("/")[-1]}',  # output directory
        num_train_epochs=cfg.trainer.num_epochs,  # number of training epochs
        per_device_train_batch_size=cfg.trainer.batch_size,  # batch size per device during training
        per_device_eval_batch_size=cfg.trainer.batch_size,
        warmup_steps=cfg.trainer.warmup_steps,  # number of warmup steps for learning rate scheduler
        logging_strategy='epoch',
        gradient_accumulation_steps=cfg.trainer.gradient_accumulation_steps,
        evaluation_strategy='epoch',
        predict_with_generate=True,
        overwrite_output_dir=True,
        save_total_limit=3,
        fp16=cfg.trainer.fp16  # mixed precision training
    )

    # Initialize Seq2Seq Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Run evaluation on the test set if prediction is enabled
    if cfg.trainer.predict:
        metrics = trainer.predict(tokenized_test_dataset).metrics
        metrics['model_name'] = cfg.model.name
        logger.info(metrics)

    # Save model if enabled
    if cfg.trainer.save_model:
        model_path = Path(get_original_cwd()) / Path(f'models/{cfg.model.name}_{cfg.trainer.num_epochs}')
        model_path.mkdir(exist_ok=True, parents=True)
        trainer.save_model(model_path)


if __name__ == '__main__':
    main()
