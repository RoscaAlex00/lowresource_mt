import logging
import os

import numpy as np
from datasets import load_dataset, load_metric
from nltk.translate.meteor_score import single_meteor_score
from sacrebleu import corpus_bleu
from transformers import AutoTokenizer, GPT2Tokenizer, EncoderDecoderModel, Trainer, TrainingArguments, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

sacrebleu = load_metric("sacrebleu")
meteor = load_metric("meteor")

logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = EncoderDecoderModel.from_encoder_decoder_pretrained("SI2M-Lab/DarijaBERT", "gpt2")
# cache is currently not supported by EncoderDecoder framework
model.decoder.config.use_cache = False
bert_tokenizer = AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")

# CLS token will work as BOS token
bert_tokenizer.bos_token = bert_tokenizer.cls_token

# SEP token will work as EOS token
bert_tokenizer.eos_token = bert_tokenizer.sep_token


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token

# set decoding params
model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
model.config.eos_token_id = gpt2_tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56
model.config.no_repeat_ngram_size = 3
model.early_stopping = True
model.length_penalty = 2.0
model.num_beams = 4

# load train and validation data
raw_datasets = load_dataset('csv', data_files='../data/sentences_nllb.csv')
raw_datasets = raw_datasets.remove_columns('darija')  # Adjust based on your CSV
raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1, seed=552)

raw_datasets['train'] = raw_datasets['train'].filter(
    lambda example: example['src'] is not None and example['tgt'] is not None)
raw_datasets['test'] = raw_datasets['test'].filter(
    lambda example: example['src'] is not None and example['tgt'] is not None)

encoder_length = 512
decoder_length = 128
batch_size = 8


# map data correctly
def map_to_encoder_decoder_inputs(batch):  # Tokenizer will automatically set [BOS] <text> [EOS]
    # use bert tokenizer here for encoder
    inputs = bert_tokenizer(batch["src"], padding="max_length", truncation=True, max_length=encoder_length)
    # force summarization <= 128
    outputs = gpt2_tokenizer(batch["tgt"], padding="max_length", truncation=True, max_length=decoder_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["labels"] = outputs.input_ids.copy()
    batch["decoder_attention_mask"] = outputs.attention_mask

    # complicated list comprehension here because pad_token_id alone is not good enough to know whether label should be excluded or not
    batch["labels"] = [
        [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in
        [zip(masks, labels) for masks, labels in zip(batch["decoder_attention_mask"], batch["labels"])]
    ]

    assert all([len(x) == encoder_length for x in inputs.input_ids])
    assert all([len(x) == decoder_length for x in outputs.input_ids])

    return batch


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode the predicted IDs and label IDs
    pred_str = gpt2_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids = np.where(labels_ids != -100, labels_ids, gpt2_tokenizer.eos_token_id)
    label_str = gpt2_tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Since we're using datasets with padding, we need to strip that padding
    # Note: Adjust according to your tokenizer's EOS token or pad token.
    pred_str = [pred.replace(gpt2_tokenizer.eos_token, "") for pred in pred_str]
    label_str = [[label.replace(gpt2_tokenizer.eos_token, "")] for label in label_str]

    # Compute BLEU and METEOR using nltk or datasets
    bleu_result = sacrebleu.compute(predictions=pred_str, references=label_str)
    meteor_result = np.mean([single_meteor_score(ref[0], pred) for ref, pred in zip(label_str, pred_str)])

    # Return the metrics
    return {
        "bleu": bleu_result["score"],
        "meteor": meteor_result,
    }


# make train dataset ready
train_dataset = raw_datasets['train'].map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size,
)
train_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# same for validation dataset
val_dataset = raw_datasets['test'].map(
    map_to_encoder_decoder_inputs, batched=True, batch_size=batch_size,
)
val_dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# set training arguments - these params are not really tuned, feel free to change
training_args = Seq2SeqTrainingArguments(
    output_dir="../model_bert_gpt/",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    do_train=True,
    do_eval=True,
    logging_steps=100,
    save_steps=2000,
    eval_steps=100,
    overwrite_output_dir=True,
    warmup_steps=100,
    save_total_limit=10,
    fp16=True,
    num_train_epochs=2,
    learning_rate=2e-7
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
def evaluate_model(tokenizer, model, dataset, metric="bleu"):
    src_texts = dataset['src']
    tgt_texts = dataset['tgt']

    model.eval()  # Ensure model is in evaluation mode

    predictions = []
    for src_text in src_texts:
        inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=encoder_length).to(model.device)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=decoder_length)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred_text)
        print(src_text)
        print(pred_text)
        print('--')

    if metric == "bleu":
        score = corpus_bleu(predictions, [[tgt] for tgt in tgt_texts]).score
    elif metric == "meteor":
        score = np.mean([single_meteor_score(tgt, pred) for tgt, pred in zip(tgt_texts, predictions)])
    else:
        raise ValueError("Unsupported metric")

    return score

# Before starting training, evaluate the untrained model
#bleu_score_pre = evaluate_model(gpt2_tokenizer, model, val_dataset, metric="bleu")
#meteor_score_pre = evaluate_model(gpt2_tokenizer, model, val_dataset, metric="meteor")
#print(f"Pre-Training BLEU Score: {bleu_score_pre}")
#print(f"Pre-Training METEOR Score: {meteor_score_pre}")

# Instantiate and configure the Trainer as before, then start training
trainer.train()

# After training, evaluate the trained model
bleu_score_post = evaluate_model(gpt2_tokenizer, model, val_dataset, metric="bleu")
meteor_score_post = evaluate_model(gpt2_tokenizer, model, val_dataset, metric="meteor")
print(f"Post-Training BLEU Score: {bleu_score_post}")
print(f"Post-Training METEOR Score: {meteor_score_post}")
