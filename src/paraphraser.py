import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
from utils import load_and_prepare_data
import pandas as pd


model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')
tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def paraphrase_sentence(sentence):
    batch = tokenizer(sentence, return_tensors='pt').to(device)
    generated_ids = model.generate(batch['input_ids'])
    paraphrased_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return paraphrased_sentence


def generate_paraphrased_targets(train_dataset):
    paraphrased_sentences = []
    source_sentences = []
    i = 0
    for example in train_dataset:
        i+=1
        original_tgt = example['tgt']  # english
        original_src = example['src']  # moroccan arabic

        # Generate paraphrase for the target sentence
        paraphrased_tgt = paraphrase_sentence(original_tgt)

        paraphrased_sentences.append(paraphrased_tgt)
        source_sentences.append(original_src)
        if i % 100 == 0 :
            print(f"Sentence:{i}")

    return pd.DataFrame({'src': source_sentences, 'tgt': paraphrased_sentences})


file_path = "../data/sentences_new.csv"
raw_datasets = load_and_prepare_data(file_path)
paraphrased_df = generate_paraphrased_targets(raw_datasets['train'])

# Save to new CSV file
output_file = "../data/paraphrased_target_data.csv"
paraphrased_df.to_csv(output_file, index=False)
print(f"Paraphrased target sentences saved to {output_file}")
