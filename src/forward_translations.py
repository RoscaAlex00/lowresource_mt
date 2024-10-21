import os
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import load_and_prepare_data


class Translator:
    def __init__(self, model_name, src_lang, tgt_lang):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load dataset and filter for training data
    dataset_path = '../data/sentences_new.csv'
    raw_datasets = load_and_prepare_data(dataset_path)

    # Initialize forward translation (Moroccan Arabic to English)
    forward_translator = Translator(
        model_name='facebook/nllb-200-3.3B',
        src_lang='ary_Arab',
        tgt_lang='eng_Latn'
    )

    # Forward translate Moroccan Arabic to English
    original_sentences = [row['src'] for row in raw_datasets['train']]
    translated_sentences = forward_translator.translate_sentences(original_sentences, source_lang='ary_Arab',
                                                                  target_lang='eng_Latn')

    # Save translated sentences to a file for back-translation
    output_file = '../data/forward_translations.csv'
    with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Moroccan Arabic', 'Translated English'])
        for original, translation in zip(original_sentences, translated_sentences):
            writer.writerow([original, translation])

    print(f"Forward translation completed and saved to {output_file}.")
