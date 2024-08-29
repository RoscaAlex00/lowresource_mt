import os
import csv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ModelEvaluator:
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

    def load_and_prepare_data(self, file_path):
        raw_datasets = load_dataset('csv', data_files=file_path)
        return raw_datasets


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluator = ModelEvaluator(
        model_name='facebook/nllb-200-3.3B',
        src_lang='eng_Latn',
        tgt_lang='ary_Arab'
    )

    # Load dataset
    dataset_path = '../data/translated_sentences.csv'
    datasets = evaluator.load_and_prepare_data(dataset_path)

    untranslated_sentences = [row['translated_eng'] for row in datasets['train']]
    original_arabic = [row['darija_ar'] for row in datasets['train']]
    print(len(untranslated_sentences))

    # Forward Translation: Moroccan Arabic to English
    back_translations = evaluator.translate_sentences(untranslated_sentences, source_lang='eng_Latn',
                                                      target_lang='ary_Arab')

    # Optionally, save these back translations to a new CSV or integrate into the dataset
    with open('../data/back_translated_sentences.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Original Arabic', 'Translated English', 'Back Translation']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for original, english_translation, back_translated in zip(original_arabic, untranslated_sentences,
                                                                  back_translations):
            writer.writerow({'Original Arabic': original, 'Translated English': english_translation,
                             'Back Translation': back_translated})

    print("Back translation completed and saved.")
