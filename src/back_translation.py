import os
import csv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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

    # Initialize back-translation (English back to Moroccan Arabic)
    back_translator = Translator(
        model_name='facebook/nllb-200-3.3B',
        src_lang='eng_Latn',
        tgt_lang='ary_Arab'
    )

    # Load the previously translated English sentences
    input_file = '../data/forward_translations.csv'
    original_sentences = []
    translated_english = []
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original_sentences.append(row['Original Moroccan Arabic'])
            translated_english.append(row['Translated English'])

    # Back-translate English to Moroccan Arabic
    back_translations = back_translator.translate_sentences(translated_english, source_lang='eng_Latn',
                                                                  target_lang='ary_Arab')

    # Save the back-translations along with the original sentences
    output_file = '../data/back_translations_new.csv'
    with open(output_file, 'w+', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Moroccan Arabic', 'Translated English', 'Back Translated Moroccan Arabic'])
        for original, english, back_translated in zip(original_sentences, translated_english, back_translations):
            writer.writerow([original, english, back_translated])

    print(f"Back translation completed and saved to {output_file}.")
