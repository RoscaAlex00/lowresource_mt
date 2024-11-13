import re
from transformers import BertForSequenceClassification, AutoTokenizer
import numpy as np
from utils import load_and_prepare_data
from datasets import load_dataset

model_name = "AMR-KELEG/Sentence-ALDi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)


def preprocess_text(arabic_text):
    """Apply preprocessing to the given Arabic text.

    Args:
        arabic_text: The Arabic text to be preprocessed.

    Returns:
        The preprocessed Arabic text.
    """
    no_urls = re.sub(
        r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b",
        "",
        arabic_text,
        flags=re.MULTILINE,
    )
    no_english = re.sub(r"[a-zA-Z]", "", no_urls)

    return no_english


def compute_ALDi(sentence):
    """Computes the ALDi score for the given sentences.

    Args:
        sentences: A list of Arabic sentences.

    Returns:
        A list of ALDi scores for the given sentences.
    """

    preprocessed_sentence = preprocess_text(sentence)

    inputs = tokenizer(
        preprocessed_sentence,
        return_tensors="pt",
        padding=True,
    )
    output = model(**inputs).logits.reshape(-1).tolist()[0]
    return max(min(output, 1), 0)


def compute_average_ALDi(sentences):
    scores = []
    i = 0
    for sent in sentences:
        scores.append(compute_ALDi(sent))

    return np.mean(scores)


if __name__ == "__main__":
    file_path = '../data/AraBench/madar.dev.mgr.0.ma.ar'
    raw_datasets = load_dataset('csv', data_files=file_path, header=None)
    # raw_datasets = raw_datasets.remove_columns('darija')
    print(raw_datasets['train']['0'])
    aldi_score = compute_average_ALDi(raw_datasets['train']['0'])

    print(f"ALDi Score: {aldi_score}")


# darija open = 0.70540
# madar = 0.7007731408450689
# bible = 0.6850881828678151