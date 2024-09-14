import torch
from datasets import load_dataset, Dataset, concatenate_datasets


def load_and_prepare_data(file_path):
    raw_datasets = load_dataset('csv', data_files=file_path)
    raw_datasets = raw_datasets.remove_columns('darija')
    raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1, seed=552)

    raw_datasets['train'] = raw_datasets['train'].filter(
        lambda example: example['src'] is not None and example['tgt'] is not None)
    raw_datasets['test'] = raw_datasets['test'].filter(
        lambda example: example['src'] is not None and example['tgt'] is not None)

    print(raw_datasets)
    return raw_datasets


def load_arabench_data(english_file, arabic_file):
    # Read the English and Arabic sentence files
    with open(english_file, 'r', encoding='utf-8') as en_file:
        english_sentences = en_file.readlines()

    with open(arabic_file, 'r', encoding='utf-8') as ar_file:
        arabic_sentences = ar_file.readlines()

    # Ensure both files have the same number of sentences
    assert len(english_sentences) == len(arabic_sentences), "Sentence counts don't match!"

    # Combine them into a list of dictionaries
    combined_data = [{'src': ar_sentence.strip(), 'tgt': en_sentence.strip()}
                     for ar_sentence, en_sentence in zip(arabic_sentences, english_sentences)]

    # Convert to a Hugging Face Dataset
    extra_dataset = Dataset.from_list(combined_data)

    return extra_dataset


def merge_datasets(base_dataset, extra_dataset):
    # Merge extra_dataset into base_dataset's training data using concatenate_datasets
    combined_train_data = concatenate_datasets([base_dataset['train'], extra_dataset])

    # Update base dataset's train split
    base_dataset['train'] = combined_train_data
    return base_dataset


def load_backtranslation_data(original_file_path, additional_file_path):
    original_datasets = load_dataset('csv', data_files=original_file_path)
    additional_datasets = load_dataset('csv', data_files=additional_file_path)
    print(additional_datasets['train'][0])

    # Remove unnecessary columns
    original_datasets = original_datasets.remove_columns('darija')

    # Split the original dataset
    split_datasets = original_datasets['train'].train_test_split(test_size=0.1, seed=552)
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


def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}:")
            print(f"Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
            print(f"Used memory: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
            print(
                f"Free memory: {(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024 ** 3:.2f} GB")
            print('-' * 40)
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    print_gpu_memory()
