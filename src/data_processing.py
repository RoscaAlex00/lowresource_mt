import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, file_path, save_path, fairseq_bin_path):
        self.file_path = file_path
        self.save_path = save_path
        self.fairseq_bin_path = fairseq_bin_path
        self.dataset = None
        self.train = None
        self.val = None
        self.test = None

    def load_data(self):
        # Load the dataset
        self.dataset = pd.read_csv(self.file_path)

    def preprocess_text(self, text):
        # Basic text preprocessing
        return text.strip()

    def preprocess(self):
        # Preprocess the dataset
        if self.dataset is None:
            self.load_data()

        self.dataset = self.dataset.dropna(subset=['darija', 'eng'])
        self.dataset['darija'] = self.dataset['darija'].apply(self.preprocess_text)
        self.dataset['eng'] = self.dataset['eng'].apply(self.preprocess_text)

    def split_data(self, test_size=0.2, val_size=0.1):
        # Split the dataset into training, validation, and test sets
        train, test = train_test_split(self.dataset, test_size=test_size, random_state=42)
        train, val = train_test_split(train, test_size=val_size, random_state=42)
        self.train = train
        self.val = val
        self.test = test

    def save_to_file(self, data, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in data:
                file.write(line + '\n')

    def save_data(self):
        # Save the split data to files
        self.save_to_file(self.train['darija'], f'{self.save_path}/train.darija')
        self.save_to_file(self.train['eng'], f'{self.save_path}/train.eng')
        self.save_to_file(self.val['darija'], f'{self.save_path}/val.darija')
        self.save_to_file(self.val['eng'], f'{self.save_path}/val.eng')
        self.save_to_file(self.test['darija'], f'{self.save_path}/test.darija')
        self.save_to_file(self.test['eng'], f'{self.save_path}/test.eng')

    def fairseq_preprocess(self):
        # Tokenization and Binarization for Fairseq
        subprocess.run([
            "fairseq-preprocess",
            "--source-lang", "darija",
            "--target-lang", "eng",
            "--trainpref", f"{self.save_path}/train",
            "--validpref", f"{self.save_path}/val",
            "--testpref", f"{self.save_path}/test",
            "--destdir", f"{self.fairseq_bin_path}/data-bin",
            "--workers", "4"
        ])
