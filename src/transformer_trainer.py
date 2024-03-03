import os
import subprocess
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sacrebleu import corpus_bleu
import nltk
from nltk.translate.meteor_score import meteor_score
from fairseq.models.transformer import TransformerModel
from nltk.tokenize import word_tokenize


class TransformerTrainer:
    def __init__(self, data_bin_path, save_dir, max_tokens=4096, lr=0.0005, dropout=0.1, max_epoch=50):
        self.data_bin_path = data_bin_path
        self.save_dir = save_dir
        self.max_tokens = max_tokens
        self.lr = lr
        self.dropout = dropout
        self.max_epoch = max_epoch
        os.makedirs(save_dir, exist_ok=True)

    def train_model(self):
        # Train the transformer model
        subprocess.run([
            "fairseq-train", self.data_bin_path,
            "--save-dir", self.save_dir,
            "--arch", "transformer",
            "--share-decoder-input-output-embed",
            "--optimizer", "adam",
            "--adam-betas", "(0.9, 0.98)",
            "--clip-norm", "0.0",
            "--lr", str(self.lr),
            "--lr-scheduler", "inverse_sqrt",
            "--warmup-updates", "4000",
            "--dropout", str(self.dropout),
            "--weight-decay", "0.0001",
            "--criterion", "label_smoothed_cross_entropy",
            "--label-smoothing", "0.1",
            "--max-tokens", str(self.max_tokens),
            "--max-epoch", str(self.max_epoch),
            "--no-progress-bar",
            "--log-file", self.save_dir + "/log_file.log",
            "--log-format", "json",
            "--log-interval", "10",
            "--save-interval", "5"
        ])

    def plot_losses(self, log_file, plot_file):
        # Extract losses from the log file and plot them
        with open(log_file, 'r') as f:
            lines = f.readlines()

        train_epochs, val_epochs= [], []
        train_losses, val_losses = [], []
        for line in lines:
            try:
                log = json.loads(line)
                if 'train_loss' in log and 'epoch' in log:
                    if log['epoch'] not in train_epochs:
                        train_losses.append(log['train_loss'])
                        train_epochs.append(log['epoch'])
                if 'valid_loss' in log and 'epoch' in log:
                    if log['epoch'] not in val_epochs:
                        val_losses.append(log['valid_loss'])
                        val_epochs.append(log['epoch'])
            except json.JSONDecodeError:
                pass

        plt.plot([float(x) for x in train_losses], label='Train Loss')
        plt.plot([float(x) for x in val_losses], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file)  # Save the plot to a file

    def load_model(self, checkpoint_file='checkpoint_best.pt'):
        # Load the trained model
        model = TransformerModel.from_pretrained(
            os.path.abspath(self.save_dir),
            checkpoint_file=checkpoint_file,
            data_name_or_path=os.path.abspath(self.data_bin_path),
        )
        return model

    def read_sentences(self, file_path):
        with open(file_path, encoding='utf-8') as file:
            return [line.strip() for line in file]

    def evaluate_model(self, model, src_file_path, tgt_file_path, metrics_file):
        # Read sentences from files
        src_sentences = self.read_sentences(src_file_path)
        tgt_sentences = self.read_sentences(tgt_file_path)

        # Evaluate the model with BLEU, chrF, METEOR, COMET, etc.
        predicted_sentences = [model.translate(src) for src in src_sentences]

        # Calculate BLEU
        bleu_score = corpus_bleu(predicted_sentences, [tgt_sentences]).score

        nltk.download('punkt')
        nltk.download('wordnet')
        # Tokenize the sentences
        predicted_tokens = [word_tokenize(sent) for sent in predicted_sentences]
        tgt_tokens = [word_tokenize(sent) for sent in tgt_sentences]

        # Calculate METEOR
        meteor_scores = [meteor_score([tgt], pred) for tgt, pred in zip(tgt_tokens, predicted_tokens)]
        avg_meteor = sum(meteor_scores) / len(meteor_scores)

        results = {
            'BLEU': bleu_score,
            'METEOR': avg_meteor,
            # Add other metrics here
        }

        # Save metrics to a file
        with open(metrics_file, 'w') as file:
            json.dump(results, file, indent=4)

        return {
            'BLEU': bleu_score,
            'METEOR': avg_meteor,
        }


if __name__ == "__main__":
    trainer = TransformerTrainer(
        data_bin_path='../data/fairseq_data/data-bin',
        save_dir='../models/transformer_model'
    )
    trainer.train_model()

    # Plot losses
    trainer.plot_losses(os.path.join(trainer.save_dir, 'log_file.log'),
                        os.path.abspath('../models/transformer_model/loss_plot.png'))

    # Load model and evaluate
    model = trainer.load_model()
    # Provide your src_sentences and tgt_sentences for evaluation
    src_file_path = '../data/processed/test.darija'
    tgt_file_path = '../data/processed/test.eng'
    results = trainer.evaluate_model(model, src_file_path, tgt_file_path,
                                     os.path.abspath('../models/transformer_model/evaluation_metrics.json'))
    print(results)
