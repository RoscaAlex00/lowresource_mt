from transformers import BertTokenizer, MarianTokenizer
import pandas as pd
import matplotlib.pyplot as plt

# Load tokenizers
tokenizer_src = BertTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
tokenizer_tgt = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
dataset = pd.read_csv('../data/sentences_new.csv').dropna()

darija_sentences = dataset['src'].tolist()
english_sentences = dataset['tgt'].tolist()
darija_lengths = [len(tokenizer_src.encode(sentence, truncation=True)) for sentence in darija_sentences]
english_lengths = [len(tokenizer_tgt.encode(sentence, truncation=True)) for sentence in english_sentences]

# Plot histograms of sentence lengths
plt.figure(figsize=(12, 6))
plt.hist(darija_lengths, bins=50, alpha=0.7, label='Darija Sentence Lengths')
plt.hist(english_lengths, bins=50, alpha=0.7, label='English Sentence Lengths')
plt.xlabel('Sentence Length (tokens)')
plt.ylabel('Frequency')
plt.title('Sentence Length Distribution for Darija and English')
plt.legend()
plt.show()

# Display statistics for sentence lengths
darija_stats = pd.DataFrame(darija_lengths, columns=['Darija Lengths']).describe()
english_stats = pd.DataFrame(english_lengths, columns=['English Lengths']).describe()

print(pd.concat([darija_stats, english_stats], axis=1))
