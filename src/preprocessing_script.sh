#!/bin/sh

# Adaptation for Darija-English data preprocessing, including tokenization and subword segmentation.

# suffix of source language files (Darija)
SRC=darija

# suffix of target language files (English)
TRG=eng

# number of merge operations for BPE
bpe_operations=8000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=./mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=./subword-nmt

# Tokenize Darija and English data
for prefix in train val test  # Note: 'valid' is changed to 'val' to match your script
do
  cat ../data/processed/$prefix.darija | \
  $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -threads 8 -l ar | \  # Assuming 'ar' works for Darija
  $mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l ar > ../data/processed/$prefix.tok.$SRC  # Using 'ar' for Arabic

  cat ../data/processed/$prefix.eng | \
  $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -threads 8 -l en | \
  $mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -a -l en > ../data/processed/$prefix.tok.$TRG
done

# Train BPE
cat ../data/processed/train.tok.$SRC ../data/processed/train.tok.$TRG | \
$subword_nmt/learn_bpe.py -s $bpe_operations > ../data/processed/model/$SRC$TRG.bpe

# Apply BPE
for prefix in train val test
do
  $subword_nmt/apply_bpe.py -c ../data/processed/model/$SRC$TRG.bpe < ../data/processed/$prefix.tok.$SRC > ../data/processed/$prefix.bpe.$SRC
  $subword_nmt/apply_bpe.py -c ../data/processed/model/$SRC$TRG.bpe < ../data/processed/$prefix.tok.$TRG > ../data/processed/$prefix.bpe.$TRG
done

# Set the source and target languages to Darija and English
SRC=darija
TGT=eng

# Specify the paths to the BPE-processed training, validation, and testing data
train_data=../data/processed/train.bpe
valid_data=../data/processed/val.bpe  # Note the change from 'vaild_data' to 'valid_data' for correctness
test_data=../data/processed/test.bpe

# Set the destination directory for the preprocessed data
data=../data/fairseq_data/bin

# Run Fairseq's preprocess command. Assuming both languages share a vocabulary, you can use the --joined-dictionary option.
# Adjust or remove the --joined-dictionary option based on your specific needs or if you decide to use separate dictionaries for each language.
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} --joined-dictionary \
    --trainpref ${train_data} --validpref ${valid_data} \
    --testpref ${test_data} \
    --destdir ${data} \
    --workers 20
