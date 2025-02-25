import pandas as pd


def convert_translation_file(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file)

    # Create a new DataFrame with the required columns
    new_df = df[["Translated Sentence", "Original"]].rename(
        columns={
            "Translated Sentence": "src",
            "Original": "tgt"
        }
    )

    # Save the new CSV file
    new_df.to_csv(output_file, index=False)


# File names
input_filename = "../results/model_opus/outputs/predictions_en_ar_finetuned.csv"
output_filename = "../data/back_translations_orig/ar_en_opus.csv"

# Convert and save
convert_translation_file(input_filename, output_filename)

print(f"File saved as {output_filename}")
