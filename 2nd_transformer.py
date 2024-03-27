import csv
import os


def tsv_to_custom_csv(input_file, output_file, wavs_directory):
    with open(input_file, 'r', encoding='utf-8') as tsv_file:
        tsv_reader = csv.DictReader(tsv_file, delimiter='\t')

        with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='|')

            # Write the CSV header with the desired column names
            csv_writer.writerow(['file_name', 'transcription'])

            for row in tsv_reader:
                # Using the 'path' column as 'file_name'
                file_name = row['path']
                sentence = row['sentence']

                # Check if the file exists in the wavs directory
                file_path = os.path.join(wavs_directory, file_name + ".mp3")
                if os.path.exists(file_path):
                    csv_writer.writerow([file_name, sentence])
                else:
                    print(f"File not found: {file_path}")


if __name__ == "__main__":
    # Replace 'train.tsv' with the actual path to your TSV file
    input_file_path = "/Users/neji.ghazouani/Downloads/speech_dataset/en/validated.tsv"

    # Replace 'metadata.csv' with the desired path for your CSV file
    output_file_path = "/Users/neji.ghazouani/Desktop/Bachelorarbeit/code/bigmetadata.csv"

    # Replace "/Users/neji.ghazouani/Downloads/speech dataset/wavs" with the actual path to the directory
    wavs_directory_path = "/Users/neji.ghazouani/Desktop/Bachelorarbeit/code/bigclips"

    tsv_to_custom_csv(input_file_path, output_file_path, wavs_directory_path)
    print("Conversion completed successfully!")
