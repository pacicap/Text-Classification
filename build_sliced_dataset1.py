import spacy

import pandas as pd
from collections import Counter
from transformers import BertTokenizer

from utils import load_jsonl_file, save_jsonl_file

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_trf")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load the JSON file
ms_data = load_jsonl_file("shared_data/dataset_1.jsonl")

# Convert to DataFrame
df = pd.DataFrame(ms_data)

# New dataset
new_data = []

# Process each text
for index, row in df.iterrows():
  index = int(index)
  print(f"Processing text {index+ 1}/{len(list(df.iterrows()))}")
  doc = nlp(row['text'])
  sentences = [sent.text for sent in doc.sents]

  # Create windows with overlap
  window = []
  for i in range(len(sentences)):
    # Add the sentence to the current window
    window.append(sentences[i])
    # Check if the total number of tokens in the current window exceeds the BERT token limit
    if len(tokenizer.tokenize(" ".join(window))) > 510:  # Saving 2 tokens for padding
      # If it does, remove the last sentence from the window
      window.pop()
      # Join the sentences in the current window to form a single text, and add it to the new dataset
      new_text = " ".join(window)
      new_data.append({'id': row["id"], 'text': new_text, 'label': row['discourse_type']})
      # Start a new window with the last sentence of the previous window
      window = [sentences[i]]
  # Add the remaining sentences in the last window to the new dataset
  if window:
    new_text = " ".join(window)  # Removed the manual addition of "[CLS]" and "[SEP]" tokens here
    new_data.append({'id': row["id"], 'text': new_text, 'label': row['discourse_type']})

# Save to a new JSON file
save_jsonl_file(new_data, "shared_data/dataset_1_sliced.jsonl")

# Count the number of monologic and dialogic datapoints
labels = [item['label'] for item in new_data]
counter = Counter(labels)

print(f"Number of monologic datapoints: {counter[0]}")
print(f"Number of dialogic datapoints: {counter[1]}")
