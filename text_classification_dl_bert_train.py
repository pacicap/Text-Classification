import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

from utils import load_jsonl_file
from visualizations import plot_confusion_matrix

class_names = ["monologic", "dialogic"]

# Constants
LABEL_MAP = {"monologic": 0, "dialogic": 1}
EXPERIMENT_NAME = "trainer-text-classification"

# Hyperparameters
LEARNING_RATE = 2e-5  # 2e-5, 3e-5, 5e-5
BATCH_SIZE = 16  # 16, 32
MAX_LENGTH = 512  # the maximum sequence length that can be processed by the BERT model
WARMUP_STEPS = 0  # 0, 100, 1000, 10000
SEED = 1234  # 42, 1234, 2021
NUM_EPOCHS = 4  # 3, 5, 10
WEIGHT_DECAY = 1e-3  # 1e-2 or 1e-3
DROP_OUT_RATE = 0.1  # 0.1 or 0.2

# implement an early stop
best_val_loss = float("inf")  # set initial best validation loss as infinity
epochs_no_improve = 0  # counter for epochs without improvement
early_stop_patience = 3  # set your patience level
early_stop = False


def set_seed(seed_value):
  """Set seed for reproducibility."""
  random.seed(seed_value)
  np.random.seed(seed_value)
  torch.manual_seed(seed_value)
  torch.cuda.manual_seed_all(seed_value)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


# Initialize device GPU, CPU, or MPS (Mac M1)
def get_device():
  """Returns the appropriate device (CUDA, MPS, or CPU)."""
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess(_texts, _tokenizer, _device, max_length=MAX_LENGTH):
  """Tokenize and preprocess texts."""
  inputs = _tokenizer(_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
  return inputs["input_ids"].to(_device), inputs["attention_mask"].to(_device)


def calculate_accuracy(_true_labels, _predictions):
  correct = sum(t == p for t, p in zip(_true_labels, _predictions))
  return correct / len(_true_labels)


def calculate_classification_metrics(_true_labels, _predictions, _num_classes):
  tp = [0] * _num_classes
  fp = [0] * _num_classes
  fn = [0] * _num_classes

  for t, p in zip(_true_labels, _predictions):
    if t == p:
      tp[t] += 1
    else:
      fp[p] += 1
      fn[t] += 1

  _precision = [tp[_i] / (tp[_i] + fp[_i]) if tp[_i] + fp[_i] > 0 else 0 for _i in range(_num_classes)]
  _recall = [tp[_i] / (tp[_i] + fn[_i]) if tp[_i] + fn[_i] > 0 else 0 for _i in range(_num_classes)]
  _f1 = [2 * (_precision[_i] * _recall[_i]) / (_precision[_i] + _recall[_i])
         if _precision[_i] + _recall[_i] > 0 else 0 for _i in range(_num_classes)]

  return _precision, _recall, _f1


# Set seed and device
set_seed(SEED)
device = get_device()
print(f"\nUsing device: {device}\n")

# Initialize model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=len(LABEL_MAP),
                                                      hidden_dropout_prob=DROP_OUT_RATE)
model.to(device)

# Load dataset
print()
dataset = load_jsonl_file("shared_data/sliced_dataset1.jsonl")

texts = [datapoint["text"] for datapoint in dataset]
labels = [row['label'] for row in dataset]

# Preprocess texts
input_ids, attention_masks = preprocess(texts, tokenizer, device, max_length=MAX_LENGTH)

# Create TensorDataset
dataset = TensorDataset(input_ids, attention_masks, torch.tensor(labels))

# Split the dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))  # 80% training
remaining = len(dataset) - train_size
val_size = int(0.5 * remaining)  # 10% validation
test_size = remaining - val_size  # 10% test
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
total_steps = len(train_dataloader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=WARMUP_STEPS,
                                            num_training_steps=total_steps)

# Set MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

print()

train_losses = []
val_losses = []

# Start MLflow run
with mlflow.start_run(run_name="My Custom Run"):
  # Training Loop
  for epoch in range(NUM_EPOCHS):

    model.train()
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
      b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

      # Clear gradients
      model.zero_grad()

      # Forward pass
      outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
      loss = outputs.loss
      total_train_loss += loss.item()

      # Backward pass
      loss.backward()

      # Update weights
      optimizer.step()
      scheduler.step()

    # Calculate average training loss for this epoch
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Print average training loss for this epoch
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Average training loss: {avg_train_loss}")

    # Validation Loop
    model.eval()

    total_val_loss = 0
    predictions, true_labels = [], []

    for batch in tqdm(val_dataloader, desc="Validating"):
      with torch.no_grad():
        b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
        loss = outputs.loss
        total_val_loss += loss.item()

        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())

    # Calculate average validation loss
    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    # Print average validation loss for this epoch
    print(f"Average validation loss: {avg_val_loss}")

    # Early stopping
    """if avg_val_loss < best_val_loss:
      # torch.save(model.state_dict(), "best_model.pth")
      best_val_loss = avg_val_loss
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1
      if epochs_no_improve == early_stop_patience:
        print("Early stopping!")
        early_stop = True

    # If early stopping is triggered, exit loop
    if early_stop:
      break"""

  """ END of training /validation loop ------------------- """

  # Evaluate and log metrics
  accuracy = calculate_accuracy(true_labels, predictions)
  mlflow.log_metric("accuracy", accuracy)

  class_names = ["monologic", "dialogic"]
  precision, recall, f1 = calculate_classification_metrics(true_labels, predictions, len(class_names))

  print("\nAccuracy:", accuracy)
  print("Training Class-wise metrics:")
  for i, class_name in enumerate(class_names):
    print(f"{class_name}: Precision = {precision[i]:.2f}, Recall = {recall[i]:.2f}, F1 = {f1[i]:.2f}")
    mlflow.log_metric(f"precision_{class_name}", precision[i])
    mlflow.log_metric(f"recall_{class_name}", recall[i])
    mlflow.log_metric(f"f1_{class_name}", f1[i])

  # Test Loop
  model.eval()
  total_test_loss = 0
  test_predictions, test_true_labels = [], []

  for batch in tqdm(test_dataloader, desc="Testing"):
    with torch.no_grad():
      b_input_ids, b_attention_mask, b_labels = [b.to(device) for b in batch]

      # Forward pass
      outputs = model(b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
      loss = outputs.loss
      total_test_loss += loss.item()

      logits = outputs.logits
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      # Store predictions and true labels
      test_predictions.extend(np.argmax(logits, axis=1).flatten())
      test_true_labels.extend(label_ids.flatten())

  plt.figure()
  plot_confusion_matrix(test_true_labels,
                        test_predictions,
                        class_names,
                        "dataset1_model_confusion_matrix.png",
                        "Confusion Matrix for BERT Model",
                        values_fontsize=22
                        )

  # Calculate average test loss
  avg_test_loss = total_test_loss / len(test_dataloader)
  print(f"Average test loss: {avg_test_loss}")

  # Evaluate and log metrics for the test set
  test_accuracy = calculate_accuracy(test_true_labels, test_predictions)
  mlflow.log_metric("test_accuracy", test_accuracy)

  test_precision, test_recall, test_f1 = calculate_classification_metrics(test_true_labels, test_predictions,
                                                                          len(class_names))
  # Get AUC ROC score
  roc_auc = roc_auc_score(test_true_labels, test_predictions)
  # Get MCC score
  mcc = matthews_corrcoef(test_true_labels, test_predictions)
  # Make confusion matrix
  cm = confusion_matrix(test_true_labels, test_predictions)
  df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

  print("\nModel: BERT\n")
  print(f"- Accuracy: {test_accuracy:.3f}")
  print(f"- Precision: {np.mean(test_precision):.3f}")
  print(f"- Recall: {np.mean(test_recall):.3f}")
  print(f"- F1 Score: {np.mean(test_f1):.3f}")
  print(f"- AUC-ROC: {roc_auc:.3f}")
  print(f"- Matthews Correlation Coefficient (MCC): {mcc:.3f}")
  print(f"- Confusion Matrix:")
  print(df_cm)
  print()

  # print("Test Class-wise metrics:")
  for i, class_name in enumerate(class_names):
    print(f"{class_name}: Precision = {test_precision[i]:.3f}, Recall = {test_recall[i]:.3f}, F1 = {test_f1[i]:.3f}")
    mlflow.log_metric(f"test_precision_{class_name}", test_precision[i])
    mlflow.log_metric(f"test_recall_{class_name}", test_recall[i])
    mlflow.log_metric(f"test_f1_{class_name}", test_f1[i])
  print()

  # Optionally, you can also log the model
  mlflow.pytorch.log_model(model, "model")

  # Save the model in the 'models' directory
  torch.save(model.state_dict(), 'models/bert_model_exp_1.pth')

# Make visualization for training and validation losses
plt.figure()
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss', color='green')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss', color='black')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('shared_images/bert_model_losses.png')

# End MLflow run
mlflow.end_run()
