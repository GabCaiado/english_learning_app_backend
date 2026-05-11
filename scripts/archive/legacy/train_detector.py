import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/slang_detection.csv"
OUTPUT_DIR = "models/slang_detector"

class SlangDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Arquivo {DATA_PATH} não encontrado. Rode o generate_dataset.py primeiro.")
        return

    print("Iniciando Fine-tuning do Slang Detector (DistilBERT)...")
    
    # 1. Carregar dados
    df = pd.read_csv(DATA_PATH)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2
    )

    # 2. Tokenização
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # 3. Preparar Datasets
    train_dataset = SlangDataset(train_encodings, train_labels)
    val_dataset = SlangDataset(val_encodings, val_labels)

    # 4. Configurar Modelo
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 5. Parâmetros de Treino
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 6. Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Treinando...")
    trainer.train()

    # 7. Salvar Modelo e Tokenizer
    print(f"Salvando modelo em {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Treinamento concluído!")

if __name__ == "__main__":
    train()
