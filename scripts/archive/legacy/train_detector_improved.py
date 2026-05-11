import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

MODEL_NAME = "distilbert-base-uncased"
TRAIN_DATA_PATH = "data/detector_train.json"
TEST_DATA_PATH = "data/detector_test.json"
OUTPUT_DIR = "models/slang_detector"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

def augment_text(text: str) -> list:
    """ Advanced Data Augmentation for Slang Pipeline """
    variations = [text]
    
    # 1. Lowercase vs Uppercase vs Capitalized
    variations.append(text.lower())
    variations.append(text.upper())
    variations.append(text.capitalize())
    
    words = text.split()
    
    if len(words) > 0:
        # 2. Random word dropout (10% chance equivalent if we just drop 1 from phrases > 3)
        if len(words) > 3:
            drop_idx = random.randint(0, len(words)-1)
            variations.append(' '.join(words[:drop_idx] + words[drop_idx+1:]))
            
        # 3. Letter rep
        aug_words = words.copy()
        for i, w in enumerate(aug_words):
            if len(w) >= 3 and random.random() < 0.3:
                mid_char = w[len(w)//2]
                aug_words[i] = w.replace(mid_char, mid_char*3)
        variations.append(' '.join(aug_words))
        
        # 4. Swaps and Typos
        aug_words2 = words.copy()
        for i, w in enumerate(aug_words2):
            if len(w) >= 4 and random.random() < 0.3:
                char_idx = random.randint(1, len(w)-2)
                l = list(w)
                l[char_idx], l[char_idx+1] = l[char_idx+1], l[char_idx]
                aug_words2[i] = "".join(l)
        variations.append(' '.join(aug_words2))
        
    return list(set(variations))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    
    return {
        'accuracy': report.get('accuracy', 0),
        'precision': report.get('weighted avg', {}).get('precision', 0),
        'recall': report.get('weighted avg', {}).get('recall', 0),
        'f1': report.get('weighted avg', {}).get('f1-score', 0)
    }

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.get("loss")
            
        return (loss, outputs) if return_outputs else loss

def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    labels = []
    for item in data:
        texts.append(item.get("text", ""))
        labels.append(1 if item.get("is_slang", False) else 0)
    return texts, labels

def train():
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        print(f"Erro: Arquivos {TRAIN_DATA_PATH} não encontrados.")
        print("Rode o script 02_generate_with_openai.py primeiro.")
        return

    print("="*60)
    print("TREINO DO SLANG DETECTOR (DistilBERT)")
    print("="*60)

    train_raw_texts, train_raw_labels = load_json_data(TRAIN_DATA_PATH)
    test_texts, test_labels = load_json_data(TEST_DATA_PATH)

    print("Gerando data augmentation...")
    augmented_texts = []
    augmented_labels = []
    for txt, lbl in zip(train_raw_texts, train_raw_labels):
        variations = augment_text(txt)
        for var in variations:
            augmented_texts.append(var)
            augmented_labels.append(lbl)
            
    zipped = list(zip(augmented_texts, augmented_labels))
    random.shuffle(zipped)
    
    val_size = int(0.1 * len(zipped))
    val_set = zipped[:val_size]
    train_set = zipped[val_size:]
    
    train_texts, train_labels = zip(*train_set) if train_set else ([], [])
    val_texts, val_labels = zip(*val_set) if val_set else ([], [])
    
    print(f"Treino (com aug): {len(train_texts)}, Val: {len(val_texts)}, Teste: {len(test_texts)}")

    class_w = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights_pt = torch.tensor(class_w, dtype=torch.float32)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

    train_dataset = SlangDataset(train_encodings, train_labels)
    val_dataset = SlangDataset(val_encodings, val_labels)
    test_dataset = SlangDataset(test_encodings, test_labels)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label={0: "FORMAL", 1: "SLANG"},
        label2id={"FORMAL": 0, "SLANG": 1}
    )

    training_args = TrainingArguments(
        output_dir='./results_detector',
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        logging_dir='./logs_detector',
        save_total_limit=2,
        seed=SEED,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_pt,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\nTreinando...")
    trainer.train()

    print("\nAvaliando conjunto de teste...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall: {test_results['eval_recall']:.4f}")

    print("\nSalvando modelo...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "="*60)
    print("TREINO CONCLUIDO!")
    print("="*60)

if __name__ == "__main__":
    train()
