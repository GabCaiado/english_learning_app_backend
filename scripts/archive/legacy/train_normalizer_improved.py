import os
import json
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    GenerationConfig
)
import evaluate
import random

MODEL_NAME = "t5-small"
TRAIN_DATA_PATH = "data/normalizer_train.json"
TEST_DATA_PATH = "data/normalizer_test.json"
OUTPUT_DIR = "models/slang_normalizer"

TASK_PREFIX = "normalize slang: "

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_and_filter_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    filtered = []
    for item in data:
        slang = item.get("slang", "").strip()
        formal = item.get("formal", "").strip()
        
        # Valid pairs must have length and be different
        if slang and formal and slang.lower() != formal.lower():
            filtered.append({"slang": slang, "formal": formal})
            
    return filtered

def train():
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        print("Erro: Arquivos de normalizer não encontrados. Rode o dataset generation primeiro.")
        return

    print("="*60)
    print("TREINO DO SLANG NORMALIZER (T5-small)")
    print("="*60)

    train_data = load_and_filter_data(TRAIN_DATA_PATH)
    test_data = load_and_filter_data(TEST_DATA_PATH)
    
    print(f"Dados Carregados -> Treino: {len(train_data)}, Teste: {len(test_data)}")

    if len(train_data) < 10:
        print("ERRO: Dataset muito pequeno para treino.")
        return

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(test_data)

    print("Tokenizando...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    max_input_length = 128
    max_target_length = 128

    def preprocess(examples):
        inputs = [TASK_PREFIX + text for text in examples["slang"]]
        targets = examples["formal"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )

        # -100 to ignore padding in loss function
        label_ids = []
        for seq in labels["input_ids"]:
            label_ids.append([
                token if token != tokenizer.pad_token_id else -100
                for token in seq
            ])

        model_inputs["labels"] = label_ids
        return model_inputs

    tokenized_train = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)

    print("Carregando modelo...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Generation config
    # T5 uses pad_token_id as decoder_start_token_id (its convention)
    gen_config = GenerationConfig(
        max_length=128,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,  # Required for T5 enc-dec generation
        bos_token_id=tokenizer.pad_token_id,            # Fallback if decoder_start_token_id missing
    )
    model.generation_config = gen_config

    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 with pad token id to fix decoding issues
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # SacreBLEU takes a list of lists of references
        decoded_labels_for_eval = [[label] for label in decoded_labels]
        
        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_for_eval)
        
        return {"bleu": result["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_normalizer",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=max_target_length,
        generation_config=gen_config,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=20,
        logging_dir="./logs_normalizer",
        save_total_limit=2,
        seed=SEED,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\nTreinando...")
    trainer.train()

    print("\nAvaliando conjunto de teste...")
    eval_results = trainer.evaluate()
    print(f"BLEU Score: {eval_results['eval_bleu']:.2f}")

    print(f"\nSalvando modelo final em {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "="*60)
    print("TREINO CONCLUIDO!")
    print("="*60)

if __name__ == "__main__":
    train()
