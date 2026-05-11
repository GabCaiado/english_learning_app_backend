import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

MODEL_NAME = "t5-small"
DATA_PATH = "data/slang_normalization.csv"
OUTPUT_DIR = "models/slang_normalizer"

TASK_PREFIX = "normalize slang: "


def train():
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Arquivo {DATA_PATH} não encontrado. Rode o generate_dataset.py primeiro.")
        return

    print("Iniciando Fine-tuning do Slang Normalizer (T5-small)...")

    # 1. Carregar dados
    df = pd.read_csv(DATA_PATH)

    # Garantia de qualidade: so treina em pares onde ha transformacao real
    before = len(df)
    df = df[df["slang_text"].str.lower().str.strip() != df["standard_text"].str.lower().str.strip()]
    after = len(df)
    print(f"Pares de treino apos filtro de qualidade: {after} (removidos {before - after} identicos)")

    if after < 10:
        print("ERRO: Dataset muito pequeno. Rode generate_dataset.py primeiro.")
        return

    raw_dataset = Dataset.from_pandas(df.reset_index(drop=True))

    # 2. Tokenizacao com prefixo de tarefa T5
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    max_input_length = 128
    max_target_length = 128

    def preprocess_function(examples):
        # Adiciona prefixo de tarefa em cada exemplo de entrada
        inputs = [TASK_PREFIX + text for text in examples["slang_text"]]
        targets = examples["standard_text"]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )

        # Tokeniza o alvo separadamente (text_target)
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )

        # Substitui padding (0) por -100 para que o loss ignore esses tokens
        label_ids = []
        for seq in labels["input_ids"]:
            label_ids.append([
                token if token != tokenizer.pad_token_id else -100
                for token in seq
            ])

        model_inputs["labels"] = label_ids
        return model_inputs

    print("Processando textos...")
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

    # Dividir em Treino (90%) e Validacao (10%)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    # 3. Configurar Modelo
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # 4. Parametros de Treino
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results_norm",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,          # T5-small funciona bem com lr mais alto que ByT5
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,          # Mais epocas pois o dataset é menor (so pares reais)
        predict_with_generate=True,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        report_to="none",            # Desativa wandb/tensorboard para simplicidade
    )

    # 5. Trainer API
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print(f"Treinando com {len(tokenized_dataset['train'])} exemplos...")
    trainer.train()

    # 6. Salvar Modelo e Tokenizer
    print(f"Salvando modelo em {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Treinamento concluído! Modelo T5-small salvo.")


if __name__ == "__main__":
    train()
