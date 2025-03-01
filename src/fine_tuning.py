import os
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from src.config import Config

def fine_tune():
    """
    Esegue il fine-tuning del modello per il trading di criptovalute utilizzando dati CSV.
    I file CSV devono avere le colonne "text" e "label" (0=SELL, 1=HOLD, 2=BUY).
    """
    data_files = {
        "train": os.path.join("data", "crypto_train.csv"),
        "test": os.path.join("data", "crypto_test.csv")
    }
    dataset = load_dataset("csv", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=Config.MAX_LENGTH)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # Carica il modello specifico
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.LLM_MODEL_NAME,
        num_labels=Config.NUM_LABELS
    )
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,  
        gradient_accumulation_steps=2,  
        fp16=False,  
        optim="adamw_torch_fused",  
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()

    # Salva il modello fine-tuned
    model.save_pretrained(Config.FINE_TUNED_MODEL_NAME)
    tokenizer.save_pretrained(Config.FINE_TUNED_MODEL_NAME)
    print("Fine-tuning completato. Modello salvato in", Config.FINE_TUNED_MODEL_NAME)

if __name__ == "__main__":
    fine_tune()
