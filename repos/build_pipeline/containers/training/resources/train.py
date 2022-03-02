from datasets import load_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import transformers
import json
import os
import numpy as np
from datasets import load_metric

transformers.logging.set_verbosity_info()

def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def main():
    hyperparamters = json.loads(os.environ['SM_HPS'])
    model_dir = os.environ['SM_MODEL_DIR']
    log_dir = os.environ['SM_MODEL_DIR']
    train_data_dir = os.environ['SM_CHANNEL_TRAIN']
    
    if hyperparamters["tokenizer_download_model"] == "disable":
        tokenizer_model = os.environ['SM_CHANNEL_TOKENIZER']
    else:
        tokenizer_model = 'distilbert-base-uncased'
    
    #gpus_per_host = int(os.environ['SM_NUM_GPUS'])
    train_texts, train_labels = read_imdb_split(train_data_dir)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_model)
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))
    
    training_args = TFTrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=hyperparamters["num_train_epochs"],              # total number of training epochs
        per_device_train_batch_size=hyperparamters["per_device_train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=hyperparamters["per_device_eval_batch_size"],   # batch size for evaluation
        warmup_steps=hyperparamters["warmup_steps"],
        weight_decay=hyperparamters["weight_decay"],               # strength of weight decay
        logging_dir=log_dir,            # directory for storing logs
        logging_steps=hyperparamters["logging_steps"],
        eval_steps=hyperparamters["eval_steps"],
        evaluation_strategy="steps"
    )

    with training_args.strategy.scope():
        model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    
    trainer = TFTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    

if __name__ == "__main__":
    main()