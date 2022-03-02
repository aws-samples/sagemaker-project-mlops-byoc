from datasets import load_dataset
from pathlib import Path
from transformers import DistilBertTokenizerFast
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import transformers
import numpy as np
from datasets import load_metric
import pandas as pd
import os
import subprocess

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

def main():
    model_dir = "/opt/ml/processing/model"
    test_path = "/opt/ml/processing/input/test"
    
    subprocess.run("tar -xvf /opt/ml/processing/model/model.tar.gz -C /opt/ml/processing/model", shell=True)
    
    test_texts, test_labels = read_imdb_split(test_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors="tf")
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        test_labels
    ))

    training_args = TFTrainingArguments(
            output_dir='./results',
            do_train = False,
            do_predict = True
        )
    model = TFDistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_dir)
    
    trainer = TFTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args                  # training arguments, defined above
    )
    
    pred, _, metric_pred = trainer.predict(test_dataset=test_dataset)
    pred_label = np.argmax(pred, axis=1)
    
    pd_pred = pd.DataFrame({
        "label": test_labels,
        "inference": pred_label
    })
    
    pd_pred.to_csv('/opt/ml/processing/output/pred.csv', index=False)
    

if __name__ == "__main__":
    main()
