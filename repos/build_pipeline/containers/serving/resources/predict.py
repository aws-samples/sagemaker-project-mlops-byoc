import tensorflow as tf
import transformers
import numpy as np
import pandas as pd
import os
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast
import io
from io import StringIO

class TransformerService:
    @classmethod
    def create_predictor(cls):
        try:
            model_dir = os.environ["SM_MODEL_DIR"]
        except KeyError:
            model_dir = "/opt/ml/model"
        
        cls.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        cls.tf_model = TFDistilBertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_dir)
        
        return cls.tf_model
        

import json
import codecs
import csv
from flask import Flask, Response, request

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def health_check():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully and crrate a predictor."""
    
    status = 200 if TransformerService.create_predictor() else 404
    return Response(response="\n", status=status, mimetype="application/json")

@app.route("/invocations", methods=["POST"])
def inference():
    """
    if not request.is_json:
        result = {"error": "Content type is not application/json"}
        return Response(response=result, status=415, mimetype="application/json")
    """
    data = None
    
    # Convert from CSV to pandas
    if request.content_type == "text/csv":
        data = request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None)
    else:
        return Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )
    
    data.columns = ["inputs"]
    sentences = data["inputs"].values.tolist()
    try:
        inputs = TransformerService.tokenizer(
                                        sentences,
                                        padding=True,
                                        truncation=True,
                                        max_length=512,
                                        return_tensors="tf"
                                    )
        
        outputs = TransformerService.tf_model(inputs)
        predictions = tf.nn.softmax(outputs.logits, axis=-1)
        labels = np.argmax(predictions, axis=1)
        pred = np.max(predictions, axis=1)
        
        pred_dict = {
            "predictions": pred.tolist(),
            "labels": labels.tolist()
        }
        
        """
        out = io.StringIO()
        pd.DataFrame(pred_dict).to_csv(out, header=False, index=False)
        result = out.getvalue()
        
        return Response(response=result, status=200, mimetype="text/csv")
        """
        return Response(response=json.dumps(pred_dict), status=200, mimetype="application/json")
    except Exception as e:
        print(str(e))
        result = {"error": f"Internal server error"}
        return Response(response=result, status=500, mimetype="application/json")
    finally:
        pass
