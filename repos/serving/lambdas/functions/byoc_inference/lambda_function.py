import json
import logging
import os

import boto3
import pandas as pd
import numpy as np
import base64
import io

logger = logging.getLogger()

client = boto3.client("sagemaker-runtime")
region = os.environ["region"]

endpoint_name = os.environ["endpoint_name"]
content_type = os.environ["content_type"]
fg_name = os.environ["fg_name"]

boto_session = boto3.Session(region_name=region)
featurestore_runtime = boto_session.client(
    service_name="sagemaker-featurestore-runtime", region_name=region
)

client_sm = boto_session.client("sagemaker-runtime", region_name=region)

def lambda_handler(event, context):
    # Get data from online feature store
    indx_id = str(event["queryStringParameters"]["index"])
    
    response = featurestore_runtime.get_record(
        FeatureGroupName=fg_name,
        RecordIdentifierValueAsString=str(indx_id),
    )
    
    try:
        if response.get("Record"):
            record = response["Record"]
            df = pd.DataFrame(record).set_index('FeatureName').transpose()
            text = df["text"].tolist()
        else:
            text = json.loads(event["queryStringParameters"]["text"])
    except:
        logging.exception(f"internal error")
        return {
                "statusCode": 500,
                "body": json.dumps(
                        {"Error": "Please input index of sentence in Feature Store or a sentence"}
                    ),
            }

    try:
        """
        body = json.dumps({"inputs": text}).encode("utf-8")
        response = client.invoke_endpoint(
                EndpointName=endpoint_name, ContentType="application/json", Accept="application/json", Body=body
            )
        body = response["Body"].read()
        msg = body.decode("utf-8")
        res = json.loads(msg)
        logging.info(f"prediction: {res}")
        """
        
        df_record = pd.DataFrame({"inputs": text})
        
        csv_file = io.StringIO()
        df_record.to_csv(csv_file, sep=",", header=False, index=False)
        payload_as_csv = csv_file.getvalue()
        
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            Body= payload_as_csv,
            ContentType = 'text/csv'
        )
        
        body = response["Body"].read()
        msg = body.decode("utf-8")
        data = json.loads(msg)
        logging.info(f"prediction: {data}")
        
        return {
            "statusCode": 200,
            "body": json.dumps({"indx_id": indx_id, "prediction": data}),
        }
    except Exception:
        logging.exception(f"internal error")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"Error": f"internal error. Check Logs for more details"}
            ),
        }