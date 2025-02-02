from fastapi import FastAPI, File, UploadFile
import json
import logging
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
import torch
import schema
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette_prometheus import metrics, PrometheusMiddleware
from fastapi.exceptions import RequestValidationError
from utils.clean_text import preprocess_text
import time

app = FastAPI(title = "Ecommerce Text Classification",
    description = " This API will classify ecommerce classes from text",
    version = "1.0.0")

logger = logging.getLogger(__name__)    
app.exception_handler(RequestValidationError)
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)


def convert_np_to_native(obj):
    if isinstance(obj, np.generic):
        # Convert NumPy scalars to native Python types
        return obj.item()
    if isinstance(obj, dict):
        # Recursively process dictionary items
        return {k: convert_np_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        # Recursively process list items
        return [convert_np_to_native(i) for i in obj]
    if isinstance(obj, tuple):
        # Convert tuples, if necessary
        return tuple(convert_np_to_native(i) for i in obj)
    return obj


MODEL_PATH = './artifacts/bert_text_classifier'
TOKENIZER_PATH = 'bert-base-uncased'

logger.info("Model loading")

model = BertForSequenceClassification.from_pretrained(MODEL_PATH, device_map="cpu")
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

logger.info("Model loading completed")


def predict_class(text):
    text = preprocess_text(text)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    
    predicted_class = torch.argmax(logits, dim=1).item()
    
    class_scores = logits.squeeze().cpu().numpy()
    
    probabilities = F.softmax(torch.tensor(class_scores), dim=0).numpy()
    
    labels = ["Household", "Books", "Electronics", "Clothing & Accessories"]

    predicted_label = labels[predicted_class]

    probabilities_with_labels = list(zip(labels, probabilities))

    return predicted_label, probabilities_with_labels


@app.get("/ping")
async def ping():
    return "200"

@app.post("/invocations")
def predict(text : str):
    try:
        start_time = time.time()
        logger.info(f"Received input:{text}")
        logger.info(f"Prediction Strted")
        predicted_class, probabilities = predict_class(text)
        logger.info(f"Prediction Completed")
        output = {"predicted_class":predicted_class,
                 "probabilities":probabilities}
        
        api_response = {"request": text,
                          "result":output}
        print(api_response)
        converted_response = convert_np_to_native(api_response)
        end_time = time.time()
        logger.info(
        f"API call ended at {time.ctime(end_time)} (Duration: {end_time - start_time:.2f}s)")
        return converted_response
    except Exception as e:
        messgae = json.dumps({'error': str(e), "request": text})
        logger.error(messgae, extra={"status_code": 500})
        error_message = e
        error_response = {"request": text,
                          "error":e}
        print(error_response)
        return JSONResponse(status_code=500,content=jsonable_encoder({"message":"Internal server error"}))