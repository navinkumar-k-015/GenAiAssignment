from fastapi import FastAPI, File, UploadFile
import json
import logging
import pandas as pd
import numpy as np
import re
import schema
from fastapi.exceptions import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette_prometheus import metrics, PrometheusMiddleware
from fastapi.exceptions import RequestValidationError
from utils.chat_llm import get_llm_response
from utils.vector_store import get_context

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


app = FastAPI()
logger = logging.getLogger(__name__)    
app.exception_handler(RequestValidationError)
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", metrics)

history = []

def update_history(history, question, answer):
    """We update the history by appending the latest question and answer."""
    history.append((question, answer))
    # Limit history size to manage prompt length for LLMs
    if len(history) > 5:  # or another appropriate size for your context
        history.pop(0)  # Remove oldest entry
    return history

def prepare_history_text(history):
    """Convert the history list into a formatted string for prompt injection."""
    return "\n".join([f"Q: {q}\nA: {a}" for q, a in history])


@app.get("/ping")
async def ping():
    return "200"

@app.post("/invocations")
def predict(question : str):
    global history
    try:
        context = get_context(question)
        print(context)
        print(history)
        logger.info(context)
        print("1")
        history_text = prepare_history_text(history)
        print(history_text)
        response = get_llm_response(history_text, context, question)
        history = update_history(history, question, response['text'])
        
        api_response = {"question": question,
                          "answer": response['text']}
        if "END" in question:
            history = []
        print(api_response)
        converted_response = convert_np_to_native(api_response)
        return converted_response
    except Exception as e:
        messgae = json.dumps({'error': str(e), "request": question})
        logger.error(messgae, extra={"status_code": 500})
        error_message = e
        error_response = {"request": question,
                          "error":e}
        print(error_response)
        return JSONResponse(status_code=500,content=jsonable_encoder({"message":"Internal server error"}))