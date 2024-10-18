from flask import Flask, Response, jsonify, request, redirect
import sys, re
import json
import requests
import ast, os
import pandas as pd
import time
from collections import defaultdict

from fastapi import FastAPI, Query, HTTPException, Body
from typing import List, Dict  # Import List and Dict from the typing module

from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv(".env"))
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from dotenv import find_dotenv, load_dotenv
load_dotenv()
load_dotenv(find_dotenv(".env"))

# app = Flask(__name__)

from pymilvus import (
    connections,
    Collection,
)
from sentence_transformers import SentenceTransformer
import logging

app = FastAPI()

#=============================Milvus Credentials==============================
milvus_host="158.175.183.91"
milvus_port="8080"
milvus_password="4XYg2XK6sMU4UuBEjHq4EhYE8mSFO3Qq" 

# Connect to Milvus
connections.connect("default", host=milvus_host,
                    port=milvus_port, secure=True,
                    server_pem_path="cert-milvus.pem",
                    server_name="localhost", user="root",
                    password=milvus_password)

def similarity_search(
    user_question: str,
    limit=3,
    milvus_connection_alias: str = "default",
    collection_name: str = "indoagri_sop",
    hf_model_id: str = 'LazarusNLP/all-indo-e5-small-v4'
) -> list:

    # Search parameters
    search_params = {
        "metric_type": "L2",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }

    collection = Collection(
        name=collection_name,
        using=milvus_connection_alias
    )
    collection.load()
    logging.debug("Collection loaded.")

    # Embedding model
    model = SentenceTransformer(hf_model_id)
    logging.debug("Embedding model loaded.")

    # Search the index for the closest vectors
    results = collection.search(
        data=[model.encode(user_question)],
        anns_field="embedding_vector",
        param=search_params,
        limit=limit,
        expr=None,
        output_field=['title'],
        consistency_level="Strong"
    )

    # Retrieving the text associated with the results ids
    results_text = collection.query(
        expr="id in {}".format(results[0].ids),
        output_fields=["id", "embedding_raw", "metadata_json"],
        consistency_level="Strong"
    )
    collection.release()
    logging.debug("Text chunks successfully retrieved.")

    return results_text

def extract_embedding_raw(result):
    data = [item['embedding_raw'] for item in result['predictions'][0]['values']]
    document_name = [item['metadata_json'] for item in result['predictions'][0]['values']]
    return data, document_name

def process_payload(payload):
    # Assuming payload is a query string, not a nested dictionary
    user_question = payload  # Directly use the query string
    limit = 10  # Set a default limit if needed
    milvus_connection_alias = 'default'  # Default Milvus connection alias

    # Perform the similarity search using the question
    results = similarity_search(user_question, limit, milvus_connection_alias)

    # Format the results
    output = []
    if isinstance(results, list) and len(results) > 1:
        output.append({'values': results[:limit]})  # Append the first 'limit' results
    else:
        output.append({'values': results})

    return {'predictions': output}



def group_embeddings_by_document(predictions):
    # Dictionary to hold the list of 'embedding_raw' for each document name
    combined_data = defaultdict(list)

    # Access 'values' inside 'predictions'
    for prediction in predictions:
        for item in prediction['values']:
            # Ensure item is a dictionary with necessary keys
            if isinstance(item, dict) and 'metadata_json' in item and 'embedding_raw' in item:
                document_name = item['metadata_json']
                combined_data[document_name].append(item['embedding_raw'])

    # Create final JSON format with 'document_name' first
    final_json = [
        {'document_name': doc_name.strip(), 'data': text_list}
        for doc_name, text_list in combined_data.items()
    ]

    # Return the final JSON
    return final_json


#=============================Credentials==============================
WX_API_KEY = "gkgRfU_LK6x0Urpl7U-PdETmrhhjTS_K7jTtQi5WCTfU"
WX_PROJECT_ID = "91340ff3-d9e8-4f93-bed4-fca896e5849e"
WX_URL = "https://us-south.ml.cloud.ibm.com"

creds = {
    "url": WX_URL,
    "apikey": WX_API_KEY 
}

#=============================wx.ai functions==============================
def send_to_watsonxai(prompt, creds=creds, project_id=WX_PROJECT_ID,
                      model_name='meta-llama/llama-3-1-70b-instruct',
                      decoding_method="greedy",
                      max_new_tokens=1000,
                      min_new_tokens=1,
                      temperature=0,
                      repetition_penalty=1.0,
                      stop_sequences=["\n\n", "\n"],
                      ):
    # Ensure the prompt is valid
    assert not any(map(lambda p: len(p) < 1, [prompt])), "Make sure none of the prompts in the inputs are empty."

    # Set model parameters
    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
        GenParams.STOP_SEQUENCES: stop_sequences
    }

    # Initialize the model with parameters
    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id
    )

    # Generate text from the model (without media_type argument)
    output_gen = model.generate_text_stream(prompt=prompt)  # This is a generator

    # Collect the output from the generator
    output = ""
    for text in output_gen:
        print (text)
        output += text  # Concatenate each part to output

    return output.strip()  # Strip any leading/trailing whitespace



def answer_from_table(user_question, data):
    if not data:
        return {'answer': "Maaf, informasi yang anda butuhkan tidak tersedia di database, silahkan coba dengan pertanyaan lain! Terima kasih."}
    else:
        prompt = f"""Berikut adalah informasi yang perlu disampaikan secara lengkap: {data}
        Berikut adalah pertanyaan dari user: {user_question}
        Jawab pertanyaan dari user dengan ramah, membantu, dan interaktif hanya menggunakan informasi yang tersedia.
        Jawaban yang diberikan harus dirangkai dengan baik dari data yang disediakan.
        Jawaban yang diberikan harus mencakup semua informasi yang telah diberikan serta lengkap.
        Hindari penggunaan new line saat menjawab.
        Jawaban:"""
        
        output = send_to_watsonxai(prompt)

        if 'REQUERY' in output.strip():
            return {"output": 'REQUERY'.strip()}  
        else:    
            return {"output": output.strip()}     


def Milvus2Text(payload):
    output = []
    for data in payload["input_data"]:
        user_question = data['values'][0][0]['user_question']
        query_result = data['values'][0][0]['query_result']
        answer = answer_from_table(user_question, query_result)
        document_name = data['values'][0][0]['query_result'][1]
        document_name = list(set(document_name))
        output.append(answer)
        print (f"ini output ya {output}")
        print (f"ini data ya {data}")
    return {'output': [{'user_question': user_question, 'query_result': query_result, 'values': output, 'document_name': document_name}]}

@app.post("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.post("/milvus_query")
def queryToMilvus():
    try:
        payload = request.get_json()  # Get JSON payload from request
        if not payload:
            return jsonify({"error": "Invalid input"}), 400

        # Log payload to inspect structure
        logging.debug(f"Received payload: {payload}")

        result = process_payload(payload)
        print(result)
        result = group_embeddings_by_document(result['predictions'])
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

class QueryRequest(BaseModel):
    user_question: str = Field(..., example="apa tugas pengawas potong buah?")

@app.post("/send_to_watsonx")
async def stream_response(request: QueryRequest):
    query = request.user_question.strip()
    
    try:
        # Log the query
        print("User Query:", query)
        
        # Simulate processing the payload with Milvus
        payload_milvus = query
        result_milvus = process_payload(payload_milvus)  # Check the return type here
        print("Result Milvus:", result_milvus)  # Debugging statement
        
        # Ensure result_milvus is the correct type (e.g., not a string)
        if isinstance(result_milvus, str):
            raise ValueError("Expected result_milvus to be a dictionary or list, but got a string.")
        
        data = extract_embedding_raw(result_milvus)  # Check the return type here
        document_name = data[1]
        # if document_name sama maka ambil distinct
        if len(set(document_name)) == 1:
            document_name = list(set(document_name))[0]
        else:
            document_name = document_name[0]

        print("Extracted Data:", data)  # Debugging statement
        print("Document Name:", document_name)  # Debugging statement
        
        # Ensure data is the correct type
        if isinstance(data, str):
            raise ValueError("Expected data to be a dictionary or list, but got a string.")
        
        # Constructing the payload for Watson
        payload_watson = {
            "input_data": [{
                "values": [[{"user_question": query, "query_result": data}]]
            }]
        }
        print("Payload to Watson:", payload_watson)  # Debugging statement

        # Call the function that sends the payload to Watson
        result_watson = Milvus2Text(payload_watson)  # Check the return type here
        print("Result Watson:", result_watson)  # Debugging statement

        # Return the result as a JSON response
        return JSONResponse(content=result_watson, status_code=200)
    
    except Exception as e:
        print("Error:", str(e))  # Log the error message
        return JSONResponse(content={"error": str(e)}, status_code=400)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

