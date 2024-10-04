from flask import Flask
import sys
import json
import requests
import ast, os
import pandas as pd
import time
from collections import defaultdict
from flasgger import Swagger

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)
swagger = Swagger(app)


from flask import Flask, request, jsonify
from pymilvus import (
    connections,
    Collection,
)
from sentence_transformers import SentenceTransformer
import logging

#=============================Milvus Credentials==============================
milvus_host = os.getenv("milvus_host")
milvus_port = os.getenv("milvus_port")
milvus_password = os.getenv("milvus_password")

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
    collection_name: str = "indoagri_query",
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
    return data

def process_payload(payload):
    output = []

    # Accessing nested structure in payload
    for data in payload['input_data']:
        for value in data['values']:
            user_question = value[0]['user_question']
            limit = value[0].get('limit', 10)  # Default limit to 10 if not provided
            milvus_connection_alias = value[0].get('milvus_connection_alias', 'default')  # Default connection alias to 'default'

            results = similarity_search(user_question, limit, milvus_connection_alias)

            # Formatting the results as needed
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
WX_API_KEY = os.getenv("WX_API_KEY")
WX_PROJECT_ID = os.getenv("WX_PROJECT_ID")
WX_URL = os.getenv("WX_URL")

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
    # Helper function for sending prompts to Watsonx.ai
    assert not any(map(lambda prompt: len(prompt) < 1, prompt)), "Make sure none of the prompts in the inputs are empty."

    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
        GenParams.STOP_SEQUENCES: stop_sequences
    }

    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id
    )

    output = model.generate_text(prompt)
    return output


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
        output.append(answer)

    return {'output': [{'user_question': user_question, 'values': output}]}

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/milvus_query", methods=["POST"])
def queryToMilvus():
    """
    Endpoint to query Milvus with the given payload.
    ---
    tags:
      - Milvus
    parameters:
      - in: body
        name: payload
        required: true
        description: JSON payload containing data for querying Milvus
        schema:
          type: object
          properties:
            input_data:
              type: array
              items:
                type: object
                properties:
                  values:
                    type: array
                    items:
                      type: array
                      items:
                        type: object
                        properties:
                          user_question:
                            type: string
                            example: "apa tugas pengawas potong buah?"
    responses:
      200:
        description: A JSON object with the processed result
        schema:
          type: object
          properties:
            document_id:
              type: string
              example: "doc123"
            embeddings:
              type: array
              items:
                type: object
                properties:
                  vector:
                    type: array
                    items:
                      type: number
                    example: [0.1, 0.5, 0.3]
      400:
        description: Invalid input
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Invalid input"
      500:
        description: Internal server error
        schema:
          type: object
          properties:
            error:
              type: string
              example: "Error details"
    """
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

    

@app.route("/send_to_watsonx", methods=["POST"])
def send_to_watsonx():
    try:
        # manggil milvus_query
        # prosess hasil milvus_query jadi data

        # buat user_question, ambil dari 
        payload_milvus= request.get_json()
        result_milvus= process_payload(payload_milvus)

        data= extract_embedding_raw(result_milvus)
        payload_watson= {
        "input_data": [{
            "values": [[{"user_question": request.get_json(), "query_result":data}]]
        }]}
        result_watson= Milvus2Text(payload_watson)

        # Return the result as a JSON response
        return jsonify(result_watson), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

