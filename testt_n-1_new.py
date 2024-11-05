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
milvus_host="161.156.196.249"
milvus_port="8080"
milvus_password="4XYg2XK6sMU4UuBEjHq4EhYE8mSFO3Qq" 

# Connect to Milvus
connections.connect("default", host=milvus_host,
                    port=milvus_port, secure=True,
                    server_pem_path="cert-milvus.pem",
                    server_name="localhost", user="root",
                    password=milvus_password)

# Load the embedding model once at the module level
embedding_model = SentenceTransformer('LazarusNLP/all-indo-e5-small-v4')
logging.debug("Embedding model loaded and ready for use.")

def similarity_search(
    user_question: str,
    limit=3,
    milvus_connection_alias: str = "default",
    collection_name: str = "indoagri_sop_final"
) -> list:
    
    # Search parameters
    search_params = {
        "metric_type": "L2",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10}
    }

    # Load collection
    collection = Collection(
        name=collection_name,
        using=milvus_connection_alias
    )
    collection.load()
    logging.debug("Collection loaded.")

    # Perform the similarity search
    results = collection.search(
        data=[embedding_model.encode(user_question)],  # Use the pre-loaded model
        anns_field="embedding_vector",
        param=search_params,
        limit=limit,
        expr=None,
        output_field=['title'],
        consistency_level="Strong"
    )

    # Gather existing IDs in the collection for checking adjacency
    available_ids = set()
    all_results = collection.query(
        expr="id >= 760",  # Adjust this based on your known ID range
        output_fields=["id"],
        consistency_level="Strong"
    )
    available_ids.update(result['id'] for result in all_results)

    # Expand result IDs to include n-1 and n+1 for each ID, checking availability
    expanded_ids = set()
    for id in results[0].ids:
        if id - 1 in available_ids:
            expanded_ids.add(id - 1)
        expanded_ids.add(id)
        if id + 1 in available_ids:
            expanded_ids.add(id + 1)

    # Query expanded results
    expr = f"id in {list(expanded_ids)}"
    results_text = collection.query(
        expr=expr,
        output_fields=["id", "embedding_raw", "document_id", "metadata_json"],
        consistency_level="Strong"
    )
    collection.release()
    logging.debug("Text chunks successfully retrieved with expanded context.")

    return results_text


def extract_embedding_raw(result):
    data = [item['embedding_raw'] for item in result['predictions'][0]['values']]
    document_name = [item['metadata_json'] for item in result['predictions'][0]['values']]
    return data, document_name

def process_payload(payload):
    # Assuming payload is a query string, not a nested dictionary
    user_question = payload  # Directly use the query string
    milvus_connection_alias = 'default'  # Default Milvus connection alias
    collection_name = "indoagri_sop_final"  # Default collection name, adjust as needed

    # Perform the similarity search using the question without a limit
    results = similarity_search(user_question, milvus_connection_alias=milvus_connection_alias, collection_name=collection_name, limit=10)

    # Format the results
    output = []
    if isinstance(results, list) and len(results) > 0:
        output.append({'values': results})  # Append all results if available
    else:
        output.append({'values': []})  # Append an empty list if no results found

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
#=============================Credentials==============================
WX_API_KEY = "ON4BdvORJc02uEl6dXyvpNTPxhlO0_LG-k8CzG8Z6Cfu"
WX_PROJECT_ID = "4cae5e78-da4f-4aed-9510-cb1fbebd13d3"
WX_URL = "https://us-south.ml.cloud.ibm.com"

creds = {
    "url": WX_URL,
    "apikey": WX_API_KEY 
}

#=============================wx.ai functions==============================
def send_to_watsonxai(prompt, creds=creds, project_id=WX_PROJECT_ID,
                    model_name='meta-llama/llama-3-1-70b-instruct', #'mistralai/mixtral-8x7b-instruct-v01',#'meta-llama/llama-3-70b-instruct', #'mistralai/mixtral-8x7b-instruct-v01',', #'meta-llama/llama-2-13b-chat', #
                    decoding_method="greedy",
                    max_new_tokens=1000,
                    min_new_tokens=1,
                    temperature=0,
                    repetition_penalty=1.0,
                    stop_sequences=["\n\n","\n"],
                    ):
    '''
    helper function for sending prompts and params to Watsonx.ai

    Args:  
        prompts:list list of text prompts
        decoding:str Watsonx.ai parameter "sample" or "greedy"
        max_new_tok:int Watsonx.ai parameter for max new tokens/response returned
        temperature:float Watsonx.ai parameter for temperature (range 0>2)
        repetition_penalty:float Watsonx.ai parameter for repetition penalty (range 1.0 to 2.0)

    Returns: None
        prints response
    '''

    assert not any(map(lambda prompt: len(prompt) < 1, prompt)), "make sure none of the prompts in the inputs prompts are empty"

    # Instantiate parameters for text generation
    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
        GenParams.STOP_SEQUENCES: stop_sequences
    }

    # Instantiate a model proxy object to send your requests
    model = Model(
        model_id=model_name,
        params=model_params,
        credentials=creds,
        project_id=project_id)


    output = model.generate_text(prompt)
    return output

def answer_from_table(user_question, data):
    if data ==[]:
        return {'answer': "Maaf, informasi yang anda butuhkan tidak tersedia di database, silahkan coba dengan pertanyaan lain! Terima kasih."}
    else:
        prompt= f"""Berikut adalah informasi yang perlu disampaikan secara lengkap: {data}
        Berikut adalah pertanyaan dari user: {user_question}
        Setelah mendapatkan {user_question} maka identifikasi apakah input yang diberikan adalah pertanyaan, jika bukan maka jawab dengan "REPHRASE".
        Jawab pertanyaan dari user  dengan ramah, membantu, dan interaktif hanya menggunakan informasi yang tersedia.
        Jawaban yang diberikan harus dirangkai dengan baik dari data yang disediakan.
        Jawaban yang diberikan harus mencakup semua informasi yang telah diberikan serta lengkap.
        Hindari penggunaan new line saat menjawab.
        Jawaban:
        """
        output = send_to_watsonxai(prompt)
        # print(prompt)

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

        def extract_page_numbers(query_result):
            pages = []
            for text_list in query_result:
                for text in text_list:
                    # Regex to find "Halaman xx dari xx"
                    matches = re.findall(r'Halaman (\d+) dari \d+', text)  # Use findall to get all matches
                    pages.extend([int(page) for page in matches])  # Convert to int and extend the list
            return pages
        
        # Accessing the query_result from the data
        query_result = data['values'][0][0]['query_result']

        # Extracting page numbers
        page_numbers = extract_page_numbers(query_result)
        print('Page Numbers:', page_numbers)  # Debugging statement


        # print (f"ini output ya {output}")
        print (f"ini data ya {data}")
    return {'output': [{'user_question': user_question, 'query_result': query_result, 'values': output, 'document_name': document_name, 'page_numbers': page_numbers}]}

from ibm_watson_machine_learning.foundation_models import Model
import os
import json

WX_API_KEY = "ON4BdvORJc02uEl6dXyvpNTPxhlO0_LG-k8CzG8Z6Cfu"
WX_PROJECT_ID = "4cae5e78-da4f-4aed-9510-cb1fbebd13d3"
WX_URL = "https://us-south.ml.cloud.ibm.com"

creds = {
    "url": WX_URL,
    "apikey": WX_API_KEY 
}

def classify_text(history: str, user_query: str):
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 5100,
        "min_new_tokens": 1,
        "repetition_penalty": 1,
        "stop_sequences": ["\n\n\n"]
    }
    llama3 = "meta-llama/llama-3-1-70b-instruct"
    model = Model(
        model_id=llama3,
        params=parameters,
        credentials=creds,
        project_id=WX_PROJECT_ID
    )

    template = f""" 

    Tentukan apakah pertanyaan baru dari pengguna relevan dengan percakapan sebelumnya.
    Output dalam JSON valid sebagai berikut
    Jawaban:
    {{
        "is_relevant": "YES/NO"
    }}

    Pilih salah satu dalam JSON. Jangan tambahkan apapun selain JSON tersebut.

    Input: 
    Pertanyaan user:
    Apa tugas mandor?

    Riwayat Percakapan:
    [Penyiraman bibit dilakukan 2 kali sehari, yaitu sejak pagi hingga pukul 11.00 dan pukul 15.00 sampai selesai. Kebutuhan air rata-rata untuk setiap bibit adalah ± 2 – 3 liter per large bag per hari tergantung umur bibit. Bila terjadi hujan minimal 8 mm pada hari sebelumnya maka tidak perlu dilakukan penyiraman pada hari itu. Penyiraman manual dilakukan dengan selang air yang dilengkapi dengan kepala gembor di ujungnya, sehingga terjadinya erosi tanah dan hilangnya pupuk granular dari dalam large bag dapat dihindari. Penyiraman dengan mengunakan instalasi kirico dapat dilakukan sekaligus dalam 2 plot (2 ha) selama 30 menit atau setara 6 mm curah hujan.]

    Jawaban:
    {{
        "is_relevant": "NO"
    }}

    Input: 
    Pertanyaan user:
    Pengendalian gulma

    Riwayat Percakapan:
    [Pengendalian gulma di pre-nursery hanya dilakukan dengan cara manual, yaitu dengan mencabuti seluruh jenis gulma yang tumbuh dalam baby bag. Bersamaaan dengan pengendalian gulma tersebut, dilakukan penambahan tanah ke dalam baby bag untuk bibit yang doyong dan tersembul akarnya. Gulma di dalam large bag dilakukan dengan cara manual setiap bulan sampai bibit cukup besar. Tidak diperbolehkan mengendalikan gulma di dalam large bag dengan menggunakan herbisida. Konsolidasi bibit (mendirikan dan menegakkan bibit doyong) dilakukan bersamaan dengan pengendalian gulma. Pemberian mulsa dapat menekan pertumbuhan gulma. Gulma di antara large bag dapat dilakukan dengan penyemprotan herbisida dengan bahan aktif glifosat 480 AS dosis 2-2,5 liter per ha blanket (konsentrasi 0,5 %) atau gramoxone konsentrasi 2 %. Nozel dari sprayer yang digunakan adalah polijet kuning/ VLV 200 dan posisinya harus lebih rendah dari permukaan large bag pada saat penyemprotan.]

    Jawaban:
    {{
            "is_relevant": "YES"
    }}

    Input: 
    Pertanyaan user:
    {user_query}

    Riwayat Pesrcakapan: 
    {history}

    Jawaban: 

    """

    # Generate text using the model
    generated_response = model.generate_text(prompt=template)
    
    # Print the generated response for debugging
    print("Generated response:", generated_response)
    
    # Attempt to parse JSON with error handling
    try:
        json_results = json.loads(generated_response)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Generated response:", generated_response)
        json_results = {"is_relevant": "NO"}  # Default response if parsing fails

    return json_results

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

session_1 = []
import time 

@app.post("/send_to_watsonx")
async def stream_response(request: QueryRequest):
    global session_1  # Access the global session_1 variable
    query = request.user_question.strip()
    
    try:
        
        # Log the query
        print("User Query:", query)

        duration_start = time.time ()

        time_start_milvus = time.time()

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
        
        time_end_milvus = time.time()

        time_start_wx_ask = time.time()

        # Constructing the payload for Watson
        query_result = list(data) + list(session_1) if session_1 else list(data)

        payload_watson = {
            "input_data": [{
                "values": [[{"user_question": query, "query_result": query_result}]]
            }]
        }
        print("Payload to Watson:", payload_watson)  # Debugging statement

        # Call the function that sends the payload to Watson
        result_watson = Milvus2Text(payload_watson)  # Check the return type here

        time_end_wx_ask = time.time()

        time_start_wx_relevance = time.time()

        output_text = result_watson['output'][0]['values'][0]['output']
        # Check relevance if there's prior history
        if session_1:
            # Classify the text to check relevance
            result_comparison = classify_text(history=session_1, user_query=query)
            print("Relevance check:", result_comparison)
            is_relevant = result_comparison.get("is_relevant") == "YES"
        else:
            # Assume relevance on the first question
            is_relevant = True

        # Update session_1 based on relevance
        if is_relevant:
            session_1.append(output_text)  # Append to session_1 if relevant
        else:
            session_1 = [output_text]  # Reset session_1 if not relevant
        
        time_end_wx_relevance =  time.time()
        duration_end = time.time ()
        
        # Print current history for debugging
        # return session_1, watson_results
        duration_wx_ask = time_end_wx_ask - time_start_wx_ask
        duration_milvus = time_end_milvus - time_start_milvus
        duration_wx_relevance = time_end_wx_relevance - time_start_wx_relevance
        total = duration_end - duration_start

        print("Result Watson:", result_watson)  # Debugging statement
        print ("ini session sebelumnya", session_1)
        print("ini session_count", len(session_1))
        print("Duration milvus:", duration_milvus) 
        print("Duration wx ask:", duration_wx_ask) 
        print("Duration wx relevance:", duration_wx_relevance) 
        print("Total:", total)

        # Return the result as a JSON response
        return JSONResponse(content=result_watson, status_code=200)
    
    except Exception as e:
        print("Error:", str(e))  # Log the error message
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

