# from flask import Flask, request, jsonify, render_template, send_from_directory
# from flask_cors import CORS,cross_origin
# import os
# import fitz
# import json
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# import mysql.connector
# from huggingface_hub import login
# import traceback


# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline




# app = Flask(__name__)
# CORS(app)

# conn = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='2005',
#     database='chatbot'
# )
# cursor = conn.cursor(dictionary=True)  # Optional: returns results as dicts

# UPLOAD_FOLDER = r"d:\Internship\D4\static\uploads"  # raw string, single backslashes
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # model = SentenceTransformer('all-MiniLM-L6-v2')
# # DIM = model.get_sentence_embedding_dimension()
# # index = faiss.IndexFlatL2(DIM)

# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# llm_model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", torch_dtype="auto"
# )
# llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)

# # Store metadata (ids → filename/page mapping)
# metadata = []

#------------------------------------------------------------------
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS, cross_origin
import os
import fitz
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import mysql.connector
import requests
import traceback
from urllib.parse import unquote
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch


# loding a new model -----------------
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


# # Hugging Face Inference API details
# # HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
# HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

# headers = {
#     "Authorization": f"Bearer {HF_TOKEN}",
#     "Content-Type": "application/json"
# }
# ----------------------------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-rw-1b"
headers = {
    "Authorization": "Bearer ",  # 🔐 use your real token
    "Content-Type": "application/json"

}

USE_LOCAL_MODEL = True
if USE_LOCAL_MODEL:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# API setup (if using Hugging Face Inference API)
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-2"
headers = {
    "Authorization": ""  # Replace with your actual token
}

# Function to query either local model or Hugging Face API
def query_llm(prompt):
    try:
        prompt = prompt[:2000]  # Truncate to safe length

        if USE_LOCAL_MODEL:
            result = pipe(prompt, max_new_tokens=300, do_sample=False)
            return result[0]['generated_text'].strip()

        else:
            # Payload for Hugging Face API
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "do_sample": False
                }
            }

            # Send request
            response = requests.post(HF_API_URL, headers=headers, json=payload)

            # Error handling
            if response.status_code != 200:
                print(f"[LLM ERROR] Status {response.status_code}")
                print(f"[LLM ERROR] Response Text: {response.text}")
                print(f"[LLM ERROR] Headers Sent: {headers}")
                print(f"[LLM ERROR] Payload Sent: {payload}")
                return "Error generating answer"

            result = response.json()
            return result[0]['generated_text'].strip()

    except Exception as e:
        print(f"[LLM EXCEPTION] {type(e).__name__}: {e}")
        traceback.print_exc()
        return "Error generating answer"
# ----------------------- original --------------------------
# def query_llm(prompt):
#     try:
#         prompt = prompt[:2000]
#         payload = {
#             "inputs": prompt,
#             "parameters": {
#                 "max_new_tokens": 300,
#                 "do_sample": False
#             }
#         }
#         response = requests.post(HF_API_URL, headers=headers, json=payload)

#         # Print detailed HTTP error if any
#         if response.status_code != 200:
#             print(f"[LLM ERROR] Status {response.status_code}")
#             print(f"[LLM ERROR] Response Text: {response.text}")
#             print(f"[LLM ERROR] Headers Sent: {headers}")
#             print(f"[LLM ERROR] Payload Sent: {payload}")
#             return "Error generating answer"

#         result = response.json()

#         # Print full response for inspection
#         print("[LLM RESPONSE]", result)

#         # Parse and clean the output
#         return result[0]['generated_text'].split("Answer:")[-1].strip() \
#             if "Answer:" in result[0]['generated_text'] else result[0]['generated_text']

#     except Exception as e:
#         print(f"[LLM EXCEPTION] {type(e).__name__}: {e}")
#         traceback.print_exc()
#         return "Error generating answer"

# Flask setup
app = Flask(__name__)
CORS(app)

# MySQL setup
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='2005',
    database='chatbot'
)
cursor = conn.cursor(dictionary=True)

# Paths
UPLOAD_FOLDER = r"d:\Internship\D4\static\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Metadata placeholder
metadata = []

# @app.route('/upload-file', methods=['POST'])
# def upload_file():
#     if 'pdf' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['pdf']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if not file.filename.lower().endswith('.pdf'):
#         return jsonify({"error": "Only PDF files allowed"}), 400

#     try:
#         # Read file content
#         file_data = file.read()

#         # Save to filesystem in binary mode
#         save_path = os.path.join(UPLOAD_FOLDER, file.filename)
#         with open(save_path, 'wb') as f:
#             f.write(file_data)

#         # Optional: log
#         print(f"File saved: {save_path}")

#         return jsonify({'message': f'{file.filename} uploaded successfully'}), 201

#     except Exception as err:
#         return jsonify({'error': str(err)}), 500

# Upload Route
@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files allowed"}), 400

    try:
        file_data = file.read()
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(save_path, 'wb') as f:
            f.write(file_data)

        # Save to DB
        cursor.execute(
            "INSERT INTO uploaded_file (filename, data) VALUES (%s, %s)",
            (file.filename, file_data)
        )
        conn.commit()

        return jsonify({'message': f'{file.filename} uploaded successfully'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

   
@app.route('/preview/<filename>', methods=['GET'])
def preview(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": f"{filename} not found"}), 404

    result = extract_text_from_pdf(filename)
    return jsonify(result)


def extract_text_from_pdf(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        doc = fitz.open(file_path)
        text = ''.join([doc.load_page(i).get_text() for i in range(doc.page_count)])
        doc.close()
        return {"text": text} if text.strip() else {"error": "No extractable text"}
    except Exception as e:
        return {"error": str(e)}
 

@app.route('/show/<path:filename>', methods=['GET'])
def showfile(filename):
    print(f"[DEBUG] Received filename: {repr(filename)}")

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    print(f"[DEBUG] Resolved file path: {file_path}")
    print(f"[DEBUG] Upload folder contents: {os.listdir(UPLOAD_FOLDER)}")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return jsonify({"error": f"{filename} not found"}), 404

    try:
        # Extract text by page
        doc = fitz.open(file_path)
        page_texts = [doc.load_page(i).get_text() for i in range(doc.page_count)]
        doc.close()

        cleaned_pages = [text.strip() for text in page_texts if text.strip()]
        if not cleaned_pages:
            return jsonify({"error": "No extractable text"}), 400

        # Generate embeddings
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(cleaned_pages, convert_to_numpy=True)

        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        metadata = [
            {
                "filename": filename,
                "page": i,
                "text": text
            } for i, text in enumerate(cleaned_pages)
        ]

        # Save index and metadata
        base_filename = os.path.splitext(filename)[0]
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'vectors')
        os.makedirs(output_dir, exist_ok=True)

        index_path = os.path.join(output_dir, f'{base_filename}.index')
        metadata_path = os.path.join(output_dir, f'{base_filename}_metadata.json')

        faiss.write_index(index, index_path)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return jsonify({
            "message": f"Extracted and saved vectors for {filename}",
            "pages": len(cleaned_pages)
        })

    except Exception as e:
        print("[ERROR] Exception occurred:", e)
        return jsonify({"error": str(e)}), 500






@app.route('/listfiles',methods=['GET'])

def get_uploaded_files():
    if not os.path.exists(UPLOAD_FOLDER):
        return []

    # List all files in the folder
    files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    return jsonify(files)


@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        cursor.execute("DELETE FROM uploaded_file WHERE filename = %s", (filename,))
        conn.commit()
        
        os.remove(file_path)
        return jsonify({"message": f"'{filename}' deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete file: {str(e)}"}), 500  # this is the delete function



# @app.route('/search', methods=['GET'])
# @cross_origin()
# def search_vectors():
#     output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'vectors')

#     filename = request.args.get('filename')
#     query = request.args.get('query')

#     if not filename:
#         return jsonify({"error": "Filename parameter is required"}), 400
#     if not query:
#         return jsonify({"error": "Query parameter is required"}), 400

#     base_filename = os.path.splitext(filename)[0]
#     index_file = os.path.join(output_dir, f'{base_filename}.index')
#     metadata_file = os.path.join(output_dir, f'{base_filename}_metadata.json')

#     if not os.path.exists(index_file) or not os.path.exists(metadata_file):
#         return jsonify({"error": f"Index or metadata files for '{filename}' not found"}), 404

#     try:
#         model = SentenceTransformer('all-MiniLM-L6-v2')
#         index = faiss.read_index(index_file)

#         with open(metadata_file, 'r', encoding='utf-8') as f:
#             metadata = json.load(f)

#         query_embedding = model.encode([query], convert_to_numpy=True)
#         D, I = index.search(query_embedding, 5)

#         results = []
#         for rank, idx in enumerate(I[0]):
#             if idx < len(metadata):
#                 result = metadata[idx]
#                 results.append({
#                     "rank": rank + 1,
#                     "filename": result["filename"],
#                     "page": result["page"],
#                     "text": result["text"],
#                     "distance": float(D[0][rank])
#                 })

#         # Save to DB (assuming cursor and conn are your database cursor and connection)
#         result_json = json.dumps(results)
#         cursor.execute(
#             "INSERT INTO search_history (query, result_json) VALUES (%s, %s)",
#             (query, result_json)
#         )
#         conn.commit()

#         return jsonify(results)

#     except Exception as e:
#         print("Search error:", str(e))
#         return jsonify({"error": str(e)}), 500


@app.route('/search', methods=['GET'])
@cross_origin()
def search_vectors():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'vectors')

    filename = request.args.get('filename')
    query = request.args.get('query')

    if not filename:
        return jsonify({"error": "Filename parameter is required"}), 400
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    filename = unquote(filename)
    base_filename = os.path.splitext(filename)[0]
    index_file = os.path.join(output_dir, f'{base_filename}.index')
    metadata_file = os.path.join(output_dir, f'{base_filename}_metadata.json')

    print(f"[INFO] Looking for index file: {index_file}")
    print(f"[INFO] Looking for metadata file: {metadata_file}")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        return jsonify({"error": f"Index or metadata files for '{filename}' not found"}), 404

    try:
        print("[INFO] Loading FAISS index ...")
        index = faiss.read_index(index_file)

        print("[INFO] Loading metadata JSON ...")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"[INFO] Encoding query: {query}")
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)

        print("[INFO] Searching top 5 results in FAISS index ...")
        D, I = index.search(query_embedding, 5)

        top_chunks = []
        results = []

        for rank, idx in enumerate(I[0]):
            if idx < len(metadata):
                result = metadata[idx]
                text = result.get("text", "")
                top_chunks.append(text)
                results.append({
                    "rank": rank + 1,
                    "filename": result.get("filename", ""),
                    "page": result.get("page", -1),
                    "text": text,
                    "distance": float(D[0][rank])
                })

        context = "\n".join(top_chunks)
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

        print("[INFO] Calling LLM for answer generation ...")
        answer = query_llm(prompt)
        print(answer)

        # Convert to JSON for DB storage
        try:
            result_json = json.dumps(results, ensure_ascii=False)
        except Exception as json_error:
            print(f"[JSON ERROR] {json_error}")
            traceback.print_exc()
            result_json = "[]"

        # Optional: Save search result to DB
        try:
            cursor.execute(
                "INSERT INTO search_history_1 (query, result_json, llm_answer) VALUES (%s, %s, %s)",
                (query, result_json, answer)
            )
            conn.commit()
        except Exception as db_error:
            print(f"[DB ERROR] {db_error}")
            traceback.print_exc()

        return jsonify({
            "query": query,
            "results": results,
            "llm_answer": answer
        })

    except Exception as e:
        print(f"[SERVER ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
#    context = '''Linear Regression
# Linear Regression is a fundamental supervised learning algorithm used primarily for predicting
# continuous numeric values. It models the relationship between a dependent variable and one or
# more independent variables by fitting a linear equation to observed data.
# Advantages:
# One of the major advantages of linear regression is its simplicity and ease of interpretation. It works
# well when there is a linear relationship between the input and output variables, and it is
# computationally efficient.
# Disadvantages:
# However, linear regression can perform poorly when the data has a non-linear relationship or when
# there are multicollinearity issues among the features. It is also sensitive to outliers, which can skew
# the results significantly.
# Logistic Regression
# Logistic Regression is used for binary classification problems. It predicts the probability that an
# instance belongs to a particular category by applying a logistic function to a linear combination of
# input features.
# Advantages:
# It is a simple and effective algorithm for binary classification tasks and provides probabilistic outputs,
# which can be useful for decision making. It also works well when the data is linearly separable.
# Disadvantages:
# Despite its usefulness, logistic regression struggles with complex relationships and non-linear data.
# It assumes a linear decision boundary and may underperform when this assumption does not hold.'''
#    query = "what is linear regression ?"
#    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
#    local_res= query_llm(prompt);
#    print("local_res ===:",local_res)
    app.run(debug=True, use_reloader=False)
