

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
    app.run(debug=True, use_reloader=False)
