from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma(embedding_function=model, persist_directory="./db/bible_v2.db")
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    # Perform search in Elasticsearch
    response = db.similarity_search(query, k=3)
    response_json = []
    for res in response:
        response_json.append(
           res.dict()
        )

    return jsonify(response_json)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
