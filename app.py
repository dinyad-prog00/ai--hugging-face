import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
import chromadb
from flask import Flask, request, jsonify, render_template


def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["author"] = record.get("author")
    metadata["tags"] = ' '.join(record.get("tags"))

    return metadata


model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')
path = "./data/quotes.json"
chroma_instance = chromadb.Client()
doc_loader = JSONLoader(path, ".[]", content_key="text",
                        metadata_func=metadata_func)
docs = doc_loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_docs = splitter.split_documents(docs)
db = Chroma.from_documents(documents=split_docs, embedding=model)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    db.similarity_search_with_score()
    response = db.similarity_search_with_score(query, k=5)
    json_response = []
    for (doc,score) in response:
        json_doc = {
            'content': doc.page_content,
            'author': doc.metadata.get('author', ''),
            'tags': doc.metadata.get('tags', ''),
            'score': score
        }
        json_response.append(json_doc)
    return jsonify(json_response)


if __name__ == '__main__':
    app.run(debug=True)
