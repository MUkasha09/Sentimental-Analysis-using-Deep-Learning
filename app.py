from flask import Flask, request, jsonify, render_template
from gensim.models import Word2Vec
import os
import numpy as np

app = Flask(__name__)

# Load model
model = None
try:
    print("Loading model...")
    model = Word2Vec.load(os.path.join(os.getcwd(), "word2vec.model"))
    print("Model loaded.")
except Exception as e:
    print("Error loading model:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similar-words', methods=['POST'])
def similar_words():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    count = int(data.get('count', 10))
    try:
        if 'vector' in data:
            # Handle case: input is a word vector
            vector = np.array(data['vector'], dtype=np.float32)
            similar = model.wv.similar_by_vector(vector, topn=count)

        elif 'word' in data:
            # Handle case: input is a word
            word = data['word'].strip().lower()
            if word not in model.wv:
                return jsonify({"error": f"'{word}' not found in model."}), 404
            similar = model.wv.most_similar(word, topn=count)

        else:
            return jsonify({"error": "Missing 'word' or 'vector' in request."}), 400

        return jsonify({
            "similar_words": [{"word": w, "similarity": float(s)} for w, s in similar]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
