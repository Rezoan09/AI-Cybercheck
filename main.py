
import os
from flask import Flask, request, jsonify
import pandas as pd
from utils import load_artifact

app = Flask(__name__)

def get_ids_model():
    clf = load_artifact("ids_model", "models")
    cols = load_artifact("ids_columns", "models")
    return clf, cols

def get_phish_model():
    pipe = load_artifact("phish_model", "models")
    return pipe

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/detect_anomaly", methods=["POST"])
def detect_anomaly():
    try:
        clf, cols = get_ids_model()
    except Exception as e:
        return jsonify({"error": f"IDS model not available. Train first. {e}"}), 400

    data = request.get_json(force=True)
    rows = None

    if isinstance(data, dict) and "rows" in data:
        rows = data["rows"]
    else:
        rows = [data]

    X = pd.DataFrame(rows)
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols].fillna(0)

    preds = clf.predict(X)
    proba = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[:, 1].tolist()
    else:
        proba = [float(p) for p in preds]

    out = []
    for p, pr in zip(preds, proba):
        out.append({"prediction": int(p), "is_threat": bool(p==1), "confidence": float(pr)})
    return jsonify({"results": out})

@app.route("/detect_phishing", methods=["POST"])
def detect_phishing():
    try:
        pipe = get_phish_model()
    except Exception as e:
        return jsonify({"error": f"Phishing model not available. Train first. {e}"}), 400

    data = request.get_json(force=True)
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Provide 'text' field with email/message content"}), 400

    pred = int(pipe.predict([text])[0])
    if hasattr(pipe, "predict_proba"):
        conf = float(pipe.predict_proba([text])[0][pred])
    else:
        conf = 0.5
    return jsonify({"prediction": pred, "is_phish": bool(pred==1), "confidence": conf})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
