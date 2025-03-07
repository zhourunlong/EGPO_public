import argparse
import json
import os
import pdb
import time
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch.nn.functional as F

from judges.pair_judge import PairJudge

pref_model_name = "vectorzhou/gemma-2-2b-it-preference_dataset_mixture2_and_safe_pku-Preference"
judge = PairJudge(pref_model_name)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (optional)

@app.route("/v1/classifications", methods=["POST"])
def classify():
    try:
        data = request.json
        if "prompts" not in data:
            return jsonify({"error": "Missing 'prompts' field"}), 400
        if "completions" not in data:
            return jsonify({"error": "Missing 'completions' field"}), 400
        shuffle_order = data.get("shuffle_order", True)
        
        results = judge.judge(
            prompts=data["prompts"],
            completions=data["completions"],
            shuffle_order=shuffle_order
        )

        return jsonify({
            "id": "classification-id",
            "object": "classification",
            "model": pref_model_name,
            "results": results,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
