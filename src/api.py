import os
import psutil
import requests
import numpy as np
import torch
import platform
import json

from pathlib import Path
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from src.config import Config
from src.rl_agent import load_rl_agent

FINBERT_MODEL_NAME = "ProsusAI/finbert"

app = Flask(__name__)

tokenizer_finbert = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
model_finbert = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)

finbert_pipeline = pipeline(
    "text-classification",
    model=model_finbert,
    tokenizer=tokenizer_finbert,
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# Carica l'agente RL
model_rl = load_rl_agent()

class ResourceManager:
    """
    Classe per gestire le risorse (RAM, CPU) e abilitare/disabilitare l'LLM in caso di superamento soglie
    """
    def __init__(self):
        self.llm_enabled = True
        self.check_resources()

    def clear_cache(self):
        """Svuota la cache di torch e python"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def check_resources(self):
        """Controlla le risorse del sistema e ritorna un dizionario con i dati di cpu e ram"""
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        resources = {
            "ram_percent": mem.percent,
            "cpu_percent": cpu_percent,
            "llm_enabled": self.llm_enabled
        }
        if mem.percent > Config.RAM_THRESHOLD or cpu_percent > Config.CPU_LIMIT_PERCENT:
            self.clear_cache()
            self.llm_enabled = False
        else:
            self.llm_enabled = True
        resources["llm_enabled"] = self.llm_enabled
        return resources

resource_manager = ResourceManager()

@app.route('/ping', methods=['GET'])
def ping():
    """Endpoint di verifica della disponibilità dell'API"""
    return jsonify({"message": "pong"})

@app.route('/analizza_llm', methods=['POST'])
def analizza_llm():
    """Analizza i testi inviati con il modello LLM FinBERT"""
    app.logger.info("Richiesta ricevuta su /analizza_llm")

    resource_data = resource_manager.check_resources()
    if not resource_data["llm_enabled"]:
        app.logger.error("Risorse insufficienti per processare la richiesta")
        return jsonify({"error": "Risorse insufficienti", "resources":resource_data}), 503

    data = request.get_json()
    app.logger.info("Dati ricevuti: %s", data)

    testi = data.get("testi", [])
    if not testi:
        app.logger.error("Nessun testo fornito nella richiesta")
        return jsonify({"error": "Nessun testo fornito"}), 400

    testi = [t.strip() for t in testi if t.strip()]
    if not testi:
        app.logger.error("Lista di testi vuota dopo il trimming")
        return jsonify({"error": "Lista di testi vuota"}), 422 # 422 Unprocessable Entity

    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    total_positive_score = 0.0
    total_negative_score = 0.0
    total_neutral_score = 0.0

    for testo in testi:
        app.logger.info("Analizzo il testo: %s", testo)
        results = finbert_pipeline(testo)
        score_dict = {r["label"].lower(): r["score"] for r in results[0]}
        app.logger.info("Risultati ottenuti: %s", score_dict)

        pos = score_dict.get("positive", 0.0)
        neg = score_dict.get("negative", 0.0)
        neu = score_dict.get("neutral", 0.0)

        total_positive_score += pos
        total_negative_score += neg
        total_neutral_score += neu

        best_label = max(score_dict, key=score_dict.get)
        if best_label == "positive":
            bullish_count += 1
        elif best_label == "negative":
            bearish_count += 1
        else:
            neutral_count += 1

    n = len(testi)
    avg_pos = total_positive_score / n
    avg_neg = total_negative_score / n
    avg_neu = total_neutral_score / n

    if avg_pos > avg_neg and avg_pos > avg_neu:
        sentiment_label = "BULLISH"
        sentiment_score = avg_pos
    elif avg_neg > avg_pos and avg_neg > avg_neu:
        sentiment_label = "BEARISH"
        sentiment_score = avg_neg
    else:
        sentiment_label = "NEUTRAL"
        sentiment_score = avg_neu

    response_data = {
        "analysis": {
            "sentiment_label": sentiment_label,
            "sentiment_score": round(float(sentiment_score), 3),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count
        }
    }
    app.logger.info("Risposta inviata: %s", response_data)
    return jsonify(response_data), 200


@app.route('/decidi', methods=['POST'])
def decidi():
    """Prende una decisione di trading basata sul prezzo, il balance, la posizione e l'analisi LLM"""
    try:
        data = request.get_json()
        prezzo = float(data.get("prezzo", 0.0))
        balance = float(data.get("balance", Config.TRADING_INITIAL_CAPITAL))
        posizione = float(data.get("posizione", 0))

        llm_score = 0.5
        llm_label = "NEUTRAL"

        resource_data = resource_manager.check_resources()
        if resource_data["llm_enabled"] and "testi" in data:
            try:
                r = requests.post(f"http://{Config.API_HOST}:{Config.API_PORT}/analizza_llm",
                                json={"testi": data["testi"]},
                                timeout=Config.LLM_TIMEOUT)
                r.raise_for_status()
                r_data = r.json()
                if "analysis" in r_data:
                    llm_label = r_data["analysis"]["sentiment_label"]
                    llm_score = float(r_data["analysis"]["sentiment_score"])
            except requests.exceptions.RequestException as e:
                app.logger.error(f"Errore analisi LLM in /decidi: {e}")

        osservazione = np.array([prezzo, balance, posizione, llm_score], dtype=np.float32).reshape(1, -1)
        azione, _ = model_rl.predict(osservazione)
        azione_int = int(azione)
        decisione_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        decisione = decisione_map.get(azione_int, "HOLD")

        return jsonify({
            "decisione": decisione,
            "llm_label": llm_label,
            "llm_score": llm_score,
            "risorse": "ok" if resource_data["llm_enabled"] else "limitato",
            "resource_data": resource_data
        })
    except Exception as e:
        app.logger.error(f"Errore non gestito in /decidi: {e}")
        return jsonify({"error": "Errore nel processo decisionale"}), 500


@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    """Chiede all'LLM di generare una risposta al prompt inviato"""
    resource_data = resource_manager.check_resources()
    if not resource_data["llm_enabled"]:
        return jsonify({"error": "Risorse insufficienti", "resources": resource_data}), 503
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Nessun prompt fornito a /ask_llm"}), 400

        results = finbert_pipeline(prompt)

        response_text = "Risposta LLM:\n"
        for res in results[0]:
            response_text += f"{res['label']}: {res['score']:.4f}\n"

        return jsonify({"response": response_text}), 200
    except Exception as e:
        app.logger.error(f"Errore non gestito in /ask_llm: {e}")
        return jsonify({"error": "Errore nella richiesta LLM"}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Mostra lo stato del Raspberry PI"""
    resource_data = resource_manager.check_resources()
    cpu_freq = psutil.cpu_freq()
    cpu_freq_current = cpu_freq.current if cpu_freq else "N/A"
    disk_usage = psutil.disk_usage('/')

    status_data = {
        "system": platform.system(),
        "node_name": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_percent": resource_data["cpu_percent"],
        "cpu_freq_current": cpu_freq_current,
        "ram_percent": resource_data["ram_percent"],
        "disk_percent": disk_usage.percent,
        "llm_enabled": resource_data["llm_enabled"]
    }
    
    return jsonify(status_data)

if __name__ == '__main__':
    print(f"Avvio server su {Config.API_HOST}:{Config.API_PORT}")
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        threaded=True,
        use_reloader=False
    )
