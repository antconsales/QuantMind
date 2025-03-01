import argparse
from src.config import Config
import logging
from src.fine_tuning import fine_tune
from src.rl_agent import train_rl_agent
from src.api import app

logging.basicConfig(level=logging.INFO)

Config.init_paths()

def main():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--fine_tune", action="store_true", help="Esegui il fine-tuning del modello di sentiment")
    parser.add_argument("--train_rl", action="store_true", help="Addestra l'agente RL")
    parser.add_argument("--run_api", action="store_true", help="Avvia il server API")
    args = parser.parse_args()

    if args.fine_tune:
        fine_tune()
    if args.train_rl:
        train_rl_agent()
    if args.run_api:
        logging.info(f"Avvio server su {Config.API_HOST}:{Config.API_PORT}")
        app.run(
            host=Config.API_HOST,
            port=Config.API_PORT,
            threaded=True,
            use_reloader=False
        )

if __name__ == "__main__":
    main()
