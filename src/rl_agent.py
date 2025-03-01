import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.trading_env import CryptoTradingEnv
from src.config import Config
import os

def train_rl_agent():
    """
    Addestra l'agente RL utilizzando l'ambiente di trading simulato.
    Utilizza dati fittizi per il mercato e segnali di sentiment.
    """
    market_data = np.linspace(100, 200, num=500)  # prezzi simulati in crescita lineare
    sentiment_data = np.random.uniform(0, 1, size=500)  # valori casuali per il sentiment

    env = CryptoTradingEnv(market_data, sentiment_data)
    check_env(env, warn=True)

    model_rl = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",
        batch_size=32,
        n_steps=256
    )
    model_rl.learn(total_timesteps=Config.RL_TOTAL_TIMESTEPS)
    
    os.makedirs("models", exist_ok=True)
    model_rl.save(Config.RL_MODEL_PATH)
    print("Agente RL addestrato e salvato in", Config.RL_MODEL_PATH)
    return model_rl

def load_rl_agent():
    """
    Carica un agente RL precedentemente addestrato dal percorso salvato in Config.RL_MODEL_PATH.
    """
    model_path = str(Config.RL_MODEL_PATH) + ".zip"
    if os.path.exists(model_path):
        model_rl = PPO.load(Config.RL_MODEL_PATH, device="cpu")
        print("Agente RL caricato da", Config.RL_MODEL_PATH)
        return model_rl
    else:
        print("Nessun modello trovato in", Config.RL_MODEL_PATH)
        return None

if __name__ == "__main__":

    train_rl_agent()
