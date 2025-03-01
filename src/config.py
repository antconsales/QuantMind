from pathlib import Path

class Config:
   
    SEED = 42
    ENVIRONMENT = "production"
    LOG_DIR = Path("logs")
    MODEL_CACHE_DIR = Path("model_cache")
    MAX_INPUT_LENGTH = 512
  
    LLM_MODEL_NAME = "ProsusAI/finbert"
    FINE_TUNED_MODEL_NAME = "crypto_sentiment_model"
    BASE_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
    NUM_LABELS = 3  # 0=SELL, 1=HOLD, 2=BUY
    MAX_LENGTH = 128
    LLM_QUANTIZED = True
    LLM_PROVIDER = "CPUExecutionProvider"

   
    MAX_CONCURRENT_REQUESTS = 2
    LLM_CACHE_SIZE = 5
    LLM_REFRESH_INTERVAL = 300  
    LLM_TIMEOUT = 3.5  
    RAM_THRESHOLD = 85  

   
    TRADING_INITIAL_CAPITAL = 10000.0
    TRADING_QUANTITY = 1
    VIRALITY_BUY_THRESHOLD = 0.68
    SENTIMENT_SELL_THRESHOLD = 0.35
    RISK_FACTOR = 0.02

    # Reinforcement Learning
    RL_TOTAL_TIMESTEPS = 10000
    RL_MODEL_PATH = Path("models/ppo_agent")
    RL_OBSERVATION_SPACE = 4 

    # API 
    API_HOST = "127.0.0.1"
    API_PORT = 7000
    TELEGRAM_ENDPOINT = "https://api.telegram.org/bot"
    COINGECKO_ENDPOINT = "https://api.coingecko.com/api/v3"

    # Social monitoring
    SOCIAL_SOURCES = {
        'twitter': {'hashtags': ['#crypto', '#bitcoin'], 'limit': 3},
        'reddit': {'subreddits': ['CryptoMoonShots', 'SatoshiStreetBets'], 'limit': 2},
        'telegram': {'channels': ['pump_group', 'crypto_alerts'], 'limit': 2}
    }

    # Raspberry Pi
    CPU_LIMIT_PERCENT = 50
    PROCESS_PRIORITY = {
        'bot': -10,
        'api': 19
    }
    ENABLE_HARDWARE_MONITORING = True

    # Path initialization
    @classmethod
    def init_paths(cls):
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
