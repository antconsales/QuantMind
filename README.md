# Crypto Trading Bot with LLM and Reinforcement Learning

This project implements a cryptocurrency trading bot that utilizes a Large Language Model (LLM) for sentiment analysis and a Reinforcement Learning (RL) agent to make trading decisions. The bot is designed to run on a Raspberry Pi, optimizing resource usage and integrating various advanced technologies.

## Key Features

-   **Sentiment Analysis with LLM (FinBERT):**
    -   Uses FinBERT, an LLM optimized for the financial sector, to analyze sentiment from text (e.g., social media, news).
    -   API endpoint `/analyze_llm` to analyze text and obtain a sentiment score.
    -   API endpoint `/ask_llm` to interact with the LLM and receive responses.
-   **Reinforcement Learning (PPO):**
    -   Uses the Proximal Policy Optimization (PPO) algorithm to train an RL agent for trading decisions (buy, sell, hold).
    -   The RL agent is based on price, balance, current position, and sentiment score from the LLM.
    -   API endpoint `/decide` for decision-making through the RL agent.
-   **Optimization for Limited Resources:**
    -   Dynamic quantization of models to reduce memory usage.
    -   Resource management (RAM, CPU) with the `ResourceManager` class to prevent crashes.
    -   System monitoring with the `/status` endpoint.
    -   Resource control and disabling of the LLM if thresholds are exceeded.
    -   Batch size and gradient accumulation to optimize training.
    -   Use of efficient optimizers.
-   **Flask API:**
    -   Implementation of a REST API with Flask to interact with the LLM and RL agent.
    -   API endpoint `/ping` to check API availability.
    -   API endpoint `/status` to display Raspberry Pi status.
-   **Fine-tuning:**
    -   Ability to fine-tune the LLM model to improve sentiment analysis accuracy.
    -   `fine_tuning.py` script for fine-tuning.
-   **Modularity:**
    -   Well-structured code divided into modules to facilitate maintenance and development.
    -   Centralized configuration in the `config.py` file.
-   **Logging:**
    -   Uses logging to track system operation and facilitate debugging.
-   **Autostart:**
    -   The system can automatically start in case of a reboot.

## Architecture

The system consists of the following main components:

1.  **`config.py`**: Contains all system configuration parameters, including paths, model parameters, and trading parameters.
2.  **`api.py`**: Implements the REST API with Flask. Manages interaction with the LLM, RL agent, and `ResourceManager`.
3.  **`rl_agent.py`**: Defines functions for training and loading the RL agent.
4.  **`fine_tuning.py`**: Contains logic for fine-tuning the LLM model.
5.  **`sentiment.py`**: Contains methods to analyze sentiment through the model.
6.  **`trading_env.py`**: Defines the trading environment.
7.  **`main.py`**: The main script that manages system startup (fine-tuning, RL training, API startup).

## Installation and Setup

### Prerequisites

-   Raspberry Pi (recommended model 4 or higher) with Raspberry Pi OS installed.
-   Python 3.9 or higher.
-   `pip` for Python package management.
-   Python virtual environment (recommended `venv`).

### Steps

1.  **Clone the repository:**

    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd llm_trade
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    Remember to create the `requirements.txt` file with the necessary Python dependencies using:

    ```bash
    pip freeze > requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file and add the following variables:

    ```
    BINANCE_API_KEY=<API_KEY>
    BINANCE_API_SECRET=<API_SECRET>
    TELEGRAM_API_TOKEN=<TELEGRAM_TOKEN>
    GPT_API_KEY=<OPENAI_API_KEY>
    ```

5.  **Run the Bot:**
    -   To start the API server:
        ```bash
        python src/main.py --run_api
        ```
    -   To train the RL agent:
        ```bash
        python src/main.py --train_rl
        ```

## Autostart

To automatically start the bot in case of a reboot:

1. **Create a startup script** `start_bot.sh`:

    ```bash
    #!/bin/bash
    source venv/bin/activate
    cd /path/to/project
    python src/main.py --run_api
    ```

2. **Make the script executable:**

    ```bash
    chmod +x start_bot.sh
    ```

3. **Create a systemd service:**

    ```bash
    sudo nano /etc/systemd/system/llm_trade.service
    ```

    Add:

    ```
    [Unit]
    Description=LLM Crypto Trading Bot
    After=network.target

    [Service]
    Type=simple
    User=<your_username>
    WorkingDirectory=/path/to/project
    ExecStart=/path/to/project/start_bot.sh
    Restart=on-failure

    [Install]
    WantedBy=multi-user.target
    ```

4. **Enable and start the service:**
    ```bash
    sudo systemctl enable llm_trade.service
    sudo systemctl start llm_trade.service
    ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Disclaimer

This project is provided "as is" without any guarantees. Cryptocurrency trading involves high risks of loss. Use this bot at your own risk.
