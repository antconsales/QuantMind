import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.config import Config

class CryptoTradingEnv(gym.Env):
    def __init__(self, market_data, llm_scores): 
        self.llm_scores = llm_scores 
        self.current_step = 0
        self.balance = Config.TRADING_INITIAL_CAPITAL
        self.position = 0
        self.risk_factor = Config.RISK_FACTOR

        
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(Config.RL_OBSERVATION_SPACE,),  
            dtype=np.float32
        )

    def _get_observation(self):
        price = self.market_data[self.current_step]
        llm_value = self.llm_scores[self.current_step]
        return np.array([
            price, 
            self.balance, 
            self.position, 
            llm_value  
        ], dtype=np.float32)

    def step(self, action):
        price = self.market_data[self.current_step]
        reward = 0
        transaction_cost = price * Config.TRADING_QUANTITY * self.risk_factor

        if action == 1:  # BUY
            cost = price * Config.TRADING_QUANTITY + transaction_cost
            if self.balance >= cost:
                self.balance -= cost
                self.position += Config.TRADING_QUANTITY
                
        elif action == 2:  # SELL
            if self.position >= Config.TRADING_QUANTITY:
                revenue = price * Config.TRADING_QUANTITY - transaction_cost
                self.balance += revenue
                self.position -= Config.TRADING_QUANTITY

        net_worth = self.balance + self.position * price
        reward = net_worth - (self.position * self.risk_factor * price)
        
        self.current_step += 1
        terminated = self.current_step >= len(self.market_data) - 1
        return self._get_observation(), reward, terminated, False, {}

    def render(self, mode="human"):
        price = self.market_data[self.current_step]
        net_worth = self.balance + self.position * price
        print(f"Step: {self.current_step} | Prezzo: {price} | Capitale: {self.balance} | Posizione: {self.position} | Net Worth: {net_worth}")
