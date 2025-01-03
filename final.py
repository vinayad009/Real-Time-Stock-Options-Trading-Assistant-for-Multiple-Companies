import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from catboost import CatBoostClassifier
import lightgbm as lgb
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import streamlit as st

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import yfinance as yf
from nsetools import Nse
import weaviate
from weaviate.classes.config import Property, Configure, DataType
from weaviate.classes.query import Filter
import joblib
from catboost import CatBoostClassifier
import lightgbm as lgb
import os
import replicate


replicate_client = replicate.Client(api_token="r8_ehjrSdCah47PyhMcXMJH3kOzFUTKCAx0GLyyV5")  # Replace with your actual API key

# Map company names to integers
company_mapping = {"INFY": 0, "RELIANCE": 1, "TCS": 2}

def fetch_yahoo_data(ticker, date):
    end_date = date
    start_date = end_date - timedelta(days=40)  # Fetch 40 market days before the given date
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Company'] = ticker
    return data


# def fetch_nse_data(symbol):
#     nse = Nse()
#     quote = nse.get_quote(symbol)
#     return pd.DataFrame([quote])


# --- 2. Data Preprocessing ---
def load_data(file_path, live=False):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data, live=False):
    """
    Preprocess the data to include technical indicators and ensure numerical values only.
    """
    data['Company'] = data['Company'].map(company_mapping)
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    if live:
        # Only use the last 40 days of data for calculations
        data = data.groupby("Company").apply(lambda x: x.iloc[-40:]).reset_index(drop=True)

        data['SMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20).mean())
        data['EMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        data['Daily_Return'] = data.groupby('Company')['Close'].transform(lambda x: x.pct_change())
        data['RSI'] = data.groupby('Company')['Close'].transform(compute_rsi)
        data = data.dropna()  # Remove rows with NaN values
        if live:
            data = data.groupby('Company').tail(1)
        return data

    else:
        # Ensure numeric columns and parse dates
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        # Technical indicators
        data['SMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20).mean())
        data['EMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        data['Daily_Return'] = data.groupby('Company')['Close'].transform(lambda x: x.pct_change())
        data['RSI'] = data.groupby('Company')['Close'].transform(compute_rsi)

        # Drop rows with NaN values after calculations
        data = data.dropna()

        # Drop non-numeric columns (e.g., `Date`) and reset index
        return data.drop(columns=['Date'], errors='ignore').reset_index(drop=True)


def compute_rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def adjust_date_range(date):
    """
    Adjust the range to the last 40 days from the given date.
    """
    end_date = date
    start_date = end_date - timedelta(days=40)
    return start_date, end_date


# --- 3. Vector Database Integration with Weaviate ---
def setup_weaviate():
    client = weaviate.connect_to_local()
    print(client.is_ready())
    try:
        client.collections.create(
            "StockData",
            description="A collection of stock market data for various companies.",
            properties=[
                Property(name="Company", data_type=DataType.TEXT),
                Property(name="Features", data_type=DataType.TEXT)
            ],
        )
    finally:
        return client


def store_data_in_weaviate(client, data):
    stocks = client.collections.get("StockData")
    for _, row in data.iterrows():
        stocks.data.insert(
            {
                "Company": row['Company'],
                "Features": row[['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']].to_json(),
            }
        )


def retrieve_rag_data(client, company):
    company_ = ""
    for key, value in company_mapping.items():
        if value == company:
            company_ = key
    stocks = client.collections.get("StockData")
    result = stocks.query.bm25(
        query=company_,
        filters=Filter.by_property("company").equal(company_),
        return_properties=["company", "features"]
    )
    return result


# --- 2. Machine Learning Model ---
def train_model(data):
    """
    Train an ensemble ML model (LightGBM + CatBoost) to predict buy/sell signals.
    """
    features = ['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']
    data.loc[:, 'Signal'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    X = data[features]
    y = data['Signal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train LightGBM model
    lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    lgb_model.fit(X_train, y_train)

    # Train CatBoost model
    cat_model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
    cat_model.fit(X_train, y_train)

    # Ensemble Predictions
    lgb_predictions = lgb_model.predict_proba(X_test)[:, 1]
    cat_predictions = cat_model.predict_proba(X_test)[:, 1]
    ensemble_predictions = (0.5 * lgb_predictions) + (0.5 * cat_predictions)
    final_predictions = (ensemble_predictions >= 0.5).astype(int)

    # Evaluate
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Ensemble Model Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", classification_report(y_test, final_predictions, target_names=['Sell', 'Buy']))

    # Save the model
    ensemble_model = {'lgb_model': lgb_model, 'cat_model': cat_model, 'weights': [0.5, 0.5]}
    joblib.dump(ensemble_model, 'final_model.pkl')
    return ensemble_model


# --- DRL Agent Training ---
class StockTradingEnv(gym.Env):
    """
    Custom Stock Trading Environment for PPO.
    """

    def __init__(self, data, initial_balance=100000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_step = 0
        self.num_shares = 0
        self.max_steps = len(data) - 1

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)

    def reset(self):
        self.current_balance = self.initial_balance
        self.current_step = 0
        self.num_shares = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']

        if action == 0:  # Buy
            self.num_shares += self.current_balance // current_price
            self.current_balance %= current_price
        elif action == 1:  # Sell
            self.current_balance += self.num_shares * current_price
            self.num_shares = 0

        self.current_step += 1
        done = self.current_step >= self.max_steps
        total_value = self.current_balance + (self.num_shares * current_price)
        reward = total_value - self.initial_balance
        next_state = self.data.iloc[self.current_step].values if not done else np.zeros_like(self.data.iloc[0].values)
        return next_state, reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.current_balance}, Shares: {self.num_shares}")

def train_drl_agent(data):
    env = make_vec_env(lambda: StockTradingEnv(data), n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/", device="cpu")  # Force CPU usage
    eval_callback = EvalCallback(env, best_model_save_path="./ppo_models/", log_path="./ppo_logs/", eval_freq=500)
    model.learn(total_timesteps=10000, callback=eval_callback)
    model.save("ppo_stock_trading_agent")
    return model


def use_trained_ppo_model(processed_live_data):
    """
    Use the trained PPO agent to make predictions on live data.
    """
    # Remove the signal (target) column from the live data
    processed_live_data.loc[:, 'Signal'] = (processed_live_data['Close'].shift(-1) > processed_live_data['Close']).astype(int)
    processed_live_data = processed_live_data.iloc[:, 1:]

    # Ensure all features are numerical and have the expected shape
    # print(processed_live_data.info())

    # Setup the trading environment with live data
    env = StockTradingEnv(processed_live_data)

    # Load the trained PPO model
    model = PPO.load("ppo_stock_trading_agent")

    # Run predictions
    obs = env.reset()
    actions = []
    for _ in range(len(processed_live_data)):  # Exclude the last step as it will be handled separately
        action, _ = model.predict(obs)  # Predict the action (Buy=0, Sell=1, Hold=2)
        actions.append(action)  # Append only the action, not the entire output
        obs, _, done, _ = env.step(action)
        # if done:
        #     break

    # Map actions to human-readable format
    # print(actions)
    action_mapping = {0: "Buy", 1: "Sell", 2: "Hold"}
    processed_live_data['Action'] = [action_mapping[int(a)] for a in actions]

    # Return the processed live data with actions
    return processed_live_data[['Close', 'Action']]


# --- 6. LLM Integration with Llama 3 ---
def explain_recommendation(data, action):
    company = ""
    # print("Explan", data['Company'])
    for key, value in company_mapping.items():
        if value == int(data['Company']):
            company = key

    prompt = f"Provide an explanation for recommending {action} for {company} on {data['Date']} based on the following data:\n{data}"
    prompt += f"\nGive the accurate values of the technical signals('SMA_20', 'EMA_20', 'Daily_Return', 'RSI') in your explanation."
    explanation = ""
    for event in replicate_client.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 0,
            "top_p": 0.9,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": 0.6,
            "system_prompt": "You are a financial expert providing insightful explanations.",
        },
    ):
        explanation += str(event)
    return explanation


# Example usage
# --- 6. Streamlit Application ---
st.title("Stock Recommendation System")

client = setup_weaviate()

# User input for date
user_date = st.date_input("Select Date", datetime(2025, 1, 1))

# Adjust date range
start_date, adjusted_end_date = adjust_date_range(user_date)

# Display information about the adjusted date range
st.write(f"Data will be fetched for the range: {start_date} to {adjusted_end_date}")
st.write(f"Original user-defined date: {user_date}")

# Tickers
tickers = st.text_input("Enter Ticker Symbols (comma-separated)", "INFY")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

# Fetch data and train model
if st.button("Fetch Data"):
    # Fetch data using the adjusted date range
    stock_data = pd.concat(
        [fetch_yahoo_data(ticker, user_date) for ticker in tickers], axis=0
    )
    if stock_data.empty:
        st.write("No data fetched. Please check ticker symbols or date range.")
    else:
        st.write("Fetched Data:", stock_data.head())

        # Preprocess and calculate indicators
        processed_data = preprocess_data(stock_data, True)
        st.write("Processed Data with Indicators:", processed_data)

        # store_data_in_weaviate(client, processed_data)

        # Store calculated indicators
        # indicators = processed_data[['Company', 'SMA_20', 'EMA_20', 'Daily_Return', 'RSI']]
        # st.write("Technical Indicators:", indicators.head())

        # Processed data for selected dates
        # Slice data based on user-provided date range
        user_processed_data = processed_data[
            (processed_data['Date'] >= pd.to_datetime(start_date)) &
            (processed_data['Date'] <= pd.to_datetime(adjusted_end_date))
            ]
        st.write("User Processed Data:", user_processed_data)

        print("USER", user_processed_data.iloc[0:1])

        rag_data = retrieve_rag_data(client, user_processed_data['Company'].iloc[0])
        print(rag_data)
        # Predict
        recommendation = use_trained_ppo_model(user_processed_data)

        st.write("Recommendation:", recommendation)

        explanation = explain_recommendation(user_processed_data.iloc[0:1], recommendation['Action'])
        st.write("Explanation:", explanation)
