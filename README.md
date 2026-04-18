# Algorithmic-Strategy-Development-on-Multi-Feature-Time-Series

### **4th Place Winning Submission for the Ebullient Securities Quant PS**

This repository contains the code for our 4th place winning submission at the **Inter IIT Tech Meet 14.0**, for the Quantitative Trading problem statement by **Ebullient Securities**.

The project's objective was to develop a profitable, automated trading strategy for two futures contracts (codenamed EBX and EBY). Our solution employs a **Proximal Policy Optimization (PPO)** agent, a state-of-the-art Reinforcement Learning algorithm, to learn optimal trading decisions. The agent is trained on a rich feature set of over 60 technical indicators derived from high-frequency tick data.

The system provides an end-to-end pipeline: from raw data processing and feature engineering to model training, evaluation, and integration with Ebullient's backtesting engine.

### Key Features
- **Data-Driven Strategy**: The PPO agent learns complex patterns from market data without hard-coded rules.
- **Rich Feature Set**: Utilizes over 60 technical indicators (RSI, MACD, Bollinger Bands, etc.) to form a comprehensive market view.
- **End-to-End Pipeline**: Automates data resampling, indicator calculation, training, and testing with simple commands.
- **Robust Backtesting**: Generates detailed performance reports, equity curves, drawdown charts, and per-trade signal files.
- **Parallelized Training**: Leverages `stable-baselines3` for efficient, parallel model training, with automatic GPU detection.

---

### A Note on Reproducibility
The strategy's performance depends on the random train/test split of trading days. Some runs may include outlier days with exceptionally high returns (e.g., day 87 in EBX or day 104 in EBY), leading to higher overall performance. Other splits might yield more moderate returns. To experiment with different splits, change the `SEED` value in the `PARAMS` dictionary.

## Quick Start

### Install Dependencies
```bash
pip install numpy pandas gymnasium stable-baselines3 torch tqdm matplotlib
```

### Prepare Data
Place your raw tick data CSV files in a folder (e.g., `EBX/`). The code will automatically read from the folder specified in `PARAMS['SOURCE_FOLDER']`.
```
EBX/
├── day1.csv
├── day2.csv
└── day3.csv
```

## Commands Explained

### Command 1: `python <Ticker>.py train`

**What Happens:**

1.  **Data Resampling** (2-3 mins)
    - Reads tick data from the `EBX/` folder.
    - Converts it to 2-minute OHLC candles.
    - Saves the resampled data to `EBX_2min/` (skips if this folder already exists).
    - Creates `train_days_EBX.txt` and `test_days_EBX.txt` by randomly selecting days for training and testing.

2.  **Indicator Calculation** (1 min)
    - Precomputes 60+ technical indicators for all training days.
    - Applies a 30-minute warmup window, discarding the first 30 minutes of data from each day to allow indicators to stabilize.

3.  **Model Training** (5-10 mins depending on CPU/GPU)
    - Launches parallel environments to accelerate training.
    - Trains a PPO model using the precomputed indicators.
    - Prints training progress with a `tqdm` bar, monitoring metrics like entropy loss, explained variance, and policy loss.
    - Automatically detects and utilizes a GPU if available.

4.  **Model Saving**
    - Saves the trained model: `Models_EBX/ppo_trading_model_EBX.zip`
    - Saves the normalization statistics: `Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl`
    - Generates training performance plots: `training_plots/EBX_training_metrics.png`
    - Generates a list of features used: `feature_info_EBX.txt`

---

### Command 2: `python <Ticker>.py test`

**What Happens:**

1.  **Model Loading**
    - Loads the trained model from `Models_EBX/ppo_trading_model_EBX.zip`.
    - Loads the corresponding normalization stats from `Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl`.
    - Verifies that both files exist before proceeding.

2.  **Per-Day Backtesting**
    - Iterates through each day in the `test_days_EBX.txt` file.
    - For each day:
        - Loads the 2-minute candle data.
        - Calculates indicators (with the 30-minute warmup).
        - The model makes decisions at each step, generating signals (BUY, SELL, EXIT).
        - Records every trade entry and exit with price and timestamp.
        - Calculates the daily Profit & Loss (P&L) in basis points (bps).

3.  **Output Generation**
    - Saves all generated signals to CSV files: `signals_EBX/day123.csv`
    - Creates price charts with trade markers: `test_trade_plots/EBX_day_1_day123.png`
    - Calculates the overall equity curve and drawdown across all test days.
    - Saves the equity plot: `test_results/EBX_equity_drawdown.png`
    - Writes a comprehensive performance report: `test_results/test_results_EBX.txt`
    - **Prints a detailed trade log to the console**, showing timestamps, prices, and positions.

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "VecNormalize file not found" | The `train` command was not run first. | Run `python <Ticker>.py train` to generate the model and normalization files. |
| All trades are losing | The model has overfit to the training data. | Train on more diverse data, adjust hyperparameters, or reduce training episodes. |
| 0 trades executed | The model learned that holding is always the safest action. | Increase the `TRADE_ENTRY_PENALTY` (e.g., from -5 to -2) to encourage trading, or check reward scaling. |

---

### Command 3: `python <Ticker>.py test 123`

**What Happens:**

1.  **Specific Day Filtering**
    - This command is a variation of the `test` command.
    - It searches for `day123` in the list of test files and runs the backtest **only on that specific day**.
    - This is extremely useful for debugging the model's behavior on a particular day without running the full backtest.

---

### Command 4: `python <Ticker>.py backtest_ebullient`

**What Happens:**

1.  **Backtest Execution**
    - Initializes the `BacktesterIIT` with a configuration file.
    - Runs Ebullient's provided market simulator.
    - It reads the signal files generated by the `test` command.
    - For each signal:
        - An `EXIT` signal closes the current position.
        - A `BUY` signal opens a long position (e.g., 100 shares).
        - A `SELL` signal opens a short position (e.g., 100 shares).
    - Prints the final backtest results from the simulator.

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| `"day(\d+)"` regex error | Signal filenames do not match the expected pattern. | Ensure signal files are named like `day1.csv`, `day2.csv` and not `day_1.csv` or other variations. |
| Config file error | The `config.json` file has a formatting issue. | Manually inspect the `config.json` file created in the root directory to ensure it is valid JSON. |

---

## Common Issues & Global Solutions

### Issue 1: "PARAMS mismatch between training and testing"
**Problem:** You changed stop-loss or other parameters for testing but didn't retrain the model.
**Explanation & Solution:**
- The model learns its exit logic based on the **training parameters** (`STOP_LOSS_TR`, `TRAIL_PCT_TR`).
- The **testing parameters** (`STOP_LOSS_TE`, `TRAIL_PCT_TE`) enforce hard rules during the backtest, which can be different.
- If you change testing params, the results will differ because the exit rules have changed, but the model's underlying "knowledge" has not. For the model to learn new behavior, you must change the training params and **retrain**.

### Issue 2: "Too many/too few trades"
**Problem:** The model's trading frequency doesn't align with expectations.
**Solutions:**
- **Too many trades:** The model is too aggressive. Increase the penalty for entering a trade.
  - *Action:* Change `TRADE_ENTRY_PENALTY` from `-5` to `-10` or `-15` and retrain.
- **Too few trades:** The model is too conservative. Decrease the penalty.
  - *Action:* Change `TRADE_ENTRY_PENALTY` from `-5` to `-2` or `0` and retrain.
- **Hitting stop-loss too often:** The stop-loss might be too tight for the market's volatility.
  - *Action:* Decrease `STOP_LOSS` from `-0.0004` to `-0.0002` (making it wider) and retrain.

---

## File Checklist After Running

```
# After 'python <Ticker>.py train'
✓ Models_EBX/ppo_trading_model_EBX.zip (2-5MB)
✓ Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl (100KB)
✓ feature_info_EBX.txt (list of features)
✓ train_days_EBX.txt (list of training day files)
✓ test_days_EBX.txt (list of testing day files)
✓ training_plots/EBX_training_metrics.png (learning curve)
✓ EBX_2min/ (folder with 2-min OHLC data)

# After 'python <Ticker>.py test'
✓ test_results/test_results_EBX.txt (performance report)
✓ test_results/EBX_equity_drawdown.png (equity curve chart)
✓ test_trade_plots/ (folder with per-day trade charts)
✓ signals_EBX/ (folder with signal CSVs for the backtester)
```
