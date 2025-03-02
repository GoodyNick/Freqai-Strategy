#!/usr/bin/env bash

# Define variables
STRATEGY="ExampleLSTMStrategy_v2"
MODEL="PyTorchLSTMRegressor_v2"
CONFIG="./user_data/configs/config-torch-lstm_v2.json"
TIMERANGE="20240401-20240501"
PAIR="BTC/USDT:USDT"
HYPEROPTLOSS="SharpeHyperOptLoss"
SPACES="buy sell"

# Check input arguments
MODE="$1"
OPTION="$2"  # Captures the optional second argument (e.g., "clean")

# ✅ Optional Cleanup if "clean" argument is added
if [[ "$OPTION" == "clean" ]]; then
    echo "🧹 Cleaning old models and results..."
    rm -rf ./user_data/models/*
    rm -rf ./user_data/backtest_results/*
    rm -f ./user_data/ridge_model.pkl
    rm -f ./user_data/feature_importances.csv
    echo "✅ Cleanup complete!"
fi

# ✅ Backtesting
if [[ "$MODE" == "backtest" ]]; then
    echo "🔄 Running Backtest..."
    freqtrade backtesting -s "$STRATEGY" --freqaimodel "$MODEL" -c "$CONFIG" --timerange="$TIMERANGE" 2>&1 | tee ./user_data/backtest_results.txt

# ✅ Plotting
elif [[ "$MODE" == "plot" ]]; then
    echo "📊 Plotting DataFrame..."
    freqtrade plot-dataframe --strategy "$STRATEGY" --freqaimodel "$MODEL" --timerange="$TIMERANGE" --config "$CONFIG" --pair "$PAIR"

# ✅ Hyperopt
elif [[ "$MODE" == "hyperopt" ]]; then
    echo "🔍 Running Hyperopt..."
    freqtrade hyperopt -s "$STRATEGY" --freqaimodel "$MODEL" -c "$CONFIG" --timerange="$TIMERANGE" --hyperopt-loss "$HYPEROPTLOSS" --spaces "$SPACES"

# ✅ Invalid Argument Handling
else
    echo "❌ Invalid option! Use: backtest [clean], plot, or hyperopt"
fi
