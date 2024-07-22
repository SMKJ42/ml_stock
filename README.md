## Getting started

if you dont already have rustup installed, follow this link https://www.rust-lang.org/learn/get-started

First, run `cargo build` in the terminal.

I have reduced the dataset to only S&P500 companies for evaluators to reduce the file size.
If you would like a complete dataset, navigate to https://www.kaggle.com/datasets/paultimothymooney/stock-market-data,
download the data, replace the existing stock_market_data folder and unzip the folder.

run `cargo run` in the terminal to build a model.

The `ml_algo/data_config.json` folder has the configuration for dates that you want to select for the the training and validation steps.

The 'training' dates include data that will be read to create training and test data sets to discover the loss of the model in the provided date range.

The validation data is when the model is used to simulate it's performance in a 'real' trading environment.
