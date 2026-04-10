## Fred Forecasting

This repository contains code for forecasting economic indicators using data from the Federal Reserve Economic Data (FRED) database. The code is written in Python and utilizes various libraries for data manipulation, visualization, and machine learning.

### Description

The main goal of this project is to develop a forecasting model that can predict future values of economic indicators based on historical data. The project includes data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

### Installation

To run the code, you will need to have Python installed on your system. You can install the required dependencies using pip. Make sure to navigate to the project directory and run the following command:

```txt
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

You will also need to obtain an API key from the FRED website to access the data. Once you have the API key, you can set it as an environment variable or directly in the code.

### Usage

To use the code, simply run the main script:

```bash
python main.py
```

This will execute the entire forecasting pipeline, including data loading, preprocessing, model training, and evaluation. You can modify the parameters and settings in the 'main.py' file to customize the forecasting process according to your needs. 'Series Id' must be a valid FRED series ID for the code to work properly.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more.
