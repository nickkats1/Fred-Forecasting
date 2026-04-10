from fredapi import Fred
import pandas as pd
import requests

# --- Fetch data from FredAPI ---

def fetch_data(series_id: str) -> pd.DataFrame:
    """Fetch data from Fred using API and return dataframe.

    Args:
        series_id: the name of the series from Fred.

    Returns:
        dataframe: A dataframe consisting of clean data from FRED.

    Raises:
        ValueError:
         - Raised if incorrect Series ID is entered.
        requests.exceptions.RequestException:
        - Raised if a network or HTTP error occurs.
    """
    try:
        fred = Fred()
        
        series = fred.get_series(series_id)
        series.name = series_id
        
        dataframe = pd.DataFrame(series)
        
        dataframe.reset_index(inplace=True)
        dataframe['Date'] = dataframe['index']
        dataframe.drop("index", inplace=True, axis=1)
        dataframe = dataframe.dropna()
        dataframe.drop_duplicates(inplace=True)
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        return dataframe
    except (ValueError, requests.exceptions.RequestException) as e:
        print(f"Incorrect Series ID entered or Invalid Api Key: {e}")
        raise

    


