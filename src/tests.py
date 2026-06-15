# tests/test_car_price.py

import pandas as pd
from src import preprocessing

def test_preprocesing():
    df = pd.DataFrame({
        "year": [2020, 2021],
        "mileage": [50000, 30000],
        "price": [20000, 25000]
    })

    X, y = preprocessing(df)

    assert len(X) == 2
    assert len(y) == 2