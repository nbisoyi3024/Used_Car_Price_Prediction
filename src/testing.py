import lightgbm as lgb
import catboost as cb

#print("LightGBM version:", lgb.__version__)
#print("CatBoost version:", cb.__version__)

# Quick test: create a tiny dataset
#X = [[1, 2], [3, 4], [5, 6], [7, 8]]
#y = [10, 20, 30, 40]

# LightGBM model
#lgb_model = lgb.LGBMRegressor()
#lgb_model.fit(X, y)
#print("LightGBM prediction:", lgb_model.predict([[2, 3]]))

# CatBoost model
#cb_model = cb.CatBoostRegressor(verbose=0)
#cb_model.fit(X, y)
#print("CatBoost prediction:", cb_model.predict([[2, 3]]))