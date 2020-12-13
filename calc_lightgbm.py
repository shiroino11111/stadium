# coding: utf-8
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgbm


def main(df1_labelencode):
    #各特徴量を結合した上でdf1_mergeをLabelEncoder

    for c in ['stage', 'home', 'away', 'stadium', 'weekday', "weather_y"]:
        le = LabelEncoder()
        le = le.fit(df1_labelencode[c])
        df1_labelencode[c] = le.transform(df1_labelencode[c])

    # 不要になった変数を省く。(特徴量選択)
    df2 = df1_labelencode.loc[:, ['id', 'y', 'year', 'home', 'stadium', 'capa', 'match_sec',
                                  'month', 'weather_y', 'holiday', 'distance_away', 'stadium_class', 
                                  'awayteam_away_avarage', 'hometeam_home_avarage', 'stadium_average',
                                  'point_5game_average_away']]

    # カラム名変更
    df2 = df2.rename({"weather_y":"weather"}, axis=1)

    # 満員率
    df2['Occupancy_rate'] = df2['y'] / df2['capa']    
    
    # 訓練データ(～14年まで)とテストデータ(14年～)にわける。
    #X_train, X_test, y_train, y_test にわける
    X_train = df2.loc[df2['year'] != 2014].drop(['id', 'y', 'Occupancy_rate'], axis=1)
    X_test = df2.loc[df2['year'] == 2014].drop(['id', 'y', 'Occupancy_rate'], axis=1)
    y_train = df2['Occupancy_rate'].loc[df2['year'] != 2014]
    y_test = df2['Occupancy_rate'].loc[df2['year'] == 2014]

    # 訓練データとテストデータにわける。
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

    print(X_train.shape)
    print(X_val.shape)

    target_Yname = "number of spector"

    # LightGBM用の変数にデータセット
    lgbm_train = lgbm.Dataset(X_train, y_train)
    lgbm_val = lgbm.Dataset(X_val, y_val, reference=lgbm_train)

    my_params  = {
            'objective': 'regression',
            'learning_rate': 0.1, # 学習率
            'max_depth': -1, # 木の数 (負の値で無制限)
            'num_leaves': 9, # 枝葉の数
            'metric': ('mean_absolute_error', 'mean_squared_error', 'rmse'),
            #メトリック https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters
            'drop_rate': 0.15,
            'verbose': 0
    }

    evaluation_results  = {}
    # lgbm.trainが、scikit-learnのmodel.fitに相当する
    regr = lgbm.train(my_params,
                          lgbm_train,
                          num_boost_round = 500, # 最大試行数
                          early_stopping_rounds=15, # この数分、連続でメトリックに変化なければ終了する
                          valid_sets = [lgbm_train, lgbm_val],
                          valid_names=['train', 'test'],
                          evals_result = evaluation_results,
                          verbose_eval = 4)

    # テストデータで回帰予測して、RMSE を出力
    y_pred = regr.predict(X_test, num_iteration=regr.best_iteration)
    mse = mean_squared_error(y_test*X_test['capa'], y_pred*X_test['capa'])
    rmse = np.sqrt(mse)
    print("rmse : {}".format(rmse))

