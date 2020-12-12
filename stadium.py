#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def main():
    def studium_latlon(studium_x):
        # スタジアムの緯度と経度
        return pd.Series([dict_studium_lat.get(studium_x), dict_studium_lon.get(studium_x)])


    def check_home_studium(df_x):
        # ホームチームがホームスタジアムで試合をしたか確認
        # ホームスタジアムでの試合なら０を返す
        if dict_stadium.get(df_x['stadium']) == df_x['home']:
            return 0
        # 横浜ＦＣがニッパツ三ツ沢球技場での試合なら０を返す
        elif df_x['stadium'] == 'ニッパツ三ツ沢球技場' and df_x['home'] == '横浜ＦＣ': 
            return 0
        # 東京ヴェルディが味の素スタジアムでの試合なら０を返す
        elif df_x['stadium'] == '味の素スタジアム' and df_x['home'] == '東京ヴェルディ': 
            return 0
        
        # ホームチームがホームスタジアムで試合したら０を返す
        else: 
            return 1

    #### 距離関連の特徴量作成
    def team_latlon(team_x):
        # チームの緯度と経度
        return pd.Series([dict_team_lat.get(team_x), dict_team_lon.get(team_x)])
    
    
    df_train = pd.read_csv('data/train.csv', encoding='utf-8')
    df_train_add = pd.read_csv('data/train_add.csv', encoding='utf-8')
    df_studium = pd.read_csv('data/stadium.csv', encoding='utf-8')
    df_condition = pd.read_csv('data/condition.csv', encoding='utf-8')
    df_condition_add = pd.read_csv('data/condition_add.csv', encoding='utf-8')

    # add_dataの統合
    df_train_concat = pd.concat([df_train, df_train_add])
    df_condition_concat =  pd.concat([df_condition, df_condition_add])

    #trainとconditionをmerge
    df = pd.merge(df_train_concat, df_condition_concat, on="id")

    #更にスタジアムをマージ
    df = pd.merge(df, df_studium, left_on="stadium", right_on='name')

    # ダブり変数、審判、選手、スコア削除
    df1 = df.loc[:, ['id', 'y', 'year', 'stage', 'match', 'gameday', 'time', 'home', 'away', 'stadium', 'tv', 'home_score', 'away_score', 'weather', 'temperature',
                     'humidity', 'capa']]

    #### カテゴリカル変数の数値化
    # ザスパの名前を統一
    df1['home'].replace(['ザスパ草津'], ['ザスパクサツ群馬'], inplace=True)
    df1['away'].replace(['ザスパ草津'], ['ザスパクサツ群馬'], inplace=True)

    # timeを時間だけにする
    df1['time'] = df1['time'].str[:2].astype('int64')

    # matchを節と日にわける。
    df1['match_sec'] = df1['match'].str[1:-4].astype('int64').astype('int64')
    df1['match_sec_day'] = df1['match'].str[-2:-1].astype('int64')

    # 月の特徴量
    df1['month'] = df1['gameday'].str[:2].astype('int64')

    # 曜日の特徴量
    df1['weekday'] =  df1['gameday'].str[6]

    # 祝日かどうか
    df1.loc[df1['gameday'].str.len() < 9, 'holiday'] = 0
    df1.loc[df1['gameday'].str.len() >= 9, 'holiday'] = 1

    # humidlyを数値に
    df1['humidity'] = df1['humidity'].str[:2].astype('int64')


    # チームの緯度経度
    df_team_latlon = pd.read_csv('data/team_latlon.csv', encoding='ANSI')
    dict_team_lat = dict(zip(df_team_latlon['team'], df_team_latlon['latitude']))
    dict_team_lon = dict(zip(df_team_latlon['team'], df_team_latlon['longitude']))
    df1[['home_latitude', 'home_longitude']] = df1['home'].apply(team_latlon)
    df1[['away_latitude', 'away_longitude']] = df1['away'].apply(team_latlon)

    # スタジアムの緯度経度
    df_stadium = pd.read_csv('data/stadium_sub.csv')
    dict_studium_lat = dict(zip(df_stadium['stadium'], df_stadium['latitude']))
    dict_studium_lon = dict(zip(df_stadium['stadium'], df_stadium['longitude']))
    df1[['studium_latitude', 'studium_longitude']] = df1['stadium'].apply(studium_latlon)

    # なぜか鹿児島県立鴨池陸上競技場だけ緯度経度がNanになるので補完
    df1['studium_latitude'].fillna((31.56483), inplace=True)
    df1['studium_longitude'].fillna((130.560144), inplace=True)

    # どのチームが所有するスタジアムか
    dict_stadium = dict(zip(df_stadium['stadium'], df_stadium['team1']))
    df1['away_game'] = df1.apply(check_home_studium, axis=1)

    # スタジアムとの距離
    distance_home = []
    distance_away = []

    for row in range(len(df1.index)):
        studium_lat_lon = (df1.loc[row, 'studium_latitude'], df1.loc[row, 'studium_longitude'])
        away_lat_lon = (df1.loc[row, 'away_latitude'], df1.loc[row, 'away_longitude'])
        dist_away = distance.euclidean(away_lat_lon, studium_lat_lon)
        distance_away.append(dist_away)
        
        if df1.loc[row,'away_game'] == 0:
            # haway_gameが0のときは、0
            # それ以外の時は、距離計算。
            distance_home.append(0)
        
        else:
            home_lat_lon = (df1.loc[row, 'home_latitude'], df1.loc[row, 'home_longitude'])
            dist_home = distance.euclidean(home_lat_lon, studium_lat_lon)
            distance_home.append(dist_home)
        
        
    df1['distance_home'] = pd.Series(distance_home)
    df1['distance_away'] = pd.Series(distance_away)

    # 各試合のhome勝ち点欄、away勝ち点欄挿入
    # 勝利3点、引き分け1点、敗戦0点

    home_points=[]
    away_points=[]


    for c in range(len(df1.index)):
        if df1.loc[c,'away_score'] > df1.loc[c,'home_score']:
            home_points.append(0)
            away_points.append(3)

        elif df1.loc[c,'away_score'] == df1.loc[c,'home_score']:
            home_points.append(1)
            away_points.append(1)

        else:
            home_points.append(3)
            away_points.append(0)
        
        
    df1['points_home'] = pd.Series(home_points)
    df1['points_away'] = pd.Series(away_points)

    # アウェイチームの直近5試合の勝ち点平均特徴量挿入

    point_5game_average_away=[                (
                # ホーム試合の勝ち点合計
                df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-1) & (df1['year'] == df1.loc[i,'year']) & (df1['home'] == df1.loc[c,'away']), 'points_home'].sum()\
                +df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-2) & (df1['year'] == df1.loc[i,'year']) & (df1['home'] == df1.loc[c,'away']), 'points_home'].sum()\
                +df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-3) & (df1['year'] == df1.loc[i,'year']) & (df1['home'] == df1.loc[c,'away']), 'points_home'].sum()\
                +df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-4) & (df1['year'] == df1.loc[i,'year']) & (df1['home'] == df1.loc[c,'away']), 'points_home'].sum()\
                +df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-5) & (df1['year'] == df1.loc[i,'year']) & (df1['home'] == df1.loc[c,'away']), 'points_home'].sum()\
                +\
                # アウェイ試合の勝ち点合計
                df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-1) & (df1['year'] == df1.loc[i,'year']) & (df1['away'] == df1.loc[c,'away']), 'points_away'].sum()\
                +df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-2) & (df1['year'] == df1.loc[i,'year']) & (df1['away'] == df1.loc[c,'away']), 'points_away'].sum()\
                +df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-3) & (df1['year'] == df1.loc[i,'year']) & (df1['away'] == df1.loc[c,'away']), 'points_away'].sum()\
                +df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-4) & (df1['year'] == df1.loc[i,'year']) & (df1['away'] == df1.loc[c,'away']), 'points_away'].sum()\
                +df1.loc[(df1['match_sec'] == df1.loc[i, 'match_sec']-5) & (df1['year'] == df1.loc[i,'year']) & (df1['away'] == df1.loc[c,'away']), 'points_away'].sum())\
                    /5\
                    for i in range(len(df1.index))]# 全部の行で処理

    df1['point_5game_average_away'] = pd.Series(point_5game_average_away)


    # 天候をグループ分け(df1_weather)

    df1_weather = df1.copy()
    df1_weather['weather'].replace(['晴のち曇'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴のち曇一時雨'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴のち曇時々雨'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴のち雨'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴一時曇'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴一時雨'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴時々曇'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴時々雨'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴時々曇'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴時々雨'], ['やや晴'], inplace=True)
    df1_weather['weather'].replace(['晴時々雪'], ['やや晴'], inplace=True)

    df1_weather['weather'].replace(['曇のち晴'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇のち雨'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇のち雪'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇のち雷雨'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇一時晴'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇一時晴一時雨'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇一時雨'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇一時雨のち晴'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇一時雷雨のち曇'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇時々晴'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇時々晴一時雨'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇時々雨'], ['やや曇'], inplace=True)
    df1_weather['weather'].replace(['曇時々雨のち晴'], ['やや曇'], inplace=True)

    df1_weather['weather'].replace(['雨のち晴'], ['やや雨'], inplace=True)
    df1_weather['weather'].replace(['雨のち曇'], ['やや雨'], inplace=True)
    df1_weather['weather'].replace(['雨のち曇時々晴'], ['やや雨'], inplace=True)
    df1_weather['weather'].replace(['雨時々晴'], ['やや雨'], inplace=True)
    df1_weather['weather'].replace(['雨時々曇'], ['やや雨'], inplace=True)

    df1_weather['weather'].replace(['雪のち雨'], ['やや雪'], inplace=True)

    df1_weather = df1_weather.loc[:, ["id", "weather"]]

    # スタジアムにおける観客数平均、満員率平均の２変数をクラスタリング(df_stadium_average) 

    df_stadium_average = df1.copy()
    df_stadium_average['Occupancy_rate'] = df_stadium_average['Occupancy_rate'] = df_stadium_average['y'] / df_stadium_average['capa']
    df_stadium_average = pd.pivot_table(df_stadium_average.query("year == 2012 or year == 2013"),
                                        index="stadium", 
                                        values=["y", 'Occupancy_rate'])

    #正規化
   

    #分散正規化のインスタンスを作成
    stdsc = StandardScaler()
    #分散正規化を実行
    stdsc.fit(df_stadium_average)
    stdsc.transform(df_stadium_average)

    max_cluster_num = 10
    data_array_class_stadium_average = stdsc.transform(df_stadium_average)

    #クラスタ分析を実行 (クラスタ数=3)
    n_clusters=3
    pred_stadium = KMeans(n_clusters=3, random_state=0).fit_predict(data_array_class_stadium_average)
    df_stadium_average["stadium_class"] = pred_stadium

    df_stadium_average = df_stadium_average.reset_index()

    #カラム変更
    df_stadium_average = df_stadium_average.loc[:, ["stadium", "stadium_class"]]

    #2012～2013のデータがない香川県立丸亀競技場を追加
    df_stadium_average.loc[58] = [ '香川県立丸亀競技場', "0"]
    df_stadium_average['stadium_class'] = df_stadium_average['stadium_class'].astype(np.int64)


    # スタジアム別平均観客動員数(df1_studium_ave)
    df1_studium_ave = df1.query("year == 2012 or year == 2013")
    df1_studium_ave = pd.pivot_table(df1_studium_ave, index="stadium", values="y")
    df1_studium_ave = df1_studium_ave.reset_index()
    df1_studium_ave = df1_studium_ave.rename({"y":"stadium_average"}, axis=1)
    df1_studium_ave.head()

    #財務関係の特徴量を挿入(
    df_finance = pd.read_csv('data/Jleague基礎データ.csv', encoding='ANSI')

    year2012 = df1.query('year == 2012')
    year2013 = df1.query('year == 2013')
    year2014 = df1.query('year == 2014')

    #前年度財務
    df_finance2011 = df_finance[['name', '2011営業収入', '2011入場料', '2011スポンサー料', '2011人件費', '2012home', '2012away']]
    df_finance2012 = df_finance[['name', '2012営業収入', '2012入場料', '2012スポンサー料', '2012人件費', '2013home', '2013away']]
    df_finance2013 = df_finance[['name', '2013営業収入', '2013入場料', '2013スポンサー料', '2013人件費', '2014home', '2014away']]

    #財務関係カラム名変更
    df_finance2011= df_finance2011.rename(columns={'2011営業収入':'last_year_income', 
                                                   '2011入場料':'last_year_entrancefee',
                                                   '2011スポンサー料':'last_year_Sponsorfee',
                                                   '2011人件費':'last_year_laborcost',
                                                   '2012home':'home_avarage',
                                                   '2012away':'away_avarage'})
    df_finance2012 =df_finance2012.rename(columns={'2012営業収入':'last_year_income',
                                                   '2012入場料':'last_year_entrancefee',
                                                   '2012スポンサー料':'last_year_Sponsorfee',
                                                   '2012人件費':'last_year_laborcost',
                                                   '2013home':'home_avarage',
                                                   '2013away':'away_avarage'})
    df_finance2013 =df_finance2013.rename(columns={'2013営業収入':'last_year_income',
                                                   '2013入場料':'last_year_entrancefee',
                                                   '2013スポンサー料':'last_year_Sponsorfee',
                                                   '2013人件費':'last_year_laborcost',
                                                   '2014home':'home_avarage',
                                                   '2014away':'away_avarage'})

    #3パターンの年度データとその前年度財務データをマージ
    year2012_finance = pd.merge(year2012, df_finance2011, left_on='home', right_on='name')
    year2012_finance  = pd.merge(year2012_finance, df_finance2011, left_on='away', right_on='name')

    year2013_finance = pd.merge(year2013,df_finance2012, left_on='home', right_on='name')
    year2013_finance = pd.merge(year2013_finance, df_finance2012, left_on='away', right_on='name')

    year2014_finance = pd.merge(year2014, df_finance2013, left_on='home', right_on='name')
    year2014_finance = pd.merge(year2014_finance, df_finance2013, left_on='away', right_on='name')

    #2012,2013,2014を縦に結合
    df_finance_add = pd.concat([year2012_finance, year2013_finance, year2014_finance], axis=0)
    #不要なカラム"name_x"と"name_y"を削除
    df_finance_add = df_finance_add.drop(['name_x', 'name_y'], axis=1)
    #カラム名変更
    df_finance_add = (df_finance_add.rename({"last_year_income_x":"last_year_income_home",
                                             "last_year_entrancefee_x":"last_year_entrancefee_home",
                                             "last_year_Sponsorfee_x":"last_year_Sponsorfee_home",
                                             "last_year_laborcost_x":"last_year_laborcost_home",
                                             "home_avarage_x":"hometeam_home_avarage",
                                             "away_avarage_x":"hometeam_away_avarage",
                                             "last_year_income_y":"last_year_income_away",
                                             "last_year_entrancefee_y":"last_year_entrancefee_away",
                                             "last_year_Sponsorfee_y":"last_year_Sponsorfee_away",
                                             "last_year_laborcost_y":"last_year_laborcost_away",
                                             "home_avarage_y":"awayteam_home_avarage",
                                             "away_avarage_y":"awayteam_away_avarage"},
                                            axis=1))
    df1 = df_finance_add

    #df1と天気とクラスタリングととスタジアム別平均観客動員数をマージして、新たなdf1_mergeとする
    df_stadium= pd.merge(df1, df1_studium_ave, on="stadium", how="outer")
    df_weather = pd.merge(df_stadium, df1_weather, on="id")
    df_stadium_class = pd.merge(df_weather, df_stadium_average, on="stadium")
    df1_merge = df_stadium_class

    #各特徴量を結合した上でdf1_mergeをLabelEncoder
   

    df1_labelencode = df1_merge.copy()

    for c in ['stage', 'home', 'away', 'stadium', 'weekday', "weather_y"]:
        le = LabelEncoder()
        le = le.fit(df1_labelencode[c])
        df1_labelencode[c] = le.transform(df1_labelencode[c])

    df1_labelencode.head(1)

    # 不要になった変数を省く。(特徴量選択)
    df2 = df1_labelencode.loc[:, ['id', 'y', 'year', 'home', 'stadium', 'capa', 'match_sec',
                                  'month', 'weather_y', 'holiday', 'distance_away', 'stadium_class', 
                                  'awayteam_away_avarage', 'hometeam_home_avarage', 'stadium_average',
                                  'point_5game_average_away']]

    # カラム名変更
    df2 = df2.rename({"weather_y":"weather"}, axis=1)

    # 満員率
    df2['Occupancy_rate'] = df2['y'] / df2['capa']

    calc_lightgbm.main(df2)


if __name__ == '__main__':
    main()
