import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.title("重回帰分析")


st.sidebar.markdown("### csvファイルの入力")
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files= False)


if uploaded_files:
    df = pd.read_csv(uploaded_files)
    df_columns = df.columns
    #データフレームを表示

    st.markdown("### 入力データ")
    st.dataframe(df.style.highlight_max(axis=0))

    #matplotlibで可視化。X軸,Y軸を選択できる
    st.markdown("### 可視化 単変量")


    #データフレームのカラムを選択オプションに設定する
    x = st.selectbox("X軸", df_columns)
    y = st.selectbox("Y軸", df_columns)

    #選択した変数を用いてmtplotlibで可視化
    fig = plt.figure(figsize= (12,8))
    plt.scatter(df[x],df[y])
    plt.xlabel(x,fontsize=18)
    plt.ylabel(y,fontsize=18)
    st.pyplot(fig)

    st.markdown("### ペアプロット(数値データのみ）")
    item = st.multiselect("可視化するカラム", df_columns) 
    

    execute_pairplot = st.button("ペアプロット描画")

    if execute_pairplot:
        df_sns = df[item]
       
        fig = sns.pairplot(df_sns)
        st.pyplot(fig)

    st.markdown("### モデリング（数値データのみ）")
    ex = st.multiselect("説明変数を選択してください（複数選択可）", df_columns)

    ob = st.selectbox("目的変数を選択してください", df_columns)

    st.markdown("#### 回帰分析結果")
    execute = st.button("実行")

    lr = linear_model.LinearRegression()
    if execute:

        
        df_ex = df[ex]
        df_ob = df[ob]
        X_train, X_test, y_train, y_test = train_test_split(df_ex.values, df_ob.values, test_size = 0.3)
        lr.fit(X_train, y_train)

    #作業中感 
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1)

        col1, col2 = st.columns(2)
        col1.metric(label="トレーニングスコア", value=lr.score(X_train, y_train))
        col2.metric(label="テストスコア", value=lr.score(X_test, y_test))

