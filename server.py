import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup as bs
import pickle
import requests
import time
import lxml
from datetime import datetime, timedelta
from pytz import timezone
import yfinance as yf
import stockstats

import unicodedata
import json
from textblob import TextBlob
from urllib.parse import quote, parse_qs, urlparse, urlencode



from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.metrics import classification_report, accuracy_score


from bs4 import BeautifulSoup as bs
def get_news(symbol, time):
    USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:65.0) Gecko/20100101 Firefox/65.0"
    headers={"user-agent" : USER_AGENT}
    target='AAPL'
    keyword = quote(symbol.encode('utf8'))
    target_param = {
        "tbm": "nws",
        "hl": "en",
        "lr": "lang_en",
        "q": keyword,
        "oq": keyword,
        "dcr": "0",
        "source": "lnt",
        "num": 5,
        "tbs": "cdr:1,cd_min:"+time+",cd_max:"+time,
    }
    url = "https://www.google.com.tw/search?" + urlencode(target_param)
    res = requests.get(url, headers=headers)
    search_list = []
    if res.status_code == 200:
        content = res.content
        soup = bs(content, "html.parser")
        items = soup.findAll("div", {"class": "g"})
        if items:
            for index, item in enumerate(items):
                    # title
                    news_title = item.find("h3", {"class": "r"}).find("a").text
                    # url
                    href = item.find("h3", {"class": "r"}).find("a").get("href")
                    news_link = href


                    # content
                    news_text = item.find("div", {"class": "st"}).text

                    # source
                    news_source = item.find("h3", {"class": "r"}).findNext('div').text.split('-')
                    news_from = news_source[0]
                    time_created = str(news_source[1])

                    # add item into json object
                    search_list.append({
                        "news_title": news_title,
                        "news_link": news_link,
                        "news_text": news_text,
                        "news_from": news_from,
                        "time_created": time_created
                    })
    else:
        print('error at '+str(i))
    return search_list, url



app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
avail_symbols= ["AAPL"]
stock_date_start = "2018-12-30"
CORS(app)


@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/api/v0.1/get_stock_price',methods=['POST'])
def get_stock_price():
    data = request.get_json(force=True)
    symbol = data["symbol"]
    if(symbol in avail_symbols):
        stock_raw = yf.download(symbol, start=stock_date_start)
        return Response(stock_raw.transpose().to_json(), mimetype='application/json')
    else:
        error = {"warn": "Symbol not preprocessed"}
        return jsonify(error)

    
@app.route('/api/v0.1/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    symbol = data["symbol"]
    stock_date_start = "2020-03-01"
    stock_with_absolute = pd.read_pickle('./AAPL/data/stock_with_absolute.pkl')
    label_abs_1d = pd.read_pickle('./AAPL/data/label_abs_1d.pkl')
    X_train, X_test, y_train, y_test = train_test_split(stock_with_absolute, label_abs_1d, test_size=0.1, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    stock_raw = yf.download(symbol, start=stock_date_start)
    stock_raw_data = stock_raw
    stock_raw_data['high_low_diff'] = (stock_raw_data['High'] - stock_raw_data['Low'])
    stock_raw_data['open_close_diff'] = (stock_raw_data['Open'] - stock_raw_data['Close'])
    stock_raw_data['high_low_diff_ratio'] = (stock_raw_data['High'] - stock_raw_data['Low']) / stock_raw_data['Close']
    stock_raw_data['open_close_diff_ratio'] = (stock_raw_data['Open'] - stock_raw_data['Close']) / stock_raw_data['Close']
    stock_stats_data = stockstats.StockDataFrame.retype(stock_raw_data)
    stock_stats_data[['change', 'open_delta','close_delta','volume_delta', 'close_-2_r','close_-6_r', 'boll', 'boll_ub', 
                      'boll_lb', 'boll_-1_d', 'boll_ub_-1_d', 'boll_lb_-1_d' , 'kdjk','kdjd','kdjj', 'macd','macds',
                      'macdh', 'rsi_6', 'rsi_12', 'wr_6', 'wr_12', 'cci', 'atr', 'dma', 'vr']]
    stock_data = pd.DataFrame(stock_stats_data)
    stock_data = stock_data.dropna()
    stock_data['boll_k_diff'] = stock_data['boll'] - stock_data['close']

    stock_with_absolute = stock_data[['change', 'open_delta','close_delta','volume_delta', 'high_low_diff_ratio', 
                                    'open_close_diff_ratio','close_-2_r','close_-6_r','kdjk','kdjd','kdjj', 'macd',
                                    'macds', 'macdh', 'rsi_6', 'rsi_12', 'wr_6', 'wr_12', 'cci', 'atr', 'dma', 'vr', 
                                    'boll_-1_d','boll_ub_-1_d', 'boll_lb_-1_d', 'boll_k_diff', 'high_low_diff', 
                                    'open_close_diff', 'open', 'high', 'low', 'close', 'adj close', 'volume', ]]
    target = stock_with_absolute.tail(2)
    tz = timezone('EST')
    EST_Timezone = datetime.now(tz)
    if not EST_Timezone.isoweekday() in range(1, 6):
        target_row = target_row.tail(1)
    else:
        trading_start = datetime(EST_Timezone.year, EST_Timezone.month, EST_Timezone.day, 9, 30,tzinfo=tz)
        trading_end = datetime(EST_Timezone.year, EST_Timezone.month, EST_Timezone.day, 16, 0, tzinfo=tz)
        if(trading_start <= EST_Timezone <= trading_end):
            target = target.head(1)
        else:
            target = target.tail(1)

    news_list, google_url = get_news("AAPL", target.index[0].strftime("%m/%d/%Y"))
    num_news = len(news_list)
    sentiment = 0
    des_sentiment = 0
    for news in news_list:
        news_title = news['news_title'].replace('...', '')
        news_des = news['news_text'].encode("ascii", "ignore").decode("ascii").replace('...', '')
        blob = TextBlob(news_title)
        des_blob = TextBlob(news_des)
        sentiment += blob.sentiment.polarity
        des_sentiment += des_blob.sentiment.polarity
    target["news_title_score"] = sentiment / num_news
    target["news_des_score"] = des_sentiment / num_news
    
    SP500_raw = yf.download("^GSPC", start=stock_date_start)
    SP500_raw_data = SP500_raw
    SP500_raw_data['high_low_diff'] = (SP500_raw['High'] - SP500_raw['Low'])
    SP500_raw_data['open_close_diff'] = (SP500_raw['Open'] - SP500_raw['Close'])
    SP500_raw_data['high_low_diff_ratio'] = (SP500_raw['High'] - SP500_raw['Low']) / SP500_raw['Close']
    SP500_raw_data['open_close_diff_ratio'] = (SP500_raw['Open'] - SP500_raw['Close']) / SP500_raw['Close']
    SP500_stats_data = stockstats.StockDataFrame.retype(SP500_raw_data)
    SP500_stats_data[['change','close_delta','volume_delta', 'close_-2_r','close_-6_r']]

    SP500_stock = pd.DataFrame(SP500_stats_data).add_prefix('sp500_')
    target = target.join(SP500_stock.loc[target.index])

    gold_raw = yf.download("GLD", start=stock_date_start)
    gold_raw_data = gold_raw
    # Add high_low_diff, open_close_diff, high_low_diff_ratio, open_close_diff_ratio
    gold_raw_data['high_low_diff'] = (gold_raw['High'] - gold_raw['Low'])
    gold_raw_data['open_close_diff'] = (gold_raw['Open'] - gold_raw['Close'])
    gold_raw_data['high_low_diff_ratio'] = (gold_raw['High'] - gold_raw['Low']) / gold_raw['Close']
    gold_raw_data['open_close_diff_ratio'] = (gold_raw['Open'] - gold_raw['Close']) / gold_raw['Close']

    # Add financial indicators
    gold_stats_data = stockstats.StockDataFrame.retype(gold_raw_data)
    gold_stats_data[['change','close_delta','volume_delta', 'close_-2_r','close_-6_r']]

    gold_stock = pd.DataFrame(gold_stats_data).add_prefix('gold_')
    target = target.join(gold_stock.loc[target.index])

    # 5 year bonds
    y5bond_raw = yf.download("^FVX", start=stock_date_start)
    y5bond_raw_data = y5bond_raw
    y5bond_raw_data['high_low_diff'] = (y5bond_raw['High'] - y5bond_raw['Low'])
    y5bond_raw_data['open_close_diff'] = (y5bond_raw['Open'] - y5bond_raw['Close'])
    y5bond_raw_data['high_low_diff_ratio'] = (y5bond_raw['High'] - y5bond_raw['Low']) / y5bond_raw['Close']
    y5bond_raw_data['open_close_diff_ratio'] = (y5bond_raw['Open'] - y5bond_raw['Close']) / y5bond_raw['Close']
    y5bond_stats_data = stockstats.StockDataFrame.retype(y5bond_raw_data)
    y5bond_stats_data[['change','close_delta', 'close_-2_r','close_-6_r']] ### no volume

    y5bond_stock = pd.DataFrame(y5bond_stats_data).drop(columns='volume').add_prefix('y5bond_')

    # 10 year bonds
    y10bond_raw = yf.download("^TNX", start=stock_date_start)
    y10bond_raw_data = y10bond_raw
    y10bond_raw_data['high_low_diff'] = (y10bond_raw['High'] - y10bond_raw['Low'])
    y10bond_raw_data['open_close_diff'] = (y10bond_raw['Open'] - y10bond_raw['Close'])
    y10bond_raw_data['high_low_diff_ratio'] = (y10bond_raw['High'] - y10bond_raw['Low']) / y10bond_raw['Close']
    y10bond_raw_data['open_close_diff_ratio'] = (y10bond_raw['Open'] - y10bond_raw['Close']) / y10bond_raw['Close']
    y10bond_stats_data = stockstats.StockDataFrame.retype(y10bond_raw_data)
    y10bond_stats_data[['change','close_delta', 'close_-2_r','close_-6_r']] ### no volume

    y10bond_stock = pd.DataFrame(y10bond_stats_data).drop(columns='volume').add_prefix('y10bond_')
    target = target.join(y10bond_stock.loc[target.index])
    target = target.join(y5bond_stock.loc[target.index])
    return_dict = {}
    return_dict["Data"] = target.transpose().to_dict()
    target = scaler.transform(target)
    return_dict["news"] = news_list
    return_dict["news_url"] = google_url
    return_dict["LGR"] = [pickle.load(open('./AAPL/LogisticRegression/LR_1d.pkl','rb')).predict(target)[0],
                          pickle.load(open('./AAPL/LogisticRegression/LR_7d.pkl','rb')).predict(target)[0],
                          pickle.load(open('./AAPL/LogisticRegression/LR_30d.pkl','rb')).predict(target)[0]]
    return_dict["DT"] = [pickle.load(open('./AAPL/DecisionTree/DT_1d.pkl','rb')).predict(target)[0],
                          pickle.load(open('./AAPL/DecisionTree/DT_7d.pkl','rb')).predict(target)[0],
                          pickle.load(open('./AAPL/DecisionTree/DT_30d.pkl','rb')).predict(target)[0]]
    return_dict["RF"] = [pickle.load(open('./AAPL/RandomForest/RFC_1d.pkl','rb')).predict(target)[0],
                          pickle.load(open('./AAPL/RandomForest/RFC_7d.pkl','rb')).predict(target)[0],
                          pickle.load(open('./AAPL/RandomForest/RFC_30d.pkl','rb')).predict(target)[0]]
    return_dict["XGB"] = [pickle.load(open('./AAPL/XGBoost/XGB_1d.pkl','rb')).predict(target)[0],
                          pickle.load(open('./AAPL/XGBoost/XGB_7d.pkl','rb')).predict(target)[0],
                          pickle.load(open('./AAPL/XGBoost/XGB_30d.pkl','rb')).predict(target)[0]]
    return jsonify(return_dict)
    
if __name__ == "__main__":
    app.run(debug=True, port=5001)