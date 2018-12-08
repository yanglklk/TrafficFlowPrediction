"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from  sklearn.cluster  import  KMeans

def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag. lag=12 12*5=60
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    attr = 'Lane 1 Flow (Veh/5 Minutes)'
    df1 = pd.read_csv(train, encoding='utf-8',usecols=[1]).fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8',usecols=[1]).fillna(0)
    data1=df1.sort_values(by=attr,axis=0).values
    data2 = df2.sort_values(by=attr, axis=0).values
    split1=culster(data1)
    split2=culster(data2)

    # scaler = StandardScaler().fit(df1[attr].values)
    # scaler 缩放 ，fit 确定缩放max min， transform 进行具体缩放 。
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    split1=scaler.transform(split1.reshape(-1,1)).reshape(1,-1)[0]
    split2=scaler.transform(split2.reshape(-1,1)).reshape(1,-1)[0]
    print(split1, split2)
    train, test = [], []
    # 长为13的连续片段 重组为ndarry
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])

    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    # shuffle 打乱顺序 内部不会 只在最外层 train.shape=(7764,13)
    np.random.shuffle(train)
    # X_train （7764，12）前期数据 y_train （7764，）预测数据
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler,split1,split2
def loadData(path):
    data=pd.read_csv(path,encoding='utf-8',usecols=[1]).fillna(0)
    sort_data=data.sort_values(by='Lane 1 Flow (Veh/5 Minutes)',axis=0).values
    # label=culster(sort_data)

    return sort_data

def culster(data):
    kmean=KMeans(n_clusters=4).fit(data)
    labels=kmean.labels_

    data=data.reshape([1,-1])
    data_label=[]
    for i in range(len(labels)):
        item = [data[0,i],labels[i]]
        data_label.append(item)
    data_label=np.array(data_label)
    split=[]
    for i in range(len(data_label)):
        if len(split)<7:
            if data_label[i,1]!=data_label[i+1,1]:
                split.append(data_label[i+1,0])
    return np.array(split)




path1=r'train.csv'
path2=r'test.csv'
process_data(path1,path2,12)
