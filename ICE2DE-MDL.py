# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:54:19 2023

@author: Duygu
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:53:49 2023

@author: Duygu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:34:16 2023

@author: Duygu
"""
# =============================================================================
# 



# =============================================================================

import os
import pandas as pd
from PyEMD import CEEMDAN, EEMD
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from hyperopt import fmin, tpe, hp
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pyhht.emd import EMD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from numpy import sqrt 
import keras
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.gridspec as gridspec
import investpy as inv
import matlab.engine



def ApEn(U, m, r) -> float:
    """Approximate_entropy."""
    "U: seri, m: embedding dimension, r: tolerance"

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))

def SampEn(L, m, r):
    """Sample entropy."""
    N = len(L)
    B = 0.0
    A = 0.0
    #Epsilon was used to avoid division by zero error
    EPSILON = 1e-12
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B + EPSILON)

def calculate_entropy_StdSapma(imf,m,r):
    approx = ApEn(imf,m,r)
    sample = SampEn(imf,m,r)
    std_sapma = np.std(imf)
    return round(approx,5), round(sample,5), round(std_sapma,5)

def calculate_entropy(imfs):
    #method = 'CEEMDAN' # Choose one of 'EMD', 'EEMD', or 'CEEMDAN'
    imf_entropy_StdDeviation = np.zeros((imfs.shape[0], 3))
    for i in range(imfs.shape[0]):
        imf_entropy_StdDeviation[i] = calculate_entropy_StdSapma(imfs[i],m,r)
    
    Entropy_StdDeviation = pd.DataFrame(imf_entropy_StdDeviation, columns = ['Approximate Entropy', 'Sample Entropy', 'Standard Deviation'])
    # Entropy_StdDeviation = Entropy_StdDeviation.fillna(method = 'ffill')
    return Entropy_StdDeviation
    # Ratio_entropy = column_ratio(imf_entropy_StdDeviation) # İlk kolon: ApEn, İkinci kolon SampEn

def selection_imfs(imfs, Ratio_entropy):
#Among the IMFs obtained in the first decomposition, IMFs with ApEn_Ratio and SampEn_Ratio > 20 (20%) are considered as high frequency.
#and undergoes second parsing -> high_frequency_IMFs. The rest are assigned to the selected_IMFs_first array.

    selected_IMFs_first = []
    high_frequency_IMFs = []
    for i in range(len(imfs)):
        # if (((Ratio_entropy.iloc[i, 0] + Ratio_entropy.iloc[i, 1] )/2) < 20):
        if ((Ratio_entropy.iloc[i, 0] < 20) and (Ratio_entropy.iloc[i, 1] < 20)):
        # if ((Ratio_entropy.iloc[i, 1] < 20)):
            selected_IMFs_first.append(imfs[i])
        else:
            high_frequency_IMFs.append(imfs[i])

    #Collecting and plotting high frequency determined IMFs
    high_frequency_IMF_sum=0
    for num in high_frequency_IMFs:
        high_frequency_IMF_sum += num
    return selected_IMFs_first, high_frequency_IMFs, high_frequency_IMF_sum

def MAPE(Y_actual,Y_Predicted):
    epsilon = 1e-10

    mape = np.mean(np.abs((Y_actual - Y_Predicted)/(Y_actual+epsilon)))*100
    return mape

def repeat_list(n, x):
    return [x] * n

def relative_root_mean_squared_error(true, pred):
    n = len(true) # update
    num = np.sum(np.square(true - pred)) / n  # update
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

def rrmse(actual, predicted):
    #errors = actual - predicted
    rmse = np.sqrt(np.mean(np.square((actual - predicted)/predicted)))
    #mse = np.mean(errors**2)
    #pred_sqr = predicted**2
    #rrmse = np.sqrt(mse/pred_sqr)
    return rrmse

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) 

    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    percentage_errors = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    return np.mean(percentage_errors) * 100


# folder_path = 'E:\TİK2-ICEEMDAN-LSTM-LSTMBATCH-GRU-KOD VE SONUCLAR\XGBoost-Multivariate-Karşılaştırma\CAC40-Scale-15122023'
folder_path = 'E:\ICEEMDAN-Makale-Sonuc-26122023\\SP500'
symbol = "^GSPC" 
# symbol = "^FCHI"#CAC40
#symbol = "^KS200" #KOSPI200 ENDEKSİ
# symbol = "^KS11" #KOSPI endeksi
# symbol = "^SET.BK" #SET Index
# symbol = "000001.SS" #SSE Index
# symbol = "^GDAXI"
# symbol = '^NDX' #NASDAQ100
# symbol = "^DJI"
# symbol = "^N225" #Nikkei225
# symbol ="601857.SS" #PetroChina Company Limited
# symbol = "600030.SS" #CITIC Securities Company Limited
#symbol = "002168.SZ" # Shenzhen Hifuture Information Technology Co., Ltd.
# symbol = "300035.SZ" #Hunan Zhongke Electric Co., Ltd.
# symbol = "300013.SZ" #Henan Xinning Modern Logistics
# symbol = "002424.SZ" #Guizhou Bailing
# symbol = '^FTSE'
# symbol = '^NDX'
# start_date = "2014-02-18"
# end_date = "2021-01-19"

start_date = "2010-01-01"
end_date = "2020-01-01"

# start_date = "1990-12-03"
# end_date = "2023-05-25"

df = yf.download(symbol, start=start_date, end=end_date).reset_index()
df = df.fillna(method='bfill')

x = df['Close'].values

#WE READ CSI500 DATA FROM CSV FILE
# df = pd.read_csv('E:\ICEEMDAN-Makale-Sonuc-26122023\CSI500_.csv',sep = ';', parse_dates = True)
# df = df.fillna(method='bfill')
# x = df['Close'].values



eng = matlab.engine.start_matlab()
matlab_path = "E:/TİK2-ICEEMDAN-LSTM-LSTMBATCH-GRU-KOD VE SONUCLAR"
eng.addpath (matlab_path, nargout= 0 )
eng.emd_(nargout=0)
#eng.ceemdan_(nargout=0) # Expects a file named myMatlabFile.m in the same directory
eng.iceemdan(nargout=0)
a,b = eng.iceemdan_imfs(x, nargout=2)
#eng.cizim_ceemdan_(nargout=0)
#a, b = eng.cizim_ceemdan_(nargout=0)
eng.quit()

iceemdan_imfs = pd.DataFrame(a)

scaler= StandardScaler()
# scaler= MinMaxScaler(feature_range=(0,1))
train_size = int(len(df) * 0.75)
test_size = len(df) - train_size

m= 2
r=0.2


Entropy_StdDeviation_first_close = calculate_entropy(np.array(iceemdan_imfs))
Entropy_StdDeviation_first_close = Entropy_StdDeviation_first_close.fillna(method='ffill') #If there is a nan value, it will fill it with the value above it.
col_sum = np.sum(Entropy_StdDeviation_first_close, axis=0)
ratios = np.divide(Entropy_StdDeviation_first_close, col_sum)
Ratio_entropy = ratios*100
Ratio_entropy_first_close = Ratio_entropy.iloc[:,:2]

selected_IMFs_first_close, high_frequency_IMFs_first_close, high_frequency_IMF_sum_first_close = selection_imfs(np.array(iceemdan_imfs), Ratio_entropy_first_close)
# =============================================================================
#PLOTING THE FIRST DECOMPOSITION RESULTS
plt.figure()

iceemdan_imfs.index = [f"IMF{i+1}" for i in range(len(iceemdan_imfs))]


colors = plt.cm.viridis(np.linspace(0, 1, len(iceemdan_imfs)+1))


fig, axes = plt.subplots(nrows=len(iceemdan_imfs)+1, ncols=1, figsize=(8, 9), sharex=True)
# Plot the original data in the first subplot
axes[0].plot(df['Close'].values)
# axs[0].plot(df)
axes[0].set_title('Original Data')

for i, (row_label, row) in enumerate(iceemdan_imfs.iterrows()):
    axes[i+1].plot(row, label=row_label, color=colors[i+1])
    #axes[i].legend()
    
    
    axes[i+1].text(-0.07, 0.5, row_label, rotation=90, verticalalignment='center', horizontalalignment='right', transform=axes[i+1].transAxes)


plt.xlabel('Trading Day')
#plt.ylabel('Y Ekseni')
plt.suptitle('First Decomposition Results', x=0.54)


plt.tight_layout()


plt.show()
# 

file_name1 = 'first_decomposition_IMFs.png'
full_path1 = os.path.join(folder_path, file_name1)
fig.savefig(full_path1)
# =============================================================================
eng_2 = matlab.engine.start_matlab()
eng_2.addpath (matlab_path, nargout= 0 )
#imfs_vmd = eng_2.vmd(high_frequency_IMF_sum_first_close, nargout=1)
#imfs_second_close = pd.DataFrame(imfs_vmd)
#imfs_second_close = np.array(imfs_second_close.T)
f,g = eng_2.iceemdan_imfs(high_frequency_IMF_sum_first_close, nargout=2)
imfs_iceemdan_second = pd.DataFrame(f)
imfs_second_close = np.array (imfs_iceemdan_second)
eng_2.quit()


Entropy_StdDeviation_second_close = calculate_entropy(imfs_second_close)
col_sum_second = np.sum(Entropy_StdDeviation_second_close, axis=0)
ratios_second = np.divide(Entropy_StdDeviation_second_close, col_sum_second)
Ratio_entropy_second = ratios_second*100
Ratio_entropy_second_close = Ratio_entropy_second.iloc[:,:2]
selected_IMFs_second_close, high_frequency_IMFs_second_close, high_frequency_IMF_sum_second_close = selection_imfs(imfs_second_close, Ratio_entropy_second_close)
# =============================================================================
# PLOTING THE SECOND DECOMPOSITION RESULTS

plt.figure()

imfs_iceemdan_second.index = [f"IMF{i+1}" for i in range(len(imfs_iceemdan_second))]


colors = plt.cm.viridis(np.linspace(0, 1, len(imfs_iceemdan_second)+1))


fig, axes = plt.subplots(nrows=len(imfs_iceemdan_second)+1, ncols=1, figsize=(8, 9), sharex=True)
# Plot the original data in the first subplot
axes[0].plot(high_frequency_IMF_sum_first_close)
# axs[0].plot(df)
axes[0].set_title('High Frequency Data')

for i, (row_label, row) in enumerate(imfs_iceemdan_second.iterrows()):
    axes[i+1].plot(row, label=row_label, color=colors[i+1])
    #axes[i].legend()
    
    
    axes[i+1].text(-0.08, 0.5, row_label, rotation=90, verticalalignment='center', horizontalalignment='right', transform=axes[i+1].transAxes)


plt.xlabel('Trading Day')
#plt.ylabel('Y Ekseni')
plt.suptitle('Second Decomposition Results', x=0.54)


plt.tight_layout()

# Grafiği gösterelim
plt.show()
# 

file_name2 = 'second_decomposition_high_frequency_data.png'
full_path2 = os.path.join(folder_path, file_name2)
fig.savefig(full_path2)
# =============================================================================
#Ploting the IMFs selected as a result of second decomposition

plt.figure()
fig, axs = plt.subplots(nrows=len(selected_IMFs_second_close), figsize=(8, 8), sharex=True)
fig.subplots_adjust(hspace=0.2)
fig.suptitle("Selected IMFs at The 2nd Decomposition")
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_IMFs_second_close)))

# Plot the IMFs in subsequent subplots
for i in range(len(selected_IMFs_second_close)):
    axs[i].plot(selected_IMFs_second_close[i], color=colors[i])
    axs[i].set_ylabel(f'IMF {i+1}', rotation=90, ha='right', va='center')  # Yatay hizayı ayarla
    axs[i].yaxis.set_label_coords(-0.1, 0.8)  # Etiketin konumunu ayarla

fig.text(0.5, 0.04, 'Trading Day', ha='center')

for ax in axs.flat:
    ax.set_xlabel('')
    
# Show the plot
plt.show()



file_name3 = 'selected_IMFs_second_decomp.png'
full_path3 = os.path.join(folder_path, file_name3)
fig.savefig(full_path3)
# =============================================================================
# Plotting the actual data with the data determined noiseless as a result of the 1st and 2nd Decomposition

selected_all_sum_imfs = sum(selected_IMFs_first_close) + sum(selected_IMFs_second_close)
denoised_test_data_hiyerarşik = selected_all_sum_imfs[train_size:len(df)]

plt.figure()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'].values, label='Original Data')
plt.plot(selected_all_sum_imfs, label='Denoised Data')
plt.xlabel('Trading Day')
plt.legend()
plt.show()

file_name4 = 'actual_and_denoised_data_hiyerarşik_duygu.png'
full_path4 = os.path.join(folder_path, file_name4)
fig.savefig(full_path4)

# =============================================================================
def train_evaluate_LSTM_Batch(X_train, y_train, X_test, y_test, y_test_orj, timestep):
    print("LSTM Batch training...")
    LSTM_Batch_model = Sequential()
    LSTM_Batch_model.add(LSTM(128, activation='relu', return_sequences=True, input_shape = (timestep,1)))
    LSTM_Batch_model.add(BatchNormalization())
    LSTM_Batch_model.add(LSTM(32, activation='relu', input_shape = (timestep,1)))
    LSTM_Batch_model.add(BatchNormalization())
    LSTM_Batch_model.add(Dropout(0.1))    
    LSTM_Batch_model.add(Dense(1))
    LSTM_Batch_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode ='min', min_delta = 0.0001)
    LSTM_Batch_model.fit(X_train, y_train,epochs=200,batch_size=16,validation_data=(X_test, y_test), callbacks=[early_stop])
    
    testpredict = LSTM_Batch_model.predict(X_test)
    testPredict_descaled = scaler.inverse_transform(testpredict)
    
    rmse = sqrt(mean_squared_error(y_test, testpredict))
    mse = mean_squared_error(y_test, testpredict)
    mae = mean_absolute_error(y_test, testpredict)
    mape = MAPE(y_test, testpredict)
    mape_yuzsuz = mape/100
    r2 = r2_score(y_test, testpredict)
    rrmse_ = relative_root_mean_squared_error(y_test, testpredict)
    mape_ = mean_absolute_percentage_error(y_test, testpredict)
    smape_ = smape(y_test, testpredict)
    
    metrics = {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R2': r2, 'RRMSE': rrmse_,'SMAPE': smape_ ,"MAPE_": mape_}
    return testPredict_descaled, metrics

def train_evaluate_LSTM(X_train, y_train, X_test, y_test, y_test_orj, timestep):
    print("LSTM training...")
    LSTM_model= Sequential()
    LSTM_model.add(LSTM(units=128,return_sequences=True, activation='relu', input_shape=(timestep, 1)))
    LSTM_model.add(Dropout(rate=0.1))
    LSTM_model.add(LSTM(units=64,return_sequences=True, activation='relu'))
    LSTM_model.add(Dropout(rate=0.1))
    LSTM_model.add(LSTM(units=16))
    LSTM_model.add(Dropout(rate=0.1))
    LSTM_model.add(Dense(units=1))
    LSTM_model.compile(optimizer='Adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode ='min', min_delta = 0.0001)
    LSTM_model.fit(X_train, y_train,epochs=200,batch_size=16,validation_data=(X_test, y_test), callbacks=[early_stop])

    testpredict = LSTM_model.predict(X_test)
    testPredict_descaled = scaler.inverse_transform(testpredict)
    
    rmse = sqrt(mean_squared_error(y_test, testpredict))
    mse = mean_squared_error(y_test, testpredict)
    mae = mean_absolute_error(y_test, testpredict)
    mape = MAPE(y_test, testpredict)
    mape_yuzsuz = mape/100
    r2 = r2_score(y_test, testpredict)
    rrmse_ = relative_root_mean_squared_error(y_test, testpredict)
    mape_ = mean_absolute_percentage_error(y_test, testpredict)
    smape_ = smape(y_test, testpredict)

    
    metrics = {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R2': r2, 'RRMSE': rrmse_, 'SMAPE':smape_ ,"MAPE_": mape_}
    return testPredict_descaled, metrics

def train_evaluate_GRU(X_train, y_train, X_test, y_test, y_test_orj, timestep):
    print("GRU training...")

    GRU_model = Sequential()
    GRU_model.add(GRU(units=128, return_sequences=True, input_shape=(timestep, 1), activation='tanh'))
    GRU_model.add(Dropout(0.2))
    GRU_model.add(GRU(units=64, return_sequences=True, input_shape=(timestep, 1), activation='tanh'))
    GRU_model.add(Dropout(0.2))
    GRU_model.add(GRU(units=32, return_sequences=True, input_shape=(timestep, 1), activation='tanh'))
    GRU_model.add(Dropout(0.2))
    GRU_model.add(GRU(units=8, activation='tanh'))
    GRU_model.add(Dropout(0.2))
    GRU_model.add(Dense(units=1))
    GRU_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode ='min', min_delta = 0.0001)
    GRU_model.fit(X_train, y_train,epochs=200,batch_size=16,validation_data=(X_test, y_test), callbacks=[early_stop])
    
    testpredict = GRU_model.predict(X_test)
    testPredict_descaled = scaler.inverse_transform(testpredict)
    
    rmse = sqrt(mean_squared_error(y_test, testpredict))
    mse = mean_squared_error(y_test, testpredict)
    mae = mean_absolute_error(y_test, testpredict)
    mape = MAPE(y_test, testpredict)
    mape_yuzsuz = mape/100
    r2 = r2_score(y_test, testpredict)
    rrmse_ = relative_root_mean_squared_error(y_test, testpredict)
    mape_ = mean_absolute_percentage_error(y_test, testpredict)
    smape_ = smape(y_test, testpredict)

    
    metrics = {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R2': r2, 'RRMSE': rrmse_, 'SMAPE': smape_ ,"MAPE_": mape_}
    return testPredict_descaled, metrics
    
def train_evaluate_SVR(X_train, y_train, X_test, y_test, y_test_orj, timestep):  
    print("SVR training...")
    
    svr = SVR(kernel = 'rbf')
    param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'epsilon': [0.1, 0.01, 0.001], 'kernel':['rbf', 'linear']}
    tscv = TimeSeriesSplit(n_splits=5)
    grid_model = GridSearchCV(estimator=svr, param_grid=param_grid, cv=tscv, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error', error_score='raise')
    grid_model.fit(X_train.reshape(-1, timestep), y_train)
    print('Best hyperparameters:', grid_model.best_params_)

    best_C = grid_model.best_params_['C']
    best_epsilon = grid_model.best_params_['epsilon']
    best_kernel = grid_model.best_params_['kernel']
    best_gamma = grid_model.best_params_['gamma']

    svr = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel, gamma = best_gamma)
    svr.fit(X_train.reshape(-1, timestep), y_train)
    testpredict = svr.predict(X_test.reshape(-1, timestep))
    testPredict_descaled = scaler.inverse_transform(testpredict.reshape(-1,1))
    
    rmse = sqrt(mean_squared_error(y_test, testpredict))
    mse = mean_squared_error(y_test, testpredict)
    mae = mean_absolute_error(y_test, testpredict)
    mape = MAPE(y_test, testpredict)
    mape_yuzsuz = mape/100
    r2 = r2_score(y_test, testpredict)
    rrmse_ = relative_root_mean_squared_error(y_test, testpredict)
    mape_ = mean_absolute_percentage_error(y_test, testpredict)
    smape_ = smape(y_test, testpredict)

    
    metrics = {'RMSE': rmse, 'MSE': mse, 'MAE': mae, 'R2': r2, 'RRMSE': rrmse_, 'SMAPE':smape_, "MAPE_": mape_}
    return testPredict_descaled, metrics

def find_best_model (X_train, y_train, X_test, y_test, y_test_orj, timestep):
    lstm_batch_predictions, lstm_batch_metrics = train_evaluate_LSTM_Batch(X_train, y_train, X_test, y_test, y_test_orj, timestep)
    lstm_predictions, lstm_metrics = train_evaluate_LSTM(X_train, y_train, X_test, y_test, y_test_orj, timestep)
    gru_predictions, gru_metrics = train_evaluate_GRU(X_train, y_train, X_test, y_test, y_test_orj, timestep)
    #svr_predictions, svr_metrics = train_evaluate_SVR(X_train, y_train, X_test, y_test, y_test_orj, timestep)
    
    models = {#'SVR': {'predictions': svr_predictions, 'metrics': svr_metrics},
              'LSTM': {'predictions': lstm_predictions, 'metrics': lstm_metrics},
              'GRU': {'predictions': gru_predictions, 'metrics': gru_metrics},
              'LSTM-Batch': {'predictions': lstm_batch_predictions, 'metrics': lstm_batch_metrics}}
    
    model_metrics_list = []
    for model_name, model_info in models.items():
        model_metrics = {f'{model_name}_Predictions': model_info['predictions'],
                         f'{model_name}_Metrics': model_info['metrics']}
        model_metrics_list.append(model_metrics)
    
  

    best_model_name = min(models, key=lambda k: models[k]['metrics']['RMSE'])
    

    return best_model_name, models[best_model_name]['predictions'], models[best_model_name]['metrics'], model_metrics_list
    

    
def each_IMFs_train (imfs_set, timestep):
    metrics_list = []
    model_metrics_list = []
    train_IMFs = imfs_set # TÜM SEÇİLEN IMFLERLE EĞİTİM YAPILSIN AMA SONUÇ KISMINDA BU ELE ALINMIYOR. ÇÜNKÜ SON IMF İÇİN SVR ELE ALINIYOR
    #Train each IMF component separately
    for i in range(len(train_IMFs)):
        print("IMF", i+1, "training...")
        dataset = train_IMFs[i]
        dataset_scaled = scaler.fit_transform(dataset.reshape(-1, 1))

        X_train, y_train, X_test, y_test, y_test_orj = [], [], [], [], []
        for j in range(timestep, train_size):
            X_train.append(dataset_scaled[j-timestep:j, 0])
            y_train.append(dataset_scaled[j, 0])
        for k in range(train_size, len(dataset)):
            X_test.append(dataset_scaled[k-timestep:k, 0])
            y_test.append(dataset_scaled[k, 0])
            y_test_orj.append(dataset[k])

        X_train, y_train, X_test, y_test, y_test_orj = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(y_test_orj)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        
        best_model_name, best_model_predictions, best_model_metrics, model_metrics = find_best_model(X_train, y_train, X_test, y_test, y_test_orj, timestep)
        print(f"IMF_{i+1} için en iyi model: {best_model_name}, En iyi modelin RMSE değeri: {best_model_metrics['RMSE']}")
        
        model_metrics_list.append(model_metrics)

       
       
        metrics_dict = {'IMF': f'IMF_{i+1}', 'BestModel': best_model_name, 'BestModelMetrics': best_model_metrics, 'BestModelPredictions': best_model_predictions}
        metrics_list.append(metrics_dict)

    metrics_df = pd.DataFrame(metrics_list)
    
    
    #Training the last IMF with svr
    data = imfs_set[-1:]
    dataset_knn = np.array(data, dtype = float).reshape(-1,1)
    dataset_knn_scaled = scaler.fit_transform(np.array(dataset_knn).reshape(-1,1))


    X_train_knn, y_train_knn, X_test_knn, y_test_knn, y_test_orj_knn = [], [], [], [], []
    for j in range(timestep, train_size):
       X_train_knn.append(dataset_knn_scaled[j-timestep:j, 0])
       y_train_knn.append(dataset_knn_scaled[j, 0])
    for k in range(train_size, len(dataset)):
        X_test_knn.append(dataset_knn_scaled[k-timestep:k, 0])
        y_test_knn.append(dataset_knn_scaled[k, 0])
        y_test_orj_knn.append(dataset_knn[k])

    X_train_knn, y_train_knn, X_test_knn, y_test_knn, y_test_orj_knn = np.array(X_train_knn), np.array(y_train_knn), np.array(X_test_knn), np.array(y_test_knn), np.array(y_test_orj_knn)
    X_train_knn = np.reshape(X_train_knn, (X_train_knn.shape[0], X_train_knn.shape[1], 1))
    X_test_knn = np.reshape(X_test_knn, (X_test_knn.shape[0], X_test.shape[1], 1))

    svr = SVR(kernel = 'rbf')
    param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01, 0.001], 'epsilon': [0.1, 0.01, 0.001], 'kernel':['rbf', 'linear']}
    tscv = TimeSeriesSplit(n_splits=5)
    grid_model = GridSearchCV(estimator=svr, param_grid=param_grid, cv=tscv, verbose=2, n_jobs=-1, scoring='neg_mean_squared_error', error_score='raise')
    grid_model.fit(X_train_knn.reshape(-1, timestep), y_train_knn)
    print('Best hyperparameters:', grid_model.best_params_)

    best_C = grid_model.best_params_['C']
    best_epsilon = grid_model.best_params_['epsilon']
    best_kernel = grid_model.best_params_['kernel']
    best_gamma = grid_model.best_params_['gamma']

    svr = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel, gamma = best_gamma)
    svr.fit(X_train_knn.reshape(-1, timestep), y_train_knn)
    test_predict_svr = svr.predict(X_test_knn.reshape(-1, timestep))
    descaled_test_predict_svr = scaler.inverse_transform(test_predict_svr.reshape(-1,1))

    rmse_svr = sqrt(mean_squared_error(y_test_knn, test_predict_svr))
    mse_svr = mean_squared_error(y_test_knn, test_predict_svr)
    mae_svr = mean_absolute_error(y_test_knn, test_predict_svr)
    mape_svr = MAPE(y_test_knn, test_predict_svr)
    mape_yuzsuz_svr = mape_svr/100
    r2_svr = (r2_score(y_test_knn, test_predict_svr)) 
    rrmse_svr = relative_root_mean_squared_error(y_test_knn, test_predict_svr)
    mape_ = mean_absolute_percentage_error(y_test_knn, test_predict_svr)
    smape_svr = smape(y_test_knn, test_predict_svr)


    print("SVR SONUÇLARI")
    print('Test RMSE: %.3f' % rmse_svr)
    print('Test MSE: %.3f' % mse_svr)
    print('Test MAE: %.3f' % mae_svr)
    print('Test MAPE: %.3f' % mape_svr)
    print('Test MAPE_100_çarpmasız: %.3f' % mape_yuzsuz_svr)
    print('Test R2: %.3f' % r2_svr)
    

    metrics_svr = {'RMSE': rmse_svr, 'MSE': mse_svr, 'MAE': mae_svr, 'R2': r2_svr, 'RRMSE': rrmse_svr, 'SMAPE':smape_svr, 'MAPE_': mape_}
    model_metrics_list.append(metrics_svr)
    metrics_list.append(metrics_svr)
    return metrics_list, model_metrics_list, descaled_test_predict_svr
    
#metric_list: best model results
#model_metric_list: all model results
timestep = 10
# timestep=15
time_step_path = 'timestep10'
time_step_folder_path = os.path.join(folder_path, time_step_path)
TEST_ACTUAL_DATA = (df['Close'][train_size:len(df)]).values

first_decomposition_train_result_metric_list, first_decomposition_train_result_model_metric_list, descaled_test_predict_svr_first_decomposition = each_IMFs_train(selected_IMFs_first_close, timestep)
second_decomposition_train_result_metric_list, second_decomposition_train_result_model_metric_list, descaled_test_predict_svr_second_decomposition = each_IMFs_train(selected_IMFs_second_close, timestep)

first_decomposition_scaled_predictions = np.zeros_like(first_decomposition_train_result_metric_list[0]['BestModelPredictions'])
for i in range(len(first_decomposition_train_result_metric_list)-2):
    
    
    first_decomposition_scaled_predictions += first_decomposition_train_result_metric_list[i]['BestModelPredictions']

second_decomposition_scaled_predictions = np.zeros_like(second_decomposition_train_result_metric_list[0]['BestModelPredictions'])
for i in range(len(second_decomposition_train_result_metric_list)-2):
    
    second_decomposition_scaled_predictions += second_decomposition_train_result_metric_list[i]['BestModelPredictions']


first_decomposition_scaled_predictions += descaled_test_predict_svr_first_decomposition
second_decomposition_scaled_predictions += descaled_test_predict_svr_second_decomposition
final_decomposition_scaled_predictions = first_decomposition_scaled_predictions + second_decomposition_scaled_predictions


# =============================================================================
# Plotting Final prediction result

plt.figure()
fig = plt.figure(figsize=(12, 6))
plt.plot(TEST_ACTUAL_DATA, label='Original Value')
plt.plot(final_decomposition_scaled_predictions, label='Prediction result of the proposed method')
plt.plot(denoised_test_data_hiyerarşik, label='Denoised Value')

plt.xlabel('Trading Day')

plt.legend()
plt.show()

file_name5 = 'predicted_and_original_hiyerarşik_yöntem.png'

full_path5 = os.path.join(time_step_folder_path, file_name5)
fig.savefig(full_path5)

# =============================================================================
# Result of training all IMFs with a single LSTM model

first_decomposition_scaled_predictions_LSTM = np.zeros_like(first_decomposition_train_result_model_metric_list[0][0]['LSTM_Predictions'])
for i in range(len(first_decomposition_train_result_model_metric_list)-1):
   
    first_decomposition_scaled_predictions_LSTM += first_decomposition_train_result_model_metric_list[i][0]['LSTM_Predictions']

second_decomposition_scaled_predictions_LSTM = np.zeros_like(second_decomposition_train_result_model_metric_list[0][0]['LSTM_Predictions'])
for i in range(len(second_decomposition_train_result_model_metric_list)-1):
    
    second_decomposition_scaled_predictions_LSTM += second_decomposition_train_result_model_metric_list[i][0]['LSTM_Predictions']

final_decomposition_scaled_predictions_LSTM = first_decomposition_scaled_predictions_LSTM + second_decomposition_scaled_predictions_LSTM


# =============================================================================
# Result of training all IMFs with a single LSTM-BN model

first_decomposition_scaled_predictions_LSTMBatch = np.zeros_like(first_decomposition_train_result_model_metric_list[0][2]['LSTM-Batch_Predictions'])
for i in range(len(first_decomposition_train_result_model_metric_list)-1):

    first_decomposition_scaled_predictions_LSTMBatch += first_decomposition_train_result_model_metric_list[i][2]['LSTM-Batch_Predictions']

second_decomposition_scaled_predictions_LSTMBatch = np.zeros_like(second_decomposition_train_result_model_metric_list[0][2]['LSTM-Batch_Predictions'])
for i in range(len(second_decomposition_train_result_model_metric_list)-1):
    
    second_decomposition_scaled_predictions_LSTMBatch += second_decomposition_train_result_model_metric_list[i][2]['LSTM-Batch_Predictions']

final_decomposition_scaled_predictions_LSTMBatch = first_decomposition_scaled_predictions_LSTMBatch + second_decomposition_scaled_predictions_LSTMBatch


# =============================================================================
# Result of training all IMFs with a single GRU model

first_decomposition_scaled_predictions_GRU = np.zeros_like(first_decomposition_train_result_model_metric_list[0][1]['GRU_Predictions'])
for i in range(len(first_decomposition_train_result_model_metric_list)-1):

    first_decomposition_scaled_predictions_GRU += first_decomposition_train_result_model_metric_list[i][1]['GRU_Predictions']

second_decomposition_scaled_predictions_GRU = np.zeros_like(second_decomposition_train_result_model_metric_list[0][1]['GRU_Predictions'])
for i in range(len(second_decomposition_train_result_model_metric_list)-1):
    
    second_decomposition_scaled_predictions_GRU += second_decomposition_train_result_model_metric_list[i][1]['GRU_Predictions']

final_decomposition_scaled_predictions_GRU = first_decomposition_scaled_predictions_GRU + second_decomposition_scaled_predictions_GRU


plt.figure()
fig = plt.figure(figsize=(12, 6))
plt.plot(TEST_ACTUAL_DATA, label='Original Value')
plt.plot(final_decomposition_scaled_predictions_LSTM, label='Prediction result of the LSTM model')
plt.plot(final_decomposition_scaled_predictions_LSTMBatch, label='Prediction result of the LSTM-Batch Normalization model')
plt.plot(final_decomposition_scaled_predictions_GRU, label='Prediction result of the GRU model')
plt.plot(final_decomposition_scaled_predictions, label='Prediction result of the proposed method')
plt.plot(denoised_test_data_hiyerarşik, label='Denoised Value')

plt.xlabel('Trading Day')
plt.legend()
plt.show()

file_name6 = 'predicted_and_original_hiyerarşik_yöntem_herIMF_GRU_egitim.png'

full_path6 = os.path.join(time_step_folder_path, file_name6)
fig.savefig(full_path6)
