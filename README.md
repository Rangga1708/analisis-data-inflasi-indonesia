Hai...kali ini kita akan menganalisis data inflasi Indonesia dari Januari 2003 sampai Juni 2019. Nantinya kita akan menggunakan data ini untuk membuat model dan memprediksi inflasi Indonesia pada Juli 2019 sampai Desember 2019. Data yang akan digunakan untuk membentuk model dapat teman-teman temukan [di sini](https://github.com/Rangga1708/analisis-data-inflasi-indonesia/blob/master/data_inflasi_indonesia_train.xlsx) sedangkan data inflasi Indonesia pada bulan Juli 2019 sampai Desember 2019 yang sebenarnya juga ada [di sini](https://github.com/Rangga1708/analisis-data-inflasi-indonesia/blob/master/data_inflasi_indonesia_test.xlsx). Untuk syntax Python dapat teman-teman temukan **di sini**.

Analisis data kali ini berbeda dengan yang sudah kita lakukan sebelumnya. Pada analisis data motor trend US, kita menggunakan analisis regresi linear sederhana. Tetapi untuk analisis data inflasi ini, kita akan menggunakan analisis runtun waktu (time series). Mengapa kita tidak menggunakan analisis regresi linear sederhana? Karena data inflasi ini nilainya hanya bergantung pada data waktu sebelumnya dan diasumsikan tidak ada variabel lain yang mempengaruhi inflasi, walaupun pada kenyataannya ada faktor ekonomi yang mempengaruhinya. Sebagai contoh, inflasi pada bulan Juni 2019 nilainya bergantung pada inflasi pada bulan Mei 2019 (bisa saja bergantung pada inflasi bulan sebelumnya lagi). Inilah mengapa analisis ini dinamakan analisis runtun waktu karena nilainya hanya bergantung pada nilai di waktu sebelumnya.

Pertama saya akan mengimport beberapa modules python yang saya butuhkan. Modules yang akan saya gunakan adalah sebagai berikut:
```Python
#import modules
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import statistics as stat
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import shapiro
from statsmodels.sandbox.stats.runs import runstest_1samp
```

## Asumsi Awal
Pada analisis runtun waktu, asumsi awal yang harus dipenuhi adalah ***stasioneritas*** data terhadap mean atau dengan kata lain mean dari data haruslah konstan. Perhatikan plot data awal berikut:
```Python
#import data
data_inflasi_train = pd.read_excel('data_inflasi_indonesia_train.xlsx')

#ubah urutan data
data_inflasi_train = data_inflasi_train.iloc[::-1].reset_index(drop = True)

#tentukan sumbu x dan y line plot
waktu = data_inflasi_train['Month']
inflasi = data_inflasi_train['Inflasi']

#buat line plot
plot_inflasi = go.Scatter(
    x = waktu,
    y = inflasi,
    mode = 'lines'
)

#atur layout
layout = {
    'title' : {
        'text' : 'Plot Awal Data Inflasi Indonesia',
        'x' : 0.5
    }
}

#buat figure
fig = go.Figure(data = plot_inflasi, layout = layout)

#tampilkan plot
fig.show()
```
<img src="Plot_Awal_Data_Inflasi_Indonesia.png" class="img-responsive" alt="">

Dari plot di atas, ternyata data tidak stasioner terhadap mean karena plot tersebut membentuk bukit dan jurang yang cukup terjal. Kita juga dapat melakukan uji hipotesis menggunakan ADF Unit Root Test dengan hipotesis nol data tidak stasioner terhadap mean. Hipotesis nol ditolak jika p-value < 0.05.
```Python
#hitung p-value adf test
result = adfuller(inflasi, autolag='AIC')
print('P-value :',result[1])
```
`P-value : 0.38987669092034716`

Karena p-value > 0.05, maka hipotesis nol tidak ditolak sehingga dapat disimpulkan bahwa data tidak stasioner terhadap mean.

Jika data tidak stasioner terhadap mean, maka analisis runtun waktu tidak dapat dilakukan. Untuk mengatasi masalah tersebut, kita perlu mentransformasi data terlebih dahulu. Kita akan coba lakukan ***differencing*** data (menghitung selisih data pada waktu ke-t dengan ke-t+1) dan melihat plotnya. 
```Python
#tentukan sumbu x dan y line plot
waktu = data_diff['t']
inflasi_diff = data_diff['Difference']

#buat line plot data differencing
plot_inflasi = go.Scatter(
    x = waktu,
    y = inflasi_diff,
    mode = 'lines'
)

#atur layout
layout = {
    'title' : {
        'text' : 'Plot Differencing Data Inflasi Indonesia',
        'x' : 0.5
    }
}

#buat figure
fig = go.Figure(data = plot_inflasi, layout = layout)

#tampilkan plot
fig.show()
```
<img src="Plot_Differencing_Data_Inflasi_Indonesia.png" class="img-responsive" alt="">

Dari plot di atas, terlihat bahwa data berada di sekitar garis y=0. Artinya data sudah stasioner terhadap mean. Untuk meyakinkan, kita lakukan ADF Unit Root Test kembali.
```Python
#hitung p-value adf test
result = adfuller(inflasi_diff, autolag='AIC')
print('P-value :',result[1])
```
`P-value : 6.303998805804485e-09`

Karena p-value < 0.05, maka hipotesis nol ditolak sehingga dapat disimpulkan bahwa data stasioner terhadap mean. Dengan demikian, analisis runtun waktu dapat dilakukan. Akan tetapi, perlu diingat bahwa data yang akan kita gunakan adalah data differencing.

## Menentukan Orde ARIMA
Salah satu model runtun waktu yang akan kita gunakan kali ini adalah model ARIMA karena model ini cukup sederhana. Untuk penjelasan terkait model ARIMA dapat teman-teman pelajari sendiri. Secara umum, model ARIMA(![p,d,q](https://latex.codecogs.com/gif.latex?p%2Cd%2Cq)) adalah sebagai berikut.

![ARIMA(p,d,q)](https://latex.codecogs.com/gif.latex?%5Calpha_0Y_t&plus;%5Calpha_1Y_%7Bt-1%7D&plus;%5Cdots&plus;%5Calpha_pY_%7Bt-p%7D%3D%5Cbeta_0%5Cvarepsilon_t&plus;%5Cbeta_1%5Cvarepsilon_%7Bt-1%7D&plus;%5Cdots&plus;%5Cbeta_q%5Cvarepsilon_%7Bt-q%7D)

dengan <br>
![Y_t](https://latex.codecogs.com/gif.latex?Y_t) : data ke-t <br>
![p](https://latex.codecogs.com/gif.latex?p) : orde AR <br>
![q](https://latex.codecogs.com/gif.latex?q) : orde MA <br>
![epsilon](https://latex.codecogs.com/gif.latex?%5Cvarepsilon_t%20%5Csim%20White%20Noise%280%2C%5Csigma%5E2%29)

Karena kita sudah melakukan differencing pada data, maka model ARIMA yang akan terbentuk menjadi

![ARIMA(p,d,q) diff](https://latex.codecogs.com/gif.latex?%5Calpha_0%20%5CDelta%20Y_t%20&plus;%20%5Calpha_1%20%5CDelta%20Y_%7Bt-1%7D%20&plus;%20%5Cdots%20&plus;%20%5Calpha_p%20%5CDelta%20Y_%7Bt-p%7D%20%3D%20%5Cbeta_0%20%5CDelta%20%5Cvarepsilon_t%20&plus;%20%5Cbeta_1%20%5CDelta%20%5Cvarepsilon_%7Bt-1%7D%20&plus;%20%5Cdots%20&plus;%20%5Cbeta_q%20%5CDelta%20%5Cvarepsilon_%7Bt-q%7D)

dengan <br>
![delta Y_t](https://latex.codecogs.com/gif.latex?%5CDelta%20Y_t%20%3D%20Y_t-Y_%7Bt-1%7D) <br>
![delta epsilon t](https://latex.codecogs.com/gif.latex?%5CDelta%20%5Cvarepsilon_t%20%3D%20%5Cvarepsilon_t-%5Cvarepsilon_%7Bt-1%7D)

Nilai ![d](https://latex.codecogs.com/gif.latex?d) pada model ARIMA di atas menandakan orde differencing (berapa kali kita melakukan differencing data). Karena kita hanya melakukan 1x differencing, maka ![d=1](https://latex.codecogs.com/gif.latex?d%3D1). Nilai yang perlu kita cari berikutnya adalah orde AR (![p](https://latex.codecogs.com/gif.latex?p)), orde MA (![q](https://latex.codecogs.com/gif.latex?q)), koefisien ![alpha](https://latex.codecogs.com/gif.latex?%5Calpha), dan koefisien ![beta](https://latex.codecogs.com/gif.latex?%5Cbeta).

### Plot ACF dan PACF
Pertama, untuk menentukan orde AR dan MA, kita perlu memperhatikan plot Autocorrelation Function (ACF) dan plot Partial Autocorrelation (PACF) terlebih dahulu.
```Python
#buat plot acf dan pacf
plt.figure(figsize=(15,4))
plt.subplot(121)
plot_acf(inflasi_diff, lags=36, ax=plt.gca())
plt.subplot(122)
plot_pacf(inflasi_diff, lags=36, ax=plt.gca())
plt.show()
```
![plot acf dan pacf](https://github.com/Rangga1708/analisis-data-inflasi-indonesia/blob/master/Plot_ACF_PACF.png)

Untuk menentukan orde AR dan MA, kita perlu melihat lag mana yang melewati arsiran persegi. Lag yang melewati arsiran persegi dan terletak paling kanan akan menjadi penentu dari orde AR dan MA. Akan tetapi, orde AR dan MA yang cukup besar sebenarnya tidak terlalu signifikan dengan orde yang lebih kecil sehingga kita cukup perhatikan empat lag paling kiri dari plot ACF dan PACF. Apalagi kalau kita lihat plot di atas, lag pertama jauh melewati arsiran persegi. Artinya orde AR dan MA yang kecil sudah cukup untuk membentuk model ARIMA.

Dari empat lag pertama masing-masing plot, dapat dilihat bahwa lag pertama dan kedua dari masing-masing plot melewati arsiran persegi. Dengan demikian, kita peroleh orde AR ![p=2](https://latex.codecogs.com/gif.latex?p%3D2) dan orde MA ![q=2](https://latex.codecogs.com/gif.latex?q%3D2). Jujur saja alasan mengapa pemilihannya seperti itu aku juga tidak terlalu mengerti hehehe.... Mungkin jika teman-teman tahu alasannya bisa share ke aku.

## Membentuk Model ARIMA
Walaupun kita sudah menentukan orde AR dan orde MA, bukan berarti model yang kita peroleh adalah ARIMA(2,1,2). Seperti yang sudah aku katakan sebelumnya, orde AR dan MA yang cukup besar tidak terlalu signifikan dengan orde yang lebih kecil. Dengan kata lain, orde yang kecil saja sudah cukup untuk membentuk model ARIMA. Tapi seberapa kecil kah? 

Kita akan menyelidiki model ARIMA dengan orde AR dan MA yang lebih kecil atau sama dengan 2 baik dengan konstanta maupun tanpa konstanta (seperti ARIMA(2,1,2), ARIMA(1,1,2), ARIMA(0,1,2), ARIMA(2,1,1), dst). Perlu diingat bahwa tidak ada ARIMA(0,1,0) karena jelas dari persamaannya saja tidak mungkin kedua orde bernilai 0. Berikut code python yang kita gunakan untuk menyelidiki beberapa model ARIMA yang dapat terbentuk.
```Python
#buat model arima
model_cons = []
model_nocons = []

k = 0
for i in range(2,-1,-1):
    for j in range(2,-1,-1):
        if i==0 and j==0:
            continue
        else:
            model1 = ARIMA(inflasi_diff,order=(j,1,i))
            model2 = ARIMA(inflasi_diff,order=(j,1,i))
            model_cons.append(model1.fit())
            model_nocons.append(model2.fit(trend='nc'))
            print(model_cons[k].summary(),'\n')
            print(model_nocons[k].summary(),'\n')
            k = k+1
```
Untuk outputnya tidak aku cantumkan di sini karena cukup panjang, tapi teman-teman dapat melihatnya di syntax yang sudah aku buat.

## Model Selection
Dari beberapa model ARIMA yang sudah terbentuk, kita akan pilih model ARIMA yang semua variabelnya signifikan. Dengan melihat p-value masing-masing variabel setiap modelnya pada hasil output pembentukan model ARIMA, variabel yang signifikan adalah variabel dengan p-value kurang dari 0.05. Berikut program untuk menentukan mana model yang semua variabelnya signifikan.
```Python
#cek apakah ada variabel yang tidak signifikan
def cek(model):
    not_sig_pvalues = [x for x in model.pvalues if x > 0.05]
    if len(not_sig_pvalues) != 0:
        return False
    else:
        return True

#model yang signifikan
model_cons_sig = []
model_nocons_sig = []
model_cons_sig_name = []
model_nocons_sig_name = []

#print model yang signifikan
for i in range(len(model_cons)):
    if cek(model_cons[i]) == True:
        model_cons_sig.append(model_cons[i])
        ar = len(model_cons[i].arparams)
        ma = len(model_cons[i].maparams)
        model_cons_sig_name.append('ARIMA(%d,1,%d) C' %(ar,ma))
        print('ARIMA(%d,1,%d) C' %(ar,ma))
    if cek(model_nocons[i]) == True:
        model_nocons_sig.append(model_nocons[i])
        ar = len(model_nocons[i].arparams)
        ma = len(model_nocons[i].maparams)
        model_nocons_sig_name.append('ARIMA(%d,1,%d) TC' %(ar,ma))
        print('ARIMA(%d,1,%d) TC' %(ar,ma))
```
Kita peroleh model yang signifikan adalah
```
ARIMA(0,1,2) TC
ARIMA(1,1,1) TC
ARIMA(0,1,1) TC
ARIMA(2,1,0) TC
ARIMA(1,1,0) TC
```
dengan TC : tanpa konstan.

## Model Testing
Dari kelima model yang signifikan, kita akan memilih model mana yang akan kita gunakan. Biasanya, dalam memilih model terbaik kita perlu membandingkan nilai statistik dari masing-masing model (R Squared, Sum Square Error, dsb). Setelah itu, kita juga perlu melakukan diagnostic checking untuk memastikan apakah model yang kita pilih benar-benar model yang terbaik. Akan tetapi, kali ini kita tidak melakukan itu karena selama ini ketika aku membuat model time series, model yang terbaik secara statistik tidak sesuai dengan keadaan yang sebenarnya. Apa maksudnya? Nanti akan aku tunjukkan di bawah.

Oleh karena itu, untuk memilih model yang terbaik, kita langsung menguji model kita menggunakan data test. Berikut data inflasi Indonesia dari bulan Juli 2019 sampai Desember 2019 yang akan kita gunakan untuk menguji model.
```Python
#import data
data_inflasi_test = pd.read_excel('data_inflasi_indonesia_test.xlsx')

#ubah urutan data
data_inflasi_test = data_inflasi_test.iloc[::-1].reset_index(drop = True)

data_inflasi_test
```

<table>
    <tr>
        <th> Month </th>
        <th> Inflasi </th>
    </tr>
    <tr>
        <td> 2019-07-01 </td> <td> 0.0332 </td>
    </tr>
    <tr>
        <td> 2019-08-01 </td> <td> 0.0349 </td>
    </tr>
    <tr>
        <td> 2019-09-01 </td> <td> 0.0339 </td>
    </tr>
    <tr>
        <td> 2019-10-01 </td> <td> 0.0313 </td>
    </tr>
    <tr>
        <td> 2019-11-01 </td> <td> 0.0300 </td>
    </tr>
    <tr>
        <td> 2019-12-01 </td> <td> 0.0272 </td>
    </tr>    
</table>

Sekarang kita akan menggunakan kelima model ARIMA yang sudah terbentuk untuk memprediksi inflasi indonesia dari bulan Juli 2019 sampai Desember 2019, lalu membandingkan nilainya dengan data asli.

1. ARIMA(0,1,2) TC
```Python
predicted_values_diff = model_nocons_sig[0].predict(start = len(inflasi_diff), end = len(inflasi_diff)+5)

predicted_values = [inflasi[197]]
n_data_awal = len(inflasi)
for i in range(len(predicted_values_diff)):
    predicted_values.append(predicted_values[i]+predicted_values_diff[n_data_awal-1+i])
    
data_inflasi_test['Forecast'] = predicted_values[1:]
data_inflasi_test['Error'] = data_inflasi_test['Inflasi']-predicted_values[1:]

data_inflasi_test
```

<table>
    <tr>
        <th> Month </th>
        <th> Inflasi </th>
        <th> Forecast </th>
        <th> Error </th>
    </tr>
    <tr>
        <td> 2019-07-01 </td> <td> 0.0332 </td> <td> 0.032653 </td> <td> 0.000547 </td>
    </tr>
    <tr>
        <td> 2019-08-01 </td> <td> 0.0349 </td> <td> 0.032915 </td> <td> 0.001985 </td>
    </tr>
    <tr>
        <td> 2019-09-01 </td> <td> 0.0339 </td> <td> 0.032915 </td> <td> 0.000985 </td>
    </tr>
    <tr>
        <td> 2019-10-01 </td> <td> 0.0313 </td> <td> 0.032915 </td> <td> -0.001615 </td>
    </tr>
    <tr>
        <td> 2019-11-01 </td> <td> 0.0300 </td> <td> 0.032915 </td> <td> -0.002915 </td>
    </tr>
    <tr>
        <td> 2019-12-01 </td> <td> 0.0272 </td> <td> 0.032915 </td> <td> -0.005715 </td>
    </tr>  
</table>

2. ARIMA(1,1,1) TC
```Python
predicted_values_diff = model_nocons_sig[1].predict(start = len(inflasi_diff), end = len(inflasi_diff)+5)

predicted_values = [inflasi[197]]
n_data_awal = len(inflasi)
for i in range(len(predicted_values_diff)):
    predicted_values.append(predicted_values[i]+predicted_values_diff[n_data_awal-1+i])
    
data_inflasi_test['Forecast'] = predicted_values[1:]
data_inflasi_test['Error'] = data_inflasi_test['Inflasi']-predicted_values[1:]

data_inflasi_test
```

<table>
    <tr>
        <th> Month </th>
        <th> Inflasi </th>
        <th> Forecast </th>
        <th> Error </th>
    </tr>
    <tr>
        <td> 2019-07-01 </td> <td> 0.0332 </td> <td> 0.032896 </td> <td> 0.000304 </td>
    </tr>
    <tr>
        <td> 2019-08-01 </td> <td> 0.0349 </td> <td> 0.032919 </td> <td> 0.001985 </td>
    </tr>
    <tr>
        <td> 2019-09-01 </td> <td> 0.0339 </td> <td> 0.032920 </td> <td> 0.000981 </td>
    </tr>
    <tr>
        <td> 2019-10-01 </td> <td> 0.0313 </td> <td> 0.032920 </td> <td> -0.001620 </td>
    </tr>
    <tr>
        <td> 2019-11-01 </td> <td> 0.0300 </td> <td> 0.032920 </td> <td> -0.002920 </td>
    </tr>
    <tr>
        <td> 2019-12-01 </td> <td> 0.0272 </td> <td> 0.032920 </td> <td> -0.005720 </td>
    </tr>  
</table>

3. ARIMA(0,1,1) TC
```Python
predicted_values_diff = model_nocons_sig[2].predict(start = len(inflasi_diff), end = len(inflasi_diff)+5)

predicted_values = [inflasi[197]]
n_data_awal = len(inflasi)
for i in range(len(predicted_values_diff)):
    predicted_values.append(predicted_values[i]+predicted_values_diff[n_data_awal-1+i])
    
data_inflasi_test['Forecast'] = predicted_values[1:]
data_inflasi_test['Error'] = data_inflasi_test['Inflasi']-predicted_values[1:]

data_inflasi_test
```

<table>
    <tr>
        <th> Month </th>
        <th> Inflasi </th>
        <th> Forecast </th>
        <th> Error </th>
    </tr>
    <tr>
        <td> 2019-07-01 </td> <td> 0.0332 </td> <td> 0.032927 </td> <td> 0.000273 </td>
    </tr>
    <tr>
        <td> 2019-08-01 </td> <td> 0.0349 </td> <td> 0.032927 </td> <td> 0.001973 </td>
    </tr>
    <tr>
        <td> 2019-09-01 </td> <td> 0.0339 </td> <td> 0.032927 </td> <td> 0.000973 </td>
    </tr>
    <tr>
        <td> 2019-10-01 </td> <td> 0.0313 </td> <td> 0.032927 </td> <td> -0.001627 </td>
    </tr>
    <tr>
        <td> 2019-11-01 </td> <td> 0.0300 </td> <td> 0.032927 </td> <td> -0.002927 </td>
    </tr>
    <tr>
        <td> 2019-12-01 </td> <td> 0.0272 </td> <td> 0.032927 </td> <td> -0.005727 </td>
    </tr>  
</table>

4. ARIMA(2,1,0) TC
```Python
predicted_values_diff = model_nocons_sig[3].predict(start = len(inflasi_diff), end = len(inflasi_diff)+5)

predicted_values = [inflasi[197]]
n_data_awal = len(inflasi)
for i in range(len(predicted_values_diff)):
    predicted_values.append(predicted_values[i]+predicted_values_diff[n_data_awal-1+i])
    
data_inflasi_test['Forecast'] = predicted_values[1:]
data_inflasi_test['Error'] = data_inflasi_test['Inflasi']-predicted_values[1:]

data_inflasi_test
```

<table>
    <tr>
        <th> Month </th>
        <th> Inflasi </th>
        <th> Forecast </th>
        <th> Error </th>
    </tr>
    <tr>
        <td> 2019-07-01 </td> <td> 0.0332 </td> <td> 0.034814 </td> <td> -0.001614 </td>
    </tr>
    <tr>
        <td> 2019-08-01 </td> <td> 0.0349 </td> <td> 0.035875 </td> <td> -0.000975 </td>
    </tr>
    <tr>
        <td> 2019-09-01 </td> <td> 0.0339 </td> <td> 0.034593 </td> <td> -0.000693 </td>
    </tr>
    <tr>
        <td> 2019-10-01 </td> <td> 0.0313 </td> <td> 0.034804 </td> <td> -0.003504 </td>
    </tr>
    <tr>
        <td> 2019-11-01 </td> <td> 0.0300 </td> <td> 0.035193 </td> <td> -0.005193 </td>
    </tr>
    <tr>
        <td> 2019-12-01 </td> <td> 0.0272 </td> <td> 0.034925 </td> <td> -0.007725 </td>
    </tr>  
</table>

5. ARIMA(1,1,0) TC
```Python
predicted_values_diff = model_nocons_sig[4].predict(start = len(inflasi_diff), end = len(inflasi_diff)+5)

predicted_values = [inflasi[197]]
n_data_awal = len(inflasi)
for i in range(len(predicted_values_diff)):
    predicted_values.append(predicted_values[i]+predicted_values_diff[n_data_awal-1+i])
    
data_inflasi_test['Forecast'] = predicted_values[1:]
data_inflasi_test['Error'] = data_inflasi_test['Inflasi']-predicted_values[1:]

data_inflasi_test
```

<table>
    <tr>
        <th> Month </th>
        <th> Inflasi </th>
        <th> Forecast </th>
        <th> Error </th>
    </tr>
    <tr>
        <td> 2019-07-01 </td> <td> 0.0332 </td> <td> 0.034637 </td> <td> -0.001437 </td>
    </tr>
    <tr>
        <td> 2019-08-01 </td> <td> 0.0349 </td> <td> 0.034000 </td> <td> 0.000900 </td>
    </tr>
    <tr>
        <td> 2019-09-01 </td> <td> 0.0339 </td> <td> 0.034221 </td> <td> -0.000321 </td>
    </tr>
    <tr>
        <td> 2019-10-01 </td> <td> 0.0313 </td> <td> 0.034144 </td> <td> -0.002844 </td>
    </tr>
    <tr>
        <td> 2019-11-01 </td> <td> 0.0300 </td> <td> 0.034171 </td> <td> -0.004171 </td>
    </tr>
    <tr>
        <td> 2019-12-01 </td> <td> 0.0272 </td> <td> 0.034162 </td> <td> -0.006962 </td>
    </tr>  
</table>

Dari tabel-tabel di atas, kita bisa lihat beberapa hal. Model ARIMA(0,1,2), ARIMA(1,1,1), dan ARIMA(0,1,1) memberikan hasil prediksi yang cenderung konstan. Nilainya tiap periode tidak berubah. Tentunya dalam kenyataannya, inflasi Indonesia setiap bulan tidak akan selalu konstan. Sedangkan untuk model ARIMA(2,1,0) dan ARIMA(1,1,0) hasil prediksinya cenderung bervariatif. Kalau kita perhatikan errornya, error dari hasil prediksi model ARIMA(1,1,0) lebih kecil dibandingkan model ARIMA (2,1,0). Dengan demikian, kita pilih model ARIMA (1,1,0) atau sama dengan AR(1) (karena orde MA nya 0) sebagai model terbaik. 

## Kesimpulan

Kita peroleh model time series

![ARIMA(1,1,0)](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20%5CDelta%20Y_t%20-%200.3466%20%5CDelta%20Y_%7Bt-1%7D%20%3D%20%5CDelta%20%5Cvarepsilon_t%20%26%5Ciff%20Y_t%20-%20Y_%7Bt-1%7D%20-%200.3466%20%28Y_%7Bt-1%7D-Y_%7Bt-2%7D%29%20%3D%20%5Cvarepsilon_t%20-%20%5Cvarepsilon_%7Bt-1%7D%5C%5C%20%26%5Ciff%20Y_t%20-%201.3466%20Y_%7Bt-1%7D%20&plus;%200.3466%20Y_%7Bt-1%7D%20%3D%20%5Cvarepsilon_t%20-%20%5Cvarepsilon_%7Bt-1%7D%5C%5C%20%26%5Ciff%20Y_t%20%3D%201.3466%20Y_%7Bt-1%7D%20-%200.3466%20Y_%7Bt-1%7D%20&plus;%20%5Cvarepsilon_t%20-%20%5Cvarepsilon_%7Bt-1%7D%5C%5C%20%5Cend%7Balign*%7D)

Sebenarnya berdasarkan nilai statistik dan diagnostic checking, model terbaik adalah ARIMA(2,1,0). Tetapi setelah aku tes, ternyata hasilnya tidak sesuai yang diinginkan sehingga aku mencoba menguji semua model sekaligus dan membandingkan nilainya. 

Jadi itulah model time series yang kita peroleh. Jika ada yang salah atau ada yang perlu didiskusikan, silakan berikan komentar dan saran. Semoga bermanfaat :)
