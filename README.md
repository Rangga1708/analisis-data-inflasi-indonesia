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
Pada analisis runtun waktu, asumsi awal yang harus dipenuhi adalah ***stasioneritas*** data terhadap mean. Artinya, mean dari data haruslah konstan. Perhatikan plot data awal berikut:
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
![Plot Awal Data Inflasi Indonesia](https://github.com/Rangga1708/analisis-data-inflasi-indonesia/blob/master/Plot_Awal_Data_Inflasi_Indonesia.png)

Dari plot di atas, ternyata data tidak stasioner terhadap mean karena plot tersebut membentuk bukit dan jurang yang cukup terjal. Kita coba lakukan ***differencing*** data (menghitung selisih data pada waktu ke-t dengan ke-t+1) dan melihat plotnya. 
