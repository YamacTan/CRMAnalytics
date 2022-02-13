#######################
# Yamac TAN - Data Science Bootcamp - Week 3 - Project 2
#######################

import numpy as np
import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# %%
###############################################
# Görev 1 - Adım 1
###############################################

df_ = pd.read_csv("Odevler/HAFTA_03/ENDUSTRI_PROJESI_2/flo_data_20K.csv")
df = df_.copy()

df.isnull().sum()
df.dropna(inplace=True)
df.dtypes
df.head(10)
df.describe().T

# %%
###############################################
# Görev 1 - Adım 2
###############################################

# Quantile parametreleriyle alakalı açıklama:
# outlier_thresholds fonksiyonunda 1.çeyrek için 0,25 ve 2.çeyrek için 0,75 parametre değerleri kullanıldığında,
# dataframein betimsel istatistiğinde gözlenen mean, std ve max değerlerinde ilk haline kıyasla çok büyük değişimler
# gözlenmiştir. Bu durum, yüksek miktarda veri değişimi belirtisi olduğundan 0,01 ve 0,99 değerleri kullanılmıştır.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquantile_range).round()
    low_limit = (quartile1 - 1.5 * interquantile_range).round()
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# %%
###############################################
# Görev 1 - Adım 3
###############################################

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

# %%
###############################################
# Görev 1 - Adım 4
###############################################

df["order_num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_omnichannel"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# %%
###############################################
# Görev 1 - Adım 5
###############################################

df.dtypes  # Değişken tiplerinin incelenmesi

# Tarih ifade eden değişkenlerin adı "date" ifadesini içermektedir.

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

# %%
###############################################
# Görev 2 - Adım 1
###############################################

df["last_order_date"].max()  # Last order date kolonunun max değerini 2021-05-30 olarak vermektedir.
analysis_date = dt.datetime(2021, 6, 2)

# %%
###############################################
# Görev 2 - Adım 2
###############################################

cltv = pd.DataFrame()
cltv["customer_id"] = df["master_id"]
cltv["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]')) / 7
cltv["frequency"] = df["order_num_total_omnichannel"]
cltv["monetary_cltv_avg"] = df["customer_value_total_omnichannel"] / df["order_num_total_omnichannel"]

cltv = cltv[cltv["frequency"] > 1]

# %%
###############################################
# Görev 3 - Adım 1
###############################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv['frequency'], cltv['recency_cltv_weekly'], cltv['T_weekly'])

# 3 aylık satış tahmini
cltv["exp_sales_3_month"] = bgf.predict(12,
                                        cltv['frequency'],
                                        cltv['recency_cltv_weekly'],
                                        cltv['T_weekly'])

# 6 aylık satış tahmini
cltv["exp_sales_6_month"] = bgf.predict(24,
                                        cltv['frequency'],
                                        cltv['recency_cltv_weekly'],
                                        cltv['T_weekly'])

cltv.nlargest(n=10, columns=["exp_sales_3_month"])
cltv.nlargest(n=10, columns=["exp_sales_6_month"])

# 3. ve 6. ay tahminlerine göre en çok satın alım gerçekleştirecek 10 kişi incelendiğinde kişilerin aynı olduğu görülmektedir.

# %%
###############################################
# Görev 3 - Adım 2
###############################################

ggf = GammaGammaFitter(penalizer_coef=0.02)
ggf.fit(cltv['frequency'], cltv['monetary_cltv_avg'])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                    cltv['monetary_cltv_avg'])


# %%
###############################################
# Görev 3 - Adım 3
###############################################

cltv["CLTV"] = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency_cltv_weekly'],
                                   cltv['T_weekly'],
                                   cltv["monetary_cltv_avg"],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)


scaler = MinMaxScaler(feature_range=(0,1))
cltv["scaled_cltv"] = scaler.fit_transform(cltv[["CLTV"]])

cltv.nlargest(n=20, columns=["CLTV"])


# %%
###############################################
# Görev 4 - Adım 1
###############################################

cltv["segment"] = pd.qcut(cltv["scaled_cltv"], 4, labels=["D", "C", "B", "A"])


# %%
###############################################
# Görev 4 - Adım 2
###############################################

# CLTV Dataframe'i segmentlere göre gruplanıp, değişkenlere göre aggregation uygulandığında; segmentlerin CLTV değerleri
# arasındaki farkın üst segmentlere çıktıkça arttığı görülmektedir. Bu durum, müşteri sınıflandırmasında ilerideki
# dönemlerde problem yaratma potansiyeline sahiptir. Bunun yanında, projeye özgü olarak 19945 müşteri
# üzerinde çalışıldığı için 4 segmente ayırmak, segment başına yaklaşık 4990 müşteri anlamına gelmektedir ve bu sayı,
# yürütülecek çalışmaların hedeflenen asıl müşteri kitlesinden (Örneğin A segmentinin ilk %50'lik kısmı) sapmasına neden
# olabilir. Segmentler özelinde alınacak aksiyonlarda daha az sayıda müşteriden oluşan daha spesifik grupların olması,
# daha başarılı sonuçlar verecektir. Bundan kaynaklı olarak bu çalışma özelinde 4'ten daha çok sayıda segment kullanılması
# daha mantıklı olacaktır.


# %%
###############################################
# Görev 4 - Adım 3
###############################################

# A Segmenti:
# Şirkete en çok getiriyi sağlayan ve ilerleyen dönemde en çok beklenen satın alma değerine sahip segment olarak,
# A segmenti müşterilerine şirkete 6 aylık süreç içerisinde bağlılıklarını ödüllendirecek ve tahmin edilen satın
# alımları garanti edecek kampanya çalışmaları yürütülebilir. Bu çalışmalara, belli bir tutar üzerinde geçerli olacak
# indirim kuponları, yapılan alışverişlerde diğer segmentlerden alışverişlerden kazanılan puanların daha yüksek bir
# katsayıyla çarpılarak biriktirilebilmesi, yılın belli günlerinde sadece bu segmente özel indirimlerin yapılması gibi
# aksiyonlar örnek gösterilebilir. Bununla birlikte tanımlanabilecek bir referans sistemi ile, çevrelerinde aynı
# alışveriş  alışkanlıklarına sahip ve A segmentine kazandırabilecekleri her bir müşteri için bir indirim kuponu
# tanımlaması yapılabilir. Tüm bu çalışmaların amacı, A segmenti müşterilerini elimizde tutabilmek ve alışveriş
# alışkanlıklarını aynı standartta ya da artırarak devam ettirmelerini sağlamaktır.

#C Segmenti:
# Bu segmentin müşterileri, 6 aylık dönemde satın alma alışkanlıklarını artırabilecek ve tahmin edilen satış
# miktarlarını geçme potansiyeli olan müşterilerdir. Buy till you die kavramından yola çıkarak, bu segment için
# alınacak aksiyonlar, C segmenti müşterilerinin dropout ya da churn olmalarını engellemek üzerine yoğunlaşmalıdır.
# Burada izlenebilecek stratejilerden ilki, satın alma sonrası drop'u önleme amacıyla müşteri özelinde çapraz satış
# yapma çalışmaları olabilir. Bu çalışmalarla müşterinin satın alma ihtiyacı yeniden tetiklenebilir. Aynı zamanda,
# bu segment müşterileri özelinde alışverişlerin analizi yapılarak ilgi duyulan kategorilere yönelik mevcut ve gelecek
# kampanyalara dair özel olarak bilgilendirme yapılabilir. Yine bu analizlerden edinilecek bilgiyle; üst segmentlerdeki
# uygulamaların aksine her üründe geçerli olmayan, sadece müşterinin ilgi duyduğu ve geçmişte satın aldığı
# company'nin ürünlerinde geçerli olan özel indirim ya da kupon uygulamaları gerçekleştirilebilir.








