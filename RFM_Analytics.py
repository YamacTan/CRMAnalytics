#######################
# Yamac TAN - Data Science Bootcamp - Week 3 - Project 1
#######################
# %%
import numpy as np
import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# %%
###############################################
# Görev 1 - Adım 1
###############################################

df_ = pd.read_csv("Odevler/HAFTA_03/ENDUSTRI_PROJESI_1/flo_data_20K.csv")
df = df_.copy()

# %%
###############################################
# Görev 1 - Adım 2
###############################################

df.head(10)  # a) İlk 10 gözlem

df.columns  # b)Degisken isimleri

df.describe().T  # c) Betimsel istatistik. Okunurlugu artırmak için transpose edilmiştir.

df.isnull().sum()  # d) Bos deger
df.dropna(inplace=True)

df.dtypes  # e) Değişken tipleri incelemesi.

# %%
###############################################
# Görev 1 - Adım 3
###############################################

df["order_num_total_omnichannel"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total_omnichannel"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# %%
###############################################
# Görev 1 - Adım 4
###############################################

df.dtypes  # Değişken tiplerinin incelenmesi

# Tarih ifade eden değişkenlerin adı "date" ifadesini içermektedir.

for col in df.columns:
    if "date" in col:
        df[col] = pd.to_datetime(df[col])

# %%
###############################################
# Görev 1 - Adım 5
###############################################

# Müsteri sayılarının toplamı unique master_id sayılarının toplamına eşittir.

df.groupby("order_channel").agg({"master_id": "count",
                                 "order_num_total_omnichannel": "mean",
                                 "customer_value_total_omnichannel": "mean"})

# %%
###############################################
# Görev 1 - Adım 6
###############################################

df["customer_value_total_omnichannel"].sort_values(ascending=False).head(10)

# %%
###############################################
# Görev 1 - Adım 7
###############################################

df["order_num_total_omnichannel"].sort_values(ascending=False).head(10)

# %%
###############################################
# Görev 1 - Adım 8
###############################################

def data_prep(dataframe):
    print("##################### Head #####################")
    print(df.head(10))
    print("##################### Variables #####################")
    print(df.columns)
    print("##################### Descriptive statistics #####################")
    print(df.describe().T)
    print("##################### Null Values #####################")
    print(df.isnull().sum())
    df.dropna(inplace=True)
    print("##################### Data Types #####################")
    print(df.dtypes)

    dataframe["order_num_total_omnichannel"] = dataframe["order_num_total_ever_online"] + dataframe[
        "order_num_total_ever_offline"]
    dataframe["customer_value_total_omnichannel"] = dataframe["customer_value_total_ever_online"] + dataframe[
        "customer_value_total_ever_offline"]

    for col in dataframe.columns:
        if "date" in col:
            dataframe[col] = pd.to_datetime(dataframe[col])


data_prep(df)

###############################################
# Görev 2 - Adım 1
###############################################

# Recency: Yenilik anlamına gelir. "Müşteri bizden en son ne zaman alışveriş yaptı?" sorusunun yanıtı olarak
# düşünülebilir. Analizin yapıldığı tarih - Müşterinin satın alım yaptığı son tarih formülüyle hesaplanır.

# Frequency: Sıklık anlamına gelir. Müşterinin toplam işlem sayısı, yani işlem sıklığıdır. Toplam alışveriş saysı ile
# ifade edilebilir.

# Monetary: Yapılan işlemler, yani satın alımlar neticesinde müşteri tarafından bırakılan toplam paradır.

# %%
###############################################
# Görev 2 - Adım 2 - 3 - 4
###############################################

# Önemli not: Müşterileri ifade eden master_id'lerin tümü unique değerler olduğundan (yani her biri tek bir müşteriyi
# ifade ettiğinden), her müşteri için son işlem tarihi belli olduğundan ve her müşterinin harcadığı toplam tutar ile
# toplam işlem sayısı önceki adımlarda hesaplandığından herhangi bir aggregation ya da groupby işlemine gerek yoktur.

df["last_order_date"].max()  # Last order date kolonunun max değerini 2021-05-30 olarak vermektedir.
todays_date = dt.datetime(2021, 6, 1)

df["date_difference"] = (todays_date - df["last_order_date"]).astype('timedelta64[D]')

# date_difference = Recency
# order_num_total_omnichannel = Toplam alışveriş sayısı = Frequency
# customer_value_total_omnichannel = Müşterinin bıraktığı toplam para = Monetary

rfm = df[['date_difference', 'order_num_total_omnichannel', 'customer_value_total_omnichannel']]
rfm.index = df["master_id"]

rfm.columns = ['recency', 'frequency', 'monetary']
rfm.describe().T

# %%
###############################################
# Görev 3 - Adım 1 - 2 - 3
###############################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["recency"], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

###############################################
# Görev 4 - Adım 1
###############################################

# RF Segment Tanımlamaları:

# Hibernating: Recency ve Frequency puanları en düşük müşteri segmentidir. Durum olarak inaktif haldedirler. [R(1–2),
# F(1–2)]

# At risk: Düşük recency ve orta-üst seviye frequency'e sahip müşteri segmentidir. Satın alım sayıları iyi seviyede
# olmasına rağmen son satın alımları üzerinden çok zaman geçtiği için risk grubu olarak değerlendirilir. [R(1–2),
# F(3–4)]

# Can't loose them: En yüksek frekans ve en düşün recency puanlarına sahip gruptur. Çok yüksek satın alım
# yaptıklarından ticari strateji olarak kaybedilmemesi ve kendilerini satın alıma teşvik edecek şeyler yapıılması
# gerekir. [R(1–2), F(4–5)]

# About to sleep: Orta seviye recency ve düşük seviyede frequency puanına sahip gruptur. [R(2–3), F(1–2)]

# Need Attention: Hem Frequency hem de recency'de ortalama puanlara sahip gruptur. R [(2–3), F(2–3)]

# Loyal Customers: Orta - üst seviye recency ve frequency puanına sahip gruptur. Alışkanlıkları iki parametrede de
# ortalamanın üstünde puanlarda olduğundan sadık müşteri grubu olarak kabul edilirler.  [R(3–4), F(4–5)]

# Promising: Yüksek recency ve düşük frequency puanına sahip gruptur. Alışveriş sayısı düşük olmasına rağmen son
# alışverişlerinin üstünden zaman geçmemiştir ve tekrar gelme ihtimalleri yüksektir. [R(3–4), F(0–1)]

# New Customers: En düşük frequency ve en yüksek recency puanına sahip gruptur. R [(4–5), F(0–1)]

# Potential Loyalists: Ortalamanın üstünde recency ve ortalamanın altında frequency puanına sahip gruptur. İşlem
# sıklıklarından kaynaklı olarak frequency puanlarının da gelecekte artma ihtimali yüksektir. [R(4–5), F(2–3)]

# Champions: Hem frequency hem recency bakımından en yüksek puanlara sahip gruptur. [R(4–5), F(4–5)]

# %%
###############################################
# Görev 4 - Adım 2
###############################################


# SEGMENTLERIN ISIMLENDIRILMESI

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

# %%
###############################################
# Görev 5 - Adım 1
###############################################

rfm.groupby('segment').agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"})

# %%
###############################################
# Görev 5 - Adım 2-A
###############################################

case_df = pd.merge(df, rfm, on="master_id")

case_df_a = case_df[((case_df["segment"] == "champions") | (case_df["segment"] == "loyal_customers"))
        &(case_df["customer_value_total_omnichannel"] / case_df["order_num_total_omnichannel"] > 250 )
        &(case_df["interested_in_categories_12"].str.contains('KADIN'))]["master_id"]

case_df_a = case_df_a.reset_index(drop=True)
case_df_a.to_csv("gorev_5_a.csv", index=False)

# %%
###############################################
# Görev 5 - Adım 2-B
###############################################

# Not: Bu adımda veritabanının ayrım yapısına uymak adına AKTIFCOCUK kategorisinin COCUK kategorsine dahil olmadığı
# varsayılmıştır.

case_df = pd.merge(df, rfm, on="master_id")

case_df_b = case_df[((case_df["segment"] == "cant_loose")
        | (case_df["segment"] == "about_to_sleep")
        | (case_df["segment"] == "new_customers"))
        &((case_df["interested_in_categories_12"].str.contains('ERKEK'))
        | (case_df["interested_in_categories_12"].str.contains('COCUK')))
        & (~case_df["interested_in_categories_12"].str.contains('AKTIFCOCUK'))]["master_id"]

case_df_b = case_df_b.reset_index(drop=True)
case_df_b.to_csv("gorev_5_b.csv", index=False)