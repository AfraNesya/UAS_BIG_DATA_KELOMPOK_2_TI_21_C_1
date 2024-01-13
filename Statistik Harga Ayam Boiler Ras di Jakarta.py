#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# In[2]:


# Inisialisasi sesi Spark
spark = SparkSession.builder.appName("LinearRegressionExample").getOrCreate()


# In[3]:


# Memanggil dataset dari CSV
data = spark.read.csv('Statistik_Harga_Komoditas_Ayam_Broiler_Ras.csv', inferSchema=True, header=True)


# In[4]:


# Menggunakan VectorAssembler untuk menggabungkan fitur-fitur ke dalam vektor
feature_columns = data.columns[2:]  # Menghilangkan kolom "Bulan" dan "Pasar_Senen_Blok_III_VI"
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_vectorized = vector_assembler.transform(data)


# In[5]:


# Menggunakan backticks (`)
df_final = df_vectorized.select("features", "`Pasar Senen Blok III - VI`")


# In[6]:


# Inisialisasi model Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="Pasar Senen Blok III - VI")


# In[7]:


# Fitting model ke data
model = lr.fit(df_final)


# In[8]:


# Menampilkan koefisien dan intersep model
print("Coefficients: " + str(model.coefficients))
print("Intercept: " + str(model.intercept))


# In[9]:


# Menampilkan statistik model
training_summary = model.summary
print("RMSE: %f" % training_summary.rootMeanSquaredError)
print("R2: %f" % training_summary.r2)


# In[10]:


# Prediksi menggunakan model
prediksi = model.transform(df_vectorized)


# In[14]:


# Hitung MSE dan MAE
evaluator_mse = RegressionEvaluator(labelCol="Pasar Senen Blok III - VI", predictionCol="prediction", metricName="mse")
evaluator_mae = RegressionEvaluator(labelCol="Pasar Senen Blok III - VI", predictionCol="prediction", metricName="mae")


# In[15]:


mse = evaluator_mse.evaluate(prediksi)
mae = evaluator_mae.evaluate(prediksi)


# In[16]:


print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")


# In[17]:


# Statistik Deskriptif
statistik_harga = data.describe().select("summary", "Pasar Senen Blok III - VI", "Pasar Sunter Podomoro", "Pasar Rawa Badak", "Pasar Grogol", "Pasar Glodok", "Pasar Minggu", "Pasar Mayestik", "Pasar Pramuka", "Pasar Kramat Jati", "Pasar Jatinegara")


# In[18]:


# Tampilkan statistik deskriptif
statistik_harga.show()


# In[19]:


import matplotlib.pyplot as plt


# In[20]:


# Konversi data dari Spark DataFrame ke Pandas DataFrame
data_pandas = data.toPandas()


# In[21]:


# Plot tren harga
plt.figure(figsize=(12, 6))
for pasar in feature_columns:
    plt.plot(data_pandas["Bulan"], data_pandas[pasar], label=pasar)
plt.xlabel("Bulan")
plt.ylabel("Harga")
plt.title("Tren Harga di Berbagai Pasar")
plt.legend()
plt.show()


# In[22]:


# Hitung matriks korelasi
korelasi = data_pandas[feature_columns].corr()


# In[23]:


# Visualisasikan matriks korelasi
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(korelasi, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriks Korelasi antar Pasar")
plt.show()


# In[24]:


# Hitung selisih harga antar pasar
selisih_harga = data_pandas[feature_columns].diff()


# In[25]:


# Visualisasikan selisih harga
plt.figure(figsize=(16, 8))
for pasar in feature_columns:
    plt.plot(data_pandas["Bulan"], selisih_harga[pasar], label=pasar)

plt.xlabel("Bulan")
plt.ylabel("Selisih Harga")
plt.title("Selisih Harga Antar Pasar")
plt.legend()
plt.show()


# In[26]:


# Visualisasikan distribusi harga
plt.figure(figsize=(16, 8))
for pasar in feature_columns:
    sns.kdeplot(data_pandas[pasar], label=pasar)

plt.xlabel("Harga")
plt.ylabel("Density")
plt.title("Distribusi Harga di Berbagai Pasar")
plt.legend()
plt.show()


# In[ ]:




