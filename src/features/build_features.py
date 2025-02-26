import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_data_outliers_removed_chauvenet.pickle")

predictor_columns = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df.info()
# subset = df[df["set"]==20]["acc_z"].plot()

for col in predictor_columns:
    df[col] = df[col].interpolate()#method="linear", inplace=True)

df.info()


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

df[df["set"]==20]["acc_z"].plot()

for s in df["set"].unique():
    start = df[df["set"]==s].index[0]
    stop = df[df["set"]==s].index[-1]
    duration = stop - start
    df.loc[df["set"]==s, "duration"] = duration.seconds


duration_df = df.groupby("category")["duration"].mean()

heavy_duration_avg_per_rep = duration_df.iloc[0]/5
medium_duration_avg_per_rep = duration_df.iloc[1]/10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
lowpass = LowPassFilter()

fs = 1000/200  # 1sec/0.2sec = 5Hz
cutoff = 1.25 # higher it is, less smooth the data will be

df_lowpass = lowpass.low_pass_filter(df_lowpass,"acc_y",fs,cutoff,order = 5)


for col in predictor_columns:
    df_lowpass = lowpass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col+"_lowpass"]
    del df_lowpass[col+"_lowpass"]
    
df_lowpass

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)#, n_components=2)

plt.figure(figsize=(10, 5))
plt.plot(range(1,len(predictor_columns)+1), pca_values, marker="o")
plt.xlabel("Principal Component")   
plt.ylabel("Explained Variance")
plt.show()


df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"]==35][["pca_1","pca_2","pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"]**2 + df_squared["acc_y"]**2 + df_squared["acc_z"]**2
gyr_r = df_squared["gyr_x"]**2 + df_squared["gyr_y"]**2 + df_squared["gyr_z"]**2

df_squared["acc_r"] = acc_r
df_squared["gyr_r"] = gyr_r

subset = df_squared[df_squared["set"]==35][["acc_r","gyr_r"]].plot(subplots=True)

df_squared
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

numabs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r","gyr_r"]

window_size = int(1000/200) # 1 sec
for col in predictor_columns:
    
        df_temporal = numabs.abstract_numerical(df_temporal, [col], window_size, "mean")
        df_temporal = numabs.abstract_numerical(df_temporal, [col], window_size,"std")

df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"]==s].copy()
    for col in predictor_columns:
    
        subset = numabs.abstract_numerical(subset, [col], window_size, "mean")
        subset = numabs.abstract_numerical(subset, [col], window_size,"std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)

subset[["acc_y","acc_y_temp_mean_ws_5","acc_y_temp_std_ws_5"]].plot()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
freqabs = FourierTransformation()

fs = int(1000/200) # 1 sec/0.2 sec = 5 Hz
ws = int(2800/200) # 2 sec , average length of a repetition

df_freq = freqabs.abstract_frequency(df_freq, ["acc_y"], ws, fs)
df_freq.columns

df_freq_list = []

for s in df_freq["set"].unique():
    print(f"Processing set {s}")
    subset = df_freq[df_freq["set"]==s].reset_index(drop=True).copy()    
    subset = freqabs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)",drop = True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()

df_freq = df_freq.iloc[::2] # to overcome highly correlated data


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

cluster_df = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2,10)
inertias = []

for k in k_values:
    subset = cluster_df[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    kmeans.fit(cluster_df[cluster_columns])
    inertias.append(kmeans.inertia_)

plt.plot(k_values, inertias, marker="o") # 5 is most optimal

subset = cluster_df[cluster_columns]
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
cluster_labels = kmeans.fit_predict(subset)
cluster_df["cluster"] = cluster_labels


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

cluster_df.to_pickle("../../data/interim/03_data_features.pickle")