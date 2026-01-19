import joblib
import pandas as pd



X, y = joblib.load("data/dataset.joblib")

df = pd.DataFrame(X)
df["label"] = y
df.columns  = [
    # MFCC means
    "mfcc1_mean", "mfcc2_mean", "mfcc3_mean", "mfcc4_mean",
    "mfcc5_mean", "mfcc6_mean", "mfcc7_mean", "mfcc8_mean",
    # Delta MFCC means
    "d_mfcc1_mean", "d_mfcc2_mean", "d_mfcc3_mean", "d_mfcc4_mean",
    "d_mfcc5_mean", "d_mfcc6_mean", "d_mfcc7_mean", "d_mfcc8_mean",
    # Delta-delta MFCC means
    "dd_mfcc1_mean", "dd_mfcc2_mean", "dd_mfcc3_mean", "dd_mfcc4_mean",
    "dd_mfcc5_mean", "dd_mfcc6_mean", "dd_mfcc7_mean", "dd_mfcc8_mean",
    # Spectral features means
    "spectral_centroid_mean", "spectral_bandwidth_mean", "f0_frame_mean",

    # MFCC stds
    "mfcc1_std", "mfcc2_std", "mfcc3_std", "mfcc4_std",
    "mfcc5_std", "mfcc6_std", "mfcc7_std", "mfcc8_std",
    # Delta MFCC stds
    "d_mfcc1_std", "d_mfcc2_std", "d_mfcc3_std", "d_mfcc4_std",
    "d_mfcc5_std", "d_mfcc6_std", "d_mfcc7_std", "d_mfcc8_std",
    # Delta-delta MFCC stds
    "dd_mfcc1_std", "dd_mfcc2_std", "dd_mfcc3_std", "dd_mfcc4_std",
    "dd_mfcc5_std", "dd_mfcc6_std", "dd_mfcc7_std", "dd_mfcc8_std",
    # Spectral features stds
    "spectral_centroid_std", "spectral_bandwidth_std", "f0_frame_std",

    # Pitch / statistics
    "f0_mean_stat", "f0_std_stat", "f0_range", "jitter", "shimmer", "f1_f2_ratio",
    #label 
    "label"

]

df.to_csv("full_dataset.csv")

#generate a small dataset to explore in Rstudio
adults = df[df["label"] == 1].sample(n=100, random_state=42)
children = df[df["label"] == 0].sample(n=100, random_state=42)
sample_df = pd.concat([adults, children], ignore_index=True)
print(sample_df.head())
print(sample_df["label"].value_counts())
sample_df.to_csv("dataset_sample_200.csv", index=False)
