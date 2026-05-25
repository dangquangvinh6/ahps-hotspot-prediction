import pickle
with open("models/best_thresholds.pkl", "rb") as f:
    thresholds = pickle.load(f)
print("Sự thật về Threshold:", thresholds)