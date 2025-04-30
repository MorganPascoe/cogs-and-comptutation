from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier




dataFrame = pd.read_csv('eyetracking_data.csv')

#separate features from classifications
#dimensions = dataFrame[["mu_fixation","tau_fixation","mu_saccade_amp","tau_saccade_amp","mu_saccade_dur","tau_saccade_dur","mu_fixation_num","tau_fixation_num","mu_saccade_num","tau_saccade_num","mu_blink_num","tau_blink_num","mu_response_time","tau_response_time","mu_correct","tau_correct"]]

#only eye
#dimensions = dataFrame[["mu_fixation","tau_fixation","mu_saccade_amp","tau_saccade_amp","mu_saccade_dur","tau_saccade_dur","mu_fixation_num","tau_fixation_num","mu_saccade_num","tau_saccade_num","mu_blink_num","tau_blink_num"]]

#no eye
#dimensions = dataFrame[["mu_response_time","tau_response_time","mu_correct","tau_correct"]]

#only mu
#dimensions = dataFrame[["mu_fixation","mu_saccade_amp","mu_saccade_dur","mu_fixation_num","mu_saccade_num","mu_blink_num","mu_response_time","mu_correct"]]

#only tau
#dimensions = dataFrame[["tau_fixation","tau_saccade_amp","tau_saccade_dur","tau_fixation_num","tau_saccade_num","tau_blink_num","tau_response_time","tau_correct"]]

#top 8 important
dimensions = dataFrame[["tau_fixation","mu_saccade_amp","tau_saccade_amp","mu_saccade_num","mu_blink_num","mu_response_time","mu_correct","tau_correct"]]


feature_names = dimensions.columns
n_features = len(feature_names)
n_classes = 3



scaler = StandardScaler()
features = scaler.fit_transform(dimensions)
labels = dataFrame['CL_level']  # integers: 1, 2, 3
accuracies = []
precisions1 = []
precisions2 = []
precisions3 = []

recalls1 = []
recalls2 = []
recalls3 = []



for _ in range(500):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Overall accuracy
    report = classification_report(y_test, preds, output_dict=True)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)


    precisions1.append(report['1']['precision'])
    precisions2.append(report['2']['precision'])
    precisions3.append(report['3']['precision'])
    recalls1.append(report['1']['recall'])
    recalls2.append(report['2']['recall'])
    recalls3.append(report['3']['recall'])





# Report average metrics
print(f"\nAverage over 500 runs:")
print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Precision low load: {np.mean(precisions1):.4f}")
print(f"Precision med load: {np.mean(precisions2):.4f}")
print(f"Precision high load: {np.mean(precisions3):.4f}")
print(f"Recall low load: {np.mean(recalls1):.4f}")
print(f"Recall med load: {np.mean(recalls2):.4f}")
print(f"Recall high load: {np.mean(recalls3):.4f}")





