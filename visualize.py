from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from keras import Sequential
# from keras import layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

dataFrame = pd.read_csv('eyetracking_data.csv')

#separate features from classifications
dimensions = dataFrame[["mu_fixation","tau_fixation","mu_saccade_amp","tau_saccade_amp","mu_saccade_dur","tau_saccade_dur","mu_fixation_num","tau_fixation_num","mu_saccade_num","tau_saccade_num","mu_blink_num","tau_blink_num","mu_response_time","tau_response_time","mu_correct","tau_correct"]]

scaler = StandardScaler()
features = scaler.fit_transform(dimensions)
labels = dataFrame['CL_level']  # integers: 1, 2, 3



# Step 1: Reduce features to 2D using PCA
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(features_2d, labels, test_size=0.2, stratify=labels)

# Step 3: Train logistic regression on 2D features
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Step 4: Create a meshgrid
x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Step 5: Predict class for each point on the grid
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

# Step 6: Plot decision boundaries and data points
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00CC00', '#0000FF'])

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.5)

# Overlay training points
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=50)
plt.title("Logistic Regression Decision Boundaries (2D PCA projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(*scatter.legend_elements(), title="Class")
plt.grid(True)
plt.tight_layout()
plt.show()
