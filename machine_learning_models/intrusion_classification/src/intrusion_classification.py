# Dataset used:Network Security Laboratory Knowledge Discovery in Databases (NSL-KDD)
# Citation: https://www.unb.ca/cic/datasets/nsl.html
# M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,”
# Submitted to Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

# NSL-KDD 4-Class Network Intrusion Classification
# Cleaning (num_outbound_cmds, su_attempted), Class Assignment (4-class), Label Encoding, Deep Neural Network, Softmax Regression

# Written in Anaconda Spyder 5.5.0 IDE using Python 3.9.18 64-bit on Windows 10

# Necessary Imports
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from keras.models import load_model

#%%

""" Load Datasets """

# CHANGE AS NEEDED
local_path_train = r'C:\Users\Leo\Desktop\Senior Design II\ml_models\intrusion_classification\data\KDDTrain+.txt'

# --- Importing Train Dataset ---
# NSL-KDD, 43 features, 125973 samples, Multiclass Classification (From text file)
KDDTrain = pd.read_csv(local_path_train, header = None) # Data with difficulty level
# Column Headings
KDDTrain.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class', 'difficulty']

# We will not utilize the 'difficulty' feature for now
KDDTrain.drop('difficulty', axis=1, inplace=True)

# CHANGE AS NEEDED
local_path_test = r'C:\Users\Leo\Desktop\Senior Design II\ml_models\intrusion_classification\data\KDDTest+.txt'

# --- Importing Test Dataset ---
# NSL-KDD, 43 features, 22544 samples, Multiclass Classification (From text file)
KDDTest = pd.read_csv(local_path_test, header = None) # Data with difficulty level
# Column Headings
KDDTest.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class', 'difficulty']

# We will not utilize the 'difficulty' feature for now
KDDTest.drop('difficulty', axis=1, inplace=True)

#%%
""" Data Cleaning """

# We drop 'num_outbound_cmds' from both training and testing dataset because every instance is equal to 0 in both datasets
KDDTrain.drop("num_outbound_cmds",axis=1,inplace=True)
KDDTest.drop("num_outbound_cmds",axis=1,inplace=True)

# We replace all instances with a value of 2 to 1 because the feature should be a binary value (0 or 1)
KDDTrain['su_attempted'] = KDDTrain['su_attempted'].replace(2, 1)
KDDTest['su_attempted'] = KDDTest['su_attempted'].replace(2, 1)

#%%

""" Class Assignment (Multi-Class) """

# Change training attack labels to their respective attack class for multiclass classification
KDDTrain['class'].replace(['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land'],'DoS',inplace=True) # 6 sub classes of DoS
KDDTrain['class'].replace(['satan', 'ipsweep', 'portsweep', 'nmap'],'Probe',inplace=True) # 4 sub classes of Probe
KDDTrain['class'].replace(['warezclient', 'guess_passwd', 'warezmaster', 'imap', 'ftp_write', 'multihop', 'phf','spy'],'R2L',inplace=True) # 8 sub classes of R2L
KDDTrain['class'].replace(['buffer_overflow', 'rootkit', 'loadmodule','perl'],'U2R',inplace=True) # 4 sub classes of U2R

# Change testing attack labels to their respective attack class for multiclass classification
KDDTest['class'].replace(['neptune', 'apache2', 'processtable', 'smurf', 'back', 'mailbomb', 'pod', 'teardrop', 'land', 'udpstorm'],'DoS',inplace=True) # 10 sub classes of DoS
KDDTest['class'].replace(['mscan', 'satan', 'saint', 'portsweep', 'ipsweep', 'nmap'],'Probe',inplace=True) # 6 sub classes of Probe
KDDTest['class'].replace(['guess_passwd', 'warezmaster', 'snmpguess', 'snmpgetattack', 'httptunnel', 'multihop', 'named', 'sendmail', 'xlock', 'xsnoop', 'ftp_write', 'worm', 'phf', 'imap'],'R2L',inplace=True) # 14 sub classes of R2L
KDDTest['class'].replace(['buffer_overflow', 'ps', 'rootkit', 'xterm', 'loadmodule', 'perl', 'sqlattack'],'U2R',inplace=True) # 7 sub classes of U2R

#%%

""" Data Preprocessing """

# Use LabelEncoding for categorical features (including 'class')

# Encode class label with LabelEncoder
label_encoder = preprocessing.LabelEncoder()
KDDTrain['class'] = label_encoder.fit_transform(KDDTrain['class'])
KDDTest['class'] = label_encoder.fit_transform(KDDTest['class'])

# Define the columns to LabelEncode
categorical_columns=['protocol_type', 'service', 'flag']

# Encode categorical columns using LabelEncoder
label_encoder = preprocessing.LabelEncoder()
for column in categorical_columns:
    KDDTrain[column] = label_encoder.fit_transform(KDDTrain[column])
    KDDTest[column] = label_encoder.transform(KDDTest[column])

# Define the columns to scale
columns_to_scale=['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

# Scale numerical columns using MinMax
scaler = MinMaxScaler()
for column in columns_to_scale:
    KDDTrain[column] = scaler.fit_transform(KDDTrain[[column]])
    KDDTest[column] = scaler.transform(KDDTest[[column]])

# Drop 'class' from X and make the Target Variable Y equal to 'class'
X_train = KDDTrain.iloc[:, :-1].values.astype('float32')
y_train = KDDTrain.iloc[:, -1].values
X_test = KDDTest.iloc[:, :-1].values.astype('float32')
y_test = KDDTest.iloc[:, -1].values

#%%

""" Class Imbalance Check """

# Check for class imbalance (if any)
KDDTrain['class'].value_counts()

#%%

""" Class Filtering (remove 'normal' class to make it 4-class) """

# Class to drop 'normal' or '4' for 4 class classification
class_to_drop = 4

# Create a mask to filter out the samples belonging to the specified class
mask_train = y_train != class_to_drop
mask_test = y_test != class_to_drop

# Filter the data based on the mask
y_train_filtered = y_train[mask_train]
X_train_filtered = X_train[mask_train]

y_test_filtered = y_test[mask_test]
X_test_filtered = X_test[mask_test]

#%%

# Do not need to run this section if you have the saved model (.keras file)
""" Deep Neural Network for 4-class Classification (Softmax Regression)""" 

# Import necessary libraries
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as pyplot

# Number of classes 0 = DoS, 1 = Probe, 2 = R2L, 3 = U2R | lexicographic order | 4 class classification
n_classes = 4
y_train_encoded = to_categorical(y_train_filtered, num_classes=n_classes)
y_test_encoded = to_categorical(y_test_filtered, num_classes=n_classes)

# Number of features in the input data (40 total features)
n_inputs = 40

# Define the input layer
visible = Input(shape=(n_inputs,))

# Hidden Layer 1
e = Dense(80, activation='relu')(visible)  # 80 neurons with ReLU activation

# Hidden layer 2
e = Dense(40, activation='relu')(e) # 40 neurons with ReLU activation

# Hidden Layer 3
e = Dense(4, activation='relu')(e) # 4 neurons with ReLU activation

# Output Layer
output = Dense(4, activation='softmax')(e) # Condensed to 4 neurons (for 4 classes)

# Define the autoencoder model
model = Model(inputs=visible, outputs=output)

# Cast the input data to float32
X_train_filtered = X_train_filtered.astype('float32')
X_test_filtered = X_test_filtered.astype('float32')

# Possible Better performance when a fixed learning rate is NOT used with Adam Optimizer, however not as stable/consistent overall
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping with a patience of 6 steps
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Fit the autoencoder model to reconstruct input with batch size of 32 and 7 epochs
history = model.fit(X_train_filtered, y_train_encoded, epochs=7, batch_size=32, verbose=2, validation_split=0.15, callbacks=[early_stopping])

# Plot training loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.plot(history.history['accuracy'], label='train_accuracy')
pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.show()

# Define a deep network model
neural_network = Model(inputs=visible, outputs=output)
plot_model(neural_network, 'nsl-kdd_4class.png', show_shapes=True)

# Save the neural_network model in Keras format
neural_network.save('nsl-kdd_4class.keras')

#--------------------------------------------------------------------------------------------------------------------
# SoftMax Regression Multiclass Classification (4 class) (for testing purposes)

# Make predictions on the test data
y_pred = neural_network.predict(X_test_filtered)

# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded true labels to class labels
y_test_classes = np.argmax(y_test_encoded, axis=1)

# Print classification report and confusion matrix on the test set
class_names = ["DoS", "Probe", "R2L", "U2R"]
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes), "\n")

# Calculate MCC
mcc_score = matthews_corrcoef(y_test_classes, y_pred_classes)
print("MCC Score:", mcc_score)

#%%

""" Run Softmax Regression & Test Performance + K-Fold Cross-Validation """

# CHANGE AS NEEDED
local_keras_path = r'C:\Users\Leo\Desktop\Senior Design II\ml_models\intrusion_classification\src\nsl-kdd_4class_final.keras'

# Load the model from file
encoder = load_model(local_keras_path)

# Encode the training and testing data
X_train_encoded = encoder.predict(X_train_filtered)
X_test_encoded = encoder.predict(X_test_filtered)

# Create a softmax regression model
smr = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, n_jobs=-1)

# Train the model on the entire training set
smr.fit(X_train_encoded, y_train_filtered)

# Evaluate the model on the test set
y_pred = smr.predict(X_test_encoded)

# Perform 10-fold cross-validation on the entire dataset
cv_scores = cross_val_score(smr, X_train_encoded, y_train_filtered, cv=10, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation scores:\n", cv_scores)
print("\nMean CV Score:", cv_scores.mean())

# Calculate Matthews correlation coefficient (MCC)
print("\nMatthews correlation coefficient (MCC):", matthews_corrcoef(y_test_filtered, y_pred))

# Print classification report and confusion matrix on the test set
print("\nClassification Report:")
report = classification_report(y_test_filtered, y_pred, digits=4) # modify digits parameter for futher decimal places
print(report)

#%%

""" Saving the Model for future use in Scoring Scripts and/or Mobile App """

# Serialize the trained model using pickle
with open('intrusionClassification.pkl', 'wb') as f:
    pickle.dump(smr, f)


#%%

# Visualize Results (Heatmap Confusion Matrix)

def plot_confusion_matrix_heatmap_with_values(cm, classes, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    sorted_classes = sorted(list(classes))  # Convert set to list and sort it
    plt.xticks(tick_marks, sorted_classes, rotation=45)
    plt.yticks(tick_marks, sorted_classes)

    fmt = '.2f' if cm.max() < 1 else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')

class_numbers = {0,1,2,3}
class_names = {'DoS', 'Probe', 'R2L', 'U2R'}

# Original Confusion Matrix
cm_original = confusion_matrix(y_test_filtered, y_pred)
plot_confusion_matrix_heatmap_with_values(cm_original, class_names, 'Intrusion Classification')

# Call plt.tight_layout() before saving
plt.tight_layout()

plt.savefig('Heatmap_400dpi.png', dpi=400)  # Saving with 400 dpi