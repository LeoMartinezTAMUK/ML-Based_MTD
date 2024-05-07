# Dataset Utilized: Network Security Laboratory Knowledge Discovery in Databases (NSL-KDD)
# Citation: https://www.unb.ca/cic/datasets/nsl.html
# M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,”
# Submitted to Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

# NSL-KDD Binary (2-class) Network Intrusion Detection
# Cleaning (num_outbound_cmds, su_attempted), Class Assignment (binary), Label Encoding, Feature Selection (MAD), Random Forest

# Written in Anaconda Spyder 5.5.0 IDE using Python 3.9.18 64-bit on Windows 10

# Necessary Imports
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score

#%%

"""Load Dataset(s)"""

# CHANGE AS NEEDED
local_path_train = r'C:\Users\Leo\Desktop\Senior Design II\ml_models\intrusion_detection\data\KDDTrain+.txt'

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
local_path_test = r'C:\Users\Leo\Desktop\Senior Design II\ml_models\intrusion_detection\data\KDDTest+.txt'

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

"""Clean Data"""

# We drop 'num_outbound_cmds' from both training and testing dataset because every instance is equal to 0 in both datasets
KDDTrain.drop("num_outbound_cmds",axis=1,inplace=True)
KDDTest.drop("num_outbound_cmds",axis=1,inplace=True)

# We replace all instances with a value of 2 to 1 because the feature should be a binary value (0 or 1)
KDDTrain['su_attempted'] = KDDTrain['su_attempted'].replace(2, 1)
KDDTest['su_attempted'] = KDDTest['su_attempted'].replace(2, 1)

#%%

"""Class Assignment (Multiclass -> Binary)"""

# Change training attack labels to their respective attack class for binary classification
KDDTrain['class'].replace(['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land', 'satan', 'ipsweep', 'portsweep', 'nmap',
                                 'warezclient', 'guess_passwd', 'warezmaster', 'imap', 'ftp_write', 'multihop', 'phf','spy', 'buffer_overflow', 'rootkit', 'loadmodule','perl'],'anomaly',inplace=True) # 22 subclasses

print("Configured training dataset for BINARY classification")

# Change testing attack labels to their respective attack class for binary classification
KDDTest['class'].replace(['neptune','apache2', 'processtable', 'smurf', 'back','mailbomb', 'pod', 'teardrop', 'land','worm', 'udpstorm',
                                'mscan', 'satan', 'saint', 'portsweep', 'ipsweep', 'nmap', 'guess_passwd', 'warezmaster', 'snmpguess', 'snmpgetattack',
                                 'httptunnel', 'multihop', 'named', 'sendmail', 'xlock', 'xsnoop', 'ftp_write', 'phf', 'imap',
                                'buffer_overflow', 'ps', 'rootkit', 'xterm', 'loadmodule', 'perl', 'sqlattack'],'anomaly',inplace=True) # 37 subclasses

print("Configured testing dataset for BINARY classification")

#%%

"""Preprocess Data"""

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

#%%

"""Feature Selection"""

# MAD (Mean Absolute Difference)

# Copy of the training dataset
nsl_kdd_df = KDDTrain

# We will use this value to ignore our target feature column during the for loop
target_column = 'class'

# Create an empty list to store the MAD results
mad_results = []

# Calculate the MAD of the target variable
target_median = np.median(nsl_kdd_df[target_column])
target_mad = np.mean(np.abs(nsl_kdd_df[target_column] - target_median))

# Loop through each feature column
for feature_column in nsl_kdd_df.columns:
    if feature_column != target_column:
        # Calculate the MADiff of the feature
        feature_median = np.median(nsl_kdd_df[feature_column])
        feature_mad = np.mean(np.abs(nsl_kdd_df[feature_column] - feature_median))

        # Calculate the MADiff ratio
        mad_ratio = feature_mad / target_mad

        # Store the results in the list
        mad_results.append({'Feature': feature_column, 'MAD_Ratio': mad_ratio})

# Convert the list of dictionaries to a DataFrame
mad_results_df = pd.DataFrame(mad_results)

# Sort the features by MADiff ratio in descending order (features with higher MAD ratios are more important)
mad_results_df = mad_results_df.sort_values(by='MAD_Ratio', ascending=False)


#%%

# Select the top features based on a threshold MAD ratio (e.g., top 95% of features)
top_percentage = 0.95
num_features_to_select = int(top_percentage * len(mad_results))
significant_features = mad_results_df.head(num_features_to_select)

#%%

# Print or use the significant features for further analysis
significant_features

#%%

# Print k amount of top features to view or use later on
k = 38
significant_features['Feature'].head(k).values

#%%

# Select top k features
topfeatures = significant_features['Feature'].head(k).values
selectedKDDTrain = KDDTrain[topfeatures]
selectedKDDTest = KDDTest[topfeatures]

#%%

# Assigning X and Y
X_train = selectedKDDTrain.values
y_train = KDDTrain['class'].values
X_test = selectedKDDTest.values
y_test = KDDTest['class'].values

# Feature count should now be equal to top k features selected
print("X Training Shape:", X_train.shape)
print("X Testing Shape:", X_test.shape)

#%%

# Gather an understanding of the severity of the imbalance (if any)
KDDTrain['class'].value_counts()

#%%

""" Run Random Forest & Test Performance + K-Fold Cross-Validation """

# Create a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=2, random_state=8, class_weight={0:1, 1:5}, n_jobs=-1)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=10, n_jobs=-1)

# Train the model on the entire training set
clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = clf.predict(X_test)

# Print classification report and confusion matrix on the test set
print("Classification Report:")
report = classification_report(y_test, y_pred, digits=4) # modify digits parameter for futher decimal places
print(report)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

# Calculate AUC
y_prob = clf.predict_proba(X_test)[:, 1]  # Get the probability of the positive class
auc_score = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc_score)

# Calculate MCC
mcc_score = matthews_corrcoef(y_test, y_pred)
print("MCC Score:", mcc_score)

#%%

""" Saving the Model for future use in Scoring Scripts and/or Mobile App """

# Serialize the trained model using pickle
with open('intrusionDetection.pkl', 'wb') as f:
    pickle.dump(clf, f)


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

class_numbers = {0,1}
class_names = {'intrusion', 'normal'}

# Original Confusion Matrix
cm_original = confusion_matrix(y_test, y_pred)
plot_confusion_matrix_heatmap_with_values(cm_original, class_names, 'Intrusion Detection')

# Call plt.tight_layout() before saving
plt.tight_layout()

plt.savefig('Heatmap_400dpi.png', dpi=400)  # Saving with 400 dpi