from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from azureml.core import Run

# Constants

dataset_name = "edu_heart_failure_dataset"
target_column = "DEATH_EVENT"

# Get dataset

run = Run.get_context()
dataset = run.input_datasets[dataset_name]

# Stratified train/test split

patients = dataset.to_pandas_dataframe()

X = patients.drop(columns=target_column)
y = patients[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# Normalization

scalable_features = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time"
]

scaler = preprocessing.StandardScaler().fit(X_train[scalable_features])

X_train[scalable_features] = scaler.transform(X_train[scalable_features])
X_test[scalable_features] = scaler.transform(X_test[scalable_features])

# Train random forest classifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
##
