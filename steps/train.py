import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import pandas as pd

from azureml.core import Run

# Constants

dataset_name = "edu_heart_failure_dataset"
target_column = "DEATH_EVENT"
primary_metric_name = "mean accuracy"

# Get command-line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, help="Number of trees in the forest")
parser.add_argument("--min_samples_split", type=float, help="Minimum fraction of samples required to split an internal node")
args = parser.parse_args()

# Log hyperparameters

run = Run.get_context()

run.log("Number of trees in the forest", int(args.n_estimators))
run.log("Minimum fraction of samples required to split an internal node", float(args.min_samples_split))

# Get dataset

url = "https://raw.githubusercontent.com/nemarona/nd00333-capstone/master/heart_failure_clinical_records_dataset.csv"
patients = pd.read_csv(url)

# Stratified train/test split

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
    "time",
]

scaler = StandardScaler().fit(X_train[scalable_features])

X_train[scalable_features] = scaler.transform(X_train[scalable_features])
X_test[scalable_features] = scaler.transform(X_test[scalable_features])

# Train random forest classifier

clf = RandomForestClassifier(
    n_estimators=args.n_estimators, min_samples_split=args.min_samples_split
)

clf.fit(X_train, y_train)

# Evaluate

test_score = clf.score(X_test, y_test)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Log primary metric

run.log(primary_metric_name, float(test_score))

# Log confusion matrix

json_value = {
       "schema_type": "confusion_matrix",
       "schema_version": "1.0.0",
       "data": {
           "class_labels": ["0", "1"],
           "matrix": cm.tolist()
       }
   }

run.log_confusion_matrix(name="confusion matrix", value=json_value)

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
