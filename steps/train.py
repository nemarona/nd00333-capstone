import argparse
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from azureml.core import Run

# Get command-line arguments

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int)
parser.add_argument("--min_samples_split", type=float)
args = parser.parse_args()

# Constants

dataset_name = "edu_heart_failure_dataset"
target_column = "DEATH_EVENT"
primary_metric_name = "mean accuracy"

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

# Log primary metric

run.log(primary_metric_name, float(test_score))

# Save model

save_path = Path("./saved_models")
save_path.mkdir(exist_ok=True)

filename = f"rf_clf_n_est_{args.n_estimators}_min_smp_split_{args.min_samples_split:.4}.joblib"
filepath = save_path / filename

joblib.dump(clf, filepath)

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
