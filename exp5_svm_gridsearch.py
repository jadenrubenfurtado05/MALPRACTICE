
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# SVM without GridSearchCV
model = make_pipeline(StandardScaler(), SVC())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Hyperparameter tuning
params = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    params,
    cv=5
)

grid.fit(X_train, y_train)

# Best model prediction
y_grid = grid.predict(X_test)

print("\nBest Parameters:", grid.best_params_)
print("Tuned SVM Accuracy:", accuracy_score(y_test, y_grid))
print("Tuned Confusion Matrix:\n", confusion_matrix(y_test, y_grid))
