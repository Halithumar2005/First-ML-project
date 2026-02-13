import joblib, os # type: ignore
from sklearn.svm import SVC# type: ignore
from sklearn.ensemble import RandomForestClassifier# type: ignore
from sklearn.neighbors import KNeighborsClassifier# type: ignore
from sklearn.metrics import accuracy_score# type: ignore
from utils.feature_extractor import extract_features

train_path = "ML Project/ML Demo Proj_Dataset/Dataset/Training_set/Training_set"
test_path  = "ML Project/ML Demo Proj_Dataset/Dataset/test/test"

print("Extracting features...")
X_train, y_train = extract_features(train_path)
X_test, y_test   = extract_features(test_path)

if X_train.size == 0 or X_test.size == 0:
    raise RuntimeError(
        f"No features extracted. Check dataset paths: '{train_path}' and '{test_path}', and ensure class subfolders contain images."
    )

models = {
    "SVM": SVC(kernel="linear", probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

best_model = None
best_accuracy = 0

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc}")

    joblib.dump(model, f"models/{name}.pkl")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        joblib.dump(best_model, "models/best_model.pkl")

print("\nBest Model Saved!")
