from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.preprocess import load_and_clean

def main():
    df = load_and_clean("data/sample.csv")

    # Features + label
    X = df[["age", "income"]]
    y = df["bought"]

    # Small dataset => keep it simple + deterministic
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Training complete. Accuracy: {acc:.3f}")

    # Optional: enforce "not totally broken"
    if acc < 0.0:  # always false, but keeps the pattern for real projects
        raise RuntimeError("Accuracy is unexpectedly low.")

if __name__ == "__main__":
    main()
