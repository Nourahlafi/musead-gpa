import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("cleaned_data.csv")
X = df[["study_hours", "attendance"]]
y = df["prev_gpa"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("depth  train_MAE  test_MAE  CV_MAE(5-fold)")
for depth in range(1, 11):
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    cv_mae = -cross_val_score(
        DecisionTreeRegressor(max_depth=depth, random_state=42),
        X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    print(f"{depth:>5}  {train_mae:>9.3f}  {test_mae:>8.3f}  {cv_mae:>14.3f}")

best = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train)
pred = best.predict(X_test)
print()
print("Chosen depth = 3")
print(f"Test MAE = {mean_absolute_error(y_test, pred):.2f}")
print(f"Test R2  = {r2_score(y_test, pred):.2f}")
