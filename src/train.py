import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from ydata_profiling import ProfileReport

data = pd.read_csv(r"D:\PYTHON\Khoa ML VietnguyenAI\personal prj\student-performance-prediction\dataset\StudentScore.csv")
profile = ProfileReport(data, title="studentcore report", explorative=True)
profile.to_file("studentcore_report.html")

target = "writing score"

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Preprocessing numerical data
num_tranformer = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing ordinal data
educationvalues = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
gendervalues = ["male", "female"]
lunchvalues = ["free/reduced", "standard"]
test_preparation = ["none", "completed"]

ord_tranformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[educationvalues, gendervalues, lunchvalues, test_preparation]))
])

# Preprocessing nominal data
nom_tranformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder())
])

# result = num_tranformer.fit_transform(x_train[["math score", "reading score"]])
# for i,j in zip(x_train[["math score", "reading score"]].values, result):
#     print("before: {}. after: {}".format(i,j))
#
# # result = ord_tranformer.fit_transform(x_train[["parental level of education", "gender", "lunch", "test preparation course"]])
# for i, j in zip(x_train[["parental level of education"]].values, result):
#     print("before: {}. after: {}".format(i,j))

# result = nom_tranformer.fit_transform(x_train[["race/ethnicity"]])
# for i, j in zip(x_train[["race/ethnicity"]].values, result.toarray()):
#     print("before: {}. after: {}".format(i,j))

preprocessor = ColumnTransformer(transformers=[
    ("num_feature", num_tranformer, ["math score", "reading score"]),
    ("ord_feature", ord_tranformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_feature", nom_tranformer, ["race/ethnicity"]),
])

regression = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor())
])

param = {
    "model__n_estimators": [100, 200, 300],
    "model__criterion": ['friedman_mse', 'squared_error', 'poisson', 'absolute_error']
}
gridsearch = GridSearchCV(estimator=regression, param_grid=param, cv=5, scoring="r2", verbose=2) # Try parameter combinations with 5-fold CV and select the best R2
gridsearch.fit(x_train, y_train)
print("Best parameters:")
print(gridsearch.best_params_)

print("Best CV R2:")
print(gridsearch.best_score_)

best_model = gridsearch.best_estimator_
y_predict = best_model.predict(x_test)

print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))