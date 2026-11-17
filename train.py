# %% [markdown]
# # Medical Insurance Charges - Project 1
# Stages 1-4 (data description, preprocessing/analysis, modeling/tuning, evaluation).
# Run top to bottom (`python train.py`) or cell by cell in VS Code / Jupyter (`# %%` cells).
# Stage 5 (deployment) lives in `app.py`.

# %%
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # remove this line if running interactively in a notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import BINARY, CATEGORICAL, FEATURES, NUMERIC, add_features, clean

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEED = 42
ROOT = Path(__file__).parent
FIG = ROOT / "figures"
MODELS = ROOT / "models"
FIG.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=120)
    plt.close()


# %% [markdown]
# ## Stage 1 - Data description
# **Source:** the public "Medical Cost Personal Datasets" insurance data (Kaggle, `insurance.csv`).
# **Problem:** predict the yearly medical charge billed by a health insurer for a person,
# from basic demographic and lifestyle information.
# **Task type:** supervised **regression**; target = `charges` (USD).
#
# | Feature | Meaning |
# |---|---|
# | age | age of the primary beneficiary (18-64) |
# | sex | female / male |
# | bmi | body mass index (kg/m^2); >= 30 is conventionally "obese" |
# | children | number of dependents covered |
# | smoker | yes / no |
# | region | US residential area: northeast, northwest, southeast, southwest |
# | charges | **target**: individual medical costs billed by insurance |

# %%
raw = pd.read_csv(ROOT / "data" / "insurance.csv")
print("Shape (rows, columns):", raw.shape)
print("\nData types:\n", raw.dtypes)
print("\nNumeric summary:\n", raw.describe().round(2).to_string())
print("\nCategorical summary:\n", raw.describe(include="object").to_string())

# %% [markdown]
# ## Stage 2 - Preprocessing and analysis
# ### 2.1 Data quality: missing values, duplicates, inconsistent entries

# %%
print("Missing values per column:\n", raw.isna().sum().to_string())
print("\nDuplicate rows:", raw.duplicated().sum())

# Inconsistent entries: trim whitespace + lowercase, then list the unique values.
df = clean(raw)
for c in ["sex", "smoker", "region"]:
    print(f"{c}: {sorted(df[c].unique())}")

# Handling:
#  * missing values: none -> nothing to impute (an imputer would still be placed
#    inside the pipeline if the app ever had to accept blanks).
#  * duplicates: 1 exact duplicate row is dropped BEFORE the split so the same
#    record cannot land in both train and test.
#  * inconsistent text: normalised by `clean()`; categories were already consistent.
df = df.drop_duplicates().reset_index(drop=True)
print("\nShape after dropping duplicates:", df.shape)

# %% [markdown]
# ### 2.2 Train / test split (done BEFORE any fitted step, to avoid data leakage)
# Everything that learns from data (scaler, encoder, model, grid search) is fitted on the
# **training set only**. The test set is touched exactly once, in Stage 4.
# The split is stratified on `smoker` because smoking is the strongest driver of cost and
# only ~20% of people smoke, so both sets get the same smoker share.

# %%
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["smoker"]
)
y_train, y_test = train_df["charges"], test_df["charges"]
print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
print("Smoker share  train/test:",
      round((train_df.smoker == 'yes').mean(), 3), "/", round((test_df.smoker == 'yes').mean(), 3))

# %% [markdown]
# ### 2.3 Exploration (training data only)

# %%
# Distributions of numeric columns
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, col in zip(axes, ["age", "bmi", "children", "charges"]):
    sns.histplot(train_df[col], kde=True, bins=30, ax=ax).set_title(f"Distribution of {col}")
save_fig("01_numeric_distributions")

# Categorical counts
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["sex", "smoker", "region"]):
    sns.countplot(x=train_df[col], ax=ax).set_title(f"Count of {col}")
save_fig("02_categorical_counts")

print("Skewness:", train_df[["age", "bmi", "children", "charges"]].apply(stats.skew).round(2).to_dict())

# %% [markdown]
# `charges` is strongly right-skewed (skew > 1). The relationship plots below show why.

# %%
# Relationship with the target
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
sns.scatterplot(data=train_df, x="age", y="charges", hue="smoker", alpha=.6, ax=axes[0, 0])
axes[0, 0].set_title("Charges vs age (three age-bands, smokers far above)")
sns.scatterplot(data=train_df, x="bmi", y="charges", hue="smoker", alpha=.6, ax=axes[0, 1])
axes[0, 1].set_title("Charges vs BMI (smoker + BMI>30 jump)")
sns.boxplot(data=train_df, x="smoker", y="charges", ax=axes[0, 2]).set_title("Charges by smoker")
sns.boxplot(data=train_df, x="sex", y="charges", ax=axes[1, 0]).set_title("Charges by sex")
sns.boxplot(data=train_df, x="region", y="charges", ax=axes[1, 1]).set_title("Charges by region")
sns.boxplot(data=train_df, x="children", y="charges", ax=axes[1, 2]).set_title("Charges by children")
save_fig("03_relationship_with_target")

# Relationships between features
num_view = train_df[["age", "bmi", "children", "charges"]].copy()
num_view["smoker"] = (train_df["smoker"] == "yes").astype(int)
plt.figure(figsize=(6, 5))
sns.heatmap(num_view.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation matrix (training data)")
save_fig("04_correlation_heatmap")

# Strength of association per feature (linear = Pearson, monotonic = Spearman)
Xf_tmp = add_features(train_df)
assoc = pd.DataFrame({
    "pearson_r": [np.corrcoef(Xf_tmp[c], y_train)[0, 1] for c in ["age", "bmi", "children", "sex", "smoker"]],
    "spearman_rho": [stats.spearmanr(Xf_tmp[c], y_train)[0] for c in ["age", "bmi", "children", "sex", "smoker"]],
}, index=["age", "bmi", "children", "sex", "smoker"])
print(assoc.round(3).sort_values("spearman_rho", key=abs, ascending=False).to_string())

# %% [markdown]
# **Findings:** `smoker` is by far the strongest driver, then `age`, then `bmi`.
# `sex`, `children` and `region` are weak. Smokers form their own, much more expensive cluster,
# and within smokers charges jump when BMI >= 30 -> an *interaction* worth engineering.

# %% [markdown]
# ### 2.4 Outliers: detect, then decide
# I detect with the IQR rule, then check whether the outliers are *errors* or *real people*.

# %%
def iqr_bounds(s):
    q1, q3 = s.quantile([.25, .75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)

for col in ["age", "bmi", "charges"]:
    lo, hi = iqr_bounds(train_df[col])
    out = train_df[(train_df[col] < lo) | (train_df[col] > hi)]
    print(f"{col:8s} bounds [{lo:9.1f}, {hi:9.1f}] -> {len(out):3d} outliers "
          f"({len(out) / len(train_df):.1%}), smokers among them: {(out.smoker == 'yes').mean():.0%}")

lo, hi = iqr_bounds(train_df["charges"])
high = train_df[train_df["charges"] > hi]
print("\nHigh-charge outliers: smoker share =", f"{(high.smoker == 'yes').mean():.0%}",
      "| obese share =", f"{(high.bmi >= 30).mean():.0%}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["age", "bmi", "charges"]):
    sns.boxplot(y=train_df[col], ax=ax).set_title(f"Boxplot of {col}")
save_fig("05_outlier_boxplots")

# %% [markdown]
# **Decision: keep the outliers; do not delete them.**
# * Every flagged `charges` outlier is a plausible value, and ~98% belong to smokers (and
#   nearly all are also obese, BMI >= 30). They are the *signal* of the problem, not noise: removing them would
#   teach the model that expensive customers do not exist and make it under-price exactly the
#   people an insurer most needs to price correctly.
# * `bmi` outliers (BMI > ~47) are extreme but medically possible; there are only a handful.
# * Instead of deleting, I reduce their influence: the target is modelled on a **log scale**
#   (below) and the model is tree-based, which is robust to extreme feature values.
#
# (The earlier version of this project removed outliers with IQR + Mahalanobis. That would
# have dropped many legitimate smokers, and it also did so before the split.)

# %% [markdown]
# ### 2.5 Target transformation
# `charges` is right-skewed, so models are trained on `log1p(charges)` and predictions are
# converted back to dollars with `expm1`. `TransformedTargetRegressor` does this inside the
# model object, so the app always receives dollars and no transform can be forgotten.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(y_train, kde=True, ax=axes[0]).set_title(f"charges (skew {stats.skew(y_train):.2f})")
sns.histplot(np.log1p(y_train), kde=True, ax=axes[1]).set_title(
    f"log1p(charges) (skew {stats.skew(np.log1p(y_train)):.2f})")
save_fig("06_target_transform")

# %% [markdown]
# ### 2.6 Feature engineering, encoding and scaling
# * `obese` = BMI >= 30 and `smoker_obese` = smoker x obese: the interaction seen in the plots.
# * `sex`, `smoker`: binary 0/1. `region`: one-hot (it has no order; label encoding would
#   invent a false ranking, and target encoding on only 4 categories is unnecessary and can
#   leak the target).
# * Numeric columns are standardised. Trees do not need scaling, but it keeps the
#   preprocessing valid if the model is swapped for a linear one.
# * Both the encoder and the scaler live in a `ColumnTransformer` that is **fit on train only**.

# %%
X_train, X_test = add_features(train_df), add_features(test_df)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("bin", "passthrough", BINARY),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
])

# %% [markdown]
# ## Stage 3 - Modeling and tuning
# ### 3.1 Choosing the model
# I compare three candidates with 5-fold CV on the **training** set only.

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

def make_model(regressor):
    return Pipeline([
        ("prep", preprocessor),
        ("model", TransformedTargetRegressor(regressor, func=np.log1p, inverse_func=np.expm1)),
    ])

candidates = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
}
rows = []
for name, reg in candidates.items():
    pipe = make_model(reg)
    r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
    mae = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    rows.append({"model": name, "cv_R2": r2.mean(), "cv_R2_std": r2.std(), "cv_MAE": mae.mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %% [markdown]
# **Choice: Gradient Boosting Regressor.** Linear regression is clearly too weak (CV R2 ~0.47):
# the data has non-linear and interaction effects (smoker x BMI, age bands) it cannot represent.
# Random Forest and *default* Gradient Boosting are statistically tied (the gap is well inside
# one CV standard deviation). I pick boosting because its depth, learning rate, subsampling and
# leaf size give several regularisation knobs to tune on only ~1,000 rows, and tuning it is
# what Stage 3 asks for. Tuning does improve it (see 3.2) to match/exceed the Random Forest.

# %% [markdown]
# ### 3.2 Grid search
# The whole pipeline is searched, so scaling/encoding are re-fitted inside every CV fold
# (no leakage between folds). Scoring is RMSE in dollars, since big under-pricing errors
# are the costly ones for an insurer.

# %%
param_grid = {
    "model__regressor__n_estimators": [100, 200, 300],
    "model__regressor__learning_rate": [0.03, 0.05, 0.1],
    "model__regressor__max_depth": [2, 3, 4],
    "model__regressor__min_samples_leaf": [5, 10, 20],
    "model__regressor__subsample": [0.8, 1.0],
}
print("Parameter grid:")
for k, v in param_grid.items():
    print(f"  {k.split('__')[-1]:18s} {v}")
print("Total combinations:", np.prod([len(v) for v in param_grid.values()]), "x 5 folds")

grid = GridSearchCV(
    make_model(GradientBoostingRegressor(random_state=SEED)),
    param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1, refit=True,
)
grid.fit(X_train, y_train)

best_params = {k.split("__")[-1]: v for k, v in grid.best_params_.items()}
print("\nBest parameters:", best_params)
print(f"Best CV RMSE: {-grid.best_score_:,.0f} USD")
best = grid.best_estimator_
cv_r2 = cross_val_score(best, X_train, y_train, cv=cv, scoring="r2")
cv_mae = -cross_val_score(best, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
print(f"CV R2 (best params): {cv_r2.mean():.3f} +/- {cv_r2.std():.3f} | CV MAE: {cv_mae.mean():,.0f} USD")

# %% [markdown]
# ## Stage 4 - Evaluation (test set, used once)

# %%
pred_test = best.predict(X_test)
pred_train = best.predict(X_train)

def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": np.sqrt(mean_squared_error(y, p)),
            "R2": r2_score(y, p)}

m_test, m_train = metrics(y_test, pred_test), metrics(y_train, pred_train)
report = pd.DataFrame({"train": m_train, "test": m_test}).T
print(report.round(3).to_string())

# Baseline for context: always predict the training mean
base = metrics(y_test, np.full(len(y_test), y_train.mean()))
print(f"\nBaseline (predict mean) test MAE: {base['MAE']:,.0f}  RMSE: {base['RMSE']:,.0f}")

# Where does it struggle?
res = test_df.assign(pred=pred_test, err=pred_test - y_test, abs_err=np.abs(pred_test - y_test))
print("\nMAE by group:")
print(res.groupby("smoker")["abs_err"].mean().round(0).to_string())
print(res.groupby(pd.cut(res.charges, [0, 5000, 15000, 30000, 1e6]), observed=True)["abs_err"]
      .agg(["count", "mean"]).round(0).to_string())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(y_test, pred_test, alpha=.6)
lims = [0, max(y_test.max(), pred_test.max())]
axes[0].plot(lims, lims, "r--")
axes[0].set(xlabel="Actual charges", ylabel="Predicted charges", title="Predicted vs actual (test)")
axes[1].scatter(pred_test, pred_test - y_test, alpha=.6)
axes[1].axhline(0, color="r", ls="--")
axes[1].set(xlabel="Predicted", ylabel="Residual", title="Residuals (test)")
sns.boxplot(data=res, x="smoker", y="abs_err", ax=axes[2]).set_title("Absolute error by smoker")
save_fig("07_evaluation")

gbm = best.named_steps["model"].regressor_
names = best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(gbm.feature_importances_, index=[n.split("__")[-1] for n in names]).sort_values()
imp.plot.barh(figsize=(7, 4), title="Feature importance")
save_fig("08_feature_importance")
print("\nTop features:", imp.sort_values(ascending=False).round(3).head(4).to_dict())

# %% [markdown]
# **Reading the results.**
# * *Performance:* the tuned model explains ~85-90% of the variance (CV R2 ~0.84, test R2 ~0.93)
#   and cuts the error from ~9,000 USD (always predicting the mean) to ~1,600 USD MAE.
# * *Overfitting:* no clear sign. Train and CV scores are close and the test score is not worse
#   than train. Test R2 being *higher* than CV R2 mostly reflects an easy 268-row split, so
#   the CV figures (R2 ~0.84, MAE ~2,250 USD) are the more conservative estimate to quote.
# * *Where it struggles:* mid-to-high charges (15k-30k USD) have the largest errors (~4,000 USD).
#   These are people near the boundary between the non-smoker and smoker cost clusters, and
#   ~2 columns (smoker, age) cannot fully separate them. Smokers are also harder than non-smokers
#   in absolute terms. The model is a reasonable estimate, not an exact quote.

# %% [markdown]
# ## Stage 5 helper - save the objects
# The fitted preprocessing object and the fitted model are saved as separate pickles.

# %%
joblib.dump(best.named_steps["prep"], MODELS / "preprocessor.pkl")
joblib.dump(best.named_steps["model"], MODELS / "model.pkl")
joblib.dump({"best_params": best_params, "cv_rmse": float(-grid.best_score_),
             "test": m_test, "train": m_train}, MODELS / "metrics.pkl")
print("Saved:", [p.name for p in MODELS.glob("*.pkl")])
