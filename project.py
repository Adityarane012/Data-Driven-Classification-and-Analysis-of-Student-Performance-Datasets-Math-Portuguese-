    # Step - 1: Load the Datasets
# 1. Import libraries
import pandas as pd
import numpy as np

# 2. Load datasets 
student_mat = pd.read_csv("student-mat.csv", sep=";")
student_por = pd.read_csv("student-por.csv", sep=";")

# Function to inspect any dataset
def inspect_dataset(df, name="Dataset"):
    print(f"\n===== {name} =====")

    # Shape & Columns
    print("Shape (rows, columns):", df.shape)
    print("Columns:", list(df.columns))

    # Data Types & Nulls
    print("\nInfo:")
    print(df.info())

    # Preview Data
    print("\nHead (first 5 rows):")
    print(df.head())
    print("\nTail (last 5 rows):")
    print(df.tail())

    # Summary Stats (numerical)
    print("\nSummary Statistics (numeric):")
    print(df.describe())

    # Missing Values
    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    # Unique Values (categorical inspection)
    print("\nUnique Values per Column:")
    print(df.nunique())

    # Duplicates
    print("\nNumber of duplicate rows:", df.duplicated().sum())

    # Value Counts Example (for some categorical columns if present)
    for col in ["school", "sex", "address", "famsize", "Pstatus"]:
        if col in df.columns:
            print(f"\nValue Counts for {col}:")
            print(df[col].value_counts())


# 3. Run inspection for both datasets
inspect_dataset(student_mat, name="Math Dataset")
inspect_dataset(student_por, name="Portuguese Dataset")
# Step - 3: Run EDA on error introduced data sets
import pandas as pd
# Load error datasets
mat_error = pd.read_csv("student-mat-error.csv", sep=";")
por_error = pd.read_csv("student-por-error.csv", sep=";")

# Basic Inspection

print("Shapes:")
print("MAT:", mat_error.shape)
print("POR:", por_error.shape, "\n")

print("Column names:")
print(mat_error.columns.tolist(), "\n")

print("Data types:")
print(mat_error.dtypes, "\n")

print("First 5 rows (MAT):")
print(mat_error.head(), "\n")

print("Summary Statistics (numeric):")
print(mat_error.describe(), "\n")

print("Summary Statistics (categorical):")
print(mat_error.describe(include=['object']), "\n")

# Null Value Check
print("Missing Values:")
print("MAT:\n", mat_error.isnull().sum(), "\n")
print("POR:\n", por_error.isnull().sum(), "\n")

# Unique Values (sample categorical inspection)
print("Unique values in 'school' (MAT):", mat_error['school'].unique())
print("Unique values in 'sex' (POR):", por_error['sex'].unique(), "\n")

# Outlier Check
print("Absences in MAT - max:", mat_error['absences'].max())
print("Some values:", mat_error['absences'].sample(10).tolist(), "\n")

# Domain Error Checks
def check_domain_errors(df, name):
    print(f"Domain Errors in {name}:")

    # Categorical rules
    categorical_rules = {
        "school": {"GP", "MS"},
        "sex": {"F", "M"},
        "address": {"U", "R"},
        "famsize": {"LE3", "GT3"},
        "Pstatus": {"T", "A"},
        "Mjob": {"teacher", "health", "services", "at_home", "other"},
        "Fjob": {"teacher", "health", "services", "at_home", "other"},
        "reason": {"home", "reputation", "course", "other"},
        "guardian": {"mother", "father", "other"},
        "schoolsup": {"yes", "no"},
        "famsup": {"yes", "no"},
        "paid": {"yes", "no"},
        "activities": {"yes", "no"},
        "nursery": {"yes", "no"},
        "higher": {"yes", "no"},
        "internet": {"yes", "no"},
        "romantic": {"yes", "no"},
    }

    for col, valid_values in categorical_rules.items():
        if col in df.columns:
            invalid = df[~df[col].isin(valid_values)][col].unique()
            if len(invalid) > 0:
                print(f"- Invalid values in {col}: {invalid}")

    # Numeric rules
    numeric_rules = {
        "age": (15, 22),
        "Medu": (0, 4),
        "Fedu": (0, 4),
        "traveltime": (1, 4),
        "studytime": (1, 4),
        "failures": (0, 4),
        "famrel": (1, 5),
        "freetime": (1, 5),
        "goout": (1, 5),
        "Dalc": (1, 5),
        "Walc": (1, 5),
        "health": (1, 5),
        "absences": (0, 93),
        "G1": (0, 20),
        "G2": (0, 20),
        "G3": (0, 20),
    }

    for col, (low, high) in numeric_rules.items():
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            invalid = df[(numeric_col < low) | (numeric_col > high)][col].unique()
            if len(invalid) > 0:
                print(f"- Out-of-range values in {col}: {invalid}")

# Run domain error checks
check_domain_errors(mat_error, "Math")
check_domain_errors(por_error, "Portuguese")
# Step - 4: Clean the datasets
import pandas as pd
import numpy as np

# Load error datasets
mat_error = pd.read_csv("student-mat-error.csv", sep=";")
por_error = pd.read_csv("student-por-error.csv", sep=";")

# Function to clean a dataset
def clean_dataset(df, name="Dataset"):
    print(f"\nCleaning {name}")

    # 1. Handle string / invalid ages
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # 2. Impute missing values
    # Numeric: use median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"- Filled missing numeric values in {col} with median ({median_val})")

    # Categorical: use mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"- Filled missing categorical values in {col} with mode ({mode_val})")

    # 3. Fix categorical inconsistencies
    if "school" in df.columns:
        df["school"] = df["school"].replace("gp", "GP")
    if "sex" in df.columns:
        df["sex"] = df["sex"].replace("U", np.nan)  # Treat as missing
        df["sex"].fillna(df["sex"].mode()[0], inplace=True)

    # 4. Handle outliers
    if "absences" in df.columns:
        df.loc[df["absences"] > 93, "absences"] = 93

    # 5. Final check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"- Dropped {duplicates} duplicate rows")

    return df

# Clean both datasets
mat_clean = clean_dataset(mat_error.copy(), "Math")
por_clean = clean_dataset(por_error.copy(), "Portuguese")

# Save cleaned versions
mat_clean.to_csv("student-mat-clean.csv", sep=";", index=False)
por_clean.to_csv("student-por-clean.csv", sep=";", index=False)

print("\nCleaned datasets saved as 'student-mat-clean.csv' and 'student-por-clean.csv'")
# Step - 5: Run EDA on clean data sets
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned datasets
mat_clean = pd.read_csv("student-mat-clean.csv", sep=";")
por_clean = pd.read_csv("student-por-clean.csv", sep=";")

# Reuse the domain error check function from before
def check_domain_errors(df, name):
    print(f"\nDomain Errors in {name}:")

    categorical_rules = {
        "school": {"GP", "MS"},
        "sex": {"F", "M"},
        "address": {"U", "R"},
        "famsize": {"LE3", "GT3"},
        "Pstatus": {"T", "A"},
        "Mjob": {"teacher", "health", "services", "at_home", "other"},
        "Fjob": {"teacher", "health", "services", "at_home", "other"},
        "reason": {"home", "reputation", "course", "other"},
        "guardian": {"mother", "father", "other"},
        "schoolsup": {"yes", "no"},
        "famsup": {"yes", "no"},
        "paid": {"yes", "no"},
        "activities": {"yes", "no"},
        "nursery": {"yes", "no"},
        "higher": {"yes", "no"},
        "internet": {"yes", "no"},
        "romantic": {"yes", "no"},
    }

    for col, valid_values in categorical_rules.items():
        if col in df.columns:
            invalid = df[~df[col].isin(valid_values)][col].unique()
            if len(invalid) > 0:
                print(f"- Invalid values in {col}: {invalid}")

    numeric_rules = {
        "age": (15, 22),
        "Medu": (0, 4),
        "Fedu": (0, 4),
        "traveltime": (1, 4),
        "studytime": (1, 4),
        "failures": (0, 4),
        "famrel": (1, 5),
        "freetime": (1, 5),
        "goout": (1, 5),
        "Dalc": (1, 5),
        "Walc": (1, 5),
        "health": (1, 5),
        "absences": (0, 93),
        "G1": (0, 20),
        "G2": (0, 20),
        "G3": (0, 20),
    }

    for col, (low, high) in numeric_rules.items():
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            invalid = df[(numeric_col < low) | (numeric_col > high)][col].unique()
            if len(invalid) > 0:
                print(f"- Out-of-range values in {col}: {invalid}")

# Run checks again
check_domain_errors(mat_clean, "Math (Cleaned)")
check_domain_errors(por_clean, "Portuguese (Cleaned)")

# Basic inspection
print("\nShapes (Cleaned):")
print("MAT:", mat_clean.shape)
print("POR:", por_clean.shape)

print("\nUnique values in 'school' (MAT):", mat_clean['school'].unique())
print("Unique values in 'sex' (POR):", por_clean['sex'].unique())

print("\nMax absences (MAT):", mat_clean['absences'].max())


# Histogram for age distribution
plt.figure(figsize=(6,4))
mat_clean["age"].hist(bins=range(15,23), edgecolor="black")
plt.title("Age Distribution - Math")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Bar plot for school distribution
plt.figure(figsize=(5,4))
mat_clean["school"].value_counts().plot(kind="bar")
plt.title("School Distribution - Math")
plt.xlabel("School")
plt.ylabel("Count")
plt.show()

# Histogram for age distribution
plt.figure(figsize=(6,4))
por_clean["age"].hist(bins=range(15,23), edgecolor="black")
plt.title("Age Distribution - Portuguese")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# Bar plot for school distribution
plt.figure(figsize=(5,4))
por_clean["school"].value_counts().plot(kind="bar")
plt.title("School Distribution - Portuguese")
plt.xlabel("School")
plt.ylabel("Count")
plt.show()

# Compare G3 final grade distribution between Math and Portuguese
plt.figure(figsize=(7,4))
mat_clean["G3"].plot(kind="kde", label="Math")
por_clean["G3"].plot(kind="kde", label="Portuguese")
plt.title("Final Grade (G3) Distribution - Cleaned Datasets")
plt.xlabel("Final Grade (0-20)")
plt.legend()
plt.show()
#Step - 6: Integrate both the datasets
import pandas as pd
import matplotlib.pyplot as plt

# Reload clean datasets
mat_clean = pd.read_csv("student-mat.csv", sep=";")
por_clean = pd.read_csv("student-por.csv", sep=";")

# Merge keys (common identifying columns)
merge_keys = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic"
]

# Merge datasets
students_merged = pd.merge(
    mat_clean, por_clean,
    on=merge_keys,
    suffixes=("_mat", "_por")
)

print("Merged shape:", students_merged.shape)
print(students_merged.head())
students_merged.to_csv("student-merged.csv", sep=";", index=False)

# 1. Scatterplot of final grades in Math vs Portuguese
plt.figure(figsize=(6,5))
plt.scatter(students_merged["G3_mat"], students_merged["G3_por"], alpha=0.6)
plt.title("Math vs Portuguese Final Grades")
plt.xlabel("Final Grade - Math (G3)")
plt.ylabel("Final Grade - Portuguese (G3)")
plt.grid(True)
plt.show()

# 2. Distribution of average grade across both subjects
students_merged["G3_avg"] = (students_merged["G3_mat"] + students_merged["G3_por"]) / 2
plt.figure(figsize=(6,4))
students_merged["G3_avg"].hist(bins=15, edgecolor="black")
plt.title("Average Final Grade Distribution (Math + Portuguese)")
plt.xlabel("Average Grade")
plt.ylabel("Frequency")
plt.show()

# 3. Compare distributions by gender
plt.figure(figsize=(6,4))
students_merged.groupby("sex")["G3_avg"].plot(kind="kde", legend=True)
plt.title("Average Grade Distribution by Gender")
plt.xlabel("Average Final Grade")
plt.show()

# 4. Correlation check
corr = students_merged[["G3_mat", "G3_por"]].corr()
print("\nCorrelation between Math and Portuguese Final Grades:")
print(corr)
#Step - 7: Feature Engineering
import pandas as pd
import numpy as np

# Assume your merged dataset is 'students_merged'

# 1. Average final grade
students_merged["G3_avg"] = (students_merged["G3_mat"] + students_merged["G3_por"]) / 2

# 2. Grade improvement (Portuguese better than Math)
students_merged["improved_por_over_math"] = students_merged["G3_por"] > students_merged["G3_mat"]

# 3. Pass/fail indicator (using G3_avg)
students_merged["passed"] = students_merged["G3_avg"] >= 10  # True if average >= 10

# 4. Grade categories
students_merged["G3_category"] = pd.cut(
    students_merged["G3_avg"], 
    bins=[0, 9, 14, 20], 
    labels=["Low", "Medium", "High"]
)

# 5. Total alcohol consumption (weekday + weekend)
students_merged["total_alc"] = students_merged["Dalc_mat"] + students_merged["Dalc_por"]
students_merged["weekend_alc"] = students_merged["Walc_mat"] + students_merged["Walc_por"]

# 6. Study effort indicator (columns without suffix)
students_merged["study_effort"] = students_merged["studytime"]  # You can also combine with traveltime or failures if desired

# 7. Absence flag for high absences
students_merged["high_absences"] = (students_merged["absences_mat"] + students_merged["absences_por"]) > 20

# 8. Family support flag
students_merged["any_support"] = (
    (students_merged["schoolsup"] == "yes") |
    (students_merged["famsup"] == "yes") |
    (students_merged["paid"] == "yes") |
    (students_merged["activities"] == "yes")
)

# 9. Average of health, social, and free time (columns with suffix)
students_merged["avg_health_social"] = (
    students_merged["health_mat"] + students_merged["health_por"] +
    students_merged["goout_mat"] + students_merged["goout_por"] +
    students_merged["freetime_mat"] + students_merged["freetime_por"]
) / 6

# 10. Binary indicator for tech access at home
students_merged["has_internet"] = (students_merged["internet"] == "yes")

# 11. Combined parental education (columns without suffix)
students_merged["parent_edu_sum"] = students_merged["Medu"] + students_merged["Fedu"]

# 12. Weekend/weekday alcohol risk flags
students_merged["alc_risk_weekday"] = students_merged["Dalc_mat"] > 3
students_merged["alc_risk_weekend"] = students_merged["Walc_mat"] > 3

# 13. High and low performer flags
students_merged["high_performer"] = students_merged["G3_avg"] >= 15
students_merged["low_performer"] = students_merged["G3_avg"] < 10

# 14. Average of first two period grades (columns with suffix)
students_merged["G1G2_avg"] = (
    students_merged["G1_mat"] + students_merged["G1_por"] +
    students_merged["G2_mat"] + students_merged["G2_por"]
) / 4

# Quick check of new features
cols_to_check = [
    "G3_avg", "G3_category", "improved_por_over_math", "high_absences",
    "any_support", "parent_edu_sum", "high_performer", "low_performer",
    "total_alc", "weekend_alc", "study_effort", "G1G2_avg"
]

print(students_merged[cols_to_check].head())
#Step - 8: Feature visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for nicer plots
sns.set(style="whitegrid")

# 1. Distribution of Average Final Grade
plt.figure(figsize=(6,4))
sns.histplot(students_merged["G3_avg"], bins=15, kde=True, color="skyblue")
plt.title("Average Final Grade Distribution")
plt.xlabel("Average Grade")
plt.ylabel("Frequency")
plt.show()

# 2. Count of G3 Categories
plt.figure(figsize=(6,4))
sns.countplot(x="G3_category", data=students_merged, palette="pastel")
plt.title("Number of Students in Each Grade Category")
plt.xlabel("Grade Category")
plt.ylabel("Count")
plt.show()

# 3. Scatter plot: Parent Education vs Average Grade
plt.figure(figsize=(6,4))
sns.scatterplot(x="parent_edu_sum", y="G3_avg", data=students_merged, hue="sex")
plt.title("Parental Education vs Average Grade")
plt.xlabel("Sum of Parental Education (Medu + Fedu)")
plt.ylabel("Average Final Grade")
plt.show()

# 4. Study effort vs Average Grade
plt.figure(figsize=(6,4))
sns.boxplot(x="studytime", y="G3_avg", data=students_merged, palette="Set2")
plt.title("Study Time vs Average Grade")
plt.xlabel("Weekly Study Time (1-4)")
plt.ylabel("Average Final Grade")
plt.show()

# 5. Total Alcohol vs Average Grade
plt.figure(figsize=(6,4))
sns.scatterplot(x="total_alc", y="G3_avg", data=students_merged, hue="sex")
plt.title("Total Alcohol Consumption vs Average Grade")
plt.xlabel("Total Alcohol Consumption (Dalc + Walc)")
plt.ylabel("Average Final Grade")
plt.show()

# 6. High Absences vs Average Grade
plt.figure(figsize=(6,4))
sns.boxplot(x="high_absences", y="G3_avg", data=students_merged)
plt.title("Impact of High Absences on Average Grade")
plt.xlabel("High Absences (True/False)")
plt.ylabel("Average Final Grade")
plt.show()

# 7. Support vs Average Grade
plt.figure(figsize=(6,4))
sns.boxplot(x="any_support", y="G3_avg", data=students_merged, palette="Set3")
plt.title("Impact of Family/School Support on Average Grade")
plt.xlabel("Has Support (True/False)")
plt.ylabel("Average Final Grade")
plt.show()

# 8. KDE by Gender for Average Grade
plt.figure(figsize=(6,4))
sns.kdeplot(data=students_merged, x="G3_avg", hue="sex", fill=True, alpha=0.4)
plt.title("Average Grade Distribution by Gender")
plt.xlabel("Average Grade")
plt.show()

# 9. Correlation Heatmap of Numeric Features
numeric_features = [
    "G3_avg", "G1G2_avg", "parent_edu_sum", "studytime",
    "total_alc", "weekend_alc", "absences_mat", "absences_por"
]
plt.figure(figsize=(8,6))
sns.heatmap(students_merged[numeric_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Numeric Features")
plt.show()
#Step - 9: Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Prepare features and target

# Selecting numeric features only for regression
features = [
    "studytime", "failures", "parent_edu_sum", 
    "total_alc", "weekend_alc", "absences_mat", "absences_por",
    "G1G2_avg"
]

X = students_merged[features]
y = students_merged["G3_avg"]

# Fill any missing values with median (just in case)
X = X.fillna(X.median())

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Initialize and train model
reg = LinearRegression()
reg.fit(X_train, y_train)

# 4. Predictions
y_pred = reg.predict(X_test)

# 5. Evaluate model

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")


# 6. Coefficients
coeff_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": reg.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nFeature Coefficients:")
print(coeff_df)

# Set style
sns.set(style="whitegrid")

# 1. Predicted vs Actual
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot([0, 20], [0, 20], 'r--')  # perfect prediction line
plt.xlabel("Actual G3_avg")
plt.ylabel("Predicted G3_avg")
plt.title("Actual vs Predicted Average Grade")
plt.grid(True)
plt.show()

# 2. Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, bins=15, kde=True, color="orange")
plt.title("Residual Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

# 3. Residuals vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.6, color="green")
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted G3_avg")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid(True)
plt.show()
#Step - 10: Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create target for classification
students_merged["pass_fail"] = students_merged["G3_avg"].apply(lambda x: 1 if x >= 10 else 0)

# Features (reuse numeric ones)
features = [
    "studytime", "failures", "parent_edu_sum", 
    "total_alc", "weekend_alc", "absences_mat", "absences_por",
    "G1G2_avg"
]

X = students_merged[features]
y = students_merged["pass_fail"]

# Fill missing
X = X.fillna(X.median())

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Logistic Regression-
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 4. Predictions & Probabilities
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

# 5. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Pass/Fail")
plt.show()

# 6. ROC Curve

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Pass/Fail Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 7. Coefficients
coeff_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": clf.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("Feature Coefficients:")
print(coeff_df)
#Step - 11: Interpretation
# Regression Interpretation

print("REGRESSION RESULTS\n")
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# RMSE and R-squared already calculated
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print("Interpretation: Lower RMSE means the predicted grades are close to actual grades.\n")

print(f"R-squared: {r2:.2f}")
print("Interpretation: R-squared shows the proportion of variance in average grades explained by the features.")
if r2 < 0.3:
    print("Model explains little of the variance.")
elif r2 < 0.6:
    print("Model explains a moderate amount of the variance.")
else:
    print("Model explains a high amount of the variance.\n")

# Feature impact
print("Feature Impact on Average Grade:")
for feat, coef in zip(features, reg.coef_):
    print(f"- {feat}: {'+' if coef > 0 else '-'} {abs(coef):.2f}")
print("Interpretation: Positive coefficient increases predicted grade; negative decreases it.\n")


# Classification Interpretation
print("CLASSIFICATION RESULTS\n")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred_class = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {accuracy:.2f}")
print("Interpretation: Proportion of students correctly classified as Pass/Fail.\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_class)
print("Confusion Matrix:")
print(cm)
print("\nInterpretation:")
print(f"- True Positives (Pass correctly predicted): {cm[1,1]}")
print(f"- True Negatives (Fail correctly predicted): {cm[0,0]}")
print(f"- False Positives (Fail predicted as Pass): {cm[0,1]}")
print(f"- False Negatives (Pass predicted as Fail): {cm[1,0]}\n")

# Classification report
report = classification_report(y_test, y_pred_class, target_names=["Fail","Pass"])
print("Detailed Classification Report:")
print(report)
print("Interpretation: Precision = accuracy of positive predictions, Recall = coverage of actual positives, F1 = balance between precision and recall.\n")

# Feature importance
print("Feature Influence on Passing (Classification Coefficients):")
for feat, coef in zip(features, clf.coef_[0]):
    print(f"- {feat}: {'+' if coef > 0 else '-'} {abs(coef):.2f}")
print("Interpretation: Positive coefficient increases odds of passing; negative decreases it.")
