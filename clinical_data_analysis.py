#Author: Shweta Joshi
#Purpose: Statistical analysis Demo
import pandas as pd
import numpy as np
import re
from datetime import date
import matplotlib.pyplot as plt
import json
from pathlib import Path

#Steps
# Import sample clinical data from Synthea on patient demographics and conditions
#Compute descriptive statistics to report age/gender distribution and top Dx by patients
OUTPUT_DIR = Path("/Users/shweta/PyCharmMiscProject/clinical data/analysis_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS_CSV = "/Users/shweta/PyCharmMiscProject/clinical data/patients.csv"
CONDITIONS_CSV = "/Users/shweta/PyCharmMiscProject/clinical data/conditions.csv"

# -----------------------------
# Helpers
# -----------------------------

def to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def compute_age_years(birthdate, as_of):
    """
    Exact age in years as of 'as_of' (date or Timestamp), handling NaT.
    """
    if pd.isna(birthdate):
        return np.nan
    b = pd.Timestamp(birthdate).date()
    a = pd.Timestamp(as_of).date()
    years = a.year - b.year - ((a.month, a.day) < (b.month, b.day))
    return years

def build_clinical_filter(
    exclude_keywords=None,
    exclude_regex=None,
    include_regex=None,
):
    """
    Returns a function f(description)->bool.
    - include_regex: if provided, descriptions must match this to be kept
    - exclude_keywords/exclude_regex: descriptions matching are removed
    """
    if exclude_keywords is None:
        # Heuristic list; tune for your org's definition of "clinical only"
        exclude_keywords = [
            "employment", "full-time", "part-time", "labor force", "occupation",
            "social", "social isolation", "limited social contact",
            "housing", "homeless", "education", "school",
            "medication review", "review due", "screening", "counseling",
            "referral", "administrative", "situation", "finding of",
        ]

    # Build one regex from keywords (case-insensitive)
    kw_pattern = r"|".join([re.escape(k) for k in exclude_keywords if k])
    kw_re = re.compile(kw_pattern, flags=re.IGNORECASE) if kw_pattern else None

    ex_re = re.compile(exclude_regex, flags=re.IGNORECASE) if exclude_regex else None
    in_re = re.compile(include_regex, flags=re.IGNORECASE) if include_regex else None

    def is_clinical(desc: str) -> bool:
        if desc is None or (isinstance(desc, float) and np.isnan(desc)):
            return False
        d = str(desc).strip()
        if not d:
            return False

        # If include_regex is set, enforce it first
        if in_re and not in_re.search(d):
            return False

        # Exclusions
        if kw_re and kw_re.search(d):
            return False
        if ex_re and ex_re.search(d):
            return False

        return True

    return is_clinical

# -----------------------------
# Load data
# -----------------------------
patients = pd.read_csv(PATIENTS_CSV)
conditions = pd.read_csv(CONDITIONS_CSV)
#print(patients.columns )
patients["BIRTHDATE"] = to_dt(patients["BIRTHDATE"])
patients["DEATHDATE"] = to_dt(patients.get("DEATHDATE"))
conditions["START"] = to_dt(conditions["START"])
conditions["STOP"] = to_dt(conditions.get("STOP"))

# -----------------------------
# Age analysis
# -----------------------------
analysis_date = pd.Timestamp(date.today())  # change if you want a fixed date
patients["AGE"] = patients["BIRTHDATE"].apply(lambda x: compute_age_years(x, analysis_date))

# Optional: exclude deceased if you want "current living population"
# patients_live = patients[patients["DEATHDATE"].isna()].copy()
patients_live = patients.copy()

# Age bins (customize as needed)
age_bins = [0, 18, 40, 65, 120]
age_labels = ["0-17", "18-39", "40-64", "65+"]
patients_live["AGE_GROUP"] = pd.cut(patients_live["AGE"], bins=age_bins, labels=age_labels, right=False)

age_summary = {
    "patients_count": int(patients_live["Id"].nunique()),
    "mean_age": float(patients_live["AGE"].mean()),
    "median_age": float(patients_live["AGE"].median()),
    "min_age": float(patients_live["AGE"].min()),
    "max_age": float(patients_live["AGE"].max()),
}
age_dist = (
    patients_live["AGE_GROUP"]
    .value_counts(dropna=False)
    .rename_axis("AGE_GROUP")
    .reset_index(name="COUNT")
)
age_dist["PCT"] = (age_dist["COUNT"] / age_dist["COUNT"].sum() * 100).round(1)

print("=== AGE SUMMARY ===")
print(age_summary)
print("\n=== AGE DISTRIBUTION (BINS) ===")
print(age_dist)

# Optional plot: age histogram
plt.figure()
patients_live["AGE"].dropna().plot(kind="hist", bins=20, title="Age distribution")
plt.xlabel("Age (years)")
plt.tight_layout()
#plt.show()

# Save age summary
with open(OUTPUT_DIR / "age_summary.json", "w") as f:
    json.dump(age_summary, f, indent=2)

# Save age distribution table
age_dist.to_csv(TABLES_DIR / "age_distribution.csv", index=False)

# -----------------------------
# Gender analysis
# -----------------------------
gender_dist = (
    patients_live.groupby("GENDER")["Id"]
    .nunique()
    .sort_values(ascending=False)
    .rename("COUNT")
    .reset_index()
)
gender_dist["PCT"] = (gender_dist["COUNT"] / gender_dist["COUNT"].sum() * 100).round(1)

print("\n=== GENDER DISTRIBUTION ===")
print(gender_dist)
gender_dist.to_csv(TABLES_DIR / "gender_distribution.csv", index=False)

# Optional plot: gender bar chart
plt.figure()
gender_dist.set_index("GENDER")["COUNT"].plot(kind="bar", title="Patients by gender")
plt.ylabel("Unique patients")
plt.tight_layout()
#plt.show()


# -----------------------------
# Diagnosis analysis (clinical-only filter)
# -----------------------------
# Build clinical-only filter (tune exclude list / regex to your needs)
is_clinical = build_clinical_filter(
    # exclude_regex example: exclude anything that clearly looks non-clinical
    exclude_regex=r"\b(employment|labor force|social|medication review|administrative)\b",
    # include_regex example (optional): keep only disorders/diseases if your text has that pattern
    # include_regex=r"\((disorder|disease)\)$",
)

conditions["IS_CLINICAL"] = conditions["DESCRIPTION"].apply(is_clinical)

# Keep only clinical diagnoses
clinical_conditions = conditions[conditions["IS_CLINICAL"]].copy()

# Top diagnoses by UNIQUE PATIENT count (preferred)
top_dx_by_patients = (
    clinical_conditions.groupby("DESCRIPTION")["PATIENT"]
    .nunique()
    .sort_values(ascending=False)
    .head(20)
    .rename("UNIQUE_PATIENTS")
    .reset_index()
)

# Top diagnoses by RECORD count (secondary)
top_dx_by_records = (
    clinical_conditions["DESCRIPTION"]
    .value_counts()
    .head(20)
    .rename_axis("DESCRIPTION")
    .reset_index(name="RECORDS")
)

print("\n=== TOP CLINICAL DIAGNOSES (by unique patients) ===")
print(top_dx_by_patients)

print("\n=== TOP CLINICAL DIAGNOSES (by record count) ===")
print(top_dx_by_records)
top_dx_by_patients.to_csv(
    TABLES_DIR / "top_clinical_diagnoses_by_patient.csv",
    index=False
)

# Optional plot: top diagnoses by unique patients
plt.figure()
top_dx_by_patients.set_index("DESCRIPTION")["UNIQUE_PATIENTS"].sort_values().plot(
    kind="barh",
    title="Top clinical diagnoses (unique patients)"
)
plt.xlabel("Unique patients")
plt.tight_layout()
#plt.show()

# -----------------------------
# Optional: Stratify diagnoses by age group / gender
# -----------------------------
# Link diagnoses to patient demographics
dx_demo = clinical_conditions.merge(
    patients_live[["Id", "GENDER", "AGE", "AGE_GROUP"]],
    left_on="PATIENT",
    right_on="Id",
    how="left"
)

# Example: top diagnoses within each gender (unique patients)
top_dx_by_gender = (
    dx_demo.groupby(["GENDER", "DESCRIPTION"])["PATIENT"]
    .nunique()
    .rename("UNIQUE_PATIENTS")
    .reset_index()
    .sort_values(["GENDER", "UNIQUE_PATIENTS"], ascending=[True, False])
)

print("\n=== (OPTIONAL) DIAGNOSES BY GENDER (unique patients) ===")
print(top_dx_by_gender.head(50))
