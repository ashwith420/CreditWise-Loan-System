import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CreditWise Loan System", page_icon="🏦")

DATA_FILE = "loan_approval_data.csv"

ONE_HOT_COLUMNS = [
    "Gender",
    "Employer_Category",
    "Property_Area",
    "Loan_Purpose",
    "Marital_Status",
    "Employment_Status",
]


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE).copy()

    # Same preprocessing used in the notebook
    df = df.drop(columns=["Applicant_ID"], errors="ignore")

    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include="object").columns.tolist()

    numerical_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    category_options = {
        column: sorted(df[column].astype(str).unique().tolist())
        for column in ONE_HOT_COLUMNS
    }
    education_options = sorted(df["Education_Level"].astype(str).unique().tolist())

    education_encoder = LabelEncoder()
    target_encoder = LabelEncoder()

    df["Education_Level"] = education_encoder.fit_transform(df["Education_Level"])
    df["Loan_Approved"] = target_encoder.fit_transform(df["Loan_Approved"])

    encoder = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded = encoder.fit_transform(df[ONE_HOT_COLUMNS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(ONE_HOT_COLUMNS),
        index=df.index
    )

    df = pd.concat([df.drop(columns=ONE_HOT_COLUMNS), encoded_df], axis=1)

    # Feature engineering
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2

    X = df.drop(columns=["DTI_Ratio", "Credit_Score", "Loan_Approved"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    return (
        model,
        scaler,
        encoder,
        education_encoder,
        target_encoder,
        X.columns.tolist(),
        category_options,
        education_options,
    )


def prepare_input(data, encoder, education_encoder, feature_names):
    row = pd.DataFrame([data])

    row["Education_Level"] = education_encoder.transform(row["Education_Level"])

    encoded = encoder.transform(row[ONE_HOT_COLUMNS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(ONE_HOT_COLUMNS)
    )

    row = pd.concat([row.drop(columns=ONE_HOT_COLUMNS), encoded_df], axis=1)

    row["Credit_Score_sq"] = row["Credit_Score"] ** 2
    row["DTI_Ratio_sq"] = row["DTI_Ratio"] ** 2

    row = row.drop(columns=["Credit_Score", "DTI_Ratio"])

    return row.reindex(columns=feature_names, fill_value=0)


st.title("🏦 CreditWise Loan System")
st.write("Enter applicant details to receive a preliminary loan-approval prediction.")
st.caption("For learning and demonstration only. Final lending decisions require human review.")

(
    model,
    scaler,
    encoder,
    education_encoder,
    target_encoder,
    feature_names,
    category_options,
    education_options,
) = train_model()

with st.form("loan_form"):
    left, right = st.columns(2)

    with left:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
        credit_score = st.number_input("Credit Score", min_value=0.0, value=650.0)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=0)

    with right:
        dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.35)
        savings = st.number_input("Savings", min_value=0.0, value=10000.0)
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=20000.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0)
        loan_term = st.number_input("Loan Term in Months", min_value=1, max_value=360, value=60)

    gender = st.selectbox("Gender", category_options["Gender"])
    employment_status = st.selectbox("Employment Status", category_options["Employment_Status"])
    marital_status = st.selectbox("Marital Status", category_options["Marital_Status"])
    education_level = st.selectbox("Education Level", education_options)
    employer_category = st.selectbox("Employer Category", category_options["Employer_Category"])
    loan_purpose = st.selectbox("Loan Purpose", category_options["Loan_Purpose"])
    property_area = st.selectbox("Property Area", category_options["Property_Area"])

    submitted = st.form_submit_button("Predict Loan Approval")

if submitted:
    applicant_data = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Gender": gender,
        "Employment_Status": employment_status,
        "Marital_Status": marital_status,
        "Education_Level": education_level,
        "Employer_Category": employer_category,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
    }

    prepared_input = prepare_input(
        applicant_data,
        encoder,
        education_encoder,
        feature_names
    )

    prediction = model.predict(scaler.transform(prepared_input))[0]
    approved_code = target_encoder.transform(["Yes"])[0]

    if prediction == approved_code:
        st.success("Preliminary Prediction: Approved")
    else:
        st.error("Preliminary Prediction: Rejected")

    st.info("This result is a model prediction and must be reviewed by a loan officer.")