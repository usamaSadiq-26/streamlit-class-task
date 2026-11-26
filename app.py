import streamlit as st
import pandas as pd
import plotly.express as px
import io


# --------------------------------------------------
# 1. PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Hypertension Risk Analytics",
    page_icon="❤️",
    layout="wide"
)

st.title("Hypertension Risk Analytics Dashboard")

st.write(
    "This application explores hypertension patterns "
    "using a cleaned dataset of 5,000 patient records."
)

st.info(
    "This application is designed for population-level "
    "exploration and research. It is not a diagnostic tool."
)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("hypertension_dataset_5000.csv")


try:
    df = load_data()

except FileNotFoundError:
    st.error("Dataset file not found. Check your file name.")
    st.stop()


# --------------------------------------------------
# 3. IDENTIFY IMPORTANT COLUMNS
# --------------------------------------------------

column_names = {
    col.lower().strip(): col for col in df.columns
}

age_col = column_names.get("age")
htn_col = column_names.get("htn_flag")

if htn_col is None:
    htn_col = column_names.get("hypertension")

gender_col = column_names.get("gender")

if gender_col is None:
    gender_col = column_names.get("sex")

categorical_variables = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_variables = df.select_dtypes(
    include="number"
).columns.tolist()


# Convert hypertension flag to numeric

if htn_col is not None:
    htn_values = df[htn_col].copy()

    if htn_values.dtype == "object":
        htn_values = htn_values.astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0}
        )
    else:
        htn_values = pd.to_numeric(htn_values, errors="coerce")
else:
    htn_values = None


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

st.sidebar.title("Hypertension Dashboard")

page = st.sidebar.radio(
    "Select Section",
    [
        "Data Overview",
        "Interactive Exploration"
    ]
)


# --------------------------------------------------
# 5. DATA OVERVIEW
# --------------------------------------------------

if page == "Data Overview":

    st.header("Data Overview")

    st.write(
        "Explore the dataset, patient population, "
        "data quality and summary visualisations."
    )


    # --------------------------------------------------
    # 5.1 DATAFRAME PREVIEW
    # --------------------------------------------------

    st.subheader("Dataset Preview")

    rows_to_show = st.selectbox(
        "Number of rows to display",
        [5, 10, 20, 50, 100]
    )

    st.dataframe(
        df.head(rows_to_show),
        use_container_width=True
    )


    # --------------------------------------------------
    # 5.2 DATASET STRUCTURE
    # --------------------------------------------------

    st.subheader("Dataset Structure")

    col1, col2 = st.columns(2)

    col1.metric(
        "Number of Rows",
        df.shape[0]
    )

    col2.metric(
        "Number of Columns",
        df.shape[1]
    )

    st.write("Column Names")
    st.write(df.columns.tolist())

    col1, col2 = st.columns(2)

    with col1:
        st.write("Numerical Variables")
        st.write(numerical_variables)

    with col2:
        st.write("Categorical Variables")
        st.write(categorical_variables)


    # --------------------------------------------------
    # 5.3 DATASET INFORMATION
    # --------------------------------------------------

    st.subheader("Dataset Information")

    with st.expander("View Dataset Information"):

        buffer = io.StringIO()

        df.info(buf=buffer)

        info_text = buffer.getvalue()

        st.text(info_text)


    # --------------------------------------------------
    # 5.4 STATISTICAL SUMMARY
    # --------------------------------------------------

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )


    # --------------------------------------------------
    # 5.5 POPULATION SUMMARY METRICS
    # --------------------------------------------------

    st.subheader("Population Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    if htn_values is not None:

        valid_htn = htn_values.dropna()

        htn_cases = int((valid_htn == 1).sum())

        prevalence = (
            (valid_htn == 1).sum() / len(valid_htn) * 100
            if len(valid_htn) > 0 else 0
        )

        col2.metric(
            "Hypertension Cases",
            f"{htn_cases:,}"
        )

        col3.metric(
            "Hypertension Prevalence",
            f"{prevalence:.1f}%"
        )

    else:
        col2.metric("Hypertension Cases", "N/A")
        col3.metric("Hypertension Prevalence", "N/A")


    if age_col is not None:

        median_age = pd.to_numeric(
            df[age_col], errors="coerce"
        ).median()

        col4.metric(
            "Median Age",
            f"{median_age:.1f} years"
            if pd.notna(median_age) else "N/A"
        )

    else:

        col4.metric("Median Age", "N/A")


    # Gender distribution

    if gender_col is not None:

        st.write("Gender Distribution")

        gender_summary = (
            df[gender_col]
            .value_counts()
            .reset_index()
        )

        gender_summary.columns = [
            "Gender",
            "Number of Records"
        ]

        st.dataframe(
            gender_summary,
            use_container_width=True
        )


    # --------------------------------------------------
    # 5.6 DATA QUALITY AND PREPROCESSING
    # --------------------------------------------------

    st.subheader("Data Quality and Preprocessing")

    with st.expander("View Data Quality Checks"):

        st.write("Missing Values")

        missing_values = df.isnull().sum()

        missing_table = pd.DataFrame({
            "Column": missing_values.index,
            "Missing Values": missing_values.values,
            "Missing Percentage": (
                missing_values.values / len(df) * 100
            ).round(2)
        })

        st.dataframe(
            missing_table,
            use_container_width=True
        )


        st.write("Duplicate Records")

        duplicate_count = df.duplicated().sum()

        st.write(
            f"Total duplicate rows: {duplicate_count}"
        )


        st.write("Data Types")

        st.dataframe(
            df.dtypes.astype(str).rename("Data Type"),
            use_container_width=True
        )


        st.write("Category Standardisation")

        for col in categorical_variables:

            if df[col].dtype == "object":

                extra_spaces = (
                    df[col].dropna().astype(str)
                    != df[col].dropna().astype(str).str.strip()
                ).sum()

                st.write(
                    f"{col}: {extra_spaces} values "
                    "contain extra spaces."
                )


        st.write(
            """
            **Preprocessing explanation**

            - Missing values were checked in all columns.
            - Duplicate records were checked.
            - Categorical variables were reviewed for inconsistent labels.
            - Numerical variables were examined using statistical summaries.
            - The cleaned dataset was used for the analysis.

            These checks help improve data quality and make
            the results easier to interpret.
            """
        )


    # --------------------------------------------------
    # 5.7 BASELINE PLOTLY VISUALISATIONS
    # --------------------------------------------------

    st.subheader("Population Visualisations")

    st.write(
        "The following charts provide a basic understanding "
        "of the patient population."
    )


    # Chart 1: Hypertension distribution

    if htn_values is not None:

        status_data = pd.DataFrame({
            "Hypertension Status": htn_values.map({
                0: "No Hypertension",
                1: "Hypertension"
            })
        }).dropna()

        status_summary = (
            status_data["Hypertension Status"]
            .value_counts()
            .reset_index()
        )

        status_summary.columns = [
            "Hypertension Status",
            "Number of Records"
        ]

        fig1 = px.bar(
            status_summary,
            x="Hypertension Status",
            y="Number of Records",
            title="Hypertension Status Distribution",
            text="Number of Records"
        )

        fig1.update_layout(
            xaxis_title="Hypertension Status",
            yaxis_title="Number of Records"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )


    # Chart 2: Age distribution

    if age_col is not None:

        age_data = pd.to_numeric(
            df[age_col], errors="coerce"
        ).dropna()

        fig2 = px.histogram(
            x=age_data,
            nbins=20,
            title="Age Distribution of the Population",
            labels={
                "x": "Age",
                "y": "Number of Records"
            }
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


    # --------------------------------------------------
    # PART 4 - CREATE BMI CATEGORIES
    # --------------------------------------------------

    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=[
            "Underweight",
            "Normal",
            "Overweight",
            "Obese"
        ],
        right=False
    )

    st.subheader("BMI Category Distribution")

    bmi_summary = (
        df["BMI_Category"]
        .value_counts(sort=False)
        .reset_index()
    )

    bmi_summary.columns = [
        "BMI Category",
        "Number of Records"
    ]

    st.dataframe(
        bmi_summary,
        use_container_width=True
    )

    fig = px.bar(
        bmi_summary,
        x="BMI Category",
        y="Number of Records",
        title="Population Distribution by BMI Category",
        text="Number of Records"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# --------------------------------------------------
# INTERACTIVE EXPLORATION PAGE
# --------------------------------------------------

elif page == "Interactive Exploration":

    st.header("Interactive Exploration")

    st.write(
        "Explore hypertension patterns using different "
        "variables and interactive charts."
    )

    st.subheader("Filter Patient Population by Age")

    age_values = sorted(df["Age"].dropna().unique())

    minimum_value = st.selectbox(
        "Minimum Age",
        age_values,
        index=0
    )

    maximum_value = st.selectbox(
        "Maximum Age",
        age_values,
        index=len(age_values) - 1
    )

    if minimum_value <= maximum_value:

        filtered_df = df[
            (df["Age"] >= minimum_value) &
            (df["Age"] <= maximum_value)
        ]

        st.write(
            f"Number of records: {len(filtered_df)}"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

    else:

        st.warning(
            "Maximum age must be greater than or equal to minimum age."
        )
        st.stop()

    st.subheader("Categorical Filtering")

    gender_options = sorted(
        filtered_df["Gender"].dropna().unique().tolist()
    )

    selected_groups = st.multiselect(
        "Select Gender Group(s)",
        gender_options,
        default=gender_options
    )

    filtered_df = filtered_df[
        filtered_df["Gender"].isin(selected_groups)
    ]

    st.write(
        f"Total records after filtering: {len(filtered_df)}"
    )

    if filtered_df.empty:
        st.warning(
            "No records match the selected filters."
        )
        st.stop()

    st.info(
        "Explore population patterns only. "
        "This application is not a diagnostic tool."
    )


# --------------------------------------------------
# 6. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Hypertension Risk Analytics Dashboard | "
    "Programming for Data Analysis"
)
