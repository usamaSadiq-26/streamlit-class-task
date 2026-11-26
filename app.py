import io
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Hypertension Risk Analytics",
    page_icon="❤️",
    layout="wide",
)

st.title("Hypertension Risk Analytics Dashboard")
st.write("Explore hypertension patterns using a cleaned dataset of 5,000 patient records.")
st.info(
    "This application supports population-level exploration and research. "
    "It is not a diagnostic tool."
)


@st.cache_data
def load_data():
    return pd.read_csv("cleaned_hypertension_dataset.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset file not found. Check your file name.")
    st.stop()


categorical_variables = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()
numerical_variables = df.select_dtypes(include="number").columns.tolist()

htn_col = next(
    (column for column in df.columns if column.lower() == "hypertension"),
    next((column for column in df.columns if column.lower() == "htn_flag"), None),
)
htn_values = None
if htn_col:
    htn_values = df[htn_col].copy()
    if htn_values.dtype == "object":
        htn_values = htn_values.astype(str).str.lower().str.strip().map(
            {"yes": 1, "no": 0}
        )
    else:
        htn_values = pd.to_numeric(htn_values, errors="coerce")

df["BMI_Category"] = pd.cut(
    df["BMI"],
    bins=[0, 18.5, 25, 30, float("inf")],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
    right=False,
)


st.sidebar.title("Hypertension Dashboard")
page = st.sidebar.radio(
    "Select Section",
    ["Data Overview", "Interactive Exploration"],
)


if page == "Data Overview":
    st.header("Data Overview")
    st.write("Explore the dataset, patient population, data quality and summary charts.")

    st.subheader("Dataset Preview")
    rows_to_show = st.selectbox(
        "Number of rows to display",
        [5, 10, 20, 50, 100],
    )
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    st.subheader("Dataset Structure")
    col1, col2 = st.columns(2)
    col1.metric("Number of Rows", df.shape[0])
    col2.metric("Number of Columns", df.shape[1])

    st.write("Column Names")
    st.write(df.columns.tolist())

    col1, col2 = st.columns(2)
    with col1:
        st.write("Numerical Variables")
        st.write(numerical_variables)
    with col2:
        st.write("Categorical Variables")
        st.write(categorical_variables)

    st.subheader("Dataset Information")
    with st.expander("View Dataset Information"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Population Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")

    if htn_values is not None:
        valid_htn = htn_values.dropna()
        cases = int((valid_htn == 1).sum())
        prevalence = (cases / len(valid_htn) * 100) if len(valid_htn) else 0
        col2.metric("Hypertension Cases", f"{cases:,}")
        col3.metric("Hypertension Prevalence", f"{prevalence:.1f}%")
    else:
        col2.metric("Hypertension Cases", "N/A")
        col3.metric("Hypertension Prevalence", "N/A")

    median_age = pd.to_numeric(df["Age"], errors="coerce").median()
    col4.metric(
        "Median Age",
        f"{median_age:.1f} years" if pd.notna(median_age) else "N/A",
    )

    st.write("Gender Distribution")
    gender_summary = (
        df["Gender"].value_counts().rename_axis("Gender").reset_index(name="Number of Records")
    )
    st.dataframe(gender_summary, use_container_width=True)

    st.subheader("Data Quality and Preprocessing")
    with st.expander("View Data Quality Checks"):
        missing_values = df.isnull().sum()
        missing_table = pd.DataFrame({
            "Column": missing_values.index,
            "Missing Values": missing_values.values,
            "Missing Percentage": (missing_values.values / len(df) * 100).round(2),
        })
        st.write("Missing Values")
        st.dataframe(missing_table, use_container_width=True)

        st.write("Duplicate Records")
        st.write(f"Total duplicate rows: {df.duplicated().sum()}")

        st.write("Data Types")
        st.dataframe(df.dtypes.astype(str).rename("Data Type"), use_container_width=True)

        st.write("Category Standardisation")
        for column in categorical_variables:
            extra_spaces = (
                df[column].dropna().astype(str)
                != df[column].dropna().astype(str).str.strip()
            ).sum()
            st.write(f"{column}: {extra_spaces} values contain extra spaces.")

        st.write(
            """
            **Preprocessing explanation**

            - Duplicate records were removed.
            - Text labels were trimmed and standardized.
            - Missing categorical values were retained as "Unknown".
            - Missing numerical values were replaced with each column median.
            - Numeric values outside the clinical validation ranges were replaced.
            - No missing values or duplicate rows remain in the cleaned file.
            """
        )

    st.subheader("Population Visualisations")

    if htn_values is not None:
        status_data = pd.DataFrame({
            "Hypertension Status": htn_values.map(
                {0: "No Hypertension", 1: "Hypertension"}
            )
        }).dropna()
        status_summary = (
            status_data["Hypertension Status"]
            .value_counts()
            .rename_axis("Hypertension Status")
            .reset_index(name="Number of Records")
        )
        fig1 = px.bar(
            status_summary,
            x="Hypertension Status",
            y="Number of Records",
            title="Hypertension Status Distribution",
            text="Number of Records",
        )
        st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        x=pd.to_numeric(df["Age"], errors="coerce").dropna(),
        nbins=20,
        title="Age Distribution of the Population",
        labels={"x": "Age", "y": "Number of Records"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("BMI Category Distribution")
    bmi_summary = (
        df["BMI_Category"]
        .value_counts(sort=False)
        .rename_axis("BMI Category")
        .reset_index(name="Number of Records")
    )
    st.dataframe(bmi_summary, use_container_width=True)
    fig = px.bar(
        bmi_summary,
        x="BMI Category",
        y="Number of Records",
        title="Population Distribution by BMI Category",
        text="Number of Records",
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.header("Interactive Exploration")
    st.write("Explore hypertension patterns using different variables and filters.")

    st.subheader("Filter Patient Population by Age")
    age_values = sorted(df["Age"].dropna().unique())
    minimum_value = st.selectbox("Minimum Age", age_values, index=0)
    maximum_value = st.selectbox(
        "Maximum Age",
        age_values,
        index=len(age_values) - 1,
    )

    if minimum_value > maximum_value:
        st.warning("Maximum age must be greater than or equal to minimum age.")
        st.stop()

    filtered_df = df[
        (df["Age"] >= minimum_value) &
        (df["Age"] <= maximum_value)
    ]
    st.write(f"Number of records: {len(filtered_df)}")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Categorical Filtering")
    gender_options = sorted(filtered_df["Gender"].dropna().unique())
    selected_groups = st.multiselect(
        "Select Gender Group(s)",
        gender_options,
        default=gender_options,
    )
    filtered_df = filtered_df[filtered_df["Gender"].isin(selected_groups)]
    st.write(f"Total records after filtering: {len(filtered_df)}")

    if filtered_df.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    st.info("Explore population patterns only. This application is not a diagnostic tool.")


st.divider()
st.caption(
    "Hypertension Risk Analytics Dashboard | Programming for Data Analysis"
)
