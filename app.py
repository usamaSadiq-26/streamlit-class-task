import io

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_hypertension_dataset.csv")

    data["Sex"] = data["Gender_Male"].map({True: "Male", False: "Female"})
    data["Hypertension_Status"] = data["Hypertension_Yes"].map(
        {True: "Hypertensive", False: "Not hypertensive"}
    )
    data["Age_Group"] = pd.cut(
        data["Age"],
        bins=[0, 39, 49, 59, 69, 120],
        labels=["<40", "40-49", "50-59", "60-69", "70+"],
        include_lowest=True,
    )
    return data


def decode_one_hot(data, prefix, column_labels=None):
    columns = [column for column in data.columns if column.startswith(prefix)]
    decoded = pd.Series("Not recorded", index=data.index, dtype="object")
    for column in columns:
        label = column.removeprefix(prefix)
        if column_labels:
            label = column_labels.get(label, label)
        decoded.loc[data[column].fillna(False)] = label
    return decoded


df = load_data()

st.set_page_config(
    page_title="Hypertension Population Explorer",
    page_icon=":material/favorite:",
    layout="wide",
)

st.title("Hypertension Population Explorer")
st.info(
    "This application helps explore population patterns in a cleaned hypertension "
    "dataset. It is for exploratory review and is not an individual diagnostic "
    "or prediction tool."
)

page = st.sidebar.radio(
    "Select Section",
    ["Data Overview", "Interactive Exploration"],
)

if page == "Data Overview":
    st.subheader("Population at a glance")

    hypertension_count = int(df["Hypertension_Yes"].sum())
    hypertension_prevalence = df["Hypertension_Yes"].mean() * 100
    stage_2_count = int(df["BP_Category_Stage 2 Hypertension"].sum())
    male_count = int(df["Gender_Male"].sum())

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Patient records", f"{len(df):,}")
    metric2.metric("Hypertension prevalence", f"{hypertension_prevalence:.1f}%")
    metric3.metric("Stage 2 BP records", f"{stage_2_count:,}")
    metric4.metric(
        "Age range",
        f"{df['Age'].min():.1f}-{df['Age'].max():.1f}",
    )

    col1, col2 = st.columns([1.15, 1])

    with col1:
        bp_categories = pd.DataFrame(
            {
                "Blood pressure category": [
                    "Normal",
                    "Stage 1 hypertension",
                    "Stage 2 hypertension",
                ],
                "Patient count": [
                    int(df["BP_Category_Normal"].sum()),
                    int(df["BP_Category_Stage 1 Hypertension"].sum()),
                    stage_2_count,
                ],
            }
        )
        fig = px.bar(
            bp_categories,
            x="Blood pressure category",
            y="Patient count",
            text="Patient count",
            color="Blood pressure category",
            title="Blood pressure category distribution",
        )
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Clinical population context**")
        st.write(
            f"- **{hypertension_count:,}** of **{len(df):,}** records "
            f"({hypertension_prevalence:.1f}%) are classified as hypertensive."
        )
        st.write(
            f"- **{male_count:,}** records are male "
            f"({male_count / len(df) * 100:.1f}%)."
        )
        st.write(
            f"- Median age is **{df['Age'].median():.1f} years** "
            f"and most records are from adults aged "
            f"{df['Age'].min():.1f}-{df['Age'].max():.1f}."
        )
        st.write(
            "These high-level measures identify the population size, outcome burden "
            "and age context before staff explore relationships."
        )

    st.subheader("Dataframe preview")
    rows_to_show = st.selectbox(
        "Number of rows to display",
        [5, 10, 20, 50, 100],
        index=0,
    )
    st.dataframe(df.head(rows_to_show), use_container_width=True)

    st.subheader("Dataset structure")
    col1, col2 = st.columns(2)
    col1.metric("Number of Rows", df.shape[0])
    col2.metric("Number of Columns", df.shape[1])

    categorical_variables = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    numerical_variables = df.select_dtypes(
        include="number"
    ).columns.tolist()

    st.write("**Numerical variables:**", ", ".join(numerical_variables))
    st.write("**Categorical variables:**", ", ".join(categorical_variables))

    with st.expander("View Dataset Information"):
        buffer = io.StringIO()
        df.drop(columns=["Sex", "Age_Group", "Hypertension_Status"]).info(
            buf=buffer
        )
        st.text(buffer.getvalue())

    st.subheader("Statistical Summary")
    st.dataframe(
        df[numerical_variables].describe().T,
        use_container_width=True,
    )

    with st.expander("Data Quality and Preprocessing"):
        missing_count = int(df.isna().sum().sum())
        duplicate_count = int(df.duplicated().sum())
        st.write(
            f"- The cleaned application dataset contains **{missing_count} missing "
            f"values** and **{duplicate_count} duplicate rows**."
        )
        st.write(
            "- Original category labels have been converted to one-hot encoded "
            "columns in the cleaned file."
        )
        st.write(
            "- Age, BMI and blood-pressure values remain numerical, with sex and "
            "hypertension status decoded as readable labels for display."
        )
        st.write(
            "- Age bands (<40, 40-49, 50-59, 60-69, 70+) were derived only to "
            "support population comparison."
        )
        st.write(
            "These checks matter because missing, duplicated or inconsistent records "
            "could distort prevalence estimates and group comparisons."
        )

else:
    st.subheader("Interactive Exploration")

    st.markdown("**Population filters**")
    age_options = sorted(
        {
            int(df["Age"].min()),
            *range(20, 90, 10),
            int(df["Age"].max()),
        }
    )
    age_options = [
        value for value in age_options
        if df["Age"].min() <= value <= df["Age"].max()
    ]

    filter1, filter2, filter3 = st.columns(3)
    with filter1:
        minimum_age = st.selectbox(
            "Minimum age",
            age_options,
            index=0,
        )
    with filter2:
        maximum_age = st.selectbox(
            "Maximum age",
            age_options,
            index=len(age_options) - 1,
        )
    with filter3:
        sex_options = ["Female", "Male"]
        selected_sexes = st.multiselect(
            "Select sex",
            sex_options,
            default=sex_options,
        )

    if minimum_age > maximum_age:
        st.warning(
            "Minimum age is greater than maximum age. "
            "Please choose a minimum age that is no greater than the maximum."
        )
        st.stop()

    if not selected_sexes:
        st.warning("Select at least one sex to display an analysis.")
        st.stop()

    filtered_df = df[
        (df["Age"] >= minimum_age)
        & (df["Age"] <= maximum_age)
        & (df["Sex"].isin(selected_sexes))
    ].copy()

    if filtered_df.empty:
        st.warning(
            "No records match the selected age and sex filters. "
            "Broaden the selection to continue."
        )
        st.stop()

    st.caption(
        f"{len(filtered_df):,} records remain after applying the population filters."
    )

    analysis_type = st.radio(
        "Choose Analysis Type",
        ["Univariate", "Bivariate", "Multivariate"],
        horizontal=True,
    )

    if analysis_type == "Univariate":
        variable_type = st.radio(
            "Variable type",
            ["Numerical", "Categorical"],
            horizontal=True,
        )

        if variable_type == "Numerical":
            numerical_options = [
                "Age",
                "BMI",
                "Waist_Circumference_cm",
                "Sleep_Hours",
                "Creatinine_mg_dL",
                "Fasting_Glucose_mg_dL",
                "Cholesterol_mg_dL",
                "Systolic_BP_mmHg",
                "Diastolic_BP_mmHg",
            ]
            variable = st.selectbox(
                "Select Numerical Variable",
                numerical_options,
            )
            fig = px.histogram(
                filtered_df,
                x=variable,
                nbins=20,
                title=f"Distribution of {variable}",
                labels={variable: variable.replace("_", " ")},
            )
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)
            st.write(
                f"This histogram shows the distribution of "
                f"{variable.replace('_', ' ').lower()} for the {len(filtered_df):,} "
                "records matching the selected filters."
            )

        else:
            categorical_options = {
                "Sex": ("direct", "Sex"),
                "Hypertension status": ("direct", "Hypertension_Status"),
                "BP category": (
                    "one_hot",
                    "BP_Category",
                    {
                        "Normal": "Normal",
                        "Stage 1 Hypertension": "Stage 1",
                        "Stage 2 Hypertension": "Stage 2",
                    },
                ),
                "Physical activity": (
                    "one_hot",
                    "Physical_Activity",
                ),
                "Stress level": (
                    "one_hot",
                    "Stress_Level",
                ),
                "Salt intake": (
                    "one_hot",
                    "Salt_Intake",
                ),
            }
            category_label = st.selectbox(
                "Select Categorical Variable",
                list(categorical_options),
            )
            category_definition = categorical_options[category_label]

            if category_definition[0] == "direct":
                counts = (
                    filtered_df[category_definition[1]]
                    .value_counts()
                    .rename_axis("Category")
                    .reset_index(name="Count")
                )
            else:
                labels = category_definition[2] if len(category_definition) == 3 else None
                decoded = decode_one_hot(
                    filtered_df,
                    category_definition[1] + "_",
                    labels,
                )
                counts = (
                    decoded.value_counts()
                    .rename_axis("Category")
                    .reset_index(name="Count")
                )

            fig = px.bar(
                counts,
                x="Category",
                y="Count",
                text="Count",
                title=f"Distribution of {category_label}",
            )
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True)
            st.write(
                f"This chart compares the number of records in each "
                f"{category_label.lower()} category within the filtered population."
            )

    elif analysis_type == "Bivariate":
        bivariate_type = st.radio(
            "Relationship type",
            ["Numerical vs numerical", "Population group vs hypertension outcome"],
            horizontal=True,
        )

        if bivariate_type == "Numerical vs numerical":
            numerical_options = [
                "Age",
                "BMI",
                "Waist_Circumference_cm",
                "Sleep_Hours",
                "Creatinine_mg_dL",
                "Fasting_Glucose_mg_dL",
                "Cholesterol_mg_dL",
                "Systolic_BP_mmHg",
                "Diastolic_BP_mmHg",
            ]
            x_col1, x_col2 = st.columns(2)
            with x_col1:
                x_variable = st.selectbox(
                    "Select X Variable",
                    numerical_options,
                    index=0,
                )
            with x_col2:
                y_variable = st.selectbox(
                    "Select Y Variable",
                    numerical_options,
                    index=1,
                )

            fig = px.scatter(
                filtered_df,
                x=x_variable,
                y=y_variable,
                title=f"{x_variable} vs {y_variable}",
                labels={
                    x_variable: x_variable.replace("_", " "),
                    y_variable: y_variable.replace("_", " "),
                },
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            correlation = filtered_df[x_variable].corr(filtered_df[y_variable])
            st.metric("Correlation", f"{correlation:.2f}")
            st.write(
                "This scatter plot shows the direction and strength of the "
                "association in the selected population; correlation does not "
                "establish causation."
            )

        else:
            group_options = {
                "Sex": ("direct", "Sex"),
                "Age group": ("direct", "Age_Group"),
                "Physical activity": (
                    "one_hot",
                    "Physical_Activity",
                ),
                "Stress level": (
                    "one_hot",
                    "Stress_Level",
                ),
                "Salt intake": (
                    "one_hot",
                    "Salt_Intake",
                ),
                "Smoking status": (
                    "one_hot",
                    "Smoking_Status",
                ),
                "Family history": (
                    "one_hot",
                    "Family_History",
                    {"HTN_Yes": "Yes"},
                ),
            }
            group_label = st.selectbox(
                "Select population group",
                list(group_options),
            )
            group_definition = group_options[group_label]

            if group_definition[0] == "direct":
                plot_df = filtered_df[
                    [group_definition[1], "Hypertension_Status"]
                ].copy()
                plot_df = plot_df.rename(
                    columns={group_definition[1]: "Population group"}
                )
            else:
                labels = group_definition[2] if len(group_definition) == 3 else None
                plot_df = pd.DataFrame(
                    {
                        "Population group": decode_one_hot(
                            filtered_df,
                            group_definition[1] + "_",
                            labels,
                        ),
                        "Hypertension_Status": filtered_df["Hypertension_Status"],
                    }
                )

            percentage_table = (
                pd.crosstab(
                    plot_df["Population group"],
                    plot_df["Hypertension_Status"],
                    normalize="index",
                )
                * 100
            ).reset_index()

            if "Hypertensive" not in percentage_table:
                percentage_table["Hypertensive"] = 0.0

            fig = px.bar(
                percentage_table,
                x="Population group",
                y="Hypertensive",
                text=percentage_table["Hypertensive"].map(
                    lambda value: f"{value:.1f}%"
                ),
                title=f"Hypertension prevalence by {group_label.lower()}",
                labels={"Hypertensive": "Hypertension prevalence (%)"},
            )
            fig.update_layout(yaxis_range=[0, 100], yaxis_ticksuffix="%", height=460)
            st.plotly_chart(fig, use_container_width=True)

            st.write(
                f"The chart recalculates hypertension percentages for each "
                f"{group_label.lower()} category using only the currently filtered "
                f"{len(filtered_df):,} records."
            )

    else:
        multivariate_numerical = [
            "Age",
            "BMI",
            "Waist_Circumference_cm",
            "Sleep_Hours",
            "Fasting_Glucose_mg_dL",
            "Cholesterol_mg_dL",
            "Systolic_BP_mmHg",
            "Diastolic_BP_mmHg",
        ]
        multi1, multi2 = st.columns(2)
        with multi1:
            x_variable = st.selectbox(
                "Select X Variable",
                multivariate_numerical,
                index=0,
                key="multivariate_x",
            )
        with multi2:
            y_variable = st.selectbox(
                "Select Y Variable",
                multivariate_numerical,
                index=1,
                key="multivariate_y",
            )

        fig = px.scatter(
            filtered_df,
            x=x_variable,
            y=y_variable,
            color="Hypertension_Status",
            title=f"{x_variable} vs {y_variable}, coloured by hypertension status",
            labels={
                x_variable: x_variable.replace("_", " "),
                y_variable: y_variable.replace("_", " "),
            },
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.write(
            "This multivariate view combines two numerical measurements with "
            "hypertension status as a colour group. It helps users compare whether "
            "the two measurements differ across the outcome in the filtered population."
        )

    st.caption(
        "Population patterns only: the findings should not be used for individual "
        "diagnosis or prediction."
    )
