import streamlit as st
import pandas as pd
import plotly.express as px
import io


@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_hypertension_dataset.csv")
    data["Sex"] = data["Gender_Male"].map({True: "Male", False: "Female"})

    age_bins = [0, 39, 49, 59, 69, 120]
    age_labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    data["Age_Group"] = pd.cut(
        data["Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
    )
    return data


df = load_data()

st.set_page_config(
    page_title="Hypertension Population Overview",
    page_icon=":material/favorite:",
    layout="wide",
)

st.title("Hypertension Population Overview")
st.caption(
    "Population exploration for GP and practice nurse review - "
    "not an individual diagnostic or prediction tool."
)

overview_tab, visualization_tab = st.tabs(
    ["Data Overview", "Interactive Visualization"]
)

with overview_tab:
    st.subheader("Population at a glance")

    hypertension_count = int(df["Hypertension_Yes"].sum())
    hypertension_prevalence = df["Hypertension_Yes"].mean() * 100
    stage_2_count = int(df["BP_Category_Stage 2 Hypertension"].sum())

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Patient records", f"{len(df):,}")
    metric2.metric("Hypertension prevalence", f"{hypertension_prevalence:.1f}%")
    metric3.metric("Patients with hypertension", f"{hypertension_count:,}")
    metric4.metric("Stage 2 BP", f"{stage_2_count:,}")

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
                    int(df["BP_Category_Stage 2 Hypertension"].sum()),
                ],
            }
        )
        bp_chart = px.bar(
            bp_categories,
            x="Blood pressure category",
            y="Patient count",
            text="Patient count",
            color="Blood pressure category",
            title="Blood pressure categories in the cleaned dataset",
        )
        bp_chart.update_layout(showlegend=False, height=420)
        st.plotly_chart(bp_chart, use_container_width=True)

    with col2:
        st.markdown("**What matters in this population**")
        st.write(
            f"- **{hypertension_prevalence:.1f}%** of the {len(df):,} records "
            "are classified as hypertensive."
        )
        st.write(
            f"- **{stage_2_count:,} records** ({stage_2_count / len(df) * 100:.1f}%) "
            "fall in the Stage 2 category."
        )
        st.write(
            f"- Median age is **{df['Age'].median():.1f} years** "
            f"(range {df['Age'].min():.1f}-{df['Age'].max():.1f})."
        )

        st.markdown("**Data preparation**")
        st.write(
            "The overview uses the cleaned 5,000-record dataset. "
            "Categorical variables are represented by one-hot encoded columns "
            "and numerical variables retain their measured values."
        )

    with st.expander("Preview data and dataset structure"):
        rows_to_show = st.selectbox(
            "Number of rows to display",
            [5, 10, 20, 50, 100],
            index=0,
        )
        st.dataframe(df.head(rows_to_show), use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Number of Rows", df.shape[0])
        col2.metric("Number of Columns", df.shape[1])
        st.write("Columns:", ", ".join(df.columns.tolist()))

        buffer = io.StringIO()
        df.drop(columns=["Sex", "Age_Group"]).info(buf=buffer)
        st.text(buffer.getvalue())

with visualization_tab:
    st.subheader("Hypertension prevalence by age group")
    st.write(
        "Compare how hypertension prevalence changes across adult age groups "
        "for the selected population."
    )

    selected_sexes = st.multiselect(
        "Sex",
        options=["Female", "Male"],
        default=["Female", "Male"],
        help="Choose one or both population groups.",
    )

    if not selected_sexes:
        st.warning("Select at least one sex to display the chart.")
        st.stop()

    filtered_df = df[df["Sex"].isin(selected_sexes)].copy()

    if filtered_df.empty:
        st.warning("No records match the selected population.")
        st.stop()

    age_summary = (
        filtered_df.groupby("Age_Group", observed=True)
        .agg(
            patient_count=("Hypertension_Yes", "count"),
            hypertension_count=("Hypertension_Yes", "sum"),
        )
        .reset_index()
    )
    age_summary["hypertension_percentage"] = (
        age_summary["hypertension_count"]
        / age_summary["patient_count"]
        * 100
    )

    prevalence_chart = px.bar(
        age_summary,
        x="Age_Group",
        y="hypertension_percentage",
        text=age_summary["hypertension_percentage"].map(lambda value: f"{value:.1f}%"),
        custom_data=["patient_count", "hypertension_count"],
        labels={
            "Age_Group": "Age group",
            "hypertension_percentage": "Hypertension prevalence (%)",
        },
        title="Hypertension prevalence by age group",
    )
    prevalence_chart.update_traces(
        hovertemplate=(
            "Age group: %{x}<br>"
            "Prevalence: %{y:.1f}%<br>"
            "Hypertensive: %{customdata[1]:,}<br>"
            "Population: %{customdata[0]:,}"
        )
    )
    prevalence_chart.update_layout(
        yaxis_range=[0, 100],
        yaxis_ticksuffix="%",
        height=460,
    )
    st.plotly_chart(prevalence_chart, use_container_width=True)

    selected_count = len(filtered_df)
    selected_hypertension_count = int(filtered_df["Hypertension_Yes"].sum())
    selected_prevalence = selected_hypertension_count / selected_count * 100

    result1, result2 = st.columns(2)
    result1.metric("Selected patient records", f"{selected_count:,}")
    result2.metric(
        "Hypertension prevalence",
        f"{selected_prevalence:.1f}% ",
        f"{selected_hypertension_count:,} patients",
    )

    st.markdown(
        f"""
        **What this shows:** In the selected population ({", ".join(selected_sexes)}),
        the chart compares hypertension prevalence across age groups. Each percentage
        is calculated from the records currently displayed, rather than from the full
        dataset.

        **Why it matters:** A GP or practice nurse can identify whether hypertension
        prevalence rises with age and compare it across sex selections when planning
        preventive reviews or interpreting screening patterns.

        **Limitation:** This application supports exploration of population patterns;
        it is not an individual diagnostic or prediction tool.
        """
    )
