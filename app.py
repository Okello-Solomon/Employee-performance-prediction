import streamlit as st
import pandas as pd
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="INX Future Inc. Employee Performance Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject stronger CSS
st.markdown("""
<style>

/* Sidebar title */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: blue;
    margin-bottom: 0px;
}

/* Radio labels (Prediction, Report) */
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: blue !important;
}

/* Selected option */
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p {
    color: darkblue !important;
    font-weight: 800 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
pipeline = joblib.load("pipeline.pkl")

# =========================================================
# SIDEBAR MENU
# =========================================================
st.sidebar.markdown("## Navigation")

menu = st.sidebar.radio(
    "",
    [
        "Prediction System",
        "Report",
        "Power BI Visualization"
    ]
)

# =========================================================
# ================= PREDICTION PAGE =======================
# =========================================================
if menu == "Prediction System":

    st.markdown(
        "<h1 style='color:blue'> 📊 Employee Performance Prediction.</h1>",
        unsafe_allow_html=True
    )

    st.markdown("""
    A machine learning-based system for predicting employee performance using key HR, behavioral, and workplace factors, powered by Extreme Gradient Boosting (XGBoost).

    **Performance Classes:**  
    - 0 = Good  
    - 1 = Excellent  
    - 2 = Outstanding  
    """)

    # =========================
    # Sidebar Inputs
    # =========================
    st.sidebar.header("Employee Profile Information")

    # Ordinal Features
    env_options = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4
    }

    EmpEnvironmentSatisfaction = st.sidebar.selectbox(
        "Employment Environment Satisfaction",
        list(env_options.keys())
    )
    EmpEnvironmentSatisfaction = env_options[EmpEnvironmentSatisfaction]

    relationship_options = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
        "Very High": 4
    }

    EmpRelationshipSatisfaction = st.sidebar.selectbox(
        "Employment Relationship Satisfaction",
        list(relationship_options.keys())
    )
    EmpRelationshipSatisfaction = relationship_options[EmpRelationshipSatisfaction]

    worklife_options = {
        "Bad": 1,
        "Good": 2,
        "Better": 3,
        "Best": 4
    }

    EmpWorkLifeBalance = st.sidebar.selectbox(
        "Employment Work Life Balance",
        list(worklife_options.keys())
    )
    EmpWorkLifeBalance = worklife_options[EmpWorkLifeBalance]

    # Binary Feature
    OverTime = st.sidebar.selectbox(
        "Over Time",
        ["No", "Yes"]
    )
    OverTime = 1 if OverTime == "Yes" else 0

    # Department
    EmpDepartment = st.sidebar.selectbox(
        "Employment Department",
        [
            "Development",
            "Sales",
            "Human Resources",
            "Data Science",
            "Research & Development",
            "Finance"
        ]
    )

    # Job Role
    EmpJobRole = st.sidebar.selectbox(
        "Employment Job Role",
        [
            "Data Scientist",
            "Delivery Manager",
            "Developer",
            "Finance Manager",
            "Healthcare Representative",
            "Human Resources",
            "Business Analyst",
            "Laboratory Technician",
            "Manager",
            "Manager R&D",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative",
            "Senior Developer",
            "Senior Manager R&D",
            "Technical Architect",
            "Technical Lead"
        ]
    )

    # =========================
    # Main Panel Inputs
    # =========================
    st.subheader("Employee Career Information")

    col1, col2 = st.columns(2)

    with col1:
        EmpLastSalaryHikePercent = st.number_input(
            "Last Salary Hike (%)",
            11.0,
            25.0,
            15.0
        )

        ExperienceYearsInCurrentRole = st.number_input(
            "Years in Current Role",
            0.0,
            18.0,
            4.0
        )

    with col2:
        YearsSinceLastPromotion = st.number_input(
            "Years Since Last Promotion",
            0.0,
            15.0,
            2.0
        )

        YearsWithCurrManager = st.number_input(
            "Years With Current Manager",
            0.0,
            17.0,
            4.0
        )

    # =========================
    # One-Hot Encoding
    # =========================

    EmpDepartment_Development = 1 if EmpDepartment == "Development" else 0

    EmpJobRole_Developer = 1 if EmpJobRole == "Developer" else 0
    EmpJobRole_Sales_Representative = 1 if EmpJobRole == "Sales Representative" else 0
    EmpJobRole_Senior_Developer = 1 if EmpJobRole == "Senior Developer" else 0

    # =========================
    # Prepare Input Data
    # =========================
    input_df = pd.DataFrame([{
        "EmpEnvironmentSatisfaction": EmpEnvironmentSatisfaction,
        "OverTime": OverTime,
        "EmpLastSalaryHikePercent": EmpLastSalaryHikePercent,
        "EmpRelationshipSatisfaction": EmpRelationshipSatisfaction,
        "EmpWorkLifeBalance": EmpWorkLifeBalance,
        "ExperienceYearsInCurrentRole": ExperienceYearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager": YearsWithCurrManager,
        "EmpDepartment_Development": EmpDepartment_Development,
        "EmpJobRole_Developer": EmpJobRole_Developer,
        "EmpJobRole_Sales Representative": EmpJobRole_Sales_Representative,
        "EmpJobRole_Senior Developer": EmpJobRole_Senior_Developer
    }])

    # =========================
    # Prediction
    # =========================
    if st.button("📊 Predict Employee Performance"):

        prediction = pipeline.predict(input_df)
        prediction_proba = pipeline.predict_proba(input_df)

        prob_good = prediction_proba[0][0]
        prob_excellent = prediction_proba[0][1]
        prob_outstanding = prediction_proba[0][2]

        pred_class = prediction[0]

        status_map = {
            0: "Good",
            1: "Excellent",
            2: "Outstanding"
        }

        description_map = {
            0: "Employee demonstrates average workplace performance.",
            1: "Employee demonstrates strong and consistent workplace performance.",
            2: "Employee demonstrates exceptional workplace performance and productivity."
        }

        # Prediction Box
        if pred_class == 2:
            background_color = "#3498db"
        elif pred_class == 1:
            background_color = "#2ecc71"
        else:
            background_color = "#f1c40f"

        st.subheader("Prediction")

        st.markdown(
            f"""
            <div style='
                background-color:{background_color};
                padding:15px;
                border-radius:8px;
                color:white;
                font-size:20px;
                font-weight:bold;
                text-align:center;
            '>
            Prediction: {status_map[pred_class]}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"**Interpretation:** {description_map[pred_class]}")

        # =========================
        # Prediction Probabilities
        # =========================
        st.subheader("Prediction Probabilities")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Good Probability",
                f"{prob_good*100:.2f}%"
            )

        with col2:
            st.metric(
                "Excellent Probability",
                f"{prob_excellent*100:.2f}%"
            )

        with col3:
            st.metric(
                "Outstanding Probability",
                f"{prob_outstanding*100:.2f}%"
            )

        # =========================
        # Performance Strength Level
        # =========================
        st.subheader("Performance Strength Level")

        highest_prob = max(
            prob_good,
            prob_excellent,
            prob_outstanding
        )

        st.progress(int(highest_prob * 100))

        if pred_class == 2:
            st.success("🌟 Outstanding Performance Potential")

        elif pred_class == 1:
            st.success("✅ Strong Performance Level")

        else:
            st.warning("⚡ Moderate Performance Level")

# =========================
# --- REPORT TAB ---
# =========================
if menu == "Report":

    st.markdown(
        "<h1 style='color:blue'> Predicting Employee Performance.</h1>",
        unsafe_allow_html=True
    )

    # Sidebar Navigation
    section = st.sidebar.radio(
        "Jump to Section",
        [
            "Introduction",
            "Problem Statement",
            "Data Description",
            "EDA",
            "Data Preprocessing",
            "Methodology",
            "Modeling & Evaluation",
            "Results & Insights",
            "Recommendations",
            "Conclusion"
        ]
    )

    # =========================
    # INTRODUCTION
    # =========================
    if section == "Introduction":

        st.markdown("## Introduction")

        st.markdown("### Background")

        st.markdown(
            "INX Future Inc. is a leading analytics and technology company known for "
            "its employee-friendly culture and innovative workforce management practices. "
            "However, the organization has recently experienced declining employee "
            "performance and reduced client satisfaction levels. These challenges "
            "have raised concerns about employee engagement, productivity, retention, "
            "and workforce stability."
        )

        st.markdown(
            "This project applies Machine Learning and Workforce Analytics to analyze "
            "employee-related factors and predict employee performance levels using "
            "historical HR data."
        )

        st.markdown("### Relevance")

        st.markdown(
            "In today’s competitive business environment, organizations rely on data-driven insights to" 
            "optimize workforce productivity and maintain service quality. Predicting employee" 
            "performance using machine learning helps companies improve talent management, " 
            "enhance employee engagement, and sustain competitive advantage in the analytics and" 
            "technology industry. "
        )

        st.markdown("**The project supports:**")

        st.markdown("- Workforce performance optimization")
        st.markdown("- Employee engagement and retention strategies")
        st.markdown("- HR analytics and workforce intelligence")
        st.markdown("- Data-driven organizational decision-making")
        st.markdown("- Employee productivity improvement")

        st.markdown("### Objective of the Analysis")
        st.markdown(
            "The main objective of this analysis is to develop a machine learning model that predicts employee performance levels and "
            "identifies the key factors influencing performance. The insights aim to support HR in improving employee productivity, "
            "retention, and overall organizational performance. "
        )

        st.markdown("**Other Objectives include:**")

        st.markdown("- Comparing multiple machine learning classification models")
        st.markdown("- Supporting strategic HR and workforce decision-making")

    # =========================
    # PROBLEM STATEMENT
    # =========================
    elif section == "Problem Statement":

        st.markdown("## Problem Statement")

        st.markdown("### Research Problem")

        st.markdown(
            "INX Future Inc. is experiencing declining employee performance and reduced "
            "client satisfaction despite maintaining a strong employer reputation. "
            "Management requires a data-driven solution capable of identifying the "
            "factors influencing employee performance and detecting workforce patterns "
            "associated with reduced productivity and attrition."
        )

        st.markdown(
            "Traditional HR evaluation methods may not fully capture the complex "
            "relationships between employee satisfaction, work environment, promotion "
            "history, engagement, and performance outcomes."
        )

        st.markdown("### Target Variable")

        st.markdown("**Predicted / Classified:**")

        st.markdown(
            "The machine learning model predicts employee performance levels using "
            "historical employee data."
        )

        st.markdown("The target classes are:")

        st.markdown("- Good")
        st.markdown("- Excellent")
        st.markdown("- Outstanding")


        st.markdown("### Business Goal")

        st.markdown(
            "The goal is to support HR departments and management teams in improving "
            "employee productivity, workforce stability, employee satisfaction, and "
            "overall organizational performance."
        )

    # =========================
    # DATA DESCRIPTION
    # =========================
    elif section == "Data Description":

        st.markdown("## Data Description")

        st.markdown("### Source of the Data")

        st.markdown(
            "The dataset used in this project is the INX Future Inc. Employee "
            "Performance Dataset provided by IABAC for employee performance analysis "
            "and predictive modeling."
        )

        st.markdown("### Features / Variables")

        st.markdown(
            "The final model used key workforce and HR-related variables selected "
            "using Recursive Feature Elimination (RFE) with XGBoost."
        )

        st.markdown("### Selected Features")

        st.markdown("- Employment Environment Satisfaction")
        st.markdown("- Over Time")
        st.markdown("- Employment Last Salary Hike Percent")
        st.markdown("- Employment Relationship Satisfaction")
        st.markdown("- Employment Work Life Balance")
        st.markdown("- Experience Years In Current Role")
        st.markdown("- Years Since Last Promotion")
        st.markdown("- Years With Current Manager")
        st.markdown("- Employment Department")
        st.markdown("- Employment Job Role")

        st.markdown("### Number of Records and Variables")

        st.markdown(
            "The dataset contains 1,200 employee records and 28 variables."
        )

        st.markdown("### Missing Values")

        st.markdown(
            "The dataset contained complete observations with no major missing values."
        )

        st.markdown("### Data Types")

        st.markdown(
            "The dataset contains categorical, ordinal, binary, and numerical variables "
            "representing employee demographics, workplace satisfaction, organizational "
            "structure, work experience, and performance indicators."
        )

# =========================
# EDA
# =========================
    elif section == "EDA":

        st.markdown("## Exploratory Data Analysis (EDA)")

        # =========================================
        # Performance Rating Distribution
        # =========================================
        st.subheader("Performance Rating Distribution (Target Class Distribution)")

        st.image("1. Performance Rating (Bar Graph, Count).png", width=700)

        st.markdown(
            "The chart shows the distribution of employee performance at INX Future Inc. "
            "Most employees (874) are rated Excellent, followed by Good (194) and "
            "Outstanding (132), with no employees in the Low category."
        )

        st.markdown(
            "This indicates that the workforce is generally high-performing, but the "
            "distribution is heavily concentrated in the Excellent group."
        )

        # =========================================
        # Employee Attrition Distribution
        # =========================================
        st.subheader("Employee Attrition Distribution")

        st.image("2. Employee Attrition Distribution.png", width=700)

        st.markdown(
            "The chart shows the employee attrition distribution at INX Future Inc. "
            "Most employees (1,022, about 85%) have stayed with the company, while "
            "178 employees (about 15%) have left."
        )

        st.markdown(
            "Although retention is generally high, there is still a noticeable attrition rate."
        )

        # =========================================
        # Age Distribution
        # =========================================
        st.subheader("Age Distribution of Employees")

        st.image("3. Age Distribution of Employees.png", width=700)

        st.markdown(
            "The histogram shows the age distribution of employees at INX Future Inc., "
            "ranging from 18 to 60 years. Most employees fall within the 30–40 age "
            "group, with an average age of around 36–37 years, indicating a "
            "predominantly mid-career workforce."
        )

        st.markdown(
            "The distribution is slightly right-skewed, meaning there are fewer older "
            "employees as age increases."
        )

        # =========================================
        # Job Satisfaction
        # =========================================
        st.subheader("Job Satisfaction Levels")

        st.image("4. Job Satisfaction Levels.png", width=700)

        st.markdown(
            "The chart shows the job satisfaction levels of employees at INX Future Inc. "
            "Most employees report high (354) or very high (378) satisfaction, indicating "
            "a generally positive work environment. However, a notable number fall into "
            "medium (237) and low (231) satisfaction levels."
        )

        st.markdown(
            "This suggests that while the majority are satisfied, a significant portion "
            "of employees may be disengaged, which could be contributing to performance "
            "issues and declining client satisfaction."
        )

        # =========================================
        # Department-wise Performance
        # =========================================
        st.subheader("Department-wise Performance Rating Distribution")

        st.image("6. Department-wise Performance Rating Distribution.png", width=700)

        st.markdown(
            "The chart shows department-wise performance ratings across the company. "
            "Most employees in all departments are rated Excellent, but there are "
            "noticeable differences between teams."
        )

        st.markdown(
            "Departments like Development and Research & Development have strong and "
            "consistent high performance, while Sales has a larger number of employees "
            "in the Good category, indicating more variation in performance."
        )

        st.markdown(
            "Smaller teams like Data Science maintain high-quality performance despite "
            "their size."
        )

        st.markdown(
            "The chart suggests that performance levels vary by department, and areas "
            "with more Good ratings may need improvement to boost overall productivity "
            "and service delivery."
        )

        # =========================================
        # Attrition vs Performance
        # =========================================
        st.subheader("Attrition vs Performance Rating")

        st.image("7. Attrition vs Performance Rating.png", width=700)

        st.markdown(
            "The chart shows the relationship between attrition and performance rating. "
            "While most employees across all performance levels remain in the company, "
            "a notable number have left, especially from the Excellent category."
        )

        st.markdown(
            "This means that even high-performing employees are leaving the organization, "
            "which is a key concern."
        )

        st.markdown(
            "The loss of top talent (Excellent and Outstanding) may be contributing "
            "to declining service delivery and client satisfaction, highlighting the "
            "need for stronger retention strategies focused on high performers."
        )

        # =========================================
        # Job Satisfaction vs Performance
        # =========================================
        st.subheader("Performance Rating by Job Satisfaction")

        st.image("8. Performance Rating by Job Satisfaction.png", width=700)

        st.markdown(
            "The chart shows the relationship between job satisfaction and performance "
            "rating at INX Future Inc. It indicates a clear positive trend where employees "
            "with higher job satisfaction tend to achieve better performance ratings, "
            "especially in the Excellent and Outstanding categories."
        )

        st.markdown(
            "Most top performers report high or very high satisfaction, while lower "
            "satisfaction levels are more mixed within the Good performance group."
        )

        st.markdown(
            "However, some high-performing employees still report low satisfaction, "
            "which may signal a risk of future attrition."
        )

        st.markdown(
            "Overall, the chart suggests that job satisfaction is strongly linked to "
            "employee performance, making it an important factor for improving "
            "productivity and retention."
        )

        # =========================================
        # Hourly Rate vs Performance
        # =========================================
        st.subheader("Hourly Rate by Performance Rating (with Mean)")

        st.image("9. Hourly Rate by Performance Rating (with Mean).png", width=700)

        st.markdown(
            "The boxplot shows the relationship between hourly rate and performance "
            "rating at INX Future Inc."
        )

        st.markdown(
            "It reveals an unexpected pattern where lower-performing employees "
            "(Good) have the highest average hourly rate (68.2), followed by "
            "Excellent (65.6), while Outstanding performers receive the lowest "
            "average rate (65.2)."
        )

        st.markdown(
            "This indicates a reverse or misaligned reward structure, where pay does "
            "not increase with performance."
        )

        st.markdown(
            "Such a gap may reduce motivation among top performers and could contribute "
            "to declining performance and service delivery issues, including reduced "
            "client satisfaction."
        )

        # =========================================
        # Distance From Home
        # =========================================
        st.subheader("Distance From Home by Performance Rating (with Mean)")

        st.image("10. Distance From Home by Performance Rating (with Mean).png", width=700)

        st.markdown(
            "The boxplot shows the relationship between distance from home and "
            "performance rating at INX Future Inc."
        )

        st.markdown(
            "It reveals a clear trend where employees who live closer to the workplace "
            "tend to perform better."
        )

        st.markdown(
            "On average, Good performers live farthest away (9.8 units), followed by "
            "Excellent performers (9.1 units), while Outstanding employees live closest "
            "(8.4 units)."
        )

        st.markdown(
            "This suggests that longer commute distances may negatively affect "
            "employee performance."
        )

        st.markdown(
            "Overall, the findings indicate that commute distance could contribute "
            "to lower performance levels, and reducing travel burden through flexible "
            "work arrangements or remote work options may improve productivity."
        )

        # =========================================
        # Key Findings
        # =========================================
        st.header("Key Findings: Trends, Outliers & Correlations")

        # =========================================
        # Skewness Heatmap
        # =========================================
        st.subheader("Skewness Heatmap of Numerical Features")

        st.image("5. Skewness Heatmap of Numerical Feature.png", width=700)

        st.markdown(
            "The skewness values show how unevenly distributed the numerical features "
            "are in the dataset."
        )

        st.markdown(
            "Some variables, like Years Since Last Promotion, Experience at Company, "
            "and Total Work Experience, are highly positively skewed, meaning most "
            "employees have lower values with a few extreme high cases (outliers)."
        )

        st.markdown(
            "Moderately skewed features, such as Distance from Home and Salary Hike "
            "Percent, show some imbalance but are less extreme."
        )

        st.markdown(
            "Meanwhile, variables like Age and Hourly Rate are fairly symmetric, "
            "indicating balanced distributions."
        )

        st.markdown(
            "Highly skewed features may require transformation (e.g., log scaling) "
            "to improve model performance and prediction reliability."
        )

        # =========================================
        # Correlation Heatmap
        # =========================================
        st.subheader("Correlation Heatmap")

        st.image("11. Correlation Heatmap.png", width=700)

        st.markdown(
            "The correlation heatmap shows that most variables have weak or near-zero "
            "correlations with each other, meaning they provide largely independent "
            "information for modeling."
        )

        st.markdown(
            "However, there are several notable positive relationships among "
            "tenure-related features."
        )

        st.markdown(
            "For example, Experience Years at This Company and Years with Current "
            "Manager show a strong positive correlation (0.76), while Experience "
            "Years at This Company and Experience Years in Current Role also "
            "show a strong correlation (0.76)."
        )

        st.markdown(
            "Additionally, Age and Total Work Experience demonstrate a logical "
            "positive relationship (0.68), indicating that older employees "
            "typically have more experience."
        )

        st.markdown(
            "Variables such as Distance from Home, Hourly Rate, and Training "
            "Times Last Year show weak correlations with most other variables, "
            "suggesting they may independently influence employee performance."
        )
    # =========================
    # DATA PREPROCESSING
    # =========================
    elif section == "Data Preprocessing":

        st.markdown("## Data Preprocessing")

        st.markdown("### Handling Missing Data")

        st.markdown(
            "The dataset was checked for missing values and no significant missing "
            "data was identified."
        )

        st.markdown("### Encoding of Variables")

        st.markdown(
            "Categorical variables were transformed into numerical format using "
            "Label Encoding and One-Hot Encoding techniques."
        )

        st.markdown("### Feature Scaling")

        st.markdown(
            "Feature scaling was not applied because the final model used was "
            "XGBoost, which is a tree-based algorithm invariant to feature scaling."
        )

        st.markdown("### Feature Selection")

        st.markdown(
            "Recursive Feature Elimination (RFE) with XGBoost was applied to identify "
            "the most influential features affecting employee performance."
        )

        st.markdown("### Target Variable Transformation")

        st.markdown("The performance classes were transformed as:")

        st.markdown("- Good = 0")
        st.markdown("- Excellent = 1")
        st.markdown("- Outstanding = 2")

    # =========================
    # METHODOLOGY
    # =========================
    elif section == "Methodology":

        st.markdown("## Methodology")

        st.markdown("### Models")

        st.markdown(
            "Several supervised machine learning algorithms were trained and compared."
        )

        st.markdown("- XGBoost")
        st.markdown("- Random Forest")
        st.markdown("- Decision Tree")
        st.markdown("- K-Nearest Neighbors (KNN)")
        st.markdown("- Support Vector Machine (SVM)")

        st.markdown(
            "Model performance was evaluated using Stratified Cross-Validation with Accuracy, and " 
            "XGBoost was selected as the final model after achieving the highest prediction accuracy "
            "among all evaluated algorithms." 
        )

        st.markdown("### Train-Test Split")

        st.markdown(
        "To evaluate model performance, the dataset was split into training and testing sets. 80% of"
        "the data was used for training the models, while 20% was reserved for testing to assess"
        "how well the model generalizes to unseen data. The split was stratified to ensure that the"
        "proportion of classes (‘Good’, ‘Excellent’ and ‘Outstanding’) was maintained in both sets, "
        "preserving the balance of the target variable. This approach allows for reliable evaluation"
        "of model performance and helps prevent overfitting, ensuring that predictions remain"
        "accurate when applied in real-world irrigation scenarios. "

        )

        st.markdown("### Feature Engineering")

        st.markdown(
            "Recursive Feature Elimination (RFE) with XGBoost was used for feature selection to identify" 
            "the most influential variables affecting employee performance. The top 10   selected" 
            "features were then used in the final model to improve prediction performance and reduce" 
            "model complexity."           
        )

    # =========================
    # MODELING & EVALUATION
    # =========================
    elif section == "Modeling & Evaluation":

        st.markdown("## Modeling & Evaluation")

        st.subheader("Classification Report (Train)")
        st.image("Classification Report (Train).png", width=700)

        st.markdown(
            "The model achieved a training accuracy of 94%, indicating that it learned the patterns in the" 
            "employee performance dataset extremely well. This means the model correctly classified" 
            "the performance levels of most employees in the training data."
        )

        st.subheader("Classification Report (Test)")
        st.image("Classification Report (Test).png", width=700)

        st.markdown(
               "The classification report shows that the model performed very well in predicting employee" 
               "performance categories (Good, Excellent, and Outstanding). The model achieved an overall" 
               "accuracy of 94%, meaning it correctly classified most employees in the test dataset."
)
        st.markdown(
            "The results indicate strong precision, recall, and F1-scores across all classes," 
            "demonstrating that the model can reliably identify employees in different performance" 
            "levels. In particular, the model showed excellent performance in predicting the ‘Excellent’" 
            "category, which had the highest number of employees in the dataset."
    )


        st.subheader("Confusion Matrix - Train Set")
        st.image("Confusion Matrix - Employee Performance (Train Set).png", width=700)

        st.markdown(
            "The training confusion matrix shows that the model performed extremely well in classifying" 
            "employee performance levels (Good, Excellent, and Outstanding). Most predictions fall on" 
            "the diagonal, indicating correct classifications, with 155 employees correctly identified as" 
            "Good (2), 669 as Excellent (3), and 107 as Outstanding (4)."
    )
        st.markdown(
            "Only a small number of misclassifications were observed, totaling 34 errors. These include" 
            "3 cases where “Good” employees were predicted as Excellent, 3 cases where Excellent" 
            "employees were predicted as Good, 17 cases where Excellent employees were predicted" 
            "as Outstanding, and 11 cases where Outstanding employees were predicted as Excellent." 
            "These errors mainly occur between adjacent classes, which is expected in ordinal" 
            "performance categories. ")

        st.markdown(
            "Overall, the matrix confirms that the model has effectively learned the distinctions between" 
            "performance categories and provides strong evidence of reliable pattern recognition for" 
            "employee performance prediction at INX Future Inc."
    )


        st.subheader("Confusion Matrix - Test Set")
        st.image("Confusion Matrix - Employee Performance (Test Set).png", width=700)

        st.markdown( 
        "The test confusion matrix shows that the model performed very well in classifying employee" 
        "performance levels (Good, Excellent, and Outstanding) on unseen data. Most predictions" 
        "fall on the diagonal, indicating correct classifications, with 25 employees correctly" 
        "identified as Good (2), 181 as Excellent (3), and 19 as Outstanding (4)."
        )
        st.markdown(
        "Only a small number of misclassifications were observed, totaling 15 errors. These include" 
        "4 cases where “Good” employees were predicted as Excellent, 3 cases where Excellent" 
        "employees were predicted as Good, and 8 cases where Outstanding employees were" 
        "predicted as Excellent. No cases were observed where Good employees were predicted as" 
        "Outstanding or Outstanding employees were predicted as Good, indicating that the model" 
        "maintained strong separation between distant performance categories. These errors" 
        "mainly occur between adjacent classes, which is expected in ordinal performance" 
        "categories."
        )
        st.markdown(
        "Overall, the matrix confirms that the model generalizes effectively to unseen employee" 
        "data and demonstrates strong predictive capability in distinguishing between different" 
        "performance levels. The results provide strong evidence that the model can reliably" 
        "support employee performance prediction and workforce decision-making at INX Future" 
        "Inc."
        )

    # =========================
    # RESULTS & INSIGHTS
    # =========================
    elif section == "Results & Insights":

        st.markdown("## Results & Insights")

        st.markdown(
            "The analysis revealed that workplace satisfaction, promotion history, "
            "work-life balance, and managerial experience are among the strongest "
            "factors influencing employee performance."
        )

        st.markdown(
            "Employees with high environment satisfaction and strong work-life balance "
            "were significantly more likely to achieve Excellent or Outstanding "
            "performance ratings."
        )

        st.markdown(
            "The analysis also showed that attrition affects high-performing employees, "
            "indicating the importance of retention and engagement strategies."
        )

        st.markdown(
            "Departments such as Sales and Finance demonstrated higher concentrations "
            "of mid-performing employees and greater performance variability."
        )

    # =========================
    # RECOMMENDATIONS
    # =========================
    elif section == "Recommendations":

        st.markdown("## Recommendations")

        st.markdown(
            "1. **Improve Employee Engagement:** \n"
            "Strengthen employee engagement initiatives to improve productivity "
            "and workplace satisfaction."
        )

        st.markdown(
            "2. **Enhance Promotion Pathways:** \n"
            "Develop clearer promotion and career advancement opportunities to "
            "reduce employee dissatisfaction."
        )

        st.markdown(
            "3. **Strengthen Retention Programs:** \n"
            "Implement targeted retention strategies for high-performing employees."
        )

        st.markdown(
            "4. **Improve Workplace Satisfaction:** \n"
            "Focus on improving work-life balance, relationship satisfaction, "
            "and workplace environment."
        )

        st.markdown(
            "5. **Use Workforce Analytics Continuously:** \n"
            "Adopt workforce analytics dashboards for continuous monitoring of "
            "employee performance and attrition trends."
        )

    # =========================
    # CONCLUSION
    # =========================
    elif section == "Conclusion":

        st.markdown("## Conclusion")

        st.markdown(
            "This project successfully demonstrated how Machine Learning and "
            "Workforce Analytics can be applied to predict employee performance "
            "and identify the major workforce factors influencing organizational "
            "productivity."
        )

        st.markdown(
            "XGBoost achieved the best predictive performance among all evaluated "
            "models, providing strong classification accuracy and reliable workforce "
            "insights."
        )

        st.markdown(
            "The findings highlight the importance of workplace satisfaction, "
            "employee engagement, career progression, and retention strategies "
            "in maintaining a productive workforce."
        )

        st.markdown(
            "Overall, the project provides a strong foundation for evidence-based "
            "HR decision-making and workforce optimization at INX Future Inc."
        )

# =========================================================
# ================= POWER BI PAGE =========================
# =========================================================
# =========================
# --- POWER BI VISUALIZATION TAB ---
# =========================
elif menu == "Power BI Visualization":

    st.markdown(
        "<h1 style='color:blue'>Employee Performance Dashboard</h1>",
        unsafe_allow_html=True
    )

    # =========================================================
    # PROJECT OVERVIEW
    # =========================================================
    st.markdown("## Project Overview")

    st.markdown(
        "This interactive Power BI dashboard provides a data-driven analysis of employee "
        "performance, attrition, satisfaction, and workforce stability using the INX Future Inc. "
        "Employee Performance dataset. The project aims to help HR leaders and decision-makers "
        "identify key factors affecting productivity, retention, employee engagement, and "
        "organizational performance."
    )


    # =========================================================
    # DETAILS SECTION
    # =========================================================
    with st.expander("View Full Dashboard Insights"):

        
        st.markdown(
        "The dashboard delivers actionable insights into performance bottlenecks, attrition risk, "
        "promotion gaps, and workplace satisfaction through interactive visual analytics and "
        "workforce segmentation. INX Future Inc. has a workforce of 1,200 employees across 6 "
        "departments, with an average employee age of 36.92 years and 11.33 years of work experience."
    )

        st.markdown(
        "The analysis highlights emerging organizational challenges related to employee retention, "
        "promotion stagnation, and maintaining high-performing talent, all of which directly impact "
        "productivity, service delivery, and client satisfaction."
    )

        st.markdown("## Workforce Structure & Demographic Insights")

        st.markdown(
            "Most employees fall within the 30–40 age group, indicating a mature workforce. "
            "Employees at this stage prioritize career growth, promotion opportunities, "
            "work-life balance, and recognition."
        )

        # =========================================================
        st.markdown("## Key Performance Indicators (KPIs)")

        st.markdown("- Total Employees: 1,200")
        st.markdown("- Departments: 6")
        st.markdown("- Average Employee Age: 36.92 Years")
        st.markdown("- Average Distance from Home: 9.17 KM")
        st.markdown("- Average Hourly Rate: 65.98")
        st.markdown("- Average Work Experience: 11.33 Years")
        st.markdown("- Attrition Rate: 14.83%")
        st.markdown("- Average Training Time: 2.79 Hrs")

        # =========================================================
        st.markdown("### Employee Performance Distribution (Donut Chart)") 
        st.markdown(
        "The employee performance analysis indicates that the majority of employees at INX Future" 
        "Inc. are performing at a high level. Most employees fall under the Excellent performance" 
        "category, accounting for approximately 72.83% of the workforce. Employees rated as Good" 
        "represent about 16.17%, while 11.00% achieved the highest rating of Outstanding."
        ) 


        # =========================================================
        st.markdown("### Employee Attrition Analysis (100% Stacked Bar Chart) ") 
        st.markdown(
        "Employee attrition highlights a key organizational risk. The ‘Good’ performance group" 
        "shows the highest attrition rate (18.56%), followed by Excellent (14.19%) and" 
        "‘Outstanding’ (13.64%).") 

        st.markdown(
        "Although 'Excellent' employees have a lower attrition rate, they account for the highest" 
        "number of exits due to their large workforce share, indicating a notable loss of core talent.") 
        st.markdown(
        "Overall, the pattern suggests retention challenges among mid-level performers and the" 
        "need to protect high-performing employees critical to productivity and service delivery.") 

        # =========================================================
        st.markdown("### Environment Satisfaction & Performance Relationship (Heatmap)") 
        st.markdown(
        "The analysis reveals a strong relationship between workplace environment satisfaction and" 
        "employee performance. Employees with High environment satisfaction recorded 310" 
        "‘Excellent’ and 54 ‘Outstanding’ ratings, while those with Very High satisfaction recorded" 
        "307 ‘Excellent’ and 51 ‘Outstanding’ ratings.") 

        st.markdown(
        "In contrast, employees with Low satisfaction recorded only 127 ‘Excellent’ ratings but a" 
        "much higher number of ‘Good’ ratings (90), while Medium satisfaction employees recorded" 
        "130 ‘Excellent’ and 98 ‘Good’ ratings. This suggests that higher workplace satisfaction is" 
        "strongly associated with better employee performance outcomes.") 


        st.markdown("### Employee Count by Performance Rating and Attrition (Clustered Column Chart)")

        st.markdown("This chart compares employee retention and attrition across performance levels.") 
        st.markdown(
        "Excellent performers had 750 stayed and 124 left, with an attrition rate of 14.19%," 
        "representing the highest number of total exits and a key loss of high-value talent.")
        st.markdown(
        "Good performers recorded the highest attrition rate at 18.56%, with 158 stayed and 36 left," 
        "indicating higher disengagement in this group. ") 
        st.markdown(
        "Outstanding performers showed the lowest attrition rate of 13.64%, with 114 stayed and" 
        "18 left, reflecting stronger retention among top performers.") 
        st.markdown(
        "Overall, attrition is more concentrated among mid and high performers rather than low" "performance groups, suggesting retention challenges are driven more by engagement and" "career growth factors than performance ability.") 



        # =========================================================
        st.markdown("### Employee Count by Department and Performance Rating (100% Stacked Bar Chart)") 
        st.markdown(
        "This 100% stacked bar chart analyzes the distribution of employee performance ratings" 
        "across departments, showing the proportion of employees classified as Excellent, Good," 
        "and Outstanding within each department.") 

        st.markdown("**Key Insights:** ") 
        st.markdown("- Development recorded one of the strongest performance profiles, with 84.21% Excellent (304 employees) and 12.19% Outstanding (44 employees), while only 3.6% Good (13 employees).") 
        st.markdown("- Data Science had the highest concentration of Excellent performers, with 85% Excellent (17 employees), 10% Outstanding (2 employees), and only 5% Good (1 employee).") 
        st.markdown("- Sales showed greater performance variation, with 67.29% Excellent (251 employees), 23.32% Good (87 employees), and 9.38% Outstanding (35 employees), indicating a larger mid-performing workforce.") 
        st.markdown("- Research & Development maintained a strong performance structure with 68.22% Excellent (234 employees) and 11.95% Outstanding (41 employees), though 19.83% (68 employees) remained in the Good category." 
) 
        st.markdown("- Human Resource demonstrated relatively balanced performance levels, with 70.37% Excellent (38 employees), 18.52% Good (10 employees), and 11.11% Outstanding (6 employees).") 
        st.markdown("- Finance recorded the weakest performance composition, with the lowest proportion of Excellent performers at 61.22% (30 employees) and the highest Good performer share at 30.61% (15 employees).") 
        st.markdown("**Strategic Insight:**") 
        st.markdown(
        "The chart shows that most departments are dominated by Excellent performers, indicating" "a generally high-performing workforce. However, departments such as Sales and Finance" "exhibit larger concentrations of ‘Good’ performers, suggesting opportunities for targeted" ",employee engagement, and performance development initiatives." ) 

        st.markdown("### Dynamic Dashboard Capabilities (Clustered Bar Chart)" ) 

        st.markdown(
        "The dashboard incorporates advanced Field Parameters, interactive filtering, and DAX" "driven analytics to enable flexible workforce exploration without duplicating visuals." ) 
        st.markdown("#### Dynamic Metrics (X-Axis Selection)" ) 
        st.markdown("Users can dynamically switch between multiple workforce metrics:" ) 
        st.markdown("- Total Employees" ) 
        st.markdown("- Attrition Rate" ) 
        st.markdown("- Average Employee Age" ) 
        st.markdown("- Average Hourly Rate" ) 
        st.markdown("- Average Work Experience" ) 
        st.markdown("- Average Training Frequency" ) 
        st.markdown("- Average Years Since Last Promotion" ) 
        st.markdown("- Average Years With Current Manager") 

        st.markdown("#### Dynamic Dimensions (Y-Axis Selection)" ) 
        st.markdown("Users can instantly segment analysis across:" ) 
        st.markdown("- Department" ) 
        st.markdown("- Job Role" ) 
        st.markdown("- Attrition Status") 
        st.markdown("- Gender" ) 
        st.markdown("- Marital Status" ) 
        st.markdown("- Education Background" ) 
        st.markdown("- Job Satisfaction" ) 
        st.markdown("- Environment Satisfaction" )
        st.markdown("- Work-Life Balance" ) 
        st.markdown("- Business Travel Frequency" )  
        st.markdown("This enables multidimensional workforce analysis within a compact dashboard layout."
)  
        st.markdown("### A histogram Employee Age Distribution (Histogram) " ) 
        st.markdown(
        "The histogram shows the age distribution of employees at INX Future Inc., ranging from 18" 
        "to 60 years. Most employees fall within the 30-40 age group, with an average age of around" 
        "36-37, indicating a predominantly mid-career workforce." ) 
        st.markdown(
        "The distribution is slightly right-skewed, meaning there are fewer older employees as age" 
        "increases."
) 

        # =========================================================
        st.markdown("## Technical Stack")

        st.markdown("- Power BI Desktop")
        st.markdown("- DAX (Dynamic KPIs, Field Parameters)")
        st.markdown("- Power Query")
        st.markdown("- Workforce Analytics & Segmentation")

        # =========================================================
        st.markdown("## Data Source")

        st.markdown(
            "INX Future Inc. Employee Performance Dataset"
        )

        st.markdown(
            "**Source:** [Employee Performance Dataset](http://data.iabac.org/exam/p2/data/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls)"
        )

    # =========================================================
    # DASHBOARD IMAGES
    # =========================================================
    st.subheader("Dashboard Preview")

    st.image("Dashboard 1.png", use_container_width=True)
    st.image("Dashboard 2.png", use_container_width=True)

    st.markdown(
        "**[Power BI Portfolio](https://github.com/Okello-Solomon/powerbi-dashboards/blob/main/README.md)**"
        )
# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "INX Future Inc. | Employee Performance Prediction System | Machine Learning Deployment 📊"
)
