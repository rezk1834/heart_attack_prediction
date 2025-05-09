import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# Function to load data - Allow user to upload file
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Create a sample dataset for demonstration if no file is uploaded
        st.warning("No data file uploaded. Creating a sample dataset for demonstration.")
        # Create sample data based on the headers provided by the user
        data = {
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'] * 20,
            'Age': np.random.randint(25, 85, 100),
            'Blood Pressure (mmHg)': [f"{np.random.randint(100, 180)}/{np.random.randint(60, 110)}" for _ in range(100)],
            'Cholesterol (mg/dL)': np.random.randint(150, 350, 100),
            'Has Diabetes': np.random.choice([True, False], 100),
            'Smoking Status': np.random.choice(['Current Smoker', 'Former Smoker', 'Never Smoked'], 100),
            'Chest Pain Type': np.random.choice(['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'], 100),
            'Treatment': np.random.choice(['Medication', 'PCI', 'CABG', 'None'], 100)
        }
        df = pd.DataFrame(data)
        
        # Add a target variable for demonstration
        df['Heart Attack Risk'] = np.random.choice([0, 1], 100, p=[0.7, 0.3])
    
    # Preprocessing
    # Split Blood Pressure column
    if 'Blood Pressure (mmHg)' in df.columns:
        split_cols = df['Blood Pressure (mmHg)'].str.split('/', expand=True)
        split_cols.columns = ['Systolic', 'Diastolic']
        
        # Convert to numeric data
        split_cols['Systolic'] = pd.to_numeric(split_cols['Systolic'])
        split_cols['Diastolic'] = pd.to_numeric(split_cols['Diastolic'])
        
        # Position the columns next to Blood Pressure column
        bp_index = df.columns.get_loc('Blood Pressure (mmHg)')
        for i, col in enumerate(split_cols.columns):
            df.insert(bp_index + 1 + i, col, split_cols[col])
        
        # Calculate BP_Ratio
        df['BP_Ratio'] = (df['Systolic'] / df['Diastolic']).round(2)
        bp_ratio_index = df.columns.get_loc('BP_Ratio')
        df.insert(bp_index + 3, 'BP_Ratio', df.pop('BP_Ratio'))
    
    # Encode categorical features
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    if 'Smoking Status' in df.columns:
        df['Smoking Status'] = df['Smoking Status'].map({
            'Never Smoked': 0, 
            'Former Smoker': 1, 
            'Current Smoker': 2
        })
    
    if 'Chest Pain Type' in df.columns:
        df['Chest Pain Type'] = df['Chest Pain Type'].map({
            'Typical Angina': 0, 
            'Atypical Angina': 1, 
            'Non-Anginal Pain': 2, 
            'Asymptomatic': 3
        })
    
    # One-hot encode Treatment if it exists
    if 'Treatment' in df.columns:
        df = pd.get_dummies(df, columns=['Treatment'], drop_first=False)
    
    # Convert boolean to 0/1
    for column in df.columns:
        if df[column].dtype == 'bool' and df[column].isin([True, False]).all():
            df[column] = df[column].map({False: 0, True: 1})
    
    # If there's no Heart Attack Risk column, create a dummy one for testing
    if 'Heart Attack Risk' not in df.columns:
        st.warning("No 'Heart Attack Risk' column found. Adding a random one for demonstration.")
        df['Heart Attack Risk'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    
    # Move Heart Attack Risk to the last column
    cols = [col for col in df.columns if col != 'Heart Attack Risk']
    df = df[cols + ['Heart Attack Risk']]
    
    return df

# Function to train models
def train_models(df):
    # Split data into features and target
    X = df.drop(columns=['Heart Attack Risk']).values
    y = df['Heart Attack Risk'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    
    # Standard scaling for KNN
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    # Train models
    models = {}
    
    # SVM
    svm = SVC(kernel='rbf', random_state=0, probability=True)
    svm.fit(X_train, y_train)
    models['SVM'] = {'model': svm, 'scaler': None}
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)  # Using a smaller n_neighbors for the sample data
    knn.fit(X_train_scaled, y_train)
    models['KNN'] = {'model': knn, 'scaler': sc}
    
    # Logistic Regression
    log_reg = LogisticRegression(random_state=0)
    log_reg.fit(X_train, y_train)
    models['Logistic Regression'] = {'model': log_reg, 'scaler': None}
    
    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    models['Gaussian NB'] = {'model': gnb, 'scaler': None}
    
    # Calculate and store accuracies
    accuracies = {}
    for name, model_info in models.items():
        if model_info['scaler'] is not None:
            pred = model_info['model'].predict(X_test_scaled)
        else:
            pred = model_info['model'].predict(X_test)
        accuracies[name] = accuracy_score(y_test, pred) * 100
    
    return models, accuracies, list(df.columns[:-1])

# Function to make prediction
def predict_heart_attack_risk(input_data, model_name, models, features):
    input_df = pd.DataFrame([input_data], columns=features)
    model_info = models[model_name]
    
    # Convert the input_df to numpy array
    input_array = input_df.values
    
    # Apply scaling if needed
    if model_info['scaler'] is not None:
        input_array = model_info['scaler'].transform(input_array)
    
    # Get prediction probability
    probability = model_info['model'].predict_proba(input_array)[0][1]
    prediction = 1 if probability >= 0.5 else 0
    
    return prediction, probability

# Main function
def main():
    st.title("❤️ Heart Attack Risk Prediction")
    st.write("This application predicts the risk of heart attack based on various health parameters.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your data (CSV file)", type="csv")
    
    # Load data
    df = load_data(uploaded_file)
    
    # Train models
    with st.spinner("Training models..."):
        models, accuracies, feature_names = train_models(df)
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Prediction", "Dataset Information", "Model Performance"]
    selection = st.sidebar.radio("Go to", pages)
    
    if selection == "Prediction":
        st.header("Predict Heart Attack Risk")
        
        # Create two columns for input form
        col1, col2 = st.columns(2)
        
        # Define input fields based on the dataset columns
        input_data = {}
        
        with col1:
            if 'Age' in df.columns:
                input_data['Age'] = st.slider("Age", 18, 100, 40)
            
            if 'Gender' in df.columns:
                gender = st.selectbox("Gender", options=["Male", "Female"])
                input_data['Gender'] = 1 if gender == "Male" else 0
            
            if 'Cholesterol (mg/dL)' in df.columns:
                input_data['Cholesterol (mg/dL)'] = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
            
            if 'Has Diabetes' in df.columns:
                input_data['Has Diabetes'] = int(st.checkbox("Has Diabetes"))
            
            if 'Smoking Status' in df.columns:
                smoking_status = st.selectbox("Smoking Status", 
                                           ["Never Smoked", "Former Smoker", "Current Smoker"])
                input_data['Smoking Status'] = {
                    "Never Smoked": 0, 
                    "Former Smoker": 1, 
                    "Current Smoker": 2
                }[smoking_status]
        
        with col2:
            if 'Systolic' in df.columns and 'Diastolic' in df.columns:
                systolic = st.slider("Systolic BP", 80, 220, 120)
                diastolic = st.slider("Diastolic BP", 40, 140, 80)
                input_data['Systolic'] = systolic
                input_data['Diastolic'] = diastolic
                
                if 'BP_Ratio' in df.columns:
                    bp_ratio = round(systolic / diastolic, 2)
                    input_data['BP_Ratio'] = bp_ratio
                    st.info(f"Calculated BP Ratio: {bp_ratio}")
            
            if 'Chest Pain Type' in df.columns:
                chest_pain = st.selectbox("Chest Pain Type", 
                                       ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
                input_data['Chest Pain Type'] = {
                    "Typical Angina": 0, 
                    "Atypical Angina": 1, 
                    "Non-Anginal Pain": 2, 
                    "Asymptomatic": 3
                }[chest_pain]
            
            # Treatment checkboxes if they exist in the dataframe
            treatment_cols = [col for col in df.columns if col.startswith('Treatment_')]
            if treatment_cols:
                st.subheader("Treatment Options")
                for col in treatment_cols:
                    treatment_name = col.replace('Treatment_', '')
                    input_data[col] = int(st.checkbox(f"{treatment_name}"))
        
        # Fill in any missing columns with default values
        for col in feature_names:
            if col not in input_data:
                if df[col].dtype == np.int64 or df[col].dtype == np.float64:
                    input_data[col] = df[col].mean()
                else:
                    input_data[col] = df[col].mode()[0]
        
        # Select model for prediction
        model_name = st.selectbox("Select Model", list(models.keys()))
        
        # Make prediction button
        if st.button("Predict"):
            prediction, probability = predict_heart_attack_risk(input_data, model_name, models, feature_names)
            
            # Show prediction result
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"⚠️ High Risk of Heart Attack (Probability: {probability:.2%})")
                st.warning("Please consult with a healthcare professional for proper evaluation.")
            else:
                st.success(f"✅ Low Risk of Heart Attack (Probability: {probability:.2%})")
                st.info("Keep maintaining a healthy lifestyle!")
            
            # Display gauge chart for probability
            fig, ax = plt.subplots(figsize=(8, 2))
            
            # Create a custom gauge chart
            risk_levels = [(0, 0.25, 'green'), (0.25, 0.5, 'yellowgreen'), 
                          (0.5, 0.75, 'orange'), (0.75, 1.0, 'red')]
            
            for i, (start, end, color) in enumerate(risk_levels):
                ax.barh(0, end-start, left=start, height=0.5, color=color, alpha=0.7)
            
            # Add a marker for the probability
            ax.plot(probability, 0, 'ko', markersize=12)
            ax.plot(probability, 0, 'wo', markersize=8)
            
            # Customize the plot
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 1)
            ax.set_yticks([])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_title('Heart Attack Risk Level')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Display the plot
            st.pyplot(fig)
    
    elif selection == "Dataset Information":
        st.header("Dataset Information")
        
        # Display dataset overview
        st.subheader("Dataset Overview")
        st.write(f"Number of records: {df.shape[0]}")
        st.write(f"Number of features: {df.shape[1] - 1}")  # Exclude target variable
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Display feature distributions
        st.subheader("Feature Distributions")
        
        # Distribution of Heart Attack Risk
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Heart Attack Risk', data=df, palette='pastel', ax=ax)
        ax.set_title('Distribution of Heart Attack Risk')
        ax.set_xlabel('Heart Attack Risk')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['No', 'Yes'])
        st.pyplot(fig)
        
        # Age distribution if exists
        if 'Age' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['Age'], kde=True, bins=15, color='skyblue', ax=ax)
            ax.set_title('Age Distribution')
            ax.set_xlabel('Age')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        numeric_df = df.select_dtypes(include=['number'])
        
        # Compute correlation matrix
        corr = numeric_df.corr()
        
        # Generate heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt=".2f", ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
    
    elif selection == "Model Performance":
        st.header("Model Performance")
        
        # Display model accuracies
        st.subheader("Model Accuracies")
        
        # Create a bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        models_df = pd.DataFrame({
            'Model': list(accuracies.keys()),
            'Accuracy (%)': list(accuracies.values())
        })
        
        # Sort by accuracy
        models_df = models_df.sort_values('Accuracy (%)', ascending=False)
        
        # Plot bar chart
        sns.barplot(x='Accuracy (%)', y='Model', data=models_df, palette='viridis', ax=ax)
        ax.set_title('Model Accuracy Comparison')
        ax.set_xlim(0, 100)
        
        # Add accuracy values as text
        for i, v in enumerate(models_df['Accuracy (%)']):
            ax.text(v + 1, i, f"{v:.2f}%", va='center')
        
        st.pyplot(fig)
        
        # Model description
        st.subheader("Model Descriptions")
        
        st.markdown("""
        **Support Vector Machine (SVM)**
        - A powerful classification algorithm that works by finding the hyperplane that best separates classes in high-dimensional space
        - Good for complex, non-linear relationships with proper kernel functions
        
        **K-Nearest Neighbors (KNN)**
        - Makes predictions based on the k closest points in the feature space
        - Simple but effective algorithm, particularly when features have similar scales (after standardization)
        
        **Logistic Regression**
        - A linear model for binary classification that estimates the probability of an event
        - Works well for linearly separable data and provides interpretable coefficients
        
        **Gaussian Naive Bayes**
        - Based on Bayes' theorem with an assumption of independence between predictors
        - Particularly useful when the dimensionality of the inputs is high
        """)

# Run the application
if __name__ == '__main__':
    main()
