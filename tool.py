import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=groq_api_key)

# Prompt templates remain unchanged

@st.cache_data
def load_data():
    file_path = 'https://raw.githubusercontent.com/forittik/updated_soca_tool/refs/heads/main/Dummy_questions.csv'
    df = pd.read_csv(file_path, header=0, encoding='ISO-8859-1')
    return df

def get_student_data(name, df):
    student_data = df[df['user_id'] == name]
    if student_data.empty:
        return None
    return student_data

def generate_single_student_summary(student_data):
    context = student_data.to_string(index=False)
    summary_chain = summary_prompt_single | llm | StrOutputParser()
    summary = summary_chain.invoke({"context": context})
    return summary

def generate_multiple_students_summary(student_data):
    context = student_data.to_string(index=False)
    summary_chain = summary_prompt_multiple | llm | StrOutputParser()
    summary = summary_chain.invoke({"context": context})
    return summary

def aggregate_student_data(df):
    # Function to convert mixed type list to numeric list and calculate mean
    def process_marks(marks_list):
        numeric_marks = []
        for mark in marks_list:
            if isinstance(mark, (int, float)):
                numeric_marks.append(mark)
            elif isinstance(mark, str):
                try:
                    numeric_marks.append(float(mark.strip()))
                except ValueError:
                    pass  # Ignore non-numeric strings
        return sum(numeric_marks) / len(numeric_marks) if numeric_marks else 0

    # Grouping by user_id and aggregating the subject scores
    aggregated_data = df.groupby('user_id').agg({
        'Marks_got_in_physics_chapters': process_marks,
        'Marks_got_in_chemistry_chapters': process_marks,
        'Marks_got_in_mathematics_chapters': process_marks,
        'productivity_yes_no': lambda x: x.iloc[-1],  # Get the last entry
        'productivity_rate': lambda x: x.iloc[-1],    # Get the last entry
        'emotional_factors': lambda x: ' '.join(x.dropna().astype(str))
    }).reset_index()
    
    return aggregated_data

def process_students(names, df):
    if isinstance(names, str):
        student_data = get_student_data(names, df)
        if student_data is None:
            return f"No data found for student: {names}"
        return generate_single_student_summary(student_data)
    elif isinstance(names, list):
        combined_data = pd.concat([get_student_data(name, df) for name in names if get_student_data(name, df) is not None])
        if combined_data.empty:
            return "No data found for the given students."
        return generate_multiple_students_summary(combined_data)

# Updated function to create performance pie chart with mean marks
def create_performance_pie_chart(student_data):
    subjects = ['Physics', 'Chemistry', 'Mathematics']
    marks = []
    for subject in subjects:
        marks_column = f'Marks_got_in_{subject.lower()}_chapters'
        marks.append(student_data[marks_column].iloc[0])
    
    fig = go.Figure(data=[go.Pie(labels=subjects, values=marks)])
    fig.update_layout(title=f"Mean Performance Distribution for {student_data['user_id'].iloc[0]}")
    return fig

st.title("B2B Dashboard")
df = load_data()
# Aggregate the student data
aggregated_df = aggregate_student_data(df)
student_names = aggregated_df['user_id'].tolist()
selected_names = st.multiselect("Select student(s) to analyze:", student_names)

if st.button("Analyze student data"):
    if selected_names:
        summary = process_students(selected_names, df)  # Use original df for detailed summary
        st.write(summary)
        
        # Create and display pie charts for each selected student
        for name in selected_names:
            student_data = aggregated_df[aggregated_df['user_id'] == name]
            if not student_data.empty:
                fig = create_performance_pie_chart(student_data)
                st.plotly_chart(fig)
    else:
        st.warning("Please select at least one student.")
