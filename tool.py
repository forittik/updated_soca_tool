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

# Define the prompt templates
summary_prompt_single = PromptTemplate.from_template("""\
Here is the data for the student:
The given structured data is complex, but the structure can be broken down as follows:
1. Column 1: user_id – A unique identifier for each student.
2. Columns 2 to 10: Subject_Scores – A collection of chapters for each subject (Physics, Chemistry, and Mathematics), along with the questions asked from each chapter and the corresponding marks obtained.
3. Column 11: Productivity_yes_no – This indicates whether the student was considered productive or not ("Yes" or "No").
4. Column 12: Productivity_rate – A numerical scale ranging from 1 to 10, reflecting the student's productivity based on their overall performance in the subjects.
5. Column 13: Emotional_factors – Captures details about any emotional or psychological elements that might have affected the student's performance.

{context}

Based on this data, generate a descriptive summary of the student's strengths, opportunities, and challenges.
Also provide some specific suggestions on how the student can improve. Avoid generic statements.
""")

summary_prompt_multiple = PromptTemplate.from_template("""\
Here is the data for the students:
The given structured data is complex, but the structure can be broken down as follows:
1. Column 1: user_id – A unique identifier for each student.
2. Columns 2 to 10: Subject_Scores – A collection of chapters for each subject (Physics, Chemistry, and Mathematics), along with the questions asked from each chapter and the corresponding marks obtained.
3. Column 11: Productivity_yes_no – This indicates whether the student was considered productive or not ("Yes" or "No").
4. Column 12: Productivity_rate – A numerical scale ranging from 1 to 10, reflecting the student's productivity based on their overall performance in the subjects.
5. Column 13: Emotional_factors – Captures details about any emotional or psychological elements that might have affected the student's performance.

{context}

Generate a detailed summary of the strengths, opportunities, and challenges for these students.
Provide specific insights for each student, and compare their strengths and areas for improvement.
Suggest ways they can learn from each other and address their challenges collaboratively where applicable.
""")

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
    # Grouping by user_id and aggregating the subject scores
    aggregated_data = df.groupby('user_id').agg(
        lambda x: list(x.dropna().astype(str)) if x.name in ['Marks_got_in_physics_chapters', 
                                                             'Marks_got_in_chemistry_chapters', 
                                                             'Marks_got_in_mathematics_chapters'] 
        else ' '.join(x.dropna().astype(str))
    ).reset_index()
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

# Function to create performance pie chart
def create_performance_pie_chart(student_data):
    subjects = ['Physics', 'Chemistry', 'Mathematics']
    marks = []
    for subject in subjects:
        marks_column = f'Marks_got_in_{subject.lower()}_chapters'
        marks.append(sum(map(int, student_data[marks_column].iloc[0])))
    
    fig = go.Figure(data=[go.Pie(labels=subjects, values=marks)])
    fig.update_layout(title=f"Performance Distribution for {student_data['user_id'].iloc[0]}")
    return fig

# Streamlit App
st.title("B2B Dashboard")
df = load_data()

# Aggregate the student data
aggregated_df = aggregate_student_data(df)
student_names = aggregated_df['user_id'].tolist()

# Sidebar Dropdown for student selection (now multiselect)
selected_names = st.sidebar.multiselect("Select student(s) to analyze:", student_names)

if st.button("Analyze Student Data"):
    if selected_names:
        summary = process_students(selected_names, aggregated_df)
        st.write(summary)
    else:
        st.warning("Please select at least one student.")

# Button to generate plots
if st.button("Show Performance Pie Chart"):
    if selected_names:
        for name in selected_names:
            student_data = get_student_data(name, aggregated_df)
            if student_data is not None:
                fig = create_performance_pie_chart(student_data)
                st.plotly_chart(fig)
    else:
        st.warning("Please select at least one student.")
