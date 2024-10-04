def create_performance_pie_chart(student_data):
    subjects = ['Physics', 'Chemistry', 'Mathematics']
    marks = []
    
    for subject in subjects:
        marks_column = f'Marks_got_in_{subject.lower()}_chapters'
        # Convert string to a list and then sum the integers
        marks_list = student_data[marks_column].iloc[0]
        # Check if the marks are stored as strings of lists
        if isinstance(marks_list, str):
            marks_list = eval(marks_list)  # Safely evaluate string to list if necessary
        
        # Sum the marks for the subject, ensuring correct integer conversion
        total_marks = sum(map(int, marks_list))
        marks.append(total_marks)

    # Calculate percentages
    total_sum = sum(marks)
    if total_sum > 0:
        percentages = [(mark / total_sum) * 100 for mark in marks]
    else:
        percentages = [0] * len(marks)

    fig = go.Figure(data=[go.Pie(labels=subjects, values=percentages)])
    fig.update_layout(title=f"Performance Distribution for {student_data['user_id'].iloc[0]}")
    return fig
