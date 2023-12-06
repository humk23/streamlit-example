import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!!!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""


import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd


import streamlit as st

# Load data
file_path = '/Users/krishum/Downloads/habit_topics-2.xlsx'
habits_data = pd.read_excel(file_path)

def create_description(row):
    habit_name = row['HABIT']
    difficulty = row['DIFFICULTY']
    topic1 = row['TOPIC']
    topic2 = row['TOPIC 2']
    description = row['DESCRIPTION']

    if pd.notna(topic2):
        return f"{habit_name} is a {difficulty} difficulty habit related to {topic1} and {topic2}."
    else:
        return f"{habit_name} is a {difficulty} difficulty habit related to {topic1}."

def recommend_habits_content_based(user_preferences, num_recommendations=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    habit_descriptions = [create_description(row) for _, row in habits_data.iterrows()]
    habit_matrix = vectorizer.fit_transform(habit_descriptions)

    user_vector = vectorizer.transform([user_preferences])

    cosine_similarities = linear_kernel(user_vector, habit_matrix).flatten()

    recommended_indices = cosine_similarities.argsort()[:-num_recommendations-1:-1]

    recommended_habits = [habits_data.iloc[i] for i in recommended_indices]
    return recommended_habits

# Streamlit app
st.title("Habit Recommender App")


# Get user input
user_preferences = st.text_input("Enter your preferences or goals:")

# Check if user input is not empty
if user_preferences:
    # Get and print recommended habits using content-based filtering
    recommended_habits_content_based = recommend_habits_content_based(user_preferences)
    
    st.subheader("Recommended Habits (Content-Based Filtering):")
    for habit in recommended_habits_content_based:
        st.write(f"- {habit['HABIT']} (Difficulty: {habit['DIFFICULTY']}, Topic: {habit['TOPIC']}, \n    Description: {habit['DESCRIPTION']})")

