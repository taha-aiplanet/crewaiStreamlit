import streamlit as st
import os
from crewai import Agent, Task, Crew, Process

from langchain_community.tools import DuckDuckGoSearchRun# Initialize the tool
search_tool = DuckDuckGoSearchRun()


# api_key = st.secrets['OPENAI_API_KEY']

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# Define the Researcher agent
researcher = Agent(
    role='Research Mathematician',
    goal='Solve the math question',
    backstory="""
        You are a Research Mathematician tasked with finding answers to math questions. Using available search tools, you need to locate and return the correct answer to the given math question. If the answer is not found, you should return 'Answer not found on web.' Your goal is to generate 100% accurate results, as this application is intended for students. Each response must include both the Question and the Answer.
    """,
    verbose=True,
    allow_delegation=True,
    tools=[search_tool]
)

# Define the Writer agent
writer = Agent(
    role='Math Professor',
    goal='Write the answer in a step-by-step format, make sure its step by step.',
    backstory="""
        You are an expert Math professor who assists students by providing step-by-step solutions to math problems. You are given both the Question and the Answer. Based on this information, you must generate an accurate, step-by-step solution within 3-4 meaningful steps that makes sense. As an expert, you must always be truthful and never provide incorrect information.
    """,
    verbose=True,
    allow_delegation=True
)

# Function to process the task
def process_question(question):
    # Create the tasks for the agents
    task1 = Task(
        description=(
            "You are given a math question: {question} and your task is to find the correct answer using available search tools. If the answer is not available, return 'Answer not found on web.' Ensure the response includes both the Question and the Answer"),
        expected_output="""
            A response containing the original Question and the Answer or 'Answer not found on web.'
        """,
        agent=researcher,
        inputs=['question']
    )

    task2 = Task(
        description=(
            "You are provided with a math question : {question}. Your task is to write a step-by-step solution that explains how to arrive at the given answer in 3-4 meaningful steps. Ensure the solution is clear, accurate, and easy to understand for students."
        ),
        expected_output="""
            A step-by-step solution to the given math question, written in 3-4 meaningful steps.
        """,
        agent=writer,
        inputs=['question']

    )

    # Instantiate the crew with a sequential process
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=2  # You can set it to 1 or 2 for different logging levels
    )

    input_data = {'question': question}
    result = crew.kickoff(inputs=input_data)
    return result

# Streamlit App
st.title("Crewai Mathpal")
st.write("Enter a math question and get a detailed step-by-step solution.")

question = st.text_input("Math Question")

if st.button("Solve"):
    if question:
        result = process_question(question)
        st.write("## Result")
        st.write(result)
    else:
        st.write("Please enter a math question.")
