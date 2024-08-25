import streamlit as st
from crewai import Agent, Task, Process, Crew
from langchain_community.llms import Ollama

# Load Local model through Ollama
llm_model = Ollama(model="stablelm-zephyr")

# Define Agents
app_developer = Agent(
    role="App Developer",
    goal="Provide information and guidance on mobile app development.",
    backstory="""You are an expert in mobile app development, with extensive experience in developing iOS and Android apps using 
        various technologies such as Swift, Kotlin, Flutter, and React Native. You can guide users through the process of app development, 
        from planning to deployment.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_model
)

web_developer = Agent(
    role="Web Developer",
    goal="Provide information and guidance on web development.",
    backstory="""You are an expert in web development, skilled in front-end and back-end technologies including HTML, CSS, JavaScript, 
        React, Node.js, and Django. You can assist with anything related to building and maintaining websites.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_model
)

ai_ml_developer = Agent(
    role="AI/ML Developer",
    goal="Provide information and guidance on artificial intelligence and machine learning.",
    backstory="""You are an AI/ML specialist with deep knowledge of machine learning algorithms, neural networks, data science, and AI development. 
        You can help with questions about AI model development, data processing, and using AI in real-world applications.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_model
)

blockchain_developer = Agent(
    role="Blockchain Developer",
    goal="Teach about blockchain technology, Hyperledger Fabric 2.5, and Web3.",
    backstory="""You are an expert Blockchain Developer with extensive knowledge of blockchain technology, Web3, and Hyperledger Fabric 2.5. 
        You are passionate about educating others on these topics and excel at breaking down complex concepts into understandable lessons. 
        Your goal is to help others grasp both the fundamentals and advanced topics in these areas.""",
    verbose=True,
    allow_delegation=False,
    llm=llm_model
)

# Streamlit App Setup
st.title("Multi-Agent Developer Chatbot")
st.write("Select the type of developer you want to interact with and ask any questions!")

# Developer Selection
developer_options = {
    "App Developer": app_developer,
    "Web Developer": web_developer,
    "AI/ML Developer": ai_ml_developer,
    "Blockchain Developer": blockchain_developer,
}

developer_choice = st.selectbox("Choose a Developer:", list(developer_options.keys()))

# User input for the chatbot
user_input = st.text_input("You: ", "")

if user_input:
    # Select the appropriate agent based on the user's choice
    selected_developer = developer_options[developer_choice]

    # Define task based on user input
    task = Task(
        description=user_input,
        expected_output="A detailed and informative response",
        agent=selected_developer,
    )

    # Create the crew with the selected developer agent and the task
    crew = Crew(
        agents=[selected_developer],
        tasks=[task],
        verbose=2,
        process=Process.sequential,
    )

    # Process user input and get the result
    with st.spinner('Processing your request...'):
        result = crew.kickoff()

    # Since the result is a string, directly display it
    st.write("Chatbot:", result)

    # Optional: Show debugging output based on user preference
    
