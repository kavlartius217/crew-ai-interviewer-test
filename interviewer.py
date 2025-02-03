__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from crewai import Agent, Task, Crew
from crewai_tools import TXTSearchTool, PDFSearchTool
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
import os
import warnings
import gc

# Suppress warnings
warnings.filterwarnings("ignore", message="Overriding of current TracerProvider is not allowed")

# Set up the environment
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Streamlit app title
st.title("Crew AI Interviewer")

# Initialize Message History class
class MessageHistory:
    def __init__(self):
        self.l1 = []
    
    def add(self, user_message, ai_response):
        self.l1.append({'role': 'user', 'content': user_message})
        self.l1.append({'role': 'assistant', 'content': ai_response})
    
    def show_history(self):
        return self.l1

# Cache initialization functions
@st.cache_resource
def initialize_llm():
    return ChatGroq(
        api_key=st.secrets['GROQ_API_KEY'],
        model="gemma2-7b-it",
        temperature=0
    )

@st.cache_resource
def initialize_agents_and_tools(jd_file, resume_file):
    jd_tool = TXTSearchTool(jd_file)
    resume_tool = PDFSearchTool(resume_file)
    
    interviewer_agent = Agent(
        role="Expert Interviewer",
        goal="Conduct a structured interview by asking relevant questions",
        backstory="You are an experienced interviewer skilled in assessing candidates.",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=False
    )
    return jd_tool, resume_tool, interviewer_agent

@st.cache_data
def save_files(job_description_file, resume_file):
    with open("Job_Description.md", "wb") as f:
        f.write(job_description_file.getbuffer())
    with open("Resume.pdf", "wb") as f:
        f.write(resume_file.getbuffer())

@st.cache_data
def generate_interview_questions(crew):
    return crew.kickoff({})

@st.cache_data
def perform_analysis(history_data, jd_tool, resume_tool):
    analysis_agent = Agent(
        role="Talent Acquisition Expert",
        goal="Evaluate candidate fit",
        backstory="You are an expert in talent acquisition",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=False
    )
    
    analysis_task = Task(
        description="Analyze the interview script and evaluate candidate's suitability",
        agent=analysis_agent,
        output_file="analysis.md"
    )
    
    crew2 = Crew(
        agents=[analysis_agent],
        tasks=[analysis_task],
        memory=True
    )
    
    return crew2.kickoff({"interview_script": history_data})

# File uploaders in sidebar
st.sidebar.header("Upload Files")
job_description_file = st.sidebar.file_uploader("Upload Job Description (MD)", type="md")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type="pdf")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = MessageHistory()

if job_description_file and resume_file:
    # Save uploaded files
    save_files(job_description_file, resume_file)
    
    # Initialize tools and agents
    jd_tool, resume_tool, interviewer_agent = initialize_agents_and_tools("Job_Description.md", "Resume.pdf")
    
    # Create interview task
    interview_task = Task(
        description=(
            "Analyze the job description and candidate's resume. "
            "Formulate 10-12 well-structured questions evaluating skills, experience, and role alignment. "
            "Include technical, situational, and behavioral questions."
        ),
        agent=interviewer_agent,
        expected_output="A structured file containing the questions only.",
        output_file='interview.md',
    )
    
    # Generate questions if not already generated
    if 'questions' not in st.session_state:
        with st.spinner('Generating interview questions...'):
            crew1 = Crew(
                agents=[interviewer_agent],
                tasks=[interview_task],
                memory=True
            )
            st.session_state.questions = generate_interview_questions(crew1)
    
    # Display generated questions
    st.header("Generated Interview Questions")
    st.write(st.session_state.questions)
    
    # Initialize Langchain for interview
    llm = initialize_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Interviewer"),
        ("system", "You have a set of questions: {question_set}. Ask them sequentially, one at a time."),
        ("system", "Only ask the next unanswered question from {question_set}."),
        ("system", "Do not repeat any question already present in chat history."),
        ("system", "Ask only the question itself, without any additional text."),
        ("system", "Never answer the questions yourself"),
        ("system", "After questions are over say Thank You"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{answer}")
    ])
    
    chain = prompt | llm
    
    def interview_chain(answer):
        ai_response = chain.invoke({
            "question_set": st.session_state.questions,
            "answer": answer,
            "chat_history": st.session_state.history.l1
        })
        st.session_state.history.add(answer, ai_response.content)
        return ai_response.content
    
    # Interview interface
    st.header("Interview Session")
    user_input = st.text_input("Your Answer:")
    if st.button("Submit"):
        with st.spinner('Processing your response...'):
            ai_response = interview_chain(user_input)
            st.write(ai_response)
            gc.collect()  # Clean up memory
    
    # Analysis section
    l1 = st.session_state.history.show_history()
    if l1 and len(l1) > 0 and l1[-1]['role'] == 'assistant' and l1[-1]['content'] == 'Thank You':
        st.header("Candidate Analysis")
        with st.spinner('Analyzing interview responses...'):
            analysis_result = perform_analysis(l1, jd_tool, resume_tool)
            st.write(analysis_result)
            gc.collect()  # Clean up memory

else:
    st.warning("Please upload both the job description and resume to proceed.")
