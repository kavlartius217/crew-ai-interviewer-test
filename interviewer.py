__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import time
import tempfile
from crewai import Agent, Task, Crew
from crewai_tools import TXTSearchTool, PDFSearchTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import google.generativeai as genai
from gtts import gTTS
import openai  # Import OpenAI module for GPT-4

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'interview_completed' not in st.session_state:
    st.session_state.interview_completed = False
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None

# Page configuration
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("AI Interviewer System")

# Sidebar for API keys and file uploads
with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    openai_key = st.text_input("Enter OpenAI API Key", type="password")
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.header("Upload Files")
    jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Set OpenAI and Groq API keys
if openai_key:
    openai.api_key = openai_key
if groq_key:
    os.environ["GROQ_API_KEY"] = groq_key

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

def play_audio(text):
    audio_path = text_to_speech(text)
    st.session_state.current_audio = audio_path
    st.audio(audio_path, format='audio/mp3')
    time.sleep(1)
    if os.path.exists(audio_path):
        os.unlink(audio_path)

def save_uploaded_file(uploaded_file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def transcribe_audio(audio_file):
    if audio_file:
        audio_path = save_uploaded_file(audio_file, "uploads")
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([audio_path, "transcribe the audio as it is"])
        return result.text
    return None

def initialize_tools():
    if jd_file and resume_file:
        jd_path = save_uploaded_file(jd_file, "uploads")
        resume_path = save_uploaded_file(resume_file, "uploads")
        jd_tool = TXTSearchTool(jd_path)
        resume_tool = PDFSearchTool(resume_path)
        return jd_tool, resume_tool
    return None, None

def create_interviewer_agent(jd_tool, resume_tool):
    return Agent(
        role="Expert Interviewer",
        goal="Conduct a structured interview by asking relevant questions based on the job description and the candidate's resume.",
        backstory="You are an experienced interviewer skilled in assessing candidates based on job requirements and their qualifications.",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=True,
        llm="openai"  # Use OpenAI GPT-4 model
    )

def create_interview_task(agent):
    return Task(
        description="Analyze the job description and resume to generate structured interview questions.",
        agent=agent,
        expected_output="A structured file containing the questions only."
    )

def create_analysis_agent(jd_tool, resume_tool):
    return Agent(
        role="Talent Acquisition Expert",
        goal="Evaluate the candidate's fit for the job based on the job description, resume, and interview script analysis.",
        backstory="An expert in talent acquisition specializing in evaluating candidates based on job descriptions and interview performance.",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=True,
        llm="openai"  # Use OpenAI GPT-4 model
    )

def create_analysis_task(agent):
    return Task(
        description="Analyze interview responses and generate a detailed candidate fit report.",
        agent=agent,
        expected_output="A detailed report assessing the candidate's suitability for the role."
    )

def setup_langchain(questions):
    return ChatGroq(
        model_name="gemma2-9b-it",
        api_key=groq_key,  # Use Groq API key here
        prompt=ChatPromptTemplate.from_messages([ 
            ("system", "You are an Interviewer."),
            ("system", "You have a set of questions: {question_set}. Ask them sequentially, one at a time."),
            ("system", "Only ask the next unanswered question from {question_set}."),
            ("system", "Do not repeat any question already present in chat history."),
            ("system", "Ask only the question itself, without any additional text."),
            ("system", "Never answer the questions yourself"),
            ("system", "After questions are over say Thank You"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{answer}")
        ])
    )

def main():
    if all([groq_key, openai_key, gemini_key, jd_file, resume_file]):
        jd_tool, resume_tool = initialize_tools()
        if jd_tool and resume_tool:
            if not st.session_state.interview_started and st.button("Start Interview"):
                st.session_state.interview_started = True
                interviewer_agent = create_interviewer_agent(jd_tool, resume_tool)
                interview_task = create_interview_task(interviewer_agent)
                crew1 = Crew(agents=[interviewer_agent], tasks=[interview_task], memory=True)
                questions = crew1.kickoff({})
                st.session_state.questions = questions
                st.session_state.chain = setup_langchain(questions)

if __name__ == "__main__":
    main()
