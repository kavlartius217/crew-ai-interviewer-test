__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import time
import tempfile
from crewai import Agent, Task, Crew
from crewai_tools import TXTSearchTool, PDFSearchTool
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import google.generativeai as genai
from gtts import gTTS

# Initialize session state variables
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'interview_completed' not in st.session_state:
    st.session_state.interview_completed = False
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None

# Page Configuration
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("AI Interviewer System")

# Sidebar for API keys and file uploads
with st.sidebar:
    st.header("Configuration")
    openai_key = st.text_input("Enter OpenAI API Key", type="password")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    gemini_key = st.text_input("Enter Gemini API Key", type="password")

    st.header("Upload Files")
    jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    audio_file = st.file_uploader("Upload Audio Response (optional)", type=["mp3", "wav", "m4a"])

# Function to Convert Text to Speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# Function to Play Audio
def play_audio(text):
    audio_path = text_to_speech(text)
    st.session_state.current_audio = audio_path
    st.audio(audio_path, format='audio/mp3')
    time.sleep(1)
    if os.path.exists(audio_path):
        os.unlink(audio_path)

# Function to Save Uploaded Files
def save_uploaded_file(uploaded_file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Function to Transcribe Audio using Gemini API
def transcribe_audio(audio_file):
    if audio_file:
        audio_path = save_uploaded_file(audio_file, "uploads")
        genai.configure(api_key=gemini_key)
        audio_data = genai.upload_file(audio_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([audio_data, "transcribe the audio as it is"])
        return result.text
    return None

# Initialize AI Tools
def initialize_tools():
    if jd_file and resume_file:
        jd_path = save_uploaded_file(jd_file, "uploads")
        resume_path = save_uploaded_file(resume_file, "uploads")

        jd_tool = TXTSearchTool(jd_path)
        resume_tool = PDFSearchTool(resume_path)
        return jd_tool, resume_tool
    return None, None

# Create Interviewer Agent
def create_interviewer_agent(jd_tool, resume_tool):
    return Agent(
        role="Interviewer",
        goal="Conduct an AI-driven interview and assess the candidate",
        backstory="A professional AI interviewer trained in various industries.",
        verbose=True,
        tools=[jd_tool, resume_tool],
        allow_delegation=False,
        llm=ChatGroq(model_name="mixtral", api_key=groq_key)
    )

# Create Interview Task
def create_interview_task(agent, jd_tool, resume_tool):
    return Task(
        description="Conduct an AI-driven job interview based on the job description and resume.",
        agent=agent,
        tools=[jd_tool, resume_tool],
        expected_output="A structured interview process with recorded responses."
    )

# Create Analysis Agent
def create_analysis_agent(jd_tool, resume_tool):
    return Agent(
        role="Interview Evaluator",
        goal="Analyze the candidate's responses and provide feedback.",
        backstory="An expert AI hiring manager who evaluates interviews.",
        verbose=True,
        tools=[jd_tool, resume_tool],
        allow_delegation=False,
        llm=ChatGroq(model_name="mixtral", api_key=groq_key)
    )

# Create Analysis Task
def create_analysis_task(agent):
    return Task(
        description="Analyze interview performance and generate feedback.",
        agent=agent,
        expected_output="Comprehensive feedback report."
    )

# Set up LangChain pipeline
def setup_langchain(questions):
    return ChatGroq(
        model_name="mixtral",
        api_key=groq_key,
        prompt=ChatPromptTemplate.from_messages([
            ("system", "You are an AI interviewer."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "Question: {question_set}. User's Answer: {answer}")
        ])
    )

# Main Function
def main():
    if all([openai_key, groq_key, gemini_key, jd_file, resume_file]):
        os.environ['OPENAI_API_KEY'] = openai_key
        jd_tool, resume_tool = initialize_tools()
        
        if jd_tool and resume_tool:
            if not st.session_state.interview_started and st.button("Start Interview"):
                st.session_state.interview_started = True
                
                # Generate Interview Questions
                interviewer_agent = create_interviewer_agent(jd_tool, resume_tool)
                interview_task = create_interview_task(interviewer_agent, jd_tool, resume_tool)
                crew1 = Crew(agents=[interviewer_agent], tasks=[interview_task], memory=True)
                questions = crew1.kickoff({})
                st.session_state.questions = questions
                st.session_state.chain = setup_langchain(questions)

                # Start with the First Question
                response = st.session_state.chain.invoke({
                    "question_set": questions,
                    "answer": "",
                    "chat_history": st.session_state.message_history
                })
                st.session_state.message_history.append({'role': 'assistant', 'content': response.content})

            # Chat Interface
            if st.session_state.interview_started:
                for i, message in enumerate(st.session_state.message_history):
                    if message['role'] == 'assistant':
                        st.write("Interviewer:", message['content'])
                        if i == len(st.session_state.message_history) - 1:
                            play_audio(message['content'])
                    else:
                        st.write("You:", message['content'])

                # Check if Interview is Completed
                if st.session_state.message_history and "Thank You" in st.session_state.message_history[-1]['content']:
                    if not st.session_state.interview_completed:
                        st.session_state.interview_completed = True
                        analysis_agent = create_analysis_agent(jd_tool, resume_tool)
                        analysis_task = create_analysis_task(analysis_agent)
                        crew2 = Crew(agents=[analysis_agent], tasks=[analysis_task], memory=True)
                        analysis = crew2.kickoff({"interview_script": st.session_state.message_history})
                        st.markdown("### Interview Analysis")
                        st.write(analysis)
                else:
                    answer_type = st.radio("Choose answer method:", ["Text", "Voice"])
                    user_input = None

                    if answer_type == "Text":
                        user_input = st.text_input("Your answer:")
                    elif audio_file and st.button("Transcribe Audio"):
                        user_input = transcribe_audio(audio_file)
                        if user_input:
                            st.write("Transcribed text:", user_input)

                    if st.button("Submit Answer") and user_input:
                        response = st.session_state.chain.invoke({
                            "question_set": st.session_state.questions,
                            "answer": user_input,
                            "chat_history": st.session_state.message_history
                        })
                        st.session_state.message_history.append({'role': 'user', 'content': user_input})
                        st.session_state.message_history.append({'role': 'assistant', 'content': response.content})
                        st.experimental_rerun()

    else:
        st.warning("Please enter API keys and upload required files.")

if __name__ == "__main__":
    main()
