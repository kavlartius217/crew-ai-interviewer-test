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
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.header("Upload Files")
    jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

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
        audio_data = genai.upload_file(audio_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([audio_data, "transcribe the audio as it is"])
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
        goal="Conduct a structured interview and assess the candidate",
        backstory="An AI interviewer skilled in job assessments.",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=True,
        llm=ChatGroq(model_name="mixtral", api_key=groq_key)
    )

def create_interview_task(agent):
    return Task(
        description="Ask structured interview questions based on job description and resume.",
        agent=agent,
        expected_output="List of structured interview questions."
    )

def create_analysis_agent(jd_tool, resume_tool):
    return Agent(
        role="Interview Evaluator",
        goal="Analyze the interview and assess candidate fitness.",
        backstory="An expert hiring manager AI.",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=True,
        llm=ChatGroq(model_name="mixtral", api_key=groq_key)
    )

def create_analysis_task(agent):
    return Task(
        description="Analyze interview responses and generate a candidate fit report.",
        agent=agent,
        expected_output="Candidate suitability report."
    )

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

def main():
    if all([groq_key, gemini_key, jd_file, resume_file]):
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
                response = st.session_state.chain.invoke({"question_set": questions, "answer": "", "chat_history": st.session_state.message_history})
                st.session_state.message_history.append({'role': 'assistant', 'content': response.content})

            if st.session_state.interview_started:
                for i, message in enumerate(st.session_state.message_history):
                    if message['role'] == 'assistant':
                        st.write("Interviewer:", message['content'])
                        if i == len(st.session_state.message_history) - 1:
                            play_audio(message['content'])
                    else:
                        st.write("You:", message['content'])

                if st.session_state.message_history and "Thank You" in st.session_state.message_history[-1]['content']:
                    if not st.session_state.interview_completed:
                        st.session_state.interview_completed = True
                        analysis_agent = create_analysis_agent(jd_tool, resume_tool)
                        analysis_task = create_analysis_task(analysis_agent)
                        crew2 = Crew(agents=[analysis_agent], tasks=[analysis_task], memory=True)
                        analysis = crew2.kickoff({"interview_script": st.session_state.message_history})
                        st.markdown("### Interview Analysis")
                        st.write(analysis)

if __name__ == "__main__":
    main()

