__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
import tempfile
import subprocess
import time

# Install streamlit_mic_recorder if not installed
try:
    from streamlit_mic_recorder import mic_recorder
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "streamlit-mic-recorder"], check=True)
    from streamlit_mic_recorder import mic_recorder

from crewai import Agent, Task, Crew
from crewai_tools import TXTSearchTool, PDFSearchTool
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import google.generativeai as genai

# Suppress regex warnings
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'interview_completed' not in st.session_state:
    st.session_state.interview_completed = False

# Page configuration
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("AI Interviewer System")

# Sidebar for API keys and file uploads
with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input("Enter Groq API Key", type="password")
    gemini_key = st.text_input("Enter Gemini API Key", type="password")
    openai_key = st.text_input("Enter OpenAI API Key", type="password")
    
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["GEMINI_API_KEY"] = gemini_key
    os.environ["OPENAI_API_KEY"] = openai_key
    
    st.header("Upload Files")
    jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt")
    resume_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

def save_uploaded_file(uploaded_file, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def transcribe_audio(audio_data):
    """Transcribes recorded audio using Google Gemini."""
    if isinstance(audio_data, dict) and "bytes" in audio_data:
        audio_bytes = audio_data["bytes"]
    else:
        raise ValueError("Invalid audio format received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    result = model.generate_content([temp_audio_path, "transcribe the audio as it is"])
    
    return result.text

def create_interviewer_agent(jd_tool, resume_tool):
    """Creates the AI Interviewer agent."""
    return Agent(
        role="Expert Interviewer",
        goal="Conduct a structured interview by asking relevant questions.",
        backstory="An experienced AI-driven interviewer designed to assess candidates based on job descriptions and resumes.",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=True
    )

def create_interview_task(agent):
    """Creates the interview task for generating questions."""
    return Task(
        description="Generate structured interview questions.",
        agent=agent,
        expected_output="A structured list of interview questions."
    )

def create_analysis_agent(jd_tool, resume_tool):
    """Creates the AI Analysis agent for evaluation."""
    return Agent(
        role="Talent Acquisition Expert",
        goal="Evaluate the candidate based on interview responses.",
        backstory="A seasoned recruitment specialist responsible for assessing candidates' suitability based on interview performance and resume analysis.",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=True
    )

def create_analysis_task(agent):
    """Creates the analysis task for candidate evaluation."""
    return Task(
        description="Analyze responses and generate a report on candidate suitability.",
        agent=agent,
        expected_output="A detailed candidate assessment report."
    )

def setup_langchain():
    """Sets up LangChain RAG model for sequential questioning."""
    return ChatGroq(
        model_name="gemma2-9b-it",
        api_key=os.environ["GROQ_API_KEY"],
        prompt=ChatPromptTemplate.from_messages([
            ("system", "You are an Interviewer."),
            ("system", "You have a set of questions: {question_set}. Ask them sequentially, one at a time."),
            ("system", "Only ask the next unanswered question from {question_set}."),
            ("system", "Do not repeat any question already present in chat history."),
            ("system", "Ask only the question itself, without any additional text."),
            ("system", "Never answer the questions yourself."),
            ("system", "After questions are over, say 'Thank You'."),
            MessagesPlaceholder(variable_name="chat_history"),
        ])
    )

def main():
    """Main function to run the AI Interviewer System."""
    if all([groq_key, gemini_key, openai_key, jd_file, resume_file]):
        jd_path = save_uploaded_file(jd_file, "uploads")
        resume_path = save_uploaded_file(resume_file, "uploads")
        jd_tool = TXTSearchTool(jd_path)
        resume_tool = PDFSearchTool(resume_path)

        if not st.session_state.interview_started and st.button("Start Interview"):
            st.session_state.interview_started = True
            interviewer_agent = create_interviewer_agent(jd_tool, resume_tool)
            interview_task = create_interview_task(interviewer_agent)
            crew1 = Crew(agents=[interviewer_agent], tasks=[interview_task], memory=True)

            # Get full question list without stripping content
            result = crew1.kickoff({})
            st.session_state.questions = result.raw.split("\n")  # Preserve full structure

            st.session_state.chain = setup_langchain()

    if st.session_state.interview_started and not st.session_state.interview_completed:
        if st.session_state.question_index < len(st.session_state.questions):
            current_question = st.session_state.questions[st.session_state.question_index].strip()
            
            if current_question:
                st.write(f"**Question {st.session_state.question_index + 1}: {current_question}**")

                audio_bytes = mic_recorder(key=f"mic_recorder_{st.session_state.question_index}")  # Unique key

                if audio_bytes:
                    transcribed_text = transcribe_audio(audio_bytes)
                    st.session_state.message_history.append({"user": transcribed_text})

                    # Ensure input is a proper string, not a dictionary
                    formatted_prompt = f"Question: {current_question}\nAnswer: {transcribed_text}"

                    response = st.session_state.chain.invoke(formatted_prompt)  # Fixed input type
                    st.session_state.message_history.append({"ai": response.content})

                    st.session_state.question_index += 1  # Move to next question

                    if response.content.strip().lower() == "thank you":
                        st.session_state.interview_completed = True

        else:
            st.session_state.interview_completed = True

    if st.session_state.interview_completed:
        analysis_agent = create_analysis_agent(jd_tool, resume_tool)
        analysis_task = create_analysis_task(analysis_agent)
        crew2 = Crew(agents=[analysis_agent], tasks=[analysis_task], memory=True)
        report = crew2.kickoff({"interview_script": st.session_state.message_history})
        st.write("### Candidate Assessment Report")
        st.write(report)

if __name__ == "__main__":
    main()
