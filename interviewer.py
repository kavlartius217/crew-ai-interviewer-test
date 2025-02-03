import streamlit as st
import os
from crewai import Agent, Task, Crew
from crewai_tools import TXTSearchTool, PDFSearchTool
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import google.generativeai as genai
from gtts import gTTS
import tempfile
import time

# Initialize session state
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'interview_completed' not in st.session_state:
    st.session_state.interview_completed = False
if 'current_audio' not in st.session_state:
    st.session_state.current_audio = None

# Page config
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

def text_to_speech(text):
    """Convert text to speech and return the audio file path"""
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

def play_audio(text):
    """Generate and play audio for the given text"""
    audio_path = text_to_speech(text)
    st.session_state.current_audio = audio_path
    st.audio(audio_path, format='audio/mp3')
    # Clean up old audio file after a delay
    time.sleep(1)  # Give time for audio to load
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

# [Previous functions remain the same: initialize_tools(), create_interviewer_agent(), 
# create_interview_task(), create_analysis_agent(), create_analysis_task(), setup_langchain()]

def main():
    if all([openai_key, groq_key, gemini_key, jd_file, resume_file]):
        # Set environment variables
        os.environ['OPENAI_API_KEY'] = openai_key
        
        # Initialize tools and agents
        jd_tool, resume_tool = initialize_tools()
        if jd_tool and resume_tool:
            # Create Start Interview button
            if not st.session_state.interview_started and st.button("Start Interview"):
                st.session_state.interview_started = True
                
                # Generate questions
                interviewer_agent = create_interviewer_agent(jd_tool, resume_tool)
                interview_task = create_interview_task(interviewer_agent, jd_tool, resume_tool)
                crew1 = Crew(agents=[interviewer_agent], tasks=[interview_task], memory=True)
                questions = crew1.kickoff({})
                st.session_state.questions = questions
                st.session_state.chain = setup_langchain(questions)
                
                # Initialize first question
                response = st.session_state.chain.invoke({
                    "question_set": questions,
                    "answer": "",
                    "chat_history": st.session_state.message_history
                })
                st.session_state.message_history.extend([
                    {'role': 'assistant', 'content': response.content}
                ])

            # Display chat interface
            if st.session_state.interview_started:
                # Display chat history and play audio for new messages
                for i, message in enumerate(st.session_state.message_history):
                    if message['role'] == 'assistant':
                        st.write("Interviewer:", message['content'])
                        # Play audio for the most recent interviewer message
                        if i == len(st.session_state.message_history) - 1:
                            play_audio(message['content'])
                    else:
                        st.write("You:", message['content'])

                # Check if interview is complete
                if st.session_state.message_history and "Thank You" in st.session_state.message_history[-1]['content']:
                    if not st.session_state.interview_completed:
                        st.session_state.interview_completed = True
                        # Run analysis
                        analysis_agent = create_analysis_agent(jd_tool, resume_tool)
                        analysis_task = create_analysis_task(analysis_agent)
                        crew2 = Crew(agents=[analysis_agent], tasks=[analysis_task], memory=True)
                        analysis = crew2.kickoff({"interview_script": st.session_state.message_history})
                        st.markdown("### Interview Analysis")
                        st.write(analysis)
                else:
                    # Input options for user's answer
                    answer_type = st.radio("Choose answer method:", ["Text", "Voice"])
                    
                    if answer_type == "Text":
                        user_input = st.text_input("Your answer:")
                    else:
                        user_input = None
                        if audio_file:
                            if st.button("Transcribe Audio"):
                                user_input = transcribe_audio(audio_file)
                                if user_input:
                                    st.write("Transcribed text:", user_input)

                    if st.button("Submit Answer"):
                        if user_input:
                            # Process answer and get next question
                            response = st.session_state.chain.invoke({
                                "question_set": st.session_state.questions,
                                "answer": user_input,
                                "chat_history": st.session_state.message_history
                            })
                            st.session_state.message_history.extend([
                                {'role': 'user', 'content': user_input},
                                {'role': 'assistant', 'content': response.content}
                            ])
                            st.experimental_rerun()
                        else:
                            st.warning("Please provide an answer through text or voice before submitting.")

    else:
        st.warning("Please provide all required API keys and upload both the job description and resume files to begin.")

if __name__ == "__main__":
    main()
