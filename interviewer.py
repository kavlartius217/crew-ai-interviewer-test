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

# Set up the environment
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Streamlit app title
st.title("Crew AI Interviewer")

# File uploaders
st.sidebar.header("Upload Files")
job_description_file = st.sidebar.file_uploader("Upload Job Description (MD)", type="md")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF)", type="pdf")

if job_description_file and resume_file:
    # Save uploaded files
    with open("Job_Description.md", "wb") as f:
        f.write(job_description_file.getbuffer())
    with open("Resume.pdf", "wb") as f:
        f.write(resume_file.getbuffer())

    # Initialize tools
    jd_tool = TXTSearchTool('Job_Description.md')
    resume_tool = PDFSearchTool('Resume.pdf')

    # Initialize Interviewer Agent
    Interviewer_agent = Agent(
        role="Expert Interviewer",
        goal="Conduct a structured interview by asking relevant questions based on the job description and the candidate's resume.",
        backstory="You are an experienced interviewer skilled in assessing candidates based on job requirements and their qualifications.",
        tools=[jd_tool, resume_tool],
        memory=True,
        verbose=True
    )

    # Initialize Interview Task
    interview_task = Task(
        description=(
            "Analyze the job description using jd_tool and the candidate's resume using resume_tool. "
            "Formulate a total of 10-12 well-structured questions that evaluate the candidate's skills, experience, and alignment with the role. "
            "Ensure a mix of technical, situational, and behavioral questions. "
        ),
        agent=Interviewer_agent,
        expected_output="A structured file containing the questions only.",
        tools=[jd_tool, resume_tool],
        output_file='interview.md',
    )

    # Run the first crew
    crew1 = Crew(
        agents=[Interviewer_agent],
        tasks=[interview_task],
        memory=True
    )

    result = crew1.kickoff({})

    # Display the generated questions
    st.header("Generated Interview Questions")
    st.write(result)

    # Initialize Langchain for interview
    llm = ChatGroq(api_key=st.secrets['GROQ_API_KEY'], model="gemma2-9b-it", temperature=0)

    class MessageHistory:
        def __init__(self):
            self.l1 = []  # Initialize as an empty list

        def add(self, user_message, ai_response):
            self.l1.append({'role': 'user', 'content': user_message})
            self.l1.append({'role': 'assistant', 'content': ai_response})

        def show_history(self):
            return self.l1

    history = MessageHistory()

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
        ai_response = chain.invoke({"question_set": result, "answer": answer, "chat_history": history.l1})
        history.add(answer, ai_response.content)
        return ai_response.content

    # Streamlit interface for interview
    st.header("Interview Session")
    user_input = st.text_input("Your Answer:")
    if st.button("Submit"):
        ai_response = interview_chain(user_input)
        st.write(ai_response)

    # Analysis Crew
    l1=history.show_history()
    if l1[-1]=='Thank You':
        st.header("Candidate Analysis")
        Interview_analysis_agent = Agent(
            role="Talent Acquisition Expert",
            goal="Evaluate the candidate's fit for the job based on the job description, resume, and interview script analysis.",
            backstory=(
                "You are an expert in talent acquisition, specializing in evaluating candidates based on "
                "their resumes, job descriptions, and interview performance. "
                "Using your analytical skills, you determine whether a candidate aligns with the job requirements."
            ),
            tools=[jd_tool, resume_tool],
            memory=True,
            verbose=True
        )

        Interview_analysis_task = Task(
            description=(
                "Analyze the interview script provided in {interview_script} to assess the candidate's fit for the role. "
                "Use the job description from the jd_tool and the candidate's resume from the resume_tool to make an informed evaluation."
            ),
            expected_output="A detailed report assessing the candidate's suitability for the role based on the job description, resume, and interview performance.",
            agent=Interview_analysis_agent,
            output_file="candidate_fit_analysis.md"
        )

        crew2 = Crew(
            agents=[Interview_analysis_agent],
            tasks=[Interview_analysis_task],
            memory=True
        )

        analysis_result = crew2.kickoff({"interview_script": history.show_history()})
        st.write(analysis_result)

else:
    st.warning("Please upload both the job description and resume to proceed.")
