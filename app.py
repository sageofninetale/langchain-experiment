import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Research Chain", page_icon="🔬", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; color: #e0e0e0; }
    section[data-testid="stMain"] > div { background-color: #0a0a0f; }
    .chain-title {
        text-align: center; font-size: 2.4rem; font-weight: 800;
        color: #ffffff; letter-spacing: 2px; padding: 1rem 0 0.2rem 0;
    }
    .chain-subtitle {
        text-align: center; font-size: 0.9rem; color: #7c8cf8;
        letter-spacing: 4px; text-transform: uppercase; margin-bottom: 1.5rem;
    }
    .step-box {
        background: rgba(124, 140, 248, 0.06);
        border: 1px solid rgba(124, 140, 248, 0.18);
        border-radius: 8px; padding: 1.2rem 1.5rem; margin: 0.8rem 0;
    }
    .step-label {
        font-size: 0.72rem; font-weight: 700; color: #7c8cf8;
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;
    }
    .step-body { color: #d0d4f0; font-size: 0.95rem; line-height: 1.7; white-space: pre-wrap; }
    .final-box {
        background: rgba(99, 220, 180, 0.06);
        border-left: 4px solid rgba(99, 220, 180, 0.5);
        border-radius: 0 8px 8px 0; padding: 1.2rem 1.5rem; margin-top: 1rem;
    }
    .final-label {
        font-size: 0.72rem; font-weight: 700; color: #63dcb4;
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;
    }
    .final-body { color: #c8f0e4; font-size: 1rem; line-height: 1.8; }
    .stTextArea label { color: #7c8cf8 !important; font-size: 0.85rem !important;
        letter-spacing: 1px !important; text-transform: uppercase !important; font-weight: 600 !important; }
    .stTextArea > div > div > textarea {
        background-color: #12121e !important; color: #e0e0e0 !important;
        border: 2px solid #2a2a4a !important; border-radius: 8px !important; }
    .stButton > button {
        background-color: #3a3af0 !important; color: #ffffff !important;
        border: none !important; border-radius: 8px !important;
        font-size: 0.95rem !important; font-weight: 700 !important;
        padding: 0.6rem 2rem !important; width: 100% !important;
        letter-spacing: 1px !important; text-transform: uppercase !important; }
    .stButton > button:hover { background-color: #5a5af8 !important; }
    #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="chain-title">🔬 Research Chain</div>', unsafe_allow_html=True)
st.markdown('<div class="chain-subtitle">Decompose · Research · Synthesise</div>', unsafe_allow_html=True)

question = st.text_area(
    "Research Question",
    placeholder="e.g. What is the future of AI in healthcare?",
    height=80,
)

if st.button("Run Research Chain"):
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            st.error("GROQ_API_KEY not set. Add it in Settings → Secrets.")
            st.stop()

        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_key, temperature=0.7)
        parser = StrOutputParser()

        decompose_prompt = PromptTemplate.from_template(
            "Break this question into exactly 3 focused sub-questions that together would fully answer it.\n"
            "Question: {question}\n"
            "Return only the 3 sub-questions numbered 1, 2, 3. Nothing else."
        )
        answer_prompt = PromptTemplate.from_template(
            "Answer each of these 3 sub-questions with 2-3 sentences each.\n"
            "Sub-questions: {sub_questions}\n"
            "Give clear factual answers."
        )
        synthesise_prompt = PromptTemplate.from_template(
            "Based on these research findings, write a clear final answer in 3-4 sentences.\n"
            "Research findings: {answers}\n"
            "Final answer:"
        )

        with st.spinner("Breaking into sub-questions..."):
            sub_questions = (decompose_prompt | llm | parser).invoke({"question": question})

        st.markdown(f'<div class="step-box"><div class="step-label">Chain 1 — Sub-questions</div><div class="step-body">{sub_questions}</div></div>', unsafe_allow_html=True)

        with st.spinner("Answering each sub-question..."):
            answers = (answer_prompt | llm | parser).invoke({"sub_questions": sub_questions})

        st.markdown(f'<div class="step-box"><div class="step-label">Chain 2 — Research findings</div><div class="step-body">{answers}</div></div>', unsafe_allow_html=True)

        with st.spinner("Synthesising final answer..."):
            final_answer = (synthesise_prompt | llm | parser).invoke({"answers": answers})

        st.markdown(f'<div class="final-box"><div class="final-label">Final Answer</div><div class="final-body">{final_answer}</div></div>', unsafe_allow_html=True)
