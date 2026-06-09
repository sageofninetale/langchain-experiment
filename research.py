import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

parser = StrOutputParser()

print("LLM ready")

# Block 2 — Prompts
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

print("Prompts ready")

# Block 3 — Chains
question = "What is the future of AI in healthcare?"

print("=== Chain 1: Breaking into sub-questions ===")
sub_questions = (decompose_prompt | llm | parser).invoke({"question": question})
print(sub_questions)

print("=== Chain 2: Answering each sub-question ===")
answers = (answer_prompt | llm | parser).invoke({"sub_questions": sub_questions})
print(answers)

print("=== Chain 3: Final answer ===")
final_answer = (synthesise_prompt | llm | parser).invoke({"answers": answers})
print(final_answer)
