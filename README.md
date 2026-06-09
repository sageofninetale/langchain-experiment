# LangChain Research Chain

A multi-step AI research pipeline built with LangChain that decomposes any question into sub-questions, answers each one, then synthesises a final answer — all using Groq's Llama 3.3 70B model.

## How it works

1. Input a research question
2. **Chain 1** — Decompose into 3 focused sub-questions
3. **Chain 2** — Answer each sub-question with 2-3 sentences
4. **Chain 3** — Synthesise a clear 3-4 sentence final answer

## Tech stack

- **LangChain Core** — prompt templates, output parsers, chain composition
- **Groq / Llama 3.3 70B** — LLM inference
- **python-dotenv** — environment variable management

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install langchain-core langchain-groq python-dotenv
   ```
3. Create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Run:
   ```bash
   python research.py
   ```

## Example output

Question: *What is the future of AI in healthcare?*

The chain breaks it down, researches each angle, and returns a synthesised paragraph answer.
