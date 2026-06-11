import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------- Page setup ----------
st.title("🤖 My First AI Chatbot")
st.caption("Built with LangChain + Groq + Streamlit")

# ---------- The chain (your existing knowledge) ----------
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=st.secrets["GROQ_API_KEY"]    # reads from Streamlit's secret store
)

prompt = ChatPromptTemplate.from_template(
    "You are a friendly, concise assistant.\n"
    "Conversation so far:\n{history}\n\n"
    "User: {question}\n"
    "Assistant:"
)
chain = prompt | llm | StrOutputParser()

# ---------- Memory (the new concept!) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []          # survives between interactions

# Re-display the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- Chat input ----------
if question := st.chat_input("Ask me anything..."):
    # show + store user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # build history string and run the chain
    history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)
    with st.chat_message("assistant"):
        answer = chain.invoke({"history": history, "question": question})
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
