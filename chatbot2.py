import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# ---------- Tools (no llm needed — safe at top) ----------
@tool
def calculator(expression: str) -> str:
    """Calculates a math expression like '847 * 392'. Use for any arithmetic."""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """Returns current weather for a city. Use when asked about weather."""
    fake_weather = {"plano": "34°C, sunny", "delhi": "31°C, humid", "london": "12°C, rain"}
    return fake_weather.get(city.lower(), f"22°C, clear in {city}")

tools_by_name = {"calculator": calculator, "get_weather": get_weather}

# ---------- Page + sidebar (creates temp) ----------
st.title("🤖 Aum AI Chatbot")
st.caption("Built with LangChain + Groq + Streamlit")

with st.sidebar:
    st.header("⚙️ Settings")
    temp = st.slider("Temperature (creativity)", 0.0, 1.5, 0.7)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- LLM (needs temp) → tools binding (needs llm) ----------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=temp,
               api_key=st.secrets["GROQ_API_KEY"])
llm_with_tools = llm.bind_tools([calculator, get_weather])

# ---------- Agent ----------
def get_answer(history: str, question: str) -> str:
    msgs = [
        SystemMessage(f"You are a friendly, concise assistant.\nConversation so far:\n{history}"),
        HumanMessage(question)
    ]
    response = llm_with_tools.invoke(msgs)
    if response.tool_calls:
        msgs.append(response)
        for tc in response.tool_calls:
            result = tools_by_name[tc["name"]].invoke(tc["args"])
            msgs.append({"role": "tool", "content": str(result), "tool_call_id": tc["id"]})
        response = llm_with_tools.invoke(msgs)
    return response.content

# ---------- Memory + redraw ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- Chat ----------
if question := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    history = "\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)
    with st.chat_message("assistant"):
        answer = get_answer(history, question)
        st.write(answer)                          # inside the bubble
    st.session_state.messages.append({"role": "assistant", "content": answer})