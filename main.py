import streamlit as st
from utils import qa_agent
from langchain.memory import ConversationBufferMemory

st.title("AI智能PDF问答工具")

with st.sidebar:
    api_key = st.text_input("请输入密钥：",type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

uploaded_file = st.file_uploader("请上传你的pdf文件",type="pdf")
question = st.text_input("请对pdf的内容进行提问",disabled=not uploaded_file)

if uploaded_file and question and not api_key:
    st.info("请输入你的密钥")

if uploaded_file and question and api_key:
    with st.spinner("AI正在思考..."):
        response = qa_agent(api_key,st.session_state["memory"],
                            uploaded_file,question)
        
    st.write("### 答案")
    st.write(response["answer"])  # 正确拼写
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0,len(st.session_state["chat_history"]),2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider() 




