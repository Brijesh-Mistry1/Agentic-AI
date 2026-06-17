import streamlit as st
import os
import sys
import json
import asyncio
from contextlib import AsyncExitStack
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="MCP RAG Chatbot", layout="wide")
st.title("MCP Semantic RAG Interface (ChromaDB + Groq)")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY environment variable. Please set it before executing.")
    st.stop()


groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF document to begin, then ask your question!"}]


async def call_mcp_agent(system_instruction: str, user_prompt: str):
    """Orchestrates an interactive chat round calling tools dynamically via MCP stdio."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["rag_server.py"],
        env=os.environ.copy()
    )
    
    async with AsyncExitStack() as stack:
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        tools_resp = await session.list_tools()
        groq_tools = []
        for t in tools_resp.tools:
            groq_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            })
            
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
        
        # Single iteration execution loop pattern for reactive tool usage handling
        while True:
            # Groq model formatting can occasionally raise a BadRequestError (tool_use_failed)
            # if it outputs malformed XML tags for tool calls. We implement a retry block.
            retries = 3
            for attempt in range(retries):
                try:
                    response = groq_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=groq_tools,
                        tool_choice="auto"
                    )
                    break
                except Exception as e:
                    if "tool_use_failed" in str(e) and attempt < retries - 1:
                        import time
                        time.sleep(0.5)
                        continue
                    raise e
            ans_msg = response.choices[0].message
            messages.append(ans_msg)
            
            if ans_msg.tool_calls:
                for tc in ans_msg.tool_calls:
                    t_name = tc.function.name
                    t_args = json.loads(tc.function.arguments)
                    
                    with st.spinner(f"Running database action: {t_name}..."):
                        mcp_call = await session.call_tool(t_name, t_args)
                        t_output = mcp_call.content[0].text
                        
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": t_name,
                        "content": t_output
                    })
                continue
            else:
                return ans_msg.content


with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload target knowledge base PDF", type=["pdf"])
    
    if uploaded_file is not None:
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        saved_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(saved_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("Index file to Vector DB"):
            sys_instruct = "You are an ingestion helper. Use the available 'ingest_pdf' tool to process and index the local PDF file."
            prompt_str = f"Please process and index the local file located here: {saved_file_path}"
            
            with st.spinner("Ingesting and indexing document content..."):
                outcome = asyncio.run(call_mcp_agent(sys_instruct, prompt_str))
                st.success(outcome)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if query_str := st.chat_input("Ask a question about your documents:"):
    st.session_state.messages.append({"role": "user", "content": query_str})
    with st.chat_message("user"):
        st.markdown(query_str)

    with st.chat_message("assistant"):
        sys_instruct = (
            "You are a helpful assistant with semantic access to a Vector Database containing uploaded documents. "
            "Use the 'query_vector_db' tool to find relevant information from the database before answering the user's question."
        )
        with st.spinner("Analyzing data vector contexts..."):
            final_llm_response = asyncio.run(call_mcp_agent(sys_instruct, query_str))
            st.markdown(final_llm_response)
            
    st.session_state.messages.append({"role": "assistant", "content": final_llm_response})
