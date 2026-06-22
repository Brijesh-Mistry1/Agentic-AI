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

st.set_page_config(page_title="MCP RAG Chatbot (Hugging Face)", layout="wide")
st.title("MCP Semantic RAG Chatbot")
st.caption("Powered by Local Hugging Face Transformers & Groq LLM")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY environment variable")
    st.stop()

groq_client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Upload a PDF document in the sidebar to index it, then ask me anything about its contents."}
    ]

if "mcp_logs" not in st.session_state:
    st.session_state.mcp_logs = []


def log_mcp_action(action: str):
    st.session_state.mcp_logs.append(action)


async def call_mcp_agent(system_instruction: str, user_prompt: str, status_placeholder):
    """
    Spawns the MCP server, queries its tools, translates them to Groq's tool schema,
    sends the prompt to Groq, intercepts any tool requests, calls the tools on the MCP server,
    and returns the final model response.
    
    All steps are rendered inside the Streamlit `status_placeholder` for visibility.
    """
    # Define connection parameters for starting the MCP server as a subprocess.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["rag_server.py"],
        env=os.environ.copy()
    )
    
    # We use AsyncExitStack to manage the lifecycles of the subprocess streams and client session.
    async with AsyncExitStack() as stack:
        log_mcp_action("Spawning MCP Server subprocess...")
        read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))

        log_mcp_action("Initializing MCP ClientSession...")
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        log_mcp_action("Requesting tool definitions from MCP Server...")
        tools_resp = await session.list_tools()
        
        # Translate MCP tools to OpenAI/Groq compatible JSON schema
        groq_tools = []
        tool_names = []
        for t in tools_resp.tools:
            tool_names.append(t.name)
            groq_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            })
        
        status_placeholder.write(f"🛠️ 4. Tools discovered: `{tool_names}`. Formatted for Groq.")
        log_mcp_action(f"Discovered tools: {tool_names}")

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ]
        
        while True:
            status_placeholder.write("🧠 5. Sending prompt and tool schema to Groq LLM...")
            log_mcp_action("Sending request to Groq LLM...")

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
                        await asyncio.sleep(0.5)
                        continue
                    raise e
                    
            ans_msg = response.choices[0].message
            messages.append(ans_msg)

            if ans_msg.tool_calls:
                for tc in ans_msg.tool_calls:
                    t_name = tc.function.name
                    t_args = json.loads(tc.function.arguments)
                    
                    status_placeholder.write(f"🎯 6. LLM requested tool execution: `{t_name}` with args: `{t_args}`")
                    log_mcp_action(f"LLM tool call request: {t_name}({t_args})")
                    
                    # Call the tool on the MCP server via the stdio channel
                    status_placeholder.write(f"⚡ 7. Sending request to MCP Server to execute `{t_name}`...")
                    mcp_call = await session.call_tool(t_name, t_args)
                    t_output = mcp_call.content[0].text
                    
                    status_placeholder.write(f"✅ 8. Tool `{t_name}` finished. Result size: {len(t_output)} chars.")
                    log_mcp_action(f"Tool {t_name} returned result.")
                    
                    # Feed the tool result back into the LLM chat history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": t_name,
                        "content": t_output
                    })
                # Loop back to let the LLM generate a response based on the tool output
                continue
            else:
                status_placeholder.write("📝 9. LLM returned final answer. Closing connection.")
                log_mcp_action("LLM generated final answer.")
                return ans_msg.content


col_chat, col_sidebar = st.columns([0.7, 0.3])

with col_sidebar:
    st.header("Control Panel")
    st.subheader("Document Indexing")
    uploaded_file = st.file_uploader("Upload PDF Knowledge Base", type=["pdf"])
    
    if uploaded_file is not None:
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        saved_file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(saved_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("Index PDF to ChromaDB", use_container_width=True):
            sys_instruct = (
                "You are an ingestion helper. Use the available 'ingest_pdf' tool to process "
                "and index the local PDF file. Return a success message when complete."
            )
            prompt_str = f"Please process and index the local file located here: {saved_file_path}"

            with st.status("Ingesting Document...", expanded=True) as status_box:
                outcome = asyncio.run(call_mcp_agent(sys_instruct, prompt_str, status_box))
                status_box.update(label="Ingestion Complete!", state="complete", expanded=False)
            st.success(outcome)

    st.subheader("MCP Session Log")
    if st.button("Clear Logs"):
        st.session_state.mcp_logs = []
        
    with st.expander("View MCP Lifecycles & Calls", expanded=True):
        if not st.session_state.mcp_logs:
            st.info("No MCP actions executed yet. Upload a PDF or ask a question to see logs.")
        else:
            for log in st.session_state.mcp_logs:
                st.write(f"- {log}")


with col_chat:
    st.write("### Chat History")

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
                "You must use the 'query_vector_db' tool to find relevant information from the database before answering. "
                "Synthesize your answer based on the context returned by the database. If the context does not contain "
                "the answer, explain that you could not find the information in the database."
            )

            with st.status("Querying vector database via MCP...", expanded=True) as status_box:
                final_llm_response = asyncio.run(call_mcp_agent(sys_instruct, query_str, status_box))
                status_box.update(label="Query and Retrieval Finished!", state="complete", expanded=False)
                
            st.markdown(final_llm_response)
            
        st.session_state.messages.append({"role": "assistant", "content": final_llm_response})
        st.rerun()
