"""
Langfuse tracing setup for LangChain.

CallbackHandler intercepts every LangChain event (LLM calls, chain starts/ends,
tool calls) and sends them to Langfuse automatically.

Usage in any agent:
    from tracing import langfuse_handler
    chain.invoke({...}, config={"callbacks": [langfuse_handler]})
"""

from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()
