import time
import yfinance as yf
from duckduckgo_search import DDGS
from itertools import islice
from openai import OpenAI
import streamlit as st
import re
import json

# Modified DuckDuckGoSearchAPIWrapper class compatible


class DuckDuckGoSearchAPIWrapper:
    def __init__(self, region='wt-wt', safesearch='Moderate', timelimit='y', backend='api'):
        self.region = region
        self.safesearch = safesearch
        self.timelimit = timelimit
        self.backend = backend

    def run(self, query, max_results=5):
        try:
            ddgs = DDGS()
            results = ddgs.text(query, region=self.region, safesearch=self.safesearch,
                                timelimit=self.timelimit, backend=self.backend)
            limited_results = list(islice(results, max_results))

            if not limited_results:
                return "No good DuckDuckGo Search Result was found"

            # Extract the ticker symbol using regex
            for result in limited_results:
                match = re.search(r'\b[A-Z]{1,5}\b', result['title'])
                if match:
                    ticker_symbol = match.group(0)
                    if ticker_symbol not in ["NYSE", "NASDAQ"]:
                        return {"ticker": ticker_symbol}
        except Exception as e:
            return f"Error retrieving data: {str(e)}"


# Describe function tools
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker",
            "description": "Given the name of a company returns its ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "The name of the company",
                    }
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_statement",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_balance_sheet",
            "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_daily_stock_performance",
            "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Ticker symbol of the company",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]

# Define tool functions


def get_ticker(inputs):
    # inputs will look like this: {"company_name":"Apple"}
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")


def get_income_statement(inputs):
    ticker = inputs['ticker']
    stock = yf.Ticker(ticker)
    income_statement = stock.income_stmt
    return json.dumps(income_statement.to_json())


def get_balance_sheet(inputs):
    ticker = inputs['ticker']
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    return json.dumps(balance_sheet.to_json())


def get_daily_stock_performance(inputs):
    ticker = inputs['ticker']
    stock = yf.Ticker(ticker)
    history = stock.history(period="3mo")
    return json.dumps(history.to_json())


# Mapping the functions
functions_map = {
    "get_ticker": get_ticker,
    "get_income_statement": get_income_statement,
    "get_balance_sheet": get_balance_sheet,
    "get_daily_stock_performance": get_daily_stock_performance,
}

# Create an assistant and get assistant ID
client = OpenAI()

assistant_id = "asst_vC1F6Bt2TKZ0EVUf9tA6B9p8"

# Define functions for message handling


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )
    messages = list(messages)
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        # because function.arguments just brings str, so convert it to json so that the function can actually use it.
        function_args = json.loads(function.arguments)
        print(
            f'Calling function:{function.name} with arg {function.arguments}')
        output = functions_map[function.name](function_args)
        output_str = json.dumps(output)
        outputs.append(
            {
                "tool_call_id": action_id,
                "output": output_str,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs
    )


def save_message(message, role):
    st.session_state["messages"].append(
        {"message": message, "role": role}
    )


def write_message(message, role, save=True):
    # shows messages in the beginning, and save them
    with st.chat_message(role):
        st.markdown(message)
    if save:
        # Note that the messages are stored in a dictionary form
        save_message(message, role)

# Displaying messages without saving them: display saved messages


def paint_history():
    for message in st.session_state["messages"]:
        write_message(message["message"], message["role"], save=False)


# ========================================================================================
st.set_page_config(
    page_title="AssistantGPT",
    page_icon="💻",
)

main_markdown = """
    # AssistantGPT
    
    Welcome to AssistantGPT.
    
    AssistantGPT will provide financial insights for the companies of your intrest for stock investment.
    
    Provide the name of the company to begin with.
    """


file_search_markdown = """
    # File Search Mode
    
    You have uploaded a file.
    
    AssistantGPT will process the file and provide insights based on its content.
    """

# Sidebar for file upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a .txt .pdf or .docx file", type=[
        "pdf", "txt", "docx"])

if uploaded_file:
    st.markdown(file_search_markdown)
    vector_store = client.beta.vector_stores.create(
        name=f"{uploaded_file.name}")
    file_paths = [f"./files/{uploaded_file.name}"]
    file_streams = [open(path, 'rb') for path in file_paths]

    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams,
    )
    assistant = client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id]
            }
        },
    )
    message_file = client.files.create(
        file=open(f"./files/{uploaded_file.name}", "rb"),
        purpose="assistants"
    )
    st.write("File uploaded successfully!")
    query = st.text_input("Ask anything about this file")
    if query:
        paint_history()
        write_message(query, "human")
        if not st.session_state.get("thread"):
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": query,
                        "attachments": [
                            {"file_id": message_file.id, "tools": [
                                {"type": "file_search"}]}
                        ]
                    }
                ]
            )
            send_message(thread.id, query)
            st.session_state["thread"] = [thread]
        else:
            thread = st.session_state["thread"][0]
            send_message(thread.id, query)
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        with st.chat_message("ai"):
            with st.spinner("Creating an answer..."):
                while get_run(run.id, thread.id).status in [
                    "queued",
                    "in_progress",
                    "requires_action",
                ]:
                    time.sleep(0.5)
            message = get_messages(thread.id)[
                0].content[0].text.value.replace("$", "\$")
            save_message(message, "ai")
            st.markdown(message)
    else:
        st.session_state["messages"] = []
        st.session_state["thread"] = []

else:
    st.markdown(main_markdown)
    query = st.text_input(
        "Write the name of the company you are interested in.")

    if query:
        paint_history()
        write_message(query, "human")
        if not st.session_state.get("thread"):
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
            st.session_state["thread"] = [thread]
        else:
            thread = st.session_state["thread"][0]
            send_message(thread.id, query)
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )
        with st.chat_message("ai"):
            with st.spinner("Creating an answer..."):
                while get_run(run.id, thread.id).status in [
                    "queued",
                    "in_progress",
                    "requires_action",
                ]:
                    if get_run(run.id, thread.id).status == "requires_action":
                        submit_tool_outputs(run.id, thread.id)
            message = get_messages(thread.id)[
                0].content[0].text.value.replace("$", "\$")
            save_message(message, "ai")
            st.markdown(message)

    else:
        st.session_state["messages"] = []
        st.session_state["thread"] = []
