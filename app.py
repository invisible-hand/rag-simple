import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from datetime import datetime
from pandasql import sqldf

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Initialize session state
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""
if 'df' not in st.session_state:
    st.session_state.df = None
if 'last_sql_query' not in st.session_state:
    st.session_state.last_sql_query = None
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

# Initialize OpenAI client
client = None

# Model name
MODEL = "gpt-4o-mini"

def execute_sql(sql_query: str) -> pd.DataFrame:
    """Execute SQL query using pandasql"""
    try:
        # Create a function that executes SQL using the current dataframe
        pysqldf = lambda q: sqldf(q, globals())
        # Execute the query
        result = pysqldf(sql_query)
        return result
    except Exception as e:
        st.error(f"Error executing SQL: {str(e)}")
        return pd.DataFrame()

def lookup_data(prompt: str) -> str:
    """Get data based on the prompt"""
    if st.session_state.df is None:
        return "No data available. Please upload a dataset."
        
    # Get column information
    cols_info = [f"{col} ({dtype})" for col, dtype in zip(st.session_state.df.columns, st.session_state.df.dtypes)]
    st.session_state.debug_messages.append(f"Available columns: {', '.join(cols_info)}")
    
    try:
        st.session_state.debug_messages.append(f"Generating SQL query for prompt: {prompt}")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate SQL queries for data analysis. Return ONLY the SQL query, no explanations."},
                {"role": "user", "content": f"Write a SQL query to {prompt}. Available columns: {', '.join(cols_info)}. The table name is 'df'."}
            ],
            temperature=0
        )
        
        sql_query = response.choices[0].message.content.strip()
        # Clean up the SQL query
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        st.session_state.last_sql_query = sql_query
        st.session_state.debug_messages.append(f"Generated SQL query: {sql_query}")
        
        # Execute query
        result = execute_sql(sql_query)
        
        if result.empty:
            st.session_state.debug_messages.append("Query returned no results")
            return "No results found."
            
        # Format output
        return result.to_string(index=False)
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        st.session_state.debug_messages.append(error_msg)
        return error_msg

def analyze_data(data: str, prompt: str) -> str:
    """Analyze the data results"""
    try:
        st.session_state.debug_messages.append(f"Analyzing data for prompt: {prompt}")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst. Provide clear, concise analysis."},
                {"role": "user", "content": f"Analyze this data to answer: {prompt}\n\nData:\n{data}"}
            ],
            temperature=0
        )
        analysis = response.choices[0].message.content.strip()
        st.session_state.debug_messages.append(f"Analysis result: {analysis}")
        return analysis
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        st.session_state.debug_messages.append(error_msg)
        return error_msg

# Define tools for the agent
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_data",
            "description": "Get data using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "The query prompt"}},
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Analyze data results",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The data to analyze"},
                    "prompt": {"type": "string", "description": "The analysis prompt"}
                },
                "required": ["data", "prompt"]
            }
        }
    }
]

def run_agent(query: str):
    """Main agent loop"""
    global client
    client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)
    
    st.session_state.debug_messages.append(f"Starting agent with query: {query}")
    
    messages = [
        {"role": "system", "content": "You are a data analyst. Use lookup_data to get data, then analyze_data to analyze it."},
        {"role": "user", "content": query}
    ]
    
    for iteration in range(3):  # Max 3 iterations
        try:
            st.session_state.debug_messages.append(f"Iteration {iteration + 1}")
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0
            )
            
            msg = response.choices[0].message
            messages.append(msg)
            
            if not msg.tool_calls:
                return msg.content
                
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                st.session_state.debug_messages.append(f"Calling {func_name} with args: {func_args}")
                
                if func_name == "lookup_data":
                    result = lookup_data(**func_args)
                elif func_name == "analyze_data":
                    result = analyze_data(**func_args)
                else:
                    result = "Unknown function"
                    
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": result})
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.session_state.debug_messages.append(error_msg)
            return error_msg
            
    return "Maximum iterations reached without final answer."

# Streamlit UI
st.title("Data Analysis Dashboard")

# Sidebar
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.OPENAI_API_KEY)
    if api_key:
        st.session_state.OPENAI_API_KEY = api_key
        client = OpenAI(api_key=api_key)
    
    uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success(f"Successfully loaded {uploaded_file.name}")
            st.write(f"Rows: {st.session_state.df.shape[0]}, Columns: {st.session_state.df.shape[1]}")
            # Display sample of the data
            st.write("### Data Sample")
            st.dataframe(st.session_state.df.head())
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

# Main area
if not st.session_state.OPENAI_API_KEY:
    st.error("Please enter your OpenAI API key")
elif st.session_state.df is None:
    st.info("Please upload a dataset")
else:
    query = st.text_area("Ask a question about your data:")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            result = run_agent(query)
            st.write("### Results")
            st.write(result)
            
            if st.session_state.last_sql_query:
                with st.expander("View SQL Query"):
                    st.code(st.session_state.last_sql_query, language="sql")
                    
            with st.expander("Debug Information"):
                for msg in st.session_state.debug_messages:
                    st.text(msg)
