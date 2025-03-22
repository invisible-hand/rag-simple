import streamlit as st
import pandas as pd
from openai import OpenAI
import json
from datetime import datetime

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Initialize session state
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""
if 'df' not in st.session_state:
    st.session_state.df = None
if 'last_sql_query' not in st.session_state:
    st.session_state.last_sql_query = None

# Initialize OpenAI client
client = None

# Model name
MODEL = "gpt-4o-mini"

def execute_query(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    """Execute a SQL-like query using pandas operations"""
    query = sql_query.lower()
    
    try:
        # Handle basic SELECT COUNT queries
        if "select count(*)" in query:
            if "where" in query:
                condition = query.split("where")[1].strip()
                if "mortgage" in condition.lower() and "'yes'" in condition.lower():
                    return pd.DataFrame({'count': [len(df[df['Mortgage'] == 'Yes'])]})
                elif "age" in condition.lower():
                    if ">" in condition:
                        age = float(condition.split(">")[1].strip())
                        return pd.DataFrame({'count': [len(df[df['Age'] > age])]})
            return pd.DataFrame({'count': [len(df)]})
            
        # Handle basic SELECT queries
        if "select * from" in query:
            return df
            
        # Handle aggregations
        if "group by" in query:
            group_cols = [col.strip() for col in query.split("group by")[1].strip().split(",")]
            if "count" in query:
                return df.groupby(group_cols).size().reset_index(name='count')
            elif "avg" in query:
                agg_col = query.split("avg(")[1].split(")")[0].strip()
                return df.groupby(group_cols)[agg_col].mean().reset_index()
                
        return df
        
    except Exception as e:
        st.error(f"Query execution error: {str(e)}")
        return pd.DataFrame()

def lookup_data(prompt: str) -> str:
    """Get data based on the prompt"""
    if st.session_state.df is None:
        return "No data available. Please upload a dataset."
        
    # Get column information
    cols_info = [f"{col} ({dtype})" for col, dtype in zip(st.session_state.df.columns, st.session_state.df.dtypes)]
    
    # Get SQL query from LLM
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate SQL queries for data analysis. Return ONLY the SQL query, no explanations."},
                {"role": "user", "content": f"Write a SQL query to {prompt}. Available columns: {', '.join(cols_info)}. Table name: data_table"}
            ],
            temperature=0
        )
        
        sql_query = response.choices[0].message.content.strip()
        st.session_state.last_sql_query = sql_query
        
        # Execute query
        result = execute_query(st.session_state.df, sql_query)
        
        if result.empty:
            return "No results found."
            
        # Format output
        if result.shape[0] == 1:  # Single row result
            return "\n".join([f"{col}: {result.iloc[0][col]}" for col in result.columns])
        else:
            return result.to_string(index=False)
            
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_data(data: str, prompt: str) -> str:
    """Analyze the data results"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a data analyst. Provide clear, concise analysis."},
                {"role": "user", "content": f"Analyze this data to answer: {prompt}\n\nData:\n{data}"}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Analysis error: {str(e)}"

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
    
    messages = [
        {"role": "system", "content": "You are a data analyst. Use lookup_data to get data, then analyze_data to analyze it."},
        {"role": "user", "content": query}
    ]
    
    for _ in range(3):  # Max 3 iterations
        try:
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
                
                if func_name == "lookup_data":
                    result = lookup_data(**func_args)
                elif func_name == "analyze_data":
                    result = analyze_data(**func_args)
                else:
                    result = "Unknown function"
                    
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": func_name, "content": result})
                
        except Exception as e:
            return f"Error: {str(e)}"
            
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
