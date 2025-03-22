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
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'data_result' not in st.session_state:
    st.session_state.data_result = None
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

# Initialize OpenAI client
client = None

# Model name
MODEL = "gpt-4o-mini"

def execute_query(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    """Execute a SQL-like query using pandas operations"""
    query = sql_query.lower()
    st.session_state.debug_messages.append(f"Executing query: {query}")
    
    try:
        # Handle basic SELECT COUNT queries
        if "select count(*)" in query:
            if "where" in query:
                condition = query.split("where")[1].strip()
                if "mortgage" in condition.lower() and "'yes'" in condition.lower():
                    result = pd.DataFrame({'count': [len(df[df['Mortgage'] == 'Yes'])]})
                    st.session_state.debug_messages.append(f"Count result: {result.iloc[0]['count']}")
                    return result
                elif "age" in condition.lower():
                    if ">" in condition:
                        age = float(condition.split(">")[1].strip())
                        result = pd.DataFrame({'count': [len(df[df['Age'] > age])]})
                        st.session_state.debug_messages.append(f"Count result: {result.iloc[0]['count']}")
                        return result
            result = pd.DataFrame({'count': [len(df)]})
            st.session_state.debug_messages.append(f"Total count: {len(df)}")
            return result
            
        # Handle basic SELECT queries
        if "select * from" in query:
            st.session_state.debug_messages.append("Returning full dataset")
            return df
            
        # Handle aggregations
        if "group by" in query:
            group_cols = [col.strip() for col in query.split("group by")[1].strip().split(",")]
            st.session_state.debug_messages.append(f"Grouping by: {group_cols}")
            
            if "count" in query:
                result = df.groupby(group_cols).size().reset_index(name='count')
                st.session_state.debug_messages.append(f"Group counts: {result.shape[0]} groups")
                return result
            elif "avg" in query:
                agg_col = query.split("avg(")[1].split(")")[0].strip()
                st.session_state.debug_messages.append(f"Calculating average of {agg_col}")
                result = df.groupby(group_cols)[agg_col].mean().reset_index()
                return result
                
        st.session_state.debug_messages.append("No specific handling for query, returning full dataset")
        return df
        
    except Exception as e:
        error_msg = f"Query execution error: {str(e)}"
        st.session_state.debug_messages.append(error_msg)
        st.error(error_msg)
        return pd.DataFrame()

def lookup_data(prompt: str) -> str:
    """Get data based on the prompt"""
    if st.session_state.df is None:
        return "No data available. Please upload a dataset."
        
    # Get column information
    cols_info = [f"{col} ({dtype})" for col, dtype in zip(st.session_state.df.columns, st.session_state.df.dtypes)]
    st.session_state.debug_messages.append(f"Available columns: {', '.join(cols_info)}")
    
    # Get SQL query from LLM
    try:
        st.session_state.debug_messages.append(f"Generating SQL query for prompt: {prompt}")
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
        st.session_state.debug_messages.append(f"Generated SQL query: {sql_query}")
        
        # Execute query
        result = execute_query(st.session_state.df, sql_query)
        st.session_state.data_result = result
        
        if result.empty:
            st.session_state.debug_messages.append("Query returned no results")
            return "No results found."
            
        # Format output
        if result.shape[0] == 1:  # Single row result
            output = "\n".join([f"{col}: {result.iloc[0][col]}" for col in result.columns])
            st.session_state.debug_messages.append(f"Single row result: {output}")
            return output
        else:
            st.session_state.debug_messages.append(f"Query returned {result.shape[0]} rows")
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
    
    st.session_state.conversation_history.append({"role": "user", "content": query})
    
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
                st.session_state.conversation_history.append({"role": "assistant", "content": msg.content})
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
st.subheader("Upload your dataset and ask questions using natural language")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.OPENAI_API_KEY)
    if api_key:
        st.session_state.OPENAI_API_KEY = api_key
        client = OpenAI(api_key=api_key)
    
    st.header("Data Upload")
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

# Tabs for Query & Analysis and Data Explorer
tab1, tab2 = st.tabs(["Query & Analysis", "Data Explorer"])

with tab1:
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
                
                if st.session_state.data_result is not None:
                    st.write("### Data Results")
                    st.dataframe(st.session_state.data_result)
                    
                    # Download button for results
                    csv_data = st.session_state.data_result.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                if st.session_state.last_sql_query:
                    with st.expander("View SQL Query"):
                        st.code(st.session_state.last_sql_query, language="sql")
                        
                with st.expander("Debug Information"):
                    for msg in st.session_state.debug_messages:
                        st.text(msg)

with tab2:
    if st.session_state.df is not None:
        st.header("Data Explorer")
        
        # Data sample
        st.subheader("Data Sample")
        sample_rows = min(1000, st.session_state.df.shape[0])
        st.dataframe(st.session_state.df.head(sample_rows))
        st.caption(f"Showing first {sample_rows} rows out of {st.session_state.df.shape[0]} total rows")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        numeric_df = st.session_state.df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe())
        else:
            st.info("No numeric columns found for summary statistics")
            
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column Name': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes.values,
            'Non-Null Count': st.session_state.df.count().values,
            'Null Count': st.session_state.df.isna().sum().values
        })
        st.dataframe(col_info)
        
        # Data exploration
        st.subheader("Data Exploration")
        categorical_cols = st.session_state.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            selected_cat_col = st.selectbox(
                "Select a categorical column to view value distribution:",
                options=[""] + categorical_cols
            )
            if selected_cat_col:
                value_counts = st.session_state.df[selected_cat_col].value_counts()
                st.bar_chart(value_counts)
                st.dataframe(value_counts)

# Conversation history
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("### Conversation History")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f"**You**: {msg['content']}")
        else:
            st.markdown(f"**Assistant**: {msg['content']}")
        st.markdown("---")
