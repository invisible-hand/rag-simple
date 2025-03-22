import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import json
import io
from datetime import datetime

# Sample dataset for demo mode
SAMPLE_DATA = {
    'Age': [25, 35, 45, 55, 65, 28, 38, 48, 58, 68],
    'Income': [30000, 45000, 60000, 75000, 90000, 35000, 50000, 65000, 80000, 95000],
    'Mortgage': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'Bachelor', 'PhD', 'Master', 'Bachelor', 'Master'],
    'Occupation': ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Manager', 'Developer', 'Professor', 'Consultant', 'Architect', 'Executive']
}

def load_sample_data():
    """Load sample dataset for demo mode"""
    return pd.DataFrame(SAMPLE_DATA)

st.set_page_config(page_title="Data Analysis Dashboard", layout="wide")

# Initialize session state variables
if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""  # Initialize empty, will be set via UI or secrets

# Initialize OpenAI client with the API key from session state
client = None

# Set model name
MODEL = "gpt-4o-mini"  # Using GPT-4 Optimized Mini model

# Example queries for demo mode
EXAMPLE_QUERIES = [
    "What is the average age in the dataset?",
    "How many people have a mortgage?",
    "What percentage of people over 40 have a mortgage?",
    "Show me the distribution of education levels",
    "What's the average income by occupation?"
]

def execute_pandas_query(df: pd.DataFrame, sql_query: str) -> pd.DataFrame:
    """
    Execute a SQL-like query using Pandas operations
    """
    # Convert SQL query to pandas operations
    query = sql_query.lower()
    
    try:
        if "select * from" in query:
            return df
        
        if "where" in query:
            # Extract conditions
            conditions = query.split("where")[1].strip()
            # Handle basic comparison operations
            if ">" in conditions:
                col, val = conditions.split(">")
                col = col.strip()
                val = float(val.strip())
                result_df = df[df[col] > val]
            elif "<" in conditions:
                col, val = conditions.split("<")
                col = col.strip()
                val = float(val.strip())
                result_df = df[df[col] < val]
            elif "=" in conditions:
                col, val = conditions.split("=")
                col = col.strip()
                val = val.strip().strip("'").strip('"')
                result_df = df[df[col] == val]
            else:
                result_df = df
        else:
            result_df = df
            
        if "group by" in query:
            group_cols = query.split("group by")[1].strip().split(",")
            group_cols = [col.strip() for col in group_cols]
            
            # Extract aggregation if present
            if "count" in query:
                result_df = result_df.groupby(group_cols).size().reset_index(name='count')
            elif "avg" in query or "mean" in query:
                agg_col = query.split("select")[1].split("from")[0].strip()
                if "as" in agg_col:
                    agg_col = agg_col.split("as")[0]
                agg_col = agg_col.replace("avg(", "").replace("mean(", "").replace(")", "").strip()
                result_df = result_df.groupby(group_cols)[agg_col].mean().reset_index()
            elif "sum" in query:
                agg_col = query.split("sum(")[1].split(")")[0].strip()
                result_df = result_df.groupby(group_cols)[agg_col].sum().reset_index()
                
        return result_df
        
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return df

def lookup_data(prompt: str) -> str:
    """
    Function called by the LLM to look up data using Pandas operations.
    Returns the query result as a string (table-like).
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "No data has been uploaded. Please upload a dataset first."

    df = st.session_state.df
    
    # Add information about the table structure
    columns_info = []
    for col, dtype in zip(df.columns, df.dtypes):
        columns_info.append(f"{col} ({dtype})")
    
    columns_str = ", ".join(columns_info)
    
    # Check if this is a query about mortgages or age
    if "mortgage" in prompt.lower() or "age" in prompt.lower():
        diagnostic_info = "\nDiagnostic information to help with your query:\n"
        
        # Check if 'Age' or similar column exists
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        if age_cols:
            for age_col in age_cols:
                diagnostic_info += f"\n{age_col} column statistics:\n"
                diagnostic_info += f"- Min: {df[age_col].min()}\n"
                diagnostic_info += f"- Max: {df[age_col].max()}\n"
                diagnostic_info += f"- Mean: {df[age_col].mean()}\n"
                diagnostic_info += f"- Number of people over 40: {(df[age_col] > 40).sum()}\n"
        
        # Check if 'Mortgage' or similar column exists
        mortgage_cols = [col for col in df.columns if 'mortgage' in col.lower()]
        if mortgage_cols:
            for mortgage_col in mortgage_cols:
                value_counts = df[mortgage_col].value_counts()
                diagnostic_info += f"\n{mortgage_col} value counts:\n"
                for value, count in value_counts.items():
                    diagnostic_info += f"- {value}: {count}\n"
        
        prompt = f"{prompt}\n\n{diagnostic_info}"
    
    # Generate the SQL query from the prompt
    sql_query = generate_sql_query(prompt, columns_info, "data_table")
    st.session_state.last_sql_query = sql_query

    try:
        # Execute the query using pandas operations
        result = execute_pandas_query(df, sql_query)
        
        if result.empty:
            sample_data = df.head(5).to_string(index=False)
            return f"""The query returned no results. Here's some information that might help:
            
1. Available columns: {columns_str}
2. Sample data (first 5 rows):
{sample_data}

Please try a different query or check if the conditions in your query match the data."""
        
        st.session_state.data_result = result
        
        # Format the results
        result_info = f"Query returned {result.shape[0]} rows and {result.shape[1]} columns.\n\n"
        
        if result.shape[0] > 100:
            result_info += f"Showing first 100 rows as sample:\n\n"
            display_result = result.head(100)
            text_result = result_info + display_result.to_string(index=False)
        else:
            text_result = result_info + result.to_string(index=False)
        
        # Add summary statistics for numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = result[numeric_cols].describe().to_string()
            text_result += f"\n\nSummary statistics for numeric columns:\n{stats}"
        
        return text_result
        
    except Exception as e:
        error_msg = str(e)
        sample_data = df.head(5).to_string(index=False)
        return f"""Error executing query: {error_msg}
        
Available columns: {columns_str}
Sample data (first 5 rows):
{sample_data}

Please try a different query that matches the actual column names and data types."""

# --------------------------
# PROMPTS
# --------------------------
SYSTEM_PROMPT = """You are a data analyst. Answer questions about the dataset using the available functions: lookup_data for SQL queries and analyze_data for analysis."""

SQL_GENERATION_PROMPT = """Generate SQL query for: "{prompt}". Available columns: {columns}. Table name: {table_name}."""

DATA_ANALYSIS_PROMPT = """Analyze: {data}
Question: "{prompt}"
Show calculations for percentages: (numerator/denominator)*100."""

# --------------------------
# FUNCTIONS (for function calling)
# --------------------------

def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    """
    Generate a valid SQL query for DuckDB based on the user's prompt.
    """
    global client
    if client is None:
        client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)
        
    # Fill in the template
    msg = SQL_GENERATION_PROMPT.format(
        prompt=prompt,
        columns=", ".join(columns),
        table_name=table_name
    )
    # Call ChatCompletion to get the SQL query
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful SQL generator."},
                {"role": "user", "content": msg}
            ],
            temperature=0
        )
        # Extract the returned SQL (the model should return only SQL)
        sql_query = response.choices[0].message.content.strip()
        # Remove any leftover fences
        sql_query = sql_query.replace("```sql", "").replace("```", "")
        return sql_query
    except Exception as e:
        st.session_state.debug_messages.append(f"Error generating SQL: {str(e)}")
        return f"Error generating SQL: {str(e)}"

def analyze_data(data: str, prompt: str) -> str:
    """
    Function called by the LLM to analyze the data retrieved from lookup_data.
    """
    global client
    if client is None:
        client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)
        
    if not data or data.strip() == "":
        return "No data available to analyze."
    
    # Ensure the data is properly formatted for analysis
    # Check if we have the data_result in session state
    if st.session_state.data_result is not None and not st.session_state.data_result.empty:
        # Format the data in a more structured way
        formatted_data = "Data for analysis:\n\n"
        
        # Add the raw data as a table
        formatted_data += data
        
        # If it's a simple query with just a few values, make it extra clear
        if st.session_state.data_result.shape[0] == 1 and st.session_state.data_result.shape[1] <= 5:
            formatted_data += "\n\nKey values extracted from the data:\n"
            for col in st.session_state.data_result.columns:
                value = st.session_state.data_result.iloc[0][col]
                formatted_data += f"- {col}: {value}\n"
            
            # If this is a percentage calculation, help with the math
            if "percentage" in prompt.lower() and st.session_state.data_result.shape[1] >= 2:
                # Try to identify numerator and denominator columns
                cols = st.session_state.data_result.columns
                if len(cols) >= 2:
                    total_col = None
                    part_col = None
                    
                    # Look for column names that suggest totals or parts
                    for col in cols:
                        col_lower = col.lower()
                        if "total" in col_lower or "all" in col_lower or "count" in col_lower:
                            total_col = col
                        elif "with" in col_lower or "has" in col_lower or "yes" in col_lower:
                            part_col = col
                    
                    # If we couldn't identify by name, use the first two columns
                    if total_col is None or part_col is None:
                        if "total" in cols[0].lower() and "with" in cols[1].lower():
                            total_col, part_col = cols[0], cols[1]
                        else:
                            total_col, part_col = cols[0], cols[1]
                    
                    # Calculate the percentage
                    total_value = float(st.session_state.data_result.iloc[0][total_col])
                    part_value = float(st.session_state.data_result.iloc[0][part_col])
                    
                    if total_value > 0:
                        percentage = (part_value / total_value) * 100
                        formatted_data += f"\nCalculation: ({part_value} / {total_value}) * 100 = {percentage:.2f}%\n"
                        formatted_data += f"\nTherefore, {percentage:.2f}% of the total have the specified attribute.\n"
        
        data = formatted_data
    
    # Format the analysis prompt
    msg = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)
    
    # If the formatted prompt is too large, truncate it but keep the important parts
    if len(msg) > 15000:  # Adjust this threshold as needed
        st.session_state.debug_messages.append(f"Warning: Analysis prompt is very large ({len(msg)} chars), truncating...")
        # Keep the beginning (instructions and data info) and the end (the actual question)
        beginning = msg[:5000]  # First 5000 chars
        end = msg[-5000:]       # Last 5000 chars
        msg = beginning + "\n...[data truncated due to size]...\n" + end
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert data analyst."},
                {"role": "user", "content": msg}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.session_state.debug_messages.append(f"Error analyzing data: {str(e)}")
        return f"Error analyzing data: {str(e)}"


# 3) Define tools for function calling (updated format for newer OpenAI API)
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_data",
            "description": "Look up data from the uploaded dataset using an SQL query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The user's question or prompt about the data."
                    }
                },
                "required": ["prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Analyze data to extract insights based on the user's prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The data output from lookup_data, formatted as text."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The user's question or prompt about the data."
                    }
                },
                "required": ["data", "prompt"]
            }
        }
    }
]

def call_function(function_name, arguments):
    """Manually dispatch the function call from the model."""
    # Log the function call for debugging
    st.session_state.debug_messages.append(f"Dispatching function: {function_name} with args: {arguments}")
    
    if function_name == "lookup_data":
        return lookup_data(**arguments)
    elif function_name == "analyze_data":
        # Check if the data is being passed correctly
        if 'data' in arguments:
            # Log the first part of the data for debugging
            data_preview = arguments['data'][:100] + "..." if len(arguments['data']) > 100 else arguments['data']
            st.session_state.debug_messages.append(f"Data being passed to analyze_data: {data_preview}")
            
            # If the data looks like it's just column names and values, expand it
            if ":" in arguments['data'] and len(arguments['data'].split("\n")) <= 2:
                st.session_state.debug_messages.append("Data appears to be in simplified format, expanding it")
                
                # Try to reconstruct from data_result if available
                if st.session_state.data_result is not None and not st.session_state.data_result.empty:
                    full_data = st.session_state.data_result.to_string(index=False)
                    arguments['data'] = full_data
        
        return analyze_data(**arguments)
    else:
        return f"Error: unknown function {function_name}"


# 4) The main agent loop
def run_agent(user_query: str):
    """
    Sends the user query + system prompt to OpenAI, checks for function calls,
    executes them, and returns the final response to display.
    """
    global client
    # Initialize the OpenAI client with the current API key
    client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)
    
    # Add information about the dataset to help the model
    dataset_info = ""
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        dataset_info = f"\nCurrent dataset information:\n"
        dataset_info += f"- Table name: data_table\n"
        dataset_info += f"- Number of rows: {df.shape[0]}\n"
        dataset_info += f"- Number of columns: {df.shape[1]}\n"
        dataset_info += f"- Column names: {', '.join(df.columns.tolist())}\n"
        
        # Add information about data types
        dataset_info += f"\nColumn data types:\n"
        for col, dtype in zip(df.columns, df.dtypes):
            dataset_info += f"- {col}: {dtype}\n"
        
        # Add sample data
        sample_data = df.head(3).to_string(index=False)
        dataset_info += f"\nSample data (first 3 rows):\n{sample_data}\n"
        
        # Add specific information about mortgage and age columns if they exist
        mortgage_cols = [col for col in df.columns if 'mortgage' in col.lower()]
        age_cols = [col for col in df.columns if 'age' in col.lower()]
        
        if mortgage_cols or age_cols:
            dataset_info += "\nSpecial column information:\n"
            
            for col in mortgage_cols:
                unique_values = df[col].unique()
                dataset_info += f"- {col} unique values: {unique_values}\n"
                
            for col in age_cols:
                dataset_info += f"- {col} range: {df[col].min()} to {df[col].max()}\n"
    
    system_message = SYSTEM_PROMPT + dataset_info
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query}
    ]
    st.session_state.conversation_history.append({"role": "user", "content": user_query})
    
    # Clear debug messages for this run
    st.session_state.debug_messages = []
    st.session_state.debug_messages.append(f"Starting agent with query: {user_query}")
    if dataset_info:
        st.session_state.debug_messages.append(f"Added dataset info to system prompt")

    max_iterations = 5  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        st.session_state.debug_messages.append(f"Iteration {iteration}")
        
        try:
            st.session_state.debug_messages.append("Calling OpenAI API...")
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # Let the model choose whether to call a function
                temperature=0
            )
            
            msg = response.choices[0].message
            messages.append(msg)  # Add the response to the conversation
            
            # Check if the model wants to call a tool
            tool_calls = msg.tool_calls
            
            if tool_calls:
                st.session_state.debug_messages.append(f"Model wants to call {len(tool_calls)} tool(s)")
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    st.session_state.debug_messages.append(
                        f"Calling function '{function_name}' with args: {function_args}"
                    )
                    
                    function_response = call_function(function_name, function_args)
                    
                    # Log the response for debugging
                    response_preview = function_response[:200] + "..." if len(function_response) > 200 else function_response
                    st.session_state.debug_messages.append(f"Function response: {response_preview}")
                    
                    # Add the function result to the messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": function_response
                    })
            else:
                # Final answer from the model
                final_response = msg.content
                st.session_state.debug_messages.append("Model provided final response")
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": final_response}
                )
                return final_response
                
        except Exception as e:
            error_msg = f"Error in run_agent: {str(e)}"
            st.session_state.debug_messages.append(error_msg)
            return f"An error occurred: {str(e)}"
    
    return "Maximum number of iterations reached without a final answer."


# 5) Loading data and cleaning columns
def load_data(uploaded_file):
    """Load the file into a Pandas DataFrame, rename columns, handle various formats."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    try:
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'parquet':
            bytes_data = uploaded_file.getvalue()
            with io.BytesIO(bytes_data) as buffer:
                try:
                    df = pd.read_parquet(buffer)
                except Exception as e1:
                    st.session_state.debug_messages.append(f"First parquet read attempt failed: {str(e1)}")
                    buffer.seek(0)
                    try:
                        df = pd.read_parquet(buffer, engine='pyarrow')
                    except Exception as e2:
                        st.session_state.debug_messages.append(f"Second parquet read attempt failed: {str(e2)}")
                        buffer.seek(0)
                        df = pd.read_parquet(buffer, engine='fastparquet')
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            return None, f"Unsupported file format: {file_extension}"

        if df.empty:
            return None, "The uploaded file contains no data."

        # Rename columns to avoid DuckDB or LLM confusion with spaces or punctuation
        safe_cols = []
        for col in df.columns:
            # Example: replace spaces with underscores, remove parentheses, etc.
            clean_col = col.strip()
            clean_col = clean_col.replace(" ", "_").replace("(", "").replace(")", "")
            clean_col = clean_col.replace("/", "_").replace("-", "_")
            safe_cols.append(clean_col)
        df.columns = safe_cols

        # Attempt numeric conversion for columns that are still object dtype
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


# --------------------------
# STREAMLIT APP LAYOUT
# --------------------------

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'data_result' not in st.session_state:
    st.session_state.data_result = None
if 'last_sql_query' not in st.session_state:
    st.session_state.last_sql_query = ""
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'debug_messages' not in st.session_state:
    st.session_state.debug_messages = []

# Title
st.title("Data Analysis Dashboard")
st.subheader("Upload any dataset and ask questions using natural language.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input with better error handling
    api_key = st.text_input(
        "Enter OpenAI API Key", 
        value=st.session_state.OPENAI_API_KEY,
        type="password",
        help="Required for analysis. Get your API key from OpenAI dashboard."
    )
    if api_key:
        st.session_state.OPENAI_API_KEY = api_key
        client = OpenAI(api_key=api_key)
    
    st.header("Mode Selection")
    demo_mode = st.checkbox("Use Demo Mode", value=True, help="Try the app with sample data")
    
    if not demo_mode:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx', 'xls', 'parquet', 'json'])
        if uploaded_file is not None:
            df, error = load_data(uploaded_file)
            if error:
                st.error(error)
            else:
                st.session_state.df = df
                st.session_state.file_name = uploaded_file.name
                st.success(f"Successfully loaded: {uploaded_file.name}")
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    else:
        st.session_state.df = load_sample_data()
        st.session_state.file_name = "sample_data.csv"
        st.success("Demo mode: Using sample dataset")
        st.write(f"Sample data rows: {st.session_state.df.shape[0]}")

    st.markdown("---")
    st.header("About")
    st.markdown("""
    This application allows you to:
    - Upload any dataset (CSV, Excel, Parquet, JSON)
    - Query the dataset using natural language
    - View query results in tabular format
    - Analyze the data with AI assistance
    - Export results for further analysis
    """)

# Tabs: Query & Analysis, Data Explorer
tab1, tab2 = st.tabs(["Query & Analysis", "Data Explorer"])

with tab1:
    st.header("Ask Questions About Your Data")
    if not st.session_state.OPENAI_API_KEY:
        st.error("Please enter your OpenAI API key in the sidebar to proceed.")
    elif st.session_state.df is None and not demo_mode:
        st.info("Please upload a dataset using the sidebar to get started.")
    else:
        st.write(f"Currently analyzing: **{st.session_state.file_name}**")

        # Example queries section
        with st.expander("📝 Example Queries (Click to try)"):
            st.write("Click any example query to try it:")
            for query in EXAMPLE_QUERIES:
                if st.button(query):
                    st.session_state.query = query

        # Query input
        query = st.text_area(
            "Enter your question about the data:",
            value=st.session_state.get("query", ""),
            placeholder="Example: Show me the summary statistics for all columns"
        )

        # Execute query button
        if st.button("Run Query & Analysis"):
            if not st.session_state.OPENAI_API_KEY:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                with st.spinner("Processing your query..."):
                    final_answer = run_agent(query)
                    st.markdown("### Analysis Results")
                    st.markdown(final_answer)

                    # If there's data from the last query, show it
                    if st.session_state.data_result is not None:
                        st.markdown("### Data Results")
                        st.dataframe(st.session_state.data_result)

                        # Download button for the results
                        csv_data = st.session_state.data_result.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_data,
                            file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

        # Show the SQL query if available
        if st.session_state.last_sql_query:
            with st.expander("View SQL Query"):
                st.code(st.session_state.last_sql_query, language="sql")

        # Show debug messages
        if st.session_state.debug_messages:
            with st.expander("Debug Information"):
                for msg in st.session_state.debug_messages:
                    st.text(msg)

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

with tab2:
    st.header("Data Explorer")
    if st.session_state.df is None and not demo_mode:
        st.info("Please upload a dataset using the sidebar to get started.")
    else:
        df = st.session_state.df
        st.markdown("### Data Sample")
        sample_rows = min(1000, df.shape[0])
        st.dataframe(df.head(sample_rows))
        st.caption(f"Showing first {sample_rows} rows out of {df.shape[0]} total rows.")

        st.markdown("### Summary Statistics")
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe())
        else:
            st.info("No numeric columns found for summary statistics.")

        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values
        })
        st.dataframe(col_info)

        st.markdown("### Data Exploration Options")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            selected_cat_col = st.selectbox(
                "Select a categorical column to view unique values:",
                options=[""] + categorical_cols
            )
            if selected_cat_col:
                unique_values = df[selected_cat_col].value_counts().reset_index()
                unique_values.columns = [selected_cat_col, 'Count']
                st.dataframe(unique_values)
        else:
            st.info("No categorical columns found.")

st.markdown("---")
st.markdown("© 2025 Data Analysis Dashboard")
