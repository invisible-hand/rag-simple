# Data Analysis Dashboard

An interactive data analysis dashboard that allows you to analyze datasets using natural language queries, powered by OpenAI's GPT models and Streamlit.

## 🌟 Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Interactive Analysis**: Get instant insights and visualizations
- **Multiple File Formats**: Support for CSV, Excel, Parquet, and JSON files
- **Demo Mode**: Try the app with sample data
- **Export Results**: Download analysis results as CSV files

## 🚀 Quick Start

1. Visit the app at [your-app-url]
2. Enter your OpenAI API key in the sidebar
3. Choose between Demo Mode or upload your own dataset
4. Start asking questions about the data!

## 💡 Example Queries

Try these example queries to get started:
- "What is the average age in the dataset?"
- "How many people have a mortgage?"
- "What percentage of people over 40 have a mortgage?"
- "Show me the distribution of education levels"
- "What's the average income by occupation?"

## 📊 Sample Dataset

The demo mode includes a sample dataset with the following columns:
- Age: Age of individuals
- Income: Annual income
- Mortgage: Whether the person has a mortgage (Yes/No)
- Education: Education level (Bachelor, Master, PhD)
- Occupation: Current job role

## 🔑 API Key Setup

1. Get your API key from [OpenAI](https://platform.openai.com/api-keys)
2. Enter it in the sidebar of the app
3. Your key is never stored and must be re-entered when you refresh the page

## 🛠️ Local Development

To run the app locally:

1. Clone this repository:
```bash
git clone [your-repo-url]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

## 📝 Requirements

- Python 3.7+
- Streamlit
- Pandas
- DuckDB
- OpenAI Python package

## 🔒 Security Note

- Never commit your API keys to the repository
- Use environment variables or Streamlit's secrets management in production
- The app is for demonstration purposes and should be properly secured before production use

## 📫 Support

For issues and feature requests, please open an issue in this repository.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 