from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sqlalchemy import create_engine, text
from llama_index.llms.ollama import Ollama
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage

app = Flask(__name__)
CORS(app)

# Initialize the LLM (Ollama model) for both chat and query responses
llm_for_sql = Ollama(model="llama3.2:latest", request_timeout=300.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Define the path for SQLite database and load data
DATABASE_PATH = "orders.db"
engine = create_engine(f"sqlite:///{DATABASE_PATH}", echo=True)

# Load orders from Excel to DataFrame for quick access
orders_df = pd.read_excel("Orders.xlsx")

def get_general_chat_response(user_input, chat_history):
    """Generate response for general chat using LLM (Ollama model)."""
    chat_context = "\n".join(chat_history[-5:] + [f"Human: {user_input}", "Bot:"])
    
    # Generate chat response using llm_for_sql
    messages = [
        ChatMessage(role="system", content="You are an assistant helping a user with general chat responses."),
        ChatMessage(role="user", content=chat_context)
    ]
    
    # Use LLM (Ollama model) for the response
    response = llm_for_sql.chat(messages)
    print(f"Generated response: ", response)
    return response["message"]["content"]

@app.route('/api/chat', methods=['POST'])
def handle_general_chat():
    """Handle general chat and context-based responses."""
    data = request.json
    message = data.get('query')
    print(f"Received message: {message}")
    if not message:
        return jsonify({"error": "Message not provided"}), 400

    # Check for order number in the message
    order_match = re.search(r"\b\d{6}\b", message)
    if order_match:
        order_number = order_match.group()
        
        # Retrieve order details using pandas
        order_details = orders_df[orders_df['ORDERNUMBER'] == int(order_number)]
        if not order_details.empty:
            order_info = order_details.iloc[0].to_dict()
            order_info_text = ", ".join(f"{k}: {v}" for k, v in order_info.items())
            
            prompt = f"User asked about order {order_number}. Details: {order_info_text}"
            response = get_general_chat_response(prompt, [])
            return jsonify({"response": response}), 200
        else:
            return jsonify({"response": f"No details found for order {order_number}."}), 404

    # General chat mode when no specific order is found
    chat_history = data.get("chat_history", [])
    response = get_general_chat_response(message, chat_history)
    return jsonify({"response": response}), 200

@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle order-specific database queries with NLP to SQL translation."""
    data = request.json
    query_str = data.get('query')
    if not query_str:
        return jsonify({"error": "Query not provided"}), 400

    try:
        # Initialize the SQLDatabase and NLSQLTableQueryEngine
        sql_database = SQLDatabase(engine)
        query_engine = NLSQLTableQueryEngine(sql_database=sql_database, tables=["orders"], llm=llm_for_sql)

        # Generate SQL query from natural language
        response = query_engine.query(query_str)
        sql_query = response.metadata['sql_query']
        print("Generated SQL Query ::>", sql_query)

        # Execute generated SQL query
        with engine.connect() as connection:
            results = connection.execute(text(sql_query)).fetchall()
        
        if results:
            print(results)
            sql_response = "\n".join([str(row) for row in results])
            # Use LLM for a more human-friendly response
            messages = [
                ChatMessage(role="system", content="You are an assistant helping a user interpret database query results."),
                ChatMessage(role="user", content=f"User's Question: {query_str}\nThe SQL query result is: {sql_response}. Please provide a concise, human-friendly answer.")
            ]
            human_response = llm_for_sql.chat(messages)
            print(f"Human response: ", human_response)
            return jsonify({"response": human_response["message"]["content"]}), 200

        return jsonify({"response": "No results found for the query."}), 404

    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": "An error occurred processing the query"}), 500

if __name__ == '__main__':
    app.run(debug=True)
