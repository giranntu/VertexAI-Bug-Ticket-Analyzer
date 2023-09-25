
import streamlit as st
from google.cloud import aiplatform, sql
from google.cloud.sql.connector import Connector

from typing import List
import os

import time
import asyncio
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

import sqlalchemy


import vertexai
# from vertexai.language_models import TextGenerationModel
from vertexai.preview.language_models import TextGenerationModel
from vertexai.language_models import TextEmbeddingModel



# Constants
COLUMNS = {
    "Ticket Number": 0,
    "Subject": 1,
    "Description": 2,
    "Resolution": 3,
    "Current Behavior": 4,
    "Solution": 5,
    "Content": 6,
    "Similarity": 7
}  # Keeping it the same as in original code... mapping to postgresDB

project_id = "rust-ry"
region = "us-central1"
instance_name = "liteon-bug-tickets-demo"
database_name = "mybugticket"
database_user = "ry-admin"     
database_password = "rust123"  

# ---- Functions (as you provided, with minor modifications) ----
def connect_with_connector() -> sqlalchemy.engine.base.Engine:
    connector = Connector()
    return sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=lambda: connector.connect(
            os.getenv("INSTANCE_CONNECTION_NAME"),
            "pg8000",
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            db=os.getenv("DB_NAME")
        )
    )

# def get_embedding_for_query(query: str):
#     model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@latest")
#     instance = {"task_type": "RETRIEVAL_QUERY", "content": query}
#     return model.get_embeddings([instance])[0].values

def get_embedding_for_query(query: str):
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko-multilingual@latest")
    instance = {
        "instances": [{
            "task_type": "RETRIEVAL_QUERY",
            "content": query
        }]
    }
    embeddings = model.get_embeddings(instance)
    return embeddings[0].values

def search_similar_tickets(user_query: str, 
                           search_method: str = "cosine",
                           similarity_threshold: float = 0.001,
                           num_matches: int = 6) -> list:
    assert user_query, "⚠️ Please input a valid search query"

    qe = get_embedding_for_query(user_query)
    distance_function = {
        "cosine": "<=>",
        "inner_product": "<#>",
        "euclidean": "<->"
    }.get(search_method, "<=>")

    with connect_with_connector().connect() as db_conn:
        query = f"""
            WITH vector_matches AS (
                SELECT ticket_number, 1 - (embedding {distance_function} :embedding) AS similarity
                FROM ticket_data
                WHERE 1 - (embedding {distance_function} :embedding) > :similarity_threshold
                ORDER BY similarity DESC
                LIMIT :num_matches
            )
            SELECT 
                t.ticket_number, t.subject, t.description, t.resolution,
                t.current_behavior, t.solution, t.content, vm.similarity
            FROM ticket_data AS t
            JOIN vector_matches AS vm ON t.ticket_number = vm.ticket_number
        """

        parameters = {
            "embedding": "[" + ",".join(map(str, qe)) + "]",
            "similarity_threshold": similarity_threshold,
            "num_matches": num_matches
        }

        results = db_conn.execute(sqlalchemy.text(query), parameters).fetchall()
        matches = [
            {column_name: row[index] for column_name, index in COLUMNS.items()}
            for row in results
        ]

    return matches

# # Sample Call
# tickets = search_similar_tickets(user_query="100M isn't work well")
# print(tickets)

#beautiful priting function, in markdown format.
def print_tickets_markdown(tickets: list):
    """
    Prints the list of tickets in a markdown format.
    """
    for idx, ticket in enumerate(tickets, start=1):
        print("#" * 3 + f" Ticket #{idx}")
        
        for key, value in ticket.items():
            # Beautify certain lengthy fields to display better.
            if key in ["Description", "Content", "Current Behavior", "Solution"]:
                value = "\n\n" + "\n".join(f"- {line}" for line in value.splitlines())

            # Format the similarity score to be more readable.
            if key == "Similarity":
                value = f"{value:.4f}"

            print(f"**{key}**: {value}\n")
        print("---\n\n")

# Sample Call
# print_tickets_markdown(tickets)


# ---- Step 1: Use Vertex AI LLM to Generate Alternative Queries ----

def generate_alternative_queries(project_id: str, location: str, base_query: str, num_alternatives: int = 5, temperature: float = 0.25):
    """
    Generate alternative queries using Vertex AI LLM.
    """
    parameters = {
        "temperature": temperature,
        "max_output_tokens": 512,
        # "top_p": 0.8,
        # "top_k": 40,
        # "stopSequence" : "Original Input:"
    }

    model = TextGenerationModel.from_pretrained("text-bison-32k")
    # prompt = f"Generate {num_alternatives} alternative search queries based on: {base_query}"
    prompt = f"""
Generate {num_alternatives} different versions of the given user question to enhance information retrieval.
Do not create anything that is unrelated to the original question, but only use the known inforamtion to form the well structurize question or ask from the differnt way as an QA engineer describing the issue.
try to give first two output using Traditional Chinese, the the rest using English.

you should only generate 5 lines of questions. and do not output other things.


Original Input: CPU Overheating Issue
Output: 
CPU過熱問題
CPU渲染任務時過熱
CPU overheating during tasks
CPU software overheating
CPU temperate rises to 90°C

Original Input: LED Status Error on Port 44
Output: 
44端口LED問題
LED狀態錯誤
LED不正常
LED port 44 issue
44端口燈亮錯誤

Original Input: Hi-Pot Test Failure
Output:
高壓測試失敗
Hi-pot test issue
Hi-pot測試不通過
Hi-pot測試有問題
高壓測試不通過


Original Input: {base_query}
Output:
    """
    
    
    response = model.predict(prompt, **parameters)
    print("response: ", response)
    # Assuming the model returns alternatives separated by newlines
    queries = response.text.split('\n')[:num_alternatives]
    print("queries: ", queries)
    return queries

# ---- Step 2 & 3: Embed and Search Queries, Apply Voting Mechanism ----

def retrieve_and_rank_tickets(base_query: str, num_alternatives: int = 5, top_n: int = 5, temperature: float = 0.25):
    alternative_queries = generate_alternative_queries(os.getenv("PROJECT_ID"), os.getenv("LOCATION"), base_query, num_alternatives, temperature)
    
    alternative_queries.append(base_query)

    # Aggregate results from all queries
    all_tickets = {}
    for query in alternative_queries:
        tickets = search_similar_tickets(query)
        
        # For each ticket, increase its score in all_tickets based on its similarity
        for ticket in tickets:
            ticket_number = ticket["Ticket Number"]
            similarity = ticket["Similarity"]
            print(query, "    ===> ", ticket_number, similarity)
            if ticket_number in all_tickets:
                all_tickets[ticket_number]["count"] += 1
                all_tickets[ticket_number]["cumulative_similarity"] += similarity
            else:
                all_tickets[ticket_number] = {
                    "ticket": ticket,
                    "count": 1,
                    "cumulative_similarity": similarity
                }
    
    # Rank by number of appearances and then by cumulative similarity
    ranked_tickets = sorted(all_tickets.values(), key=lambda x: (x["count"], x["cumulative_similarity"]), reverse=True)
    
    # Return the top N tickets based on the ranking
    return [ticket["ticket"] for ticket in ranked_tickets[:top_n]]


def analyze_bug_using_tickets(project_id: str, location: str, user_query: str, top_tickets: list):
    """
    Generate a detailed report based on the user query and top tickets using the Vertex AI LLM.
    """
    # Create the prompt structure
    tickets_str = ""
    for idx, ticket in enumerate(top_tickets, start=1):
        tickets_str += f"""
Ticket {ticket['Ticket Number']}:
- Subject: {ticket['Subject']}
- Description: {ticket['Description']}
- Resolution (if available): {ticket['Resolution']}
- Solution (if available): {ticket['Solution']}
- Current Behavior: {ticket['Current Behavior']}
-----
        """
    
    prompt = f"""
Based on the user's query:
"{user_query}"

And the related bug tickets:
{tickets_str}

Please perform the following tasks:

1. Summarize each ticket briefly.
2. Highlight which tickets are most relevant to the current user query.
3. Provide a detailed analysis of the potential issue based on these past occurrences. If similar issues have been encountered before, describe their nature and how they were resolved.
4. Recommend resolutions or further steps for the current issue based on this historical data. If a similar resolution isn't found, suggest preliminary steps for investigating and potentially resolving this issue.

Remember, precision and detail are crucial. Lives depend on this analysis.
    """

    # Set parameters for LLM
    parameters = {
        "temperature": 0.4,
        "max_output_tokens": 1024
    }

    # model = TextGenerationModel.from_pretrained("text-bison@001")
    model = TextGenerationModel.from_pretrained("text-bison-32k")
    response = model.predict(prompt, **parameters)
    
    return response.text

def analyze_bug_using_tickets_tc(project_id: str, location: str, user_query: str, top_tickets: list):
    """
    Generate a detailed report based on the user query and top tickets using the Vertex AI LLM.
    """
    # Create the prompt structure
    tickets_str = ""
    for idx, ticket in enumerate(top_tickets, start=1):
        tickets_str += f"""
Ticket {ticket['Ticket Number']}:
- Subject: {ticket['Subject']}
- Description: {ticket['Description']}
- Resolution (if available): {ticket['Resolution']}
- Solution (if available): {ticket['Solution']}
- Current Behavior: {ticket['Current Behavior']}
-----
        """
    
    prompt = f"""
Based on the user's query:
"{user_query}"

And the related bug tickets:
{tickets_str}

Please perform the following tasks:

1. Summarize each ticket briefly.
2. Highlight which tickets are most relevant to the current user query.
3. Provide a detailed analysis of the potential issue based on these past occurrences. If similar issues have been encountered before, describe their nature and how they were resolved.
4. Recommend resolutions or further steps for the current issue based on this historical data. If a similar resolution isn't found, suggest preliminary steps for investigating and potentially resolving this issue.

Remember, precision and detail are crucial. Lives depend on this analysis.
用台灣的語言回答，僅有專有名詞及專業術語使用英文，讓整體的語句越流暢越好
    """

    # Set parameters for LLM
    parameters = {
        "temperature": 0.4,
        "max_output_tokens": 1024
    }

    # model = TextGenerationModel.from_pretrained("text-bison@001")
    model = TextGenerationModel.from_pretrained("text-bison-32k")
    response = model.predict(prompt, **parameters)
    
    return response.text



# ... (above is all functions: connect_with_connector, get_embedding_for_query, search_similar_tickets... etc.)

# Streamlit UI

# Initialize Streamlit's session state for language choice
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

st.title("Advanced Ticket Analysis System")
st.sidebar.header("Configuration")


# Move language selection to sidebar


# User Inputs
user_query = st.text_input("Enter your issue/query:", "Ripple Noise test fail")
search_method = st.sidebar.selectbox("Search Method", ["cosine", "inner_product", "euclidean"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 0.9, 0.5)
num_matches = st.sidebar.slider("Number of Matches", 1, 20, 6)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.001)
language_selection = st.sidebar.radio(
    "Choose Analysis Language", ["English", "Traditional Chinese"]
)

# If the Analyze button is pressed
if st.button("Analyze"):
    # Generate Alternative Queries
    alternative_queries = generate_alternative_queries(project_id, region, user_query, 5 , temperature)
    alternative_queries.append(user_query)

    # Search similar tickets for each query
    all_found_tickets = []
    for query in alternative_queries:
        tickets = search_similar_tickets(query, search_method, similarity_threshold, num_matches)
        all_found_tickets.extend(tickets)

    # Display matched ticket details in Streamlit
    st.subheader("Top Matched Ticket Details")
    for idx, ticket in enumerate(all_found_tickets[:10], start=1):
        st.markdown(f"### Ticket {idx}: {ticket['Ticket Number']}")
        st.write(f"**Subject**: {ticket['Subject']}")
        st.write(f"**Similarity Score**: {ticket['Similarity']:.4f}")

        # Use Streamlit's expander for each section
        with st.expander("Description"):
            st.write(ticket['Description'])

        if ticket.get('Resolution'):
            with st.expander("Resolution"):
                st.write(ticket['Resolution'])

        if ticket.get('Solution'):
            with st.expander("Solution"):
                st.write(ticket['Solution'])

        if ticket.get('Current Behavior'):
            with st.expander("Current Behavior"):
                st.write(ticket['Current Behavior'])

        # Can continue this for other fields if necessary
        st.write("---")

    

    # Analyze bug using top tickets
    st.subheader("Detailed Analysis and Recommendations")
    # Use the value of language_selection to decide the language
    if language_selection == "English":
        detailed_report = analyze_bug_using_tickets(project_id, region, user_query, all_found_tickets[:5])
        st.write(detailed_report)
    elif language_selection == "Traditional Chinese":
        detailed_report_tc = analyze_bug_using_tickets_tc(project_id, region, user_query, all_found_tickets[:5])
        st.write(detailed_report_tc)



st.warning("""Please review and validate AI-generated results. Human judgment is critical, especially in crucial situations.

           請查看並驗證人工智慧產生的結果。人類的判斷至關重要，尤其是在關鍵情況下 ver 0924 1037。""")

if __name__ == "__main__":
    st.write("Streamlit Application for Advanced Ticket Analysis")
