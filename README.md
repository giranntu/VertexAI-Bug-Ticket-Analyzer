# VertexBugTicketAnalyzer

A tool for analyzing bug tickets using Vertex LLM. This repository contains a Jupyter notebook for analysis, a sample dataset, and a Streamlit application for visualization and insights.

# Advanced Ticket Analysis System

A Streamlit-based ticket analysis system that utilizes the Vertex AI language model for analyzing and generating alternative queries based on the user's input. It searches a database for similar tickets and provides a detailed analysis and recommendation.

## Setup & Installation

### 1. Clone the Repository

```bash

git clone https://github.com/giranntu/VertexAI-Bug-Ticket-Analyzer.git

cd VertexAI-Bug-Ticket-Analyzer

```

### 2. Install Dependencies

Before you can run the application, you need to install the required dependencies:

```bash

pip install -r requirements.txt

```

### 3. Environment Variables

Ensure the following environment variables are set:

- `INSTANCE_CONNECTION_NAME`: The name of the instance connection.

- `DB_USER`: Database user.

- `DB_PASS`: Database password.

- `DB_NAME`: Name of the database.

These variables are used to establish a connection to your PostgreSQL database.

### 4. Run the Application

You can start the application using:

```bash

streamlit run streamlit_bug_ticket_analysis.py

```

Once the application is running, you can navigate to the provided URL in your browser to interact with the system.

## Features

- **Query Input**: Enter your issue or query to start the analysis.

- **Ticket Matching**: The system will find similar tickets based on the input query.

- **Detailed Analysis**: A comprehensive report based on the matched tickets is provided.

- **Language Support**: Provides analysis in both English and Traditional Chinese.

## Caution

Always review and validate AI-generated results. Human judgment is crucial, especially in vital situations.

## Contributing

Please read the contribution guidelines before submitting a pull request.

## License

This project is licensed under the MIT License.
