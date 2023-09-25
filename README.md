
![VertexBugTicketAnalyzer Logo](./image%20(6).png)

# VertexBugTicketAnalyzer

A comprehensive tool for analyzing bug tickets using Vertex LLM. This repository encompasses a Jupyter notebook for meticulous analysis, an exemplar dataset, and a Streamlit application geared for an immersive visualization experience and insightful data extrapolations.

## Advanced Ticket Analysis System

Leverage the power of the Vertex AI language model with this Streamlit-based ticket analysis system. Not only does it analyze, but it also crafts alternative queries from user inputs. Dive deep into a database, unearth similar tickets, and receive in-depth analysis and informed recommendations.

## 🚀 Setup & Installation

### 1. **Clone the Repository**

```bash

git clone https://github.com/giranntu/VertexAI-Bug-Ticket-Analyzer.git

cd VertexAI-Bug-Ticket-Analyzer

```

### 2. **Dependency Installation**

Kickstart your application journey by installing the vital dependencies:

```bash

pip install -r requirements.txt

```

### 3. **Configuring Environment Variables**

Configuration is key. Ensure these environment variables are impeccably set:

- `INSTANCE_CONNECTION_NAME`: Denotes the instance connection's unique name.

- `DB_USER`: The trusted database user.

- `DB_PASS`: Secure database password.

- `DB_NAME`: The distinctive name of your database.

Utilize these variables for a seamless connection with your PostgreSQL database.

### 4. **Ignite the Application**

Initialize the application with:

```bash

streamlit run streamlit_bug_ticket_analysis.py

```

Post ignition, employ your browser to interact with the system via the provided URL.

## 🌟 Features

- **Query Input**: Embark on your analysis by inputting your issue or query.

- **Ticket Matching**: A robust system hunts down tickets that resonate with your input query.

- **Detailed Analysis**: Dive into exhaustive reports generated from matched tickets.

- **Multilingual Support**: Flexibility in analysis with support in both English and Traditional Chinese.

## ⚠️ Caution

While AI is transformative, always scrutinize and validate its results. In mission-critical situations, human discernment remains unparalleled.

## 🤝 Contributing

Prior to submitting a pull request, kindly acquaint yourself with the [contribution guidelines](LINK_TO_GUIDELINES).

## 📜 License

This endeavor is fortified under the MIT License.


