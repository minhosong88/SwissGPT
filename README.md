# SwissGPT

SwissGPT is a versatile AI-powered toolset, akin to a Swiss knife, that can be used for a variety of tasks. It comprises seven specialized projects, each designed to handle specific types of data and queries, all integrated into a single platform. The main page, `Home.py`, serves as a hub, providing links to the seven AI Web service, using Langchain, GPT-4, Whisper, FastAPI, Streamlit, Pinecone, Hugging Face… and more.

## Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [DocumentGPT](#documentgpt)
  - [QuizGPT](#quizgpt)
  - [PrivateGPT](#privategpt)
  - [SiteGPT](#sitegpt)
  - [MeetingGPT](#meetinggpt)
  - [InvestorGPT](#investorgpt)
  - [AssistantGPT](#assistantgpt)
  - [ChefGPT](#chefgpt)
- [Installation and Setup](#installation-and-setup)
- [Usage Instructions](#usage-instructions)
- [Contributing](#contributing)
- [Contact Information](#contact-information)

## Overview

SwissGPT is designed to provide a comprehensive set of AI tools that cater to different needs, from document analysis to generating investment insights. Each tool within SwissGPT leverages advanced language models to deliver specialized functionalities.

## Projects

### DocumentGPT

DocumentGPT is a chatbot that answers questions based on the content of uploaded documents. It supports PDF, TXT, and DOCX files.

**Features:**
- Document Embedding
- Interactive Chat Interface
- Memory Management

**Key Files:**
- `01_DocumentGPT.py`

### QuizGPT

QuizGPT generates quizzes based on the content of uploaded documents or Wikipedia articles. It helps users test their knowledge and study effectively.

**Features:**
- Generates Multiple Choice Questions
- Supports File and Wikipedia Inputs

**Key Files:**
- `02_QuizGPT.py`

### PrivateGPT

PrivateGPT provides a secure way to ask questions about sensitive documents. It ensures data privacy and integrity.

**Features:**
- Secure Document Processing
- Interactive Chat Interface

**Key Files:**
- `03_PrivateGPT.py`

### SiteGPT

SiteGPT allows users to ask questions about the content of a website by analyzing its sitemap. It retrieves and processes website data for query answering.

**Features:**
- Website Content Analysis
- Interactive Chat Interface

**Key Files:**
- `04_SiteGPT.py`

### MeetingGPT

MeetingGPT transcribes and summarizes meeting recordings. It also provides a chat interface to ask questions about the meeting content.

**Features:**
- Audio Transcription
- Meeting Summarization
- Interactive Q&A

**Key Files:**
- `05_MeetingGPT.py`

### InvestorGPT

InvestorGPT provides financial insights for companies based on their stock performance, income statements, and other financial data.

**Features:**
- Financial Data Analysis
- Investment Recommendations

**Key Files:**
- `06_InvestorGPT.py`

### AssistantGPT

AssistantGPT offers various assistance, such as financial insights and daily stock performance, based on company data.

**Features:**
- Financial Insights
- Stock Performance Analysis

**Key Files:**
- `07_AssistantGPT.py`

### ChefGPT

ChefGPT provides recipes based on given ingredients and allows users to save their favorite recipes. It focuses on providing Indian recipes.

**Features:**
- Recipe Recommendations
- Favorite Recipes Management

**Key Files:**
- `08_ChefGPT.py`
- `jwtforchef.py`

## Installation and Setup

**Prerequisites:**
- Python 3.8 or higher
- Required packages listed in `requirements.txt`

**Steps:**

1. Clone the repository:
    ```sh
    git clone https://github.com/minhosong88/SwissGPT.git
    ```

2. Navigate to the project directory:
    ```sh
    cd SwissGPT
    ```

3. Install the necessary dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables as needed (e.g., for API keys).

## Usage Instructions

1. Run the main application:
    ```sh
    streamlit run Home.py
    ```

2. Use the links on the home page to navigate to the different projects.

3. Follow the instructions on each project's page to upload files or input data and interact with the AI tools.

## Contributing

**Guidelines for contributing:**

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a Pull Request.

## Contact Information

If you have any questions or feedback, feel free to reach out:

- **Name:** Minho Song
- **Email:** hominsong@naver.com
