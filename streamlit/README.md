# Amazon Q Streamlit App

This Streamlit application provides a chat interface to interact with Amazon Q, AWS's generative AI-powered assistant.

## Prerequisites

- Python 3.8 or higher
- AWS account with Amazon Q Business enabled
- AWS credentials configured locally

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Configure your AWS credentials:

```bash
aws configure
```

3. Run the Streamlit app:

```bash
Create .venv using requirements.txt and run $> source .venv/bin/activate
$> streamlit run bedrock_chat_app.py
```

## Features

- Chat interface for Bedrock
- Conversation history tracking
- Ability to start new conversations
- AWS region configuration

## AWS Authentication

You would need to provide AWS Access and Secret key in the streamlit app

## Notes

