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
streamlit run amazon_q_app.py
```

## Features

- Chat interface for Amazon Q
- Conversation history tracking
- Ability to start new conversations
- AWS region configuration

## AWS Authentication

This app uses the default AWS credential provider chain. Make sure you have valid AWS credentials configured with permissions to access Amazon Q Business.

## Notes

- Amazon Q Business is a paid service. Check the [AWS pricing page](https://aws.amazon.com/q/pricing/) for details.
- The app uses the `qbusiness` AWS SDK endpoint to communicate with Amazon Q.
