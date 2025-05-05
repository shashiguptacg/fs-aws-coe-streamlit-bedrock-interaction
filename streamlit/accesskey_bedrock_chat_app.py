import streamlit as st
import boto3
import json
import uuid
import os
from botocore.exceptions import ClientError

# Set page configuration
st.set_page_config(
    page_title="Amazon Bedrock Chat Interface",
    page_icon="💬",
    layout="centered"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# App title and description
st.title("Amazon Bedrock Chat Interface - FS AWS CoE")
st.markdown("Ask questions and get answers from Amazon Bedrock AI models")

# Sidebar for AWS configuration
with st.sidebar:
    st.header("AWS Configuration")
    aws_region = st.text_input("AWS Region", "us-east-1")
    
    # AWS Credentials input
    st.subheader("AWS Credentials")
    aws_access_key = st.text_input("AWS Access Key ID", type="password")
    aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
    aws_session_token = st.text_input("AWS Session Token (optional)", type="password")
    
    # Model selection
    st.subheader("Model Settings")
    model_options = {
        "Amazon Titan Text Express": "amazon.titan-text-express-v1",
        "Claude Instant": "anthropic.claude-instant-v1",
        "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "Llama 2 Chat 13B": "meta.llama2-13b-chat-v1"
    }
    
    selected_model = st.selectbox(
        "Select AI Model",
        options=list(model_options.keys()),
        index=0
    )
    
    st.info("""
    ℹ️ **Important**: You need to enable model access in the AWS Bedrock console before using a model.
    If you get an "AccessDeniedException", it means you haven't enabled access to that specific model.
    """)
    
    with st.expander("How to enable model access in AWS Bedrock"):
        st.markdown("""
        1. Go to the [AWS Bedrock Console](https://console.aws.amazon.com/bedrock)
        2. Click on "Model access" in the left navigation
        3. Click "Manage model access"
        4. Select the models you want to use
        5. Click "Save changes"
        6. Wait a few minutes for access to be granted
        
        Note: Some models may require a subscription or additional approval from AWS.
        Amazon Titan models are typically enabled by default for all Bedrock users.
        """)
    
    
    model_id = model_options[selected_model]
    
    max_tokens = st.slider("Max Response Tokens", 100, 4096, 1024)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app demonstrates integration with Amazon Bedrock, 
    AWS's foundation model service for generative AI.
    """)

# Initialize Bedrock client
@st.cache_resource
def get_bedrock_client(region_name, aws_access_key=None, aws_secret_key=None, aws_session_token=None):
    if aws_access_key and aws_secret_key:
        # Use provided credentials
        if aws_session_token:
            return boto3.client(
                'bedrock-runtime', 
                region_name=region_name,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                aws_session_token=aws_session_token
            )
        else:
            return boto3.client(
                'bedrock-runtime', 
                region_name=region_name,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
    else:
        # Use default credentials from ~/.aws/credentials
        return boto3.client('bedrock-runtime', region_name=region_name)

# Function to check if a model is accessible
def check_model_access(client, model_id):
    try:
        # Try to invoke the model with a minimal request
        if "anthropic" in model_id:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "temperature": 0.7,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        elif "meta.llama" in model_id:
            request_body = {
                "prompt": "Human: Hello\nAssistant: ",
                "max_gen_len": 10,
                "temperature": 0.7
            }
        else:
            request_body = {
                "inputText": "Hello",
                "textGenerationConfig": {
                    "maxTokenCount": 10,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
            
        client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        return True
    except ClientError as e:
        if "AccessDeniedException" in str(e):
            return False
        else:
            # If it's another type of error, we'll assume the model is accessible
            # but there's another issue
            return True

# Function to send message to Bedrock and get response
def send_message_to_bedrock(client, model_id, messages, max_tokens, temperature):
    try:
        # Format the prompt based on the model
        if "anthropic" in model_id:
            # Claude models use a specific format
            conversation = []
            for msg in messages:
                role = "user" if msg["role"] == "user" else "assistant"
                conversation.append({"role": role, "content": msg["content"]})
            
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": conversation
            }
        elif "meta.llama" in model_id:
            # Llama models use a different format
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"Human: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "
            
            request_body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature
            }
        else:
            # Default format for Titan and other models
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "
            
            request_body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            }
        
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Parse the response based on the model
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        if "anthropic" in model_id:
            return response_body['content'][0]['text']
        elif "meta.llama" in model_id:
            return response_body['generation']
        else:
            return response_body['results'][0]['outputText']
            
    except ClientError as e:
        st.error(f"Error sending message: {e}")
        return "I encountered an error processing your request. Please check your AWS credentials and permissions."

# Initialize Bedrock client
bedrock_client = get_bedrock_client(
    aws_region,
    aws_access_key=aws_access_key if aws_access_key else None,
    aws_secret_key=aws_secret_key if aws_secret_key else None,
    aws_session_token=aws_session_token if aws_session_token else None
)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Check model access when model is selected
if "model_access_checked" not in st.session_state:
    st.session_state.model_access_checked = {}

if model_id not in st.session_state.model_access_checked and aws_access_key and aws_secret_key:
    has_access = check_model_access(bedrock_client, model_id)
    st.session_state.model_access_checked[model_id] = has_access
    if not has_access:
        st.warning(f"⚠️ You don't have access to {selected_model}. Please enable access in the AWS Bedrock console or select a different model.")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Check if credentials are provided
    if not aws_access_key or not aws_secret_key:
        st.error("Please provide your AWS credentials in the sidebar to use this app.")
    else:
        # Check model access
        has_access = st.session_state.model_access_checked.get(model_id, False)
        if not has_access and model_id in st.session_state.model_access_checked:
            st.error(f"You don't have access to {selected_model}. Please enable access in the AWS Bedrock console or select a different model.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response from Bedrock
            with st.chat_message("assistant"):
                with st.spinner(f"{selected_model} is thinking..."):
                    response = send_message_to_bedrock(
                        bedrock_client,
                        model_id,
                        st.session_state.messages,
                        max_tokens,
                        temperature
                    )
            
            if response:
                st.write(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Failed to get response from the model")

# Reset conversation button
if st.button("Start New Conversation"):
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

# Add a note about AWS credentials
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Note on AWS Credentials
Your credentials are used only for API calls and are not stored. 
For security, you can also set these as environment variables:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_SESSION_TOKEN`
""")

# Check for environment variables at startup
if not aws_access_key:
    env_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    if env_access_key:
        st.sidebar.success("✅ AWS Access Key ID found in environment variables")
        
if not aws_secret_key:
    env_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if env_secret_key:
        st.sidebar.success("✅ AWS Secret Access Key found in environment variables")