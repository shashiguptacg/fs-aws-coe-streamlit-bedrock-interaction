import streamlit as st
import boto3
import json
from botocore.exceptions import ClientError

# Set page configuration
st.set_page_config(
    page_title="Amazon Q Chat Interface",
    page_icon="💬",
    layout="centered"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# App title and description
st.title("Amazon Q Chat Interface")
st.markdown("Ask questions and get answers from Amazon Q")

# Sidebar for AWS configuration
with st.sidebar:
    st.header("AWS Configuration")
    aws_region = st.text_input("AWS Region", "us-east-1")
    application_id = st.text_input("Amazon Q Application ID", "")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app demonstrates integration with Amazon Q, 
    AWS's generative AI-powered assistant.
    """)

# Initialize Amazon Q client
@st.cache_resource
def get_q_client(region_name):
    return boto3.client('qconnect', region_name=region_name)

# Function to start a new conversation
def start_conversation(client, app_id):
    try:
        # Create a session with Amazon Q Connect
        response = client.create_session(
            assistantId=app_id
        )
        return response.get('sessionId')
    except ClientError as e:
        st.error(f"Error starting conversation: {e}")
        return None

# Function to send message to Amazon Q and get response
def send_message_to_q(client, session_id, message):
    try:
        response = client.send_message(
            sessionId=session_id,
            text=message
        )
        return response
    except ClientError as e:
        st.error(f"Error sending message: {e}")
        return None

# Initialize Q client
q_client = get_q_client(aws_region)

# Start a new conversation if needed
if st.session_state.conversation_id is None and application_id:
    st.session_state.conversation_id = start_conversation(q_client, application_id)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask Amazon Q..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response from Amazon Q
    with st.chat_message("assistant"):
        with st.spinner("Amazon Q is thinking..."):
            if st.session_state.conversation_id:
                response = send_message_to_q(
                    q_client, 
                    st.session_state.conversation_id, 
                    prompt
                )
                
                if response and 'text' in response:
                    answer = response['text']
                    st.write(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Failed to get response from Amazon Q")
            else:
                st.error("No active conversation with Amazon Q")

# Reset conversation button
if st.button("Start New Conversation") and application_id:
    st.session_state.conversation_id = start_conversation(q_client, application_id)
    st.session_state.messages = []
    st.rerun()
