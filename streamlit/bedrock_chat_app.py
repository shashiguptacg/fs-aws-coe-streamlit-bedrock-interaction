import streamlit as st
import boto3
import json
import uuid
import os
import time
import pandas as pd
from botocore.exceptions import ClientError

# Set page configuration
st.set_page_config(
    page_title="Amazon Bedrock Chat Interface - Capgemini FS AWS CoE",
    page_icon="💬",
    layout="centered"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "trace_logs" not in st.session_state:
    st.session_state.trace_logs = []

if "performance_metrics" not in st.session_state:
    st.session_state.performance_metrics = []

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

# Function to extract token counts from response
def extract_token_counts(model_id, response_body):
    input_tokens = 0
    output_tokens = 0
    
    try:
        if "anthropic" in model_id and "usage" in response_body:
            usage = response_body['usage']
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
        elif "amazon" in model_id and "amazon-bedrock-invocationMetrics" in response_body:
            metrics = response_body['amazon-bedrock-invocationMetrics']
            input_tokens = metrics.get('inputTokenCount', 0)
            output_tokens = metrics.get('outputTokenCount', 0)
        elif "meta.llama" in model_id:
            # Llama models don't always return token counts
            # Estimate based on text length (rough approximation)
            prompt_length = len(str(response_body.get('prompt', '')))
            generation_length = len(str(response_body.get('generation', '')))
            input_tokens = prompt_length // 4  # Very rough approximation
            output_tokens = generation_length // 4  # Very rough approximation
    except Exception:
        pass
        
    return input_tokens, output_tokens

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
        
        # Get the user's query (last message)
        user_query = messages[-1]["content"] if messages else ""
        
        # Record start time for tracing
        start_time = time.time()
        
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Parse the response based on the model
        response_body = json.loads(response['body'].read().decode('utf-8'))
        
        # Extract token counts
        input_tokens, output_tokens = extract_token_counts(model_id, response_body)
        
        # Get response text based on the model
        if "anthropic" in model_id:
            response_text = response_body['content'][0]['text']
        elif "meta.llama" in model_id:
            response_text = response_body['generation']
        else:
            response_text = response_body['results'][0]['outputText']
        
        # Create trace log
        trace_log = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": model_id,
            "request": request_body,
            "response": response_body,
            "elapsed_time": round(elapsed_time, 2),
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
        # Add to trace logs
        st.session_state.trace_logs.append(trace_log)
        
        # Add to performance metrics
        performance_metric = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": selected_model,
            "query_length": len(user_query),
            "response_length": len(response_text),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "response_time": round(elapsed_time, 2),
            "tokens_per_second": round((input_tokens + output_tokens) / elapsed_time, 2) if elapsed_time > 0 else 0
        }
        st.session_state.performance_metrics.append(performance_metric)
        
        return response_text
            
    except ClientError as e:
        # Record error in trace logs
        error_log = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": model_id,
            "error": str(e),
            "request": request_body if 'request_body' in locals() else None
        }
        st.session_state.trace_logs.append(error_log)
        
        st.error(f"Error sending message: {e}")
        return "I encountered an error processing your request. Please check your AWS credentials and permissions."

# Initialize Bedrock client
bedrock_client = get_bedrock_client(
    aws_region,
    aws_access_key=aws_access_key if aws_access_key else None,
    aws_secret_key=aws_secret_key if aws_secret_key else None,
    aws_session_token=aws_session_token if aws_session_token else None
)

# Check model access when model is selected
if "model_access_checked" not in st.session_state:
    st.session_state.model_access_checked = {}

if model_id not in st.session_state.model_access_checked and aws_access_key and aws_secret_key:
    has_access = check_model_access(bedrock_client, model_id)
    st.session_state.model_access_checked[model_id] = has_access
    if not has_access:
        st.warning(f"⚠️ You don't have access to {selected_model}. Please enable access in the AWS Bedrock console or select a different model.")

# Create tabs for chat, metrics, and trace
chat_tab, metrics_tab, trace_tab = st.tabs(["Chat", "Performance Metrics", "Bedrock Trace"])

with chat_tab:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Check if credentials are provided
        if not aws_access_key or not aws_secret_key:
            st.error("Please provide your AWS credentials in the sidebar to use this app.")
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

with metrics_tab:
    st.subheader("Bedrock Performance Metrics")
    
    if not st.session_state.performance_metrics:
        st.info("No performance metrics available yet. Send a message to generate metrics.")
    else:
        # Add button to clear metrics
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Clear Metrics"):
                st.session_state.performance_metrics = []
                st.rerun()
        
        # Create a DataFrame from the metrics
        df = pd.DataFrame(st.session_state.performance_metrics)
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Avg Response Time", 
                f"{df['response_time'].mean():.2f}s",
                delta=f"{df['response_time'].iloc[-1] - df['response_time'].mean():.2f}s" if len(df) > 1 else None
            )
        
        with col2:
            st.metric(
                "Avg Total Tokens", 
                f"{df['total_tokens'].mean():.0f}",
                delta=f"{df['total_tokens'].iloc[-1] - df['total_tokens'].mean():.0f}" if len(df) > 1 else None
            )
            
        with col3:
            st.metric(
                "Avg Tokens/Second", 
                f"{df['tokens_per_second'].mean():.2f}",
                delta=f"{df['tokens_per_second'].iloc[-1] - df['tokens_per_second'].mean():.2f}" if len(df) > 1 else None
            )
        
        # Display metrics table
        st.subheader("Detailed Metrics")
        
        # Format the DataFrame for display
        display_df = df.copy()
        display_df = display_df.sort_values(by='timestamp', ascending=False)
        
        # Add model color coding
        def highlight_model(val):
            if "Claude" in val:
                return 'background-color: #e6f3ff'
            elif "Titan" in val:
                return 'background-color: #fff2e6'
            elif "Llama" in val:
                return 'background-color: #e6ffe6'
            return ''
        
        # Display the styled table
        st.dataframe(
            display_df,
            column_config={
                "timestamp": "Time",
                "model": "Model",
                "query_length": "Query Length (chars)",
                "response_length": "Response Length (chars)",
                "input_tokens": "Input Tokens",
                "output_tokens": "Output Tokens",
                "total_tokens": "Total Tokens",
                "response_time": "Response Time (s)",
                "tokens_per_second": "Tokens/Second"
            },
            use_container_width=True
        )
        
        # Add visualization
        st.subheader("Visualizations")
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select Visualization",
            ["Response Time by Model", "Token Usage by Model", "Tokens per Second by Model"]
        )
        
        if viz_type == "Response Time by Model":
            st.bar_chart(df, x="model", y="response_time")
        elif viz_type == "Token Usage by Model":
            token_df = df.melt(
                id_vars=["model", "timestamp"],
                value_vars=["input_tokens", "output_tokens"],
                var_name="Token Type",
                value_name="Token Count"
            )
            st.bar_chart(token_df, x="model", y="Token Count", color="Token Type")
        else:  # Tokens per Second
            st.bar_chart(df, x="model", y="tokens_per_second")

with trace_tab:
    st.subheader("Bedrock API Trace Logs")
    
    if not st.session_state.trace_logs:
        st.info("No trace logs available yet. Send a message to generate trace logs.")
    else:
        # Add button to clear trace logs
        if st.button("Clear Trace Logs"):
            st.session_state.trace_logs = []
            st.rerun()
        
        # Display trace logs in reverse order (newest first)
        for i, log in enumerate(reversed(st.session_state.trace_logs)):
            with st.expander(f"Trace {len(st.session_state.trace_logs) - i}: {log.get('timestamp', 'Unknown time')} - {log.get('model_id', 'Unknown model')}"):
                if "error" in log:
                    st.error(f"Error: {log['error']}")
                    if log['request']:
                        st.subheader("Request")
                        st.json(log['request'])
                else:
                    st.subheader("Request")
                    st.json(log['request'])
                    
                    st.subheader("Response")
                    st.json(log['response'])
                    
                    st.subheader("Performance")
                    st.write(f"Elapsed time: {log['elapsed_time']} seconds")
                    st.write(f"Parameters: Max tokens = {log['parameters']['max_tokens']}, Temperature = {log['parameters']['temperature']}")
                    
                    # Calculate token usage if available in the response
                    if "anthropic" in log['model_id'] and "usage" in log['response']:
                        usage = log['response']['usage']
                        st.write(f"Input tokens: {usage.get('input_tokens', 'N/A')}")
                        st.write(f"Output tokens: {usage.get('output_tokens', 'N/A')}")
                    elif "amazon" in log['model_id'] and "amazon-bedrock-invocationMetrics" in log['response']:
                        metrics = log['response']['amazon-bedrock-invocationMetrics']
                        st.write(f"Input tokens: {metrics.get('inputTokenCount', 'N/A')}")
                        st.write(f"Output tokens: {metrics.get('outputTokenCount', 'N/A')}")

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