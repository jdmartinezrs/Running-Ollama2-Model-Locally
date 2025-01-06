import requests
import json
import streamlit as st  # Asegúrate de importar streamlit aquí

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def send_message_to_llm(user_message):
    payload = {
        "model": "llama2",
        "messages": [
            {"role": "user", "content": user_message}
        ]
    }
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response_text = response.text
        final_message = ""
        
        # Process JSON stream response
        for line in response_text.splitlines():
            try:
                json_fragment = json.loads(line)
                content = json_fragment.get('message', {}).get('content', '')
                if content.strip():
                    final_message += content.strip() + " "
            except json.JSONDecodeError:
                continue
                
        return final_message.strip()
    
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
