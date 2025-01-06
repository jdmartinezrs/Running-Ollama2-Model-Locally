import streamlit as st
from localollama2 import send_message_to_llm, initialize_session_state

def main():
    st.title("🦙 llama2 Chat")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask something..."):
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_message_to_llm(user_input)
                st.write(response)
        
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
