// JavaScript for Customer Support Chatbot with enhanced UI
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    
    // Typing indicator element
    let typingIndicator = null;
    
    // Function to create typing indicator
    function showTypingIndicator() {
        if (typingIndicator) return;
        
        typingIndicator = document.createElement('div');
        typingIndicator.id = 'typing-indicator';
        typingIndicator.className = 'message bot-message';
        typingIndicator.innerHTML = `
            <div class="typing">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to remove typing indicator
    function hideTypingIndicator() {
        if (typingIndicator) {
            typingIndicator.remove();
            typingIndicator = null;
        }
    }
    
    // Function to add message to chat
    function addMessage(text, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to send message
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        addMessage(message, true);
        messageInput.value = '';
        
        // Disable input while processing
        messageInput.disabled = true;
        sendButton.disabled = true;
        
        // Show typing indicator
        showTypingIndicator();
        
        try {
            // Simulate network delay for better UX
            await new Promise(resolve => setTimeout(resolve, 800));
            
            // Send message to server
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });
            
            if (response.ok) {
                const data = await response.json();
                // Remove typing indicator before adding response
                hideTypingIndicator();
                addMessage(data.response);
            } else {
                hideTypingIndicator();
                addMessage('Sorry, I encountered an error. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            hideTypingIndicator();
            addMessage('Sorry, I encountered an error. Please try again.');
        } finally {
            // Re-enable input
            messageInput.disabled = false;
            sendButton.disabled = false;
            messageInput.focus();
        }
    }
    
    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Focus input on load
    messageInput.focus();
    
    // Initialize chat with welcome message after a brief delay
    setTimeout(() => {
        addMessage('Hello! How can I help you today?');
    }, 500);
});

// Add CSS for typing indicator
const style = document.createElement('style');
style.textContent = `
    .typing {
        display: flex;
        align-items: center;
        height: 20px;
    }
    
    .typing span {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        background-color: rgba(255, 255, 255, 0.7);
        margin: 0 2px;
        display: inline-block;
        animation: typing 1.4s infinite both;
    }
    
    .typing span:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing span:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0% { transform: scale(0); }
        50% { transform: scale(1); }
        100% { transform: scale(0); }
    }
`;
document.head.appendChild(style);