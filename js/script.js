// Function to handle when the page is loaded
document.addEventListener('DOMContentLoaded', function () {
  const messageInput = document.getElementById('message');
  messageInput.focus(); // Set focus on the input field
});

// Function to handle sending message
async function sendMessage() {
  const messageInput = document.getElementById('message');
  const message = messageInput.value.trim();
  if (message === '') return;

  // Display the user message
  displayMessage(message, 'user');

  // Clear the input
  messageInput.value = '';

  try {
    // Send the message to the backend AI (replace 'http://localhost:3000/api/chat' with your actual backend endpoint)
    const response = await fetch('http://localhost:3000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: message }),
    });

    if (!response.ok) {
      throw new Error('Network response was not ok');
    }

    const data = await response.json();
    // console.log('data: ' + data);

    const aiMessage = data.response; // Assuming response is a string
    // console.log(aiMessage);

    // Display the AI message with appropriate class
    displayMessage(aiMessage, 'ai'); // Ensure 'ai' is passed as sender
  } catch (error) {
    console.error('Error:', error);
    displayMessage('Error: Unable to get response from the server', 'ai');
  }
}

// Function to handle Enter key press
function handleKeyDown(event) {
  if (event.key === 'Enter') {
    sendMessage();
  }
}

// Function to display messages in the chat history
function displayMessage(message, sender) {
  const chatHistory = document.getElementById('chat-history');
  const messageContainer = document.createElement('div');
  messageContainer.classList.add(
    'message-container',
    sender === 'ai' ? 'ai' : 'user'
  );
  const messageElement = document.createElement('div');
  messageElement.classList.add('message');
  messageElement.textContent = message;
  messageContainer.appendChild(messageElement);
  chatHistory.appendChild(messageContainer);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}
