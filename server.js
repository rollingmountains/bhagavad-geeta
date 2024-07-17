import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { initializeLangChain } from './langchain.js';

const app = express();
const PORT = process.env.PORT || 3000;

// __dirname workaround for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const corsOptions = {
  origin: 'https://bhagavadgeeta.netlify.app/', // frontend URL
  optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));
app.use(bodyParser.json());

// Serve the static HTML file
app.use(express.static(path.join(__dirname)));

let chainInstance = initializeLangChain; // Variable to hold the LangChain sequence instance

// Endpoint to handle user messages
app.post('/api/chat', async (req, res) => {
  const userMessage = req.body.message;

  try {
    // Check if LangChain sequence is initialized
    if (!chainInstance) {
      throw new Error('LangChain sequence is not initialized.');
    }

    // Invoke the LangChain sequence with user input
    const response = await chainInstance.invoke({ question: userMessage });
    // console.log(response);

    // Send the AI message back to the frontend
    res.json({ response: response });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Something went wrong' });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
