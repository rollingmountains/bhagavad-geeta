import { ChatOpenAI } from '@langchain/openai';
import { StringOutputParser } from '@langchain/core/output_parsers';
import {
  RunnableSequence,
  RunnablePassthrough,
} from '@langchain/core/runnables';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { retriever } from './utils/retriever.js';
import { combineDocuments } from './utils/combine-retrieved-doc.js';
import { BufferMemory } from 'langchain/memory';
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis';
import { ConversationChain } from 'langchain/chains';
import * as dotenv from 'dotenv';

dotenv.config();

const openaiApiKey = process.env.OPENAI_API_KEY;
const upstashUrl = process.env.UPSTASH_URL;
const upstashToken = process.env.UPSTASH_REDIS_TOKEN;

// Create standalone question prompt template
const standaloneQuestionTemplate = `Given a question, convert it into a standalone question.
question: {input} 
standalone question:`;

const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(
  standaloneQuestionTemplate
);

// Create answer prompt
const answerTemplate = `You are an expert answering complex questions in simple terms so a layman can understand. Use the context, which also includes conversation history, to synthesize the answer. If you don't know the answer, say 'I am sorry. I do not know the answer.' Do not make up the answer.
context: {context}
chat_history: {chat_history}
question: {input}
answer:`;

const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate);

// Initialize the model
const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
  apiKey: openaiApiKey,
});

const parser = new StringOutputParser();

const memory = new BufferMemory({
  chatHistory: new UpstashRedisChatMessageHistory({
    sessionId: '123',
    config: {
      url: upstashUrl,
      token: upstashToken,
    },
  }),
});

const chatHistoryChain = new ConversationChain({
  llm: model,
  memory,
});

const standAloneQuestionChain = RunnableSequence.from([
  standaloneQuestionPrompt,
  model,
  parser,
]);

const retrieverChain = RunnableSequence.from([
  (prevResult) => prevResult.standalone_question,
  retriever,
  combineDocuments,
]);

const answerChain = RunnableSequence.from([answerPrompt, model, parser]);

const chain = RunnableSequence.from([
  {
    standalone_question: standAloneQuestionChain,
    original_input: new RunnablePassthrough(),
  },
  {
    context: retrieverChain,
    input: ({ original_input }) => original_input.input,
    chat_history: async () => {
      const history = await memory.loadMemoryVariables();
      return history.chat_history || '';
    },
  },
  async ({ context, input, chat_history }) => {
    console.log('Context:', context);
    console.log('Input:', input);
    console.log('Chat History:', chat_history);
    return { context, input, chat_history };
  },
  answerChain,
  async ({ input, context, chat_history }) => {
    const chatResponse = await chatHistoryChain.invoke({
      input: ({ original_input }) => original_input.input,
    });

    console.log('Chat History Chain Response:', chatResponse);

    // Update chat history
    const updatedChatHistory = chat_history + '\n' + chatResponse.response;

    await memory.chatHistory.add({
      type: 'human',
      data: { content: input },
    });

    await memory.chatHistory.add({
      type: 'ai',
      data: { content: chatResponse.response },
    });

    return {
      input,
      context,
      chat_history: updatedChatHistory,
    };
  },
  async () => {
    const updatedMemory = await memory.loadMemoryVariables();
    console.log('Updated Memory Variables:', updatedMemory);
    return updatedMemory;
  },
]);

const response = await chain.invoke({
  input: 'who is krishna?',
});
console.log(response);
