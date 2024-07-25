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

// Instantiate the model
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

const output = await chatHistoryChain.call({
  input: 'I like to read scriptures and I am 20 years old',
});

console.log(output);
