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

// export function combineDocuments(docs) {
//   return docs.map((doc) => doc.pageContent).join('\n\n');
// }

//create standaloneprompt template
const standaloneQuestionTemplate = `Given a question, convert it into a standalone question.
question: {question} standalone question:`;

//Standalone prompt
const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(
  standaloneQuestionTemplate
);

//Create Answer prompt
const answerTemplate = `You are an expert answering complex in simple terms so as a layman can understand. Use the context and synthesis the answer. If you really don't know the answer just say 'I am sorry. I do not know the answer.' Do not make up the answer.
context: {context}
question: {question}
answer:  `;

const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate);

function combineRetrieverDoc(docs) {
  return docs.map((doc) => doc.pageContent).join('\n');
}

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

const chain = () =>
  RunnableSequence.from([
    chatHistoryChain,
    {
      standalone_question: standAloneQuestionChain,
      original_input: new RunnablePassthrough(),
    },
    {
      context: retrieverChain,
      question: ({ original_input }) => original_input.question,
    },
    answerChain,
  ]);

export const initializeLangChain = chain();
