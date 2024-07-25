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

// Standalone question template
const standaloneQuestionTemplate = `Given some conversation history if any and a question, convert it into a standalone question.
{chat_history}
Human: {input}
AI:`;

// Standalone prompt
const standaloneQuestionPrompt = ChatPromptTemplate.fromTemplate(
  standaloneQuestionTemplate
);

// Answer prompt
const answerTemplate = `You are an expert and helpful bot who can answer a question in based on the context provided and coversation history such that even a layman can understand. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer just say 'I am sorry. I do not know the answer.' Do not make up the answer.
context: {context}
{chat_history}
Human: {input}
AI:`;

const answerPrompt = ChatPromptTemplate.fromTemplate(answerTemplate);

// Combine retrieved documents
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

const chat_history = await memory.loadMemoryVariables();

// const chatHistoryChain = new ConversationChain({
//   llm: model,
//   memory,
// });

const standAloneQuestionChain = RunnableSequence.from([
  standaloneQuestionPrompt,
  model,
  parser,
]);

const retrieverChain = RunnableSequence.from([
  (prevResult) => prevResult.standalone_question,
  retriever,
  combineRetrieverDoc,
]);

const answerChain = RunnableSequence.from([answerPrompt, model, parser]);

const chain = () =>
  RunnableSequence.from([
    {
      standalone_question: standAloneQuestionChain,
      original_input: new RunnablePassthrough(),
    },
    {
      context: retrieverChain,
      input: ({ original_input }) => original_input.input,
      chat_history: ({ original_input }) => original_input.chat_history,
    },
    async (context) => {
      const result = await answerChain.invoke(context);
      // console.log('input: ', context.input);
      // console.log('context: ', context.context);
      // console.log('chat_history: ', context.chat_history);
      return { input: context.input, output: result };
    },
    async ({ input, output }) => {
      await memory.saveContext({ input: input }, { output: output });
      return output;
    },
  ]);

export const initializeLangChain = chain();
export const chatHistoryMemory = chat_history;
