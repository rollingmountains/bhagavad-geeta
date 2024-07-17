import { OpenAIEmbeddings } from '@langchain/openai';
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase';
import { createClient } from '@supabase/supabase-js';

import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import path from 'path';

// Get the directory name of the current module file
const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const openaiApiKey = process.env.OPENAI_API_KEY;
const sbApiKey = process.env.SUPABASE_API_KEY;
const subUrl = process.env.SUPABASE_URL;

console.log('openapi: ', openaiApiKey);
console.log('supaapi: ', sbApiKey);
console.log('supaurl: ', subUrl);

//Setup embedding and Supabase connection
const embeddings = new OpenAIEmbeddings({ openaiApiKey });
const client = createClient(subUrl, sbApiKey);

//Instantiate the vectorstore
const vectorStore = new SupabaseVectorStore(embeddings, {
  client,
  tableName: 'documents',
  queryName: 'match_documents',
});

//Create retriever
const retriever = vectorStore.asRetriever({ k: 2 });

export { retriever };
