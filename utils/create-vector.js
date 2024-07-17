import { ChatOpenAI } from '@langchain/openai';
import { EPubLoader } from 'langchain/document_loaders/fs/epub';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Document } from '@langchain/core/documents';
import { OpenAIEmbeddings } from '@langchain/openai';
import { SupabaseVectorStore } from '@langchain/community/vectorstores/supabase';
import { createClient } from '@supabase/supabase-js';

import * as dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import path from 'path';

// Get the directory name of the current module file
const __dirname = dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, './.env') });

//Data load, transform and embed

//Upload the epub document
const loader = new EPubLoader('../God TalkswithArjuna.epub', {
  splitChapters: true,
});

const docs = await loader.load();

const propagateChapterMetadata = (docs) => {
  let lastDefinedChapter = null;

  docs.forEach((doc, index) => {
    if (doc.metadata && doc.metadata.chapter) {
      lastDefinedChapter = doc.metadata.chapter;
    } else if (lastDefinedChapter) {
      if (!doc.metadata) {
        doc.metadata = {};
      }
      doc.metadata.source = './God TalkswithArjuna.epub';
      doc.metadata.chapter = lastDefinedChapter;
    }

    // console.log(`Chapter ${index + 1}:`);
    // if (doc.pageContent === undefined) {
    //   console.log('Content:', doc.pageContent);
    // }
    // console.log('Metadata:', doc.metadata);
    // console.log('--------------------');
  });

  return docs;
};

const formattedDoc = propagateChapterMetadata(docs);

const removeChapters = [
  'Praise for Paramahansa Yogananda’s commentary on the Bhagavad Gita…',
  'Acknowledgments',
  'Back Cover',
  'About the Author',
  'Paramahansa Yogananda: A Yogi in Life and Death',
  'Aims and Ideals of Self-Realization Fellowship',
  'Autobiography of a Yogi',
  'Other Books by Paramahansa Yogananda',
  'Additional Resources on the Kriya Yoga Teachings of Paramahansa Yogananda',
  'Terms Associated With Self-Realization Fellowship',
  'Self-Realization Fellowship Lessons',
  'Acknowledgments',
  'Notes',
];

const parsedFilteredDoc = formattedDoc
  .filter(
    (doc) =>
      doc.pageContent !== undefined &&
      !removeChapters.includes(doc.metadata.chapter)
  )
  .map((doc) => ({
    pageContent: doc.pageContent
      .replace(/\n/g, ' ')
      .replace(/\+/g, '')
      .replace(/\[part[^\[\]]*\]/g, '')
      .replace(/\[\.{2}\/images\]/g, ''),
    metadata: doc.metadata,
  }));

// console.log(parsedFilteredDoc);

const splitReadyDoc = parsedFilteredDoc.map(
  (doc) =>
    new Document({
      pageContent: doc.pageContent,
      metadata: {
        source: doc.metadata.source,
        chapter: doc.metadata.chapter,
      },
    })
);

// console.log('Split Ready: ', splitReadyDoc);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  separators: ['\n\n', '\n', ' ', ''],
  chunkOverlap: 50,
});

const splitDoc = await splitter.splitDocuments(splitReadyDoc);
// console.log('Split Doc: ', splitDoc);

const openaiApiKey = process.env.OPENAI_API_KEY;

// Instantiate the model
const model = new ChatOpenAI({
  modelName: 'gpt-3.5-turbo',
  temperature: 0.7,
  apiKey: openaiApiKey,
});

const sbApiKey = process.env.SUPABASE_API_KEY;
const subUrl = process.env.SUPABASE_URL;

//Setup embedding and Supabase connection
const embeddings = new OpenAIEmbeddings({ openaiApiKey });
const client = createClient(subUrl, sbApiKey);

await SupabaseVectorStore.fromDocuments(splitDoc, embeddings, {
  client,
  tableName: 'documents',
});
