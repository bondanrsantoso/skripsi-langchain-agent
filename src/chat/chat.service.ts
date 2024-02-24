import { DynamicStructuredTool, DynamicTool } from '@langchain/core/tools';
import { OpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';
import { z } from 'zod';
import { Milvus } from '@langchain/community/vectorstores/milvus';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { PromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { formatDocumentsAsString } from 'langchain/util/document';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { VectorStore } from '@langchain/core/vectorstores';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

const logger = new Logger();

@Injectable()
export class ChatService {
  contextSearchTool(boardId: number | null) {
    const searchUrl = new URL(
      boardId !== null
        ? `/boards/${boardId}/artifact_contexts`
        : '/artifact_contexts',
      process.env.SERVER_ADDRESS,
    );

    return new DynamicTool({
      name: 'database-search-tool',
      description: 'look up relevant context in the database',
      // schema: z.object({
      //   search: z.any().describe('Search term. it only accept string'),
      // }),
      func: async (search) => {
        // if (typeof search !== 'object') {
        //   search = Object.values(search)[0];
        // }
        return axios
          .get(searchUrl.href, {
            params: {
              search,
            },
          })
          .then((res) => JSON.stringify(res.data));
      },
    });
  }

  async releaseCollection(collectionName: string) {
    const milvusURL = new URL(process.env.MILVUS_URL);
    const milvusAddress = `${milvusURL.host}`;

    const milvus = new MilvusClient({ address: milvusAddress });
    return await milvus.releaseCollection({
      collection_name: collectionName,
    });
  }

  async vectorDirectSearchTool(
    collectionName: string,
    toolName = 'data-search-tool',
  ) {
    const openAIEmbeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: 'text-embedding-3-large',
    });

    const milvusURL = new URL(process.env.MILVUS_URL);
    const milvusAddress = `${milvusURL.host}`;

    const milvus = new MilvusClient({ address: milvusAddress });
    await milvus.loadCollection({
      collection_name: collectionName,
    });

    const chainTool = new DynamicTool({
      name: toolName,
      description:
        'Data Search tool, most useful to retrieve facts and data from indexed documuents',
      func: async (search) => {
        try {
          const vector = await openAIEmbeddings.embedQuery(search);

          const includedCategories = ['NarrativeText', 'ListItem'];

          const searchResult = await milvus.search({
            collection_name: collectionName,
            vector,
            filter: `category in ${JSON.stringify(includedCategories)}`,
            limit: 10,
          });

          if (searchResult.status.code !== 0) {
            logger.error('Vector search query error', searchResult);

            return "Can't seem to provide answer right now";
          }

          const formatted = searchResult.results.map((r) => ({
            content: r.langchain_text,
            metadata: {
              filename: r.filename,
              page_number: r.page_number,
            },
          }));

          return JSON.stringify(formatted);
        } catch (error) {
          logger.error('Vector store error');

          return "Can't seem to provide answer right now";
        }
      },
    });

    return chainTool;
  }

  async vectorSearchTool(
    collectionName: string,
    toolName = 'data-search-tool',
  ) {
    const model = new OpenAI({
      temperature: 0,
      maxConcurrency: 5,
      modelName: process.env.LLM_MODEL || 'gpt-3.5-turbo-1106',
      openAIApiKey: process.env.OPENAI_API_KEY,
    });
    const openAIEmbeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: 'text-embedding-3-large',
    });

    let vectorStore: VectorStore | null = null;
    try {
      vectorStore = await Milvus.fromExistingCollection(openAIEmbeddings, {
        collectionName: collectionName,
      });
    } catch (error) {
      logger.error('Unable to use Milvus vector store');
      // vectorStore = await HNSWLib.fromTexts(['no data'], {}, openAIEmbeddings);
    }

    const questionPrompt = PromptTemplate.fromTemplate(
      `Answer by saying the relevant context. If you don't know the answer, just say that you don't know, don't try to make up an answer.
      Answers are presented in Bahasa Indonesia. Only use English if Bahasa Indonesia isn't possible
      ----------------
      CONTEXT: {context}
      ----------------
      QUESTION: {question}
      ----------------
      Helpful Answer:`,
    );

    /* Create the chain */
    const chain = RunnableSequence.from([
      {
        question: (input: { question: string }) => input.question,
        context: async (input: { question: string }) => {
          if (vectorStore !== null) {
            try {
              const retriever = vectorStore.asRetriever();
              const relevantDocs = await retriever.getRelevantDocuments(
                input.question,
              );
              let serialized = formatDocumentsAsString(relevantDocs);

              const sourceFiles = relevantDocs.map(
                (d) => d.metadata?.filename || '',
              );
              const uniqueSources = [...new Set(sourceFiles)];

              serialized = serialized + '\nSources: ' + uniqueSources.join(',');
              return serialized;
            } catch (error) {
              logger.error('Vector store error');

              return "Can't seem to provide answer right now";
            }
          }

          return 'Data search tool is unavailable';
        },
      },
      questionPrompt,
      model,
      new StringOutputParser(),
    ]);

    const chainTool = new DynamicTool({
      name: toolName,
      description:
        'Data Search tool, most useful to retrieve facts and data from indexed documuents',
      func: async (search) => {
        const answer = await chain.invoke({ question: search });

        return answer;
      },
    });

    return chainTool;
  }

  noteSearchTool(boardId: number | null) {
    const searchUrl = new URL(
      boardId !== null ? `/boards/${boardId}/board_notes` : `/board_notes`,
      process.env.SERVER_ADDRESS,
    );

    return new DynamicTool({
      name: 'note-search-tool',
      description: 'look up relevant notes in the database',
      // schema: z.object({
      //   search: z.any().describe('Search term. it only accept string'),
      // }),
      func: async (search) => {
        // if (typeof search !== 'object') {
        //   search = Object.values(search)[0];
        // }
        return axios
          .get(searchUrl.href, {
            params: {
              search,
            },
          })
          .then((res) => JSON.stringify(res.data));
      },
    });
  }

  // vectorSearchTool()
}
