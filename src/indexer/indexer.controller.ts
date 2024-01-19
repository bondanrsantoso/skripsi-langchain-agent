import {
  Body,
  ConsoleLogger,
  Controller,
  Get,
  Post,
  Req,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Request, response } from 'express';
import {
  MIME_CSV,
  MIME_DOCX,
  MIME_HTML,
  MIME_JSON,
  MIME_PDF,
  MIME_PPT,
  MIME_PPTX,
  MIME_TXT,
  MIME_XLS,
  MIME_XML,
} from 'src/constants/filetype';
import { DocumentParserService } from 'src/document-parser/document-parser.service';

import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { Document } from 'langchain/document';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { DocxLoader } from 'langchain/document_loaders/fs/docx';
import { CSVLoader } from 'langchain/document_loaders/fs/csv';
import { JSONLoader } from 'langchain/document_loaders/fs/json';
import { PPTXLoader } from 'langchain/document_loaders/fs/pptx';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { DocumentLoader } from 'langchain/dist/document_loaders/base';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { JsonOutputParser } from '@langchain/core/output_parsers';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import {
  RunnableMap,
  RunnableLambda,
  RunnablePassthrough,
} from '@langchain/core/runnables';
import { VectorStoreRetriever } from '@langchain/core/vectorstores';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { AgentExecutor, createOpenAIFunctionsAgent } from 'langchain/agents';
import { createRetrieverTool } from 'langchain/tools/retriever';

const logger = new ConsoleLogger();

const documentIndexerSystemPrompt = `
Extract contextual information from this following document and format it as 
a JSON array and nothing else. The content of the 'field' and 'value' are 
preferably presented in Bahasa Indonesia, with English as fallback. for example:

[
  {{
    "field": "Ringkasan Dokumen",
    "value": "Secara umum dokumen ini membahas tentang proyek produksi video senilai Rp2 Miliar"
  }},
  {{
    "field": "Anggaran",
    "value": "Rp2 Miliar"
  }},
  {{
    "field": "Stakeholder",
    "value": "PT Maju Jaya Bersama"
  }}
]

You, as a language model are free to extract as much of any kind contextual information deemed 
valuable as possible, regardless of the example fields mentioned above.

Here are the documents: {documents}
`;

const documentIndexerAgentPrompt = `
you are a helpful agent who will answer in JSON
`;

@Controller('indexer')
export class IndexerController {
  constructor(private documentParser: DocumentParserService) {}

  @Get()
  getIndexer(@Req() req: Request) {
    logger.debug(req.headers);

    return { foo: 'bar' };
  }

  @Post('splitDocs')
  @UseInterceptors(FileInterceptor('file', { dest: '.uploads/' }))
  async parseDocument(
    @UploadedFile() file: Express.Multer.File,
    @Body('mimetype') mime: string,
  ) {
    let parsedText: string;
    const mimeType = file?.mimetype.trim();

    // Initialize default text (which is empty string)
    parsedText = '';

    try {
      if (mimeType === MIME_DOCX) {
        parsedText = await this.documentParser.parseWord(file.path);
      } else if (mimeType === MIME_PPTX || mimeType === MIME_PDF) {
        parsedText = await this.documentParser.parseDocument(file.path);
      }
    } catch (error) {
      logger.error('Failed parsing document', error);
    }

    const indexedMimes = [
      MIME_DOCX,
      MIME_PDF,
      MIME_CSV,
      MIME_JSON,
      MIME_PPTX,
      MIME_HTML,
      MIME_XML,
      MIME_TXT,
    ];

    if (!indexedMimes.includes(mimeType)) {
      return {
        parsed_text: parsedText,
        llm_response: [],
      };
    }

    let docs: Document<Record<string, any>>[] | null = null;
    let loader: DocumentLoader | null = null;

    if (mimeType === MIME_PDF) {
      loader = new PDFLoader(file.path, { splitPages: true });
    } else if (mimeType === MIME_DOCX) {
      loader = new DocxLoader(file.path);
    } else if (mimeType === MIME_CSV) {
      loader = new CSVLoader(file.path);
    } else if (mimeType === MIME_JSON) {
      loader = new JSONLoader(file.path);
    } else if (mimeType === MIME_PPTX) {
      loader = new PPTXLoader(file.path);
    } else {
      loader = new TextLoader(file.path);
    }

    docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splitDocs = await splitter.splitDocuments(docs);

    return {
      parsed_text: parsedText,
      chunks: splitDocs.map((d) => d.pageContent),
    };
  }

  @Post('extractContext')
  async extractContext(@Body('text') text: string) {
    const prompt = ChatPromptTemplate.fromMessages([
      ['ai', documentIndexerSystemPrompt],
      ['human', '{instruction}'],
    ]);

    // initialize model
    const model = new ChatOpenAI({
      temperature: 0,
      maxConcurrency: 5,
      modelName: process.env.LLM_MODEL || 'gpt-3.5-turbo-1106',
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const setupAndRetrieval = RunnableMap.from({
      documents: new RunnableLambda({
        func: (input: string) => Promise.resolve(text),
      }),
      instruction: new RunnablePassthrough(),
    });

    const chain = setupAndRetrieval
      .pipe(prompt)
      .pipe(model)
      .pipe(new JsonOutputParser());

    const llmResponse = await chain.invoke(
      'Ekstrak informasi dari seluruh dokumen yang diberikan',
    );

    return llmResponse;
  }

  @Post('indexDocument')
  @UseInterceptors(FileInterceptor('file', { dest: '.uploads/' }))
  async indexDocument(@UploadedFile() file: Express.Multer.File) {
    let parsedText: string;
    const mimeType = file.mimetype.trim();

    // Initialize default text (which is empty string)
    parsedText = '';

    try {
      if (mimeType === MIME_DOCX) {
        parsedText = await this.documentParser.parseWord(file.path);
      } else if (mimeType === MIME_PPTX || mimeType === MIME_PDF) {
        parsedText = await this.documentParser.parseDocument(file.path);
      }
    } catch (error) {
      logger.error('Failed parsing document', error);
    }

    const indexedMimes = [
      MIME_DOCX,
      MIME_PDF,
      MIME_CSV,
      MIME_JSON,
      MIME_PPTX,
      MIME_HTML,
      MIME_XML,
      MIME_TXT,
    ];

    if (!indexedMimes.includes(mimeType)) {
      return {
        parsed_text: parsedText,
        llm_response: [],
      };
    }

    let docs: Document<Record<string, any>>[] | null = null;
    let loader: DocumentLoader | null = null;

    if (mimeType === MIME_PDF) {
      loader = new PDFLoader(file.path, { splitPages: true });
    } else if (mimeType === MIME_DOCX) {
      loader = new DocxLoader(file.path);
    } else if (mimeType === MIME_CSV) {
      loader = new CSVLoader(file.path);
    } else if (mimeType === MIME_JSON) {
      loader = new JSONLoader(file.path);
    } else if (mimeType === MIME_PPTX) {
      loader = new PPTXLoader(file.path);
    } else {
      loader = new TextLoader(file.path);
    }

    docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splitDocs = await splitter.splitDocuments(docs);

    // let retriever: VectorStoreRetriever = undefined;

    // try {
    //   const vectorstores = await HNSWLib.fromDocuments(
    //     splitDocs,
    //     new OpenAIEmbeddings({
    //       maxConcurrency: 10,
    //       openAIApiKey: process.env.OPENAI_API_KEY,
    //     }),
    //   );

    //   retriever = vectorstores.asRetriever(docs.length);
    // } catch (error) {
    //   logger.error('Failed to process document embedding', error);
    // }
    const prompt = ChatPromptTemplate.fromMessages([
      ['ai', documentIndexerSystemPrompt],
      ['human', '{instruction}'],
    ]);

    // initialize model
    const model = new ChatOpenAI({
      temperature: 0,
      maxConcurrency: 5,
      modelName: process.env.LLM_MODEL || 'gpt-3.5-turbo-1106',
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    let responses = [];

    // const max = 3;
    // let i = 0;
    for (const doc of splitDocs) {
      // if (i++ > max) break;
      try {
        const setupAndRetrieval = RunnableMap.from({
          documents: new RunnableLambda({
            func: (input: string) => Promise.resolve(doc.pageContent),
          }),
          instruction: new RunnablePassthrough(),
        });

        const chain = setupAndRetrieval
          .pipe(prompt)
          .pipe(model)
          .pipe(new JsonOutputParser());

        const llmResponse = await chain.invoke(
          'Ekstrak informasi dari seluruh dokumen yang diberikan',
        );

        responses = responses.concat(llmResponse);
      } catch (error) {
        logger.error('Failed to index document content', error);
      }
    }

    return {
      parsed_text: parsedText,
      llm_response: responses,
    };
  }

  @Post('indexDocument_v2')
  @UseInterceptors(FileInterceptor('file', { dest: 'uploads/' }))
  async indexDocumentV2(@UploadedFile() file: Express.Multer.File) {
    let parsedText: string;
    const mimeType = file.mimetype.trim();

    // Initialize default text (which is empty string)
    parsedText = '';

    try {
      if (mimeType === MIME_DOCX) {
        parsedText = await this.documentParser.parseWord(file.path);
      } else if (mimeType === MIME_PPTX || mimeType === MIME_PDF) {
        parsedText = await this.documentParser.parseDocument(file.path);
      }
    } catch (error) {
      logger.error('Failed parsing document', error);
    }

    const indexedMimes = [
      MIME_DOCX,
      MIME_PDF,
      MIME_CSV,
      MIME_JSON,
      MIME_PPTX,
      MIME_HTML,
      MIME_XML,
      MIME_TXT,
    ];

    if (!indexedMimes.includes(mimeType)) {
      return {
        parsed_text: parsedText,
        llm_response: [],
      };
    }

    let docs: Document<Record<string, any>>[] | null = null;
    let loader: DocumentLoader | null = null;

    if (mimeType === MIME_PDF) {
      loader = new PDFLoader(file.path, { splitPages: true });
    } else if (mimeType === MIME_DOCX) {
      loader = new DocxLoader(file.path);
    } else if (mimeType === MIME_CSV) {
      loader = new CSVLoader(file.path);
    } else if (mimeType === MIME_JSON) {
      loader = new JSONLoader(file.path);
    } else if (mimeType === MIME_PPTX) {
      loader = new PPTXLoader(file.path);
    } else {
      loader = new TextLoader(file.path);
    }

    docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splitDocs = await splitter.splitDocuments(docs);

    const vectorstores = await HNSWLib.fromDocuments(
      splitDocs,
      new OpenAIEmbeddings({
        maxConcurrency: 10,
        openAIApiKey: process.env.OPENAI_API_KEY,
      }),
    );

    const retriever = vectorstores.asRetriever(docs.length);

    // try {
    // } catch (error) {
    //   logger.error('Failed to process document embedding', error);
    // }
    const prompt = ChatPromptTemplate.fromMessages([
      ['system', documentIndexerAgentPrompt],
      ['human', '{instruction}'],
      ['ai', '{agent_scratchpad}'],
    ]);

    // initialize model
    const model = new ChatOpenAI({
      temperature: 0,
      maxConcurrency: 5,
      modelName: process.env.LLM_MODEL || 'gpt-3.5-turbo-1106',
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const retrieverTool = createRetrieverTool(retriever, {
      name: 'document_search',
      description: 'use this tool to look up informations from documents',
    });

    const agent = await createOpenAIFunctionsAgent({
      llm: model,
      tools: [retrieverTool],
      prompt,
    });

    const agentExecutor = new AgentExecutor({
      agent,
      tools: [retrieverTool],
      returnIntermediateSteps: true,
      maxIterations: 3,
    });

    const agentResponse = await agentExecutor.invoke({
      instruction: 'Cari dan rangkum 1 dokumen saja',
    });

    logger.debug('Agent output', agentResponse);

    return {
      parsed_text: parsedText,
      llm_response: [],
    };
  }
}
