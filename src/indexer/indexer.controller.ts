import {
  Body,
  ConsoleLogger,
  Controller,
  Delete,
  Get,
  Param,
  Post,
  Query,
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
import { UnstructuredLoader } from 'langchain/document_loaders/fs/unstructured';
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
import * as fs from 'fs';
import * as path from 'path';
import { IndexerService } from './indexer.service';
import { formatDocumentsAsString } from 'langchain/util/document';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

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
  constructor(
    private documentParser: DocumentParserService,
    private indexerService: IndexerService,
  ) {}

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
      } else if (
        [MIME_CSV, MIME_JSON, MIME_HTML, MIME_XML, MIME_TXT].includes(mimeType)
      ) {
        parsedText = fs.readFileSync(file.path, { encoding: 'utf-8' });
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
      chunkSize: 3000,
      chunkOverlap: 1500,
    });

    const splitDocs = await splitter.splitDocuments(docs);

    return {
      parsed_text: parsedText,
      chunks: splitDocs.map((d) => d.pageContent),
    };
  }

  @Post('indexDocumentVector')
  @UseInterceptors(
    FileInterceptor('file', { dest: '.uploads/', preservePath: true }),
  )
  async parseDocumentUnstructured(
    @UploadedFile() file: Express.Multer.File,
    @Body('file_id') fileId: number,
    @Body('board_id') boardId: number | null = null,
    @Body('user_id') userId: number | null = null,
  ) {
    const targetPath = path.join(file.path, '..', file.originalname);
    // const targetFileName = file.path.substring(0, 5) + '-' + file.originalname;
    fs.renameSync(file.path, targetPath);
    const loader = new UnstructuredLoader(targetPath, {
      apiUrl: process.env.UNSTRUCTURED_URL,
      strategy: 'fast',
      xmlKeepTags: true,
    });

    let docs = await loader.load();
    docs = docs.map((d) => {
      // inject file id into the metadata
      if (!d.metadata) {
        d.metadata = {};
      }
      d.metadata.file_id = fileId;
      d.metadata.user_id = [userId];

      return d;
    });

    // const indexProcesses = [
    //   this.indexerService.createVectorStore('artifacts', docs),
    // ];

    // if (boardId) {
    //   indexProcesses.push(
    //     this.indexerService.createVectorStore(`board_${boardId}`, docs),
    //   );
    // }
    // if (userId) {
    //   indexProcesses.push(
    //     this.indexerService.createVectorStore(`user_${userId}`, docs),
    //   );
    // }

    // await Promise.allSettled(indexProcesses);
    await this.indexerService.createVectorStore('artifacts', docs);

    return { parsed_text: formatDocumentsAsString(docs), docs };
  }

  @Post('extractContext')
  async extractContext(
    @Body('text') text: string,
    @Body('source_filename') sourceFilename: string,
    @Body('uploader') uploader: string,
  ) {
    const prompt = ChatPromptTemplate.fromMessages([
      ['ai', documentIndexerSystemPrompt],
      ['human', `source file name: ${sourceFilename}`],
      ['human', `uploaded by: ${uploader || 'N/A'}`],
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

  @Delete('removeIndex/:file_id')
  async removeIndex(
    @Param('file_id') fileId: number,
    @Query('board_id') boardId: string | null = null,
    @Query('user_id') userId: string | null = null,
    @Req() req,
  ) {
    const indexProcesses = [];
    if (boardId) {
      indexProcesses.push(
        this.indexerService.removeFileIndex(`board_${boardId}`, fileId),
      );
    }
    if (userId) {
      indexProcesses.push(
        this.indexerService.removeFileIndex(`user_${userId}`, fileId),
      );
    }

    await Promise.allSettled(indexProcesses);

    return {
      message: 'OK',
    };
  }

  @Get('search')
  async testSearch(@Query('q') searchQuery: string) {
    const searchResult = await this.indexerService.searchFile(
      'artifacts',
      searchQuery,
    );

    return searchResult;
  }
}
