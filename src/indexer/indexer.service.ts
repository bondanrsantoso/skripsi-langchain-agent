import { OpenAIEmbeddings } from '@langchain/openai';
import { Injectable } from '@nestjs/common';
import { Document } from 'langchain/document';
import { Milvus } from '@langchain/community/vectorstores/milvus';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';

@Injectable()
export class IndexerService {
  async removeFileIndex(collection: string, fileId: number | string) {
    const milvusURL = new URL(process.env.MILVUS_URL);
    const milvusAddress = `${milvusURL.host}`;

    const milvus = new MilvusClient({ address: milvusAddress });
    await milvus.loadCollection({
      collection_name: collection,
    });

    const items = await milvus.query({
      collection_name: collection,
      filter: `file_id == '${fileId}'`,
    });

    await milvus.delete({
      collection_name: collection,
      ids: items.data.map((i) => i['langchain_primaryid']),
    });
  }

  async createVectorStore(
    collection: string,
    documents: Document<Record<string, any>>[],
  ) {
    const milvusURL = new URL(process.env.MILVUS_URL);
    const milvusAddress = `${milvusURL.host}`;

    const milvus = new MilvusClient({ address: milvusAddress });

    const collectionExists = await milvus.hasCollection({
      collection_name: collection,
    });

    // if(!collectionExists){
    //   milvus.createCollection(collection)
    // }

    const openAIEmbeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const sanitizedDocuments = documents
      .filter((d) => typeof d.pageContent !== 'undefined')
      .map((d, i) => {
        d.metadata = {
          file_id: d.metadata?.file_id,
          filename: d.metadata?.filename,
          page_number: d.metadata?.page_number || 0,
          filetype: d.metadata?.filetype,
          category: d.metadata?.category || 'unknown',
          // text_as_html: d.metadata?.text_as_html || '',
        };

        if (i === 0 && !collectionExists.value) {
          d.metadata.category = d.metadata.category.padEnd(1024, '\n');
        }

        return d;
      });
    return await Milvus.fromDocuments(sanitizedDocuments, openAIEmbeddings, {
      collectionName: collection,
      textFieldMaxLength: 4096,
    });
  }
}
