import { OpenAIEmbeddings } from '@langchain/openai';
import { Injectable } from '@nestjs/common';
import { Document } from 'langchain/document';
import { Milvus } from '@langchain/community/vectorstores/milvus';
import { DataType, MilvusClient } from '@zilliz/milvus2-sdk-node';

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

  async searchFile(collection: string, query: string) {
    const milvusURL = new URL(process.env.MILVUS_URL);
    const milvusAddress = `${milvusURL.host}`;

    const milvus = new MilvusClient({ address: milvusAddress });
    await milvus.loadCollection({
      collection_name: collection,
    });

    const fileId = 13;

    // const items = await milvus.query({
    //   collection_name: collection,
    //   filter: `file_id == '${fileId}'`,
    // });

    const openAIEmbeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: 'text-embedding-3-large',
    });

    const vector = await openAIEmbeddings.embedQuery(query);

    const includedCategories = ['NarrativeText', 'ListItem'];

    const searchResult = await milvus.search({
      collection_name: collection,
      vector,
      filter: `file_id == ${fileId} && category in ${JSON.stringify(
        includedCategories,
      )}`,
      limit: 10,
    });

    await milvus.releaseCollection({
      collection_name: collection,
    });

    return searchResult;
  }

  async createVectorStore(
    collection: string,
    documents: Document<Record<string, any>>[],
  ) {
    const milvusURL = new URL(process.env.MILVUS_URL);
    const milvusAddress = `${milvusURL.host}`;

    const milvus = new MilvusClient({ address: milvusAddress });

    const { value: collectionExists } = await milvus.hasCollection({
      collection_name: collection,
    });

    if (!collectionExists) {
      await milvus.createCollection({
        collection_name: collection,
        fields: [
          {
            name: 'langchain_primaryid',
            data_type: DataType.Int64,
            autoID: true,
            is_primary_key: true,
          },
          {
            name: 'langchain_text',
            data_type: DataType.VarChar,
            max_length: 65535,
          },
          {
            name: 'langchain_vector',
            data_type: DataType.FloatVector,
            dim: 3072,
          },
          {
            name: 'file_id',
            data_type: DataType.Int64,
          },
          {
            name: 'filename',
            data_type: DataType.VarChar,
            max_length: 512,
          },
          {
            name: 'page_number',
            data_type: DataType.Int32,
          },
          {
            name: 'filetype',
            data_type: DataType.VarChar,
            max_length: 128,
          },
          {
            name: 'category',
            data_type: DataType.VarChar,
            max_length: 32,
          },
          {
            name: 'user_id',
            data_type: DataType.Array,
            element_type: DataType.Int64,
            max_capacity: 255,
          },
        ],
        enable_dynamic_field: true,
      });

      await milvus.createIndex({
        collection_name: collection,
        index_name: 'category_index',
        field_name: 'category',
      });

      await milvus.createIndex({
        collection_name: collection,
        index_name: 'langchain_vector',
        field_name: 'langchain_vector',
        extra_params: {
          index_type: 'HNSW',
          metric_type: 'L2',
          params: '{"M":"8","efConstruction":"64"}',
        },
      });

      await milvus.createIndex({
        collection_name: collection,
        index_name: 'file_id',
        field_name: 'file_id',
      });
    }

    const openAIEmbeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
      modelName: 'text-embedding-3-large',
    });

    const sanitizedDocuments = documents.filter(
      (d) => typeof d.pageContent !== 'undefined',
    );

    const documentContent = sanitizedDocuments.map((d) => d.pageContent);

    const embeddings = await openAIEmbeddings.embedDocuments(documentContent);

    const milvusCompatibleData = sanitizedDocuments.map((d, i) => ({
      ...d.metadata,
      langchain_text: d.pageContent,
      langchain_vector: embeddings[i],
    }));

    return milvus.insert({
      collection_name: collection,
      data: milvusCompatibleData,
    });
  }
}
