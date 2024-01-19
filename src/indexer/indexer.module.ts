import { Module } from '@nestjs/common';
import { IndexerController } from './indexer.controller';
import { DocumentParserService } from 'src/document-parser/document-parser.service';

@Module({
  controllers: [IndexerController],
  providers: [DocumentParserService],
})
export class IndexerModule {}
