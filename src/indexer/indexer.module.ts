import { Module } from '@nestjs/common';
import { IndexerController } from './indexer.controller';
import { DocumentParserService } from 'src/document-parser/document-parser.service';
import { IndexerService } from './indexer.service';

@Module({
  controllers: [IndexerController],
  providers: [DocumentParserService, IndexerService],
})
export class IndexerModule {}
