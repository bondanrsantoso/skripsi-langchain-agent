import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { IndexerModule } from './indexer/indexer.module';
import { ConfigModule } from '@nestjs/config';
import { DocumentParserService } from './document-parser/document-parser.service';

@Module({
  imports: [IndexerModule, ConfigModule.forRoot()],
  controllers: [AppController],
  providers: [AppService, DocumentParserService],
})
export class AppModule {}
