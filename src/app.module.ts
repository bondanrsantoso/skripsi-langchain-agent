import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { IndexerModule } from './indexer/indexer.module';
import { ConfigModule } from '@nestjs/config';
import { DocumentParserService } from './document-parser/document-parser.service';
import { ChatModule } from './chat/chat.module';

@Module({
  imports: [IndexerModule, ConfigModule.forRoot(), ChatModule],
  controllers: [AppController],
  providers: [AppService, DocumentParserService],
})
export class AppModule {}
