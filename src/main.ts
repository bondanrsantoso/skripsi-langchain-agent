import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import 'dotenv/config';
import { ConsoleLogger } from '@nestjs/common';

const logger = new ConsoleLogger();

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  const port = process.env.PORT || 6996;

  await app.listen(port);
  logger.log('Listening on port ' + port);
  logger.log('http://localhost:' + port);
}
bootstrap();
