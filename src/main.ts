import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ConsoleLogger } from '@nestjs/common';

const logger = new ConsoleLogger();

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  let port = ((process.env.PORT as any) || 6996) - 0;

  let isListening = false;

  while (!isListening) {
    try {
      await app.listen(port);
      logger.log('Listening on port ' + port);
      logger.log('http://localhost:' + port);
      isListening = true;
    } catch (error) {
      logger.error(`Failed listening on port ${port}`);
      port++;
    }
  }
}
bootstrap();
