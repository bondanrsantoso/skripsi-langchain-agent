import { Body, Controller, Logger, Post } from '@nestjs/common';
import { ChatService } from './chat.service';
// import { ChatPromptTemplate } from '@langchain/core/prompts';
import { AgentExecutor, createStructuredChatAgent } from 'langchain/agents';
import { ChatOpenAI } from '@langchain/openai';
import { pull } from 'langchain/hub';
import type { ChatPromptTemplate } from '@langchain/core/prompts';
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from '@langchain/core/messages';

enum Roles {
  ROLE_HUMAN = 'human',
  ROLE_AI = 'ai',
  ROLE_SYSTEM = 'system',
}

interface ChatMessage {
  content: string;
  role: Roles;
}

const aiPromptTemplate = `
You are a helpful agent who can asnwer question by looking up context and notes 
using tools such as "context-search-tool" and "context-search-tool". You may only
use those tools at most 5 times
`;

const logger = new Logger();

@Controller('chat')
export class ChatController {
  constructor(private chatService: ChatService) {}

  @Post('getAnswer')
  async getAnswer(
    @Body('prompt') prompt: string,
    @Body('history') history: ChatMessage[],
    @Body('board_id') boardId: number | null,
    @Body('user_id') userId: number | null,
  ) {
    // const promptTemplate = ChatPromptTemplate.fromMessages([
    //   ['system', aiPromptTemplate],
    //   ['system', '{tools}'],
    //   ['human', '{input}'],
    //   ['ai', '{agent_scratchpad}'],
    // ]);

    const promptTemplate = await pull<ChatPromptTemplate>(
      'hwchase17/structured-chat-agent',
    );

    // promptTemplate.promptMessages.push(
    //   new SystemMessage(
    //     'Instruction override to at least look up relevant information from the database at least once',
    //   ),
    // );

    promptTemplate.promptMessages.push(
      new SystemMessage(
        'Your answer should be preferably in Bahasa Indonesia. English is only used as fallback.' +
          'Always use document-search-tool to look up references before coming up with your answer.' +
          'When using any references, cite the filename and page number it belongs to.' +
          'When answering wthout using any tool, put a disclaimer after the answer.' +
          'important: always format the final answer in HTML format.' +
          'important: All tools are only available to be used for 5 times.',
      ),
    );

    const tools = [
      await this.chatService.vectorDirectSearchTool(
        'artifacts',
        'document-search-tool',
      ),
      // this.chatService.noteSearchTool(boardId),
    ];

    // if (boardId) {
    //   tools.push(
    //     await this.chatService.vectorDirectSearchTool(
    //       `board_${boardId}`,
    //       'context-document-search-tool',
    //     ),
    //   );
    // }
    // if (userId) {
    //   tools.push(
    //     await this.chatService.vectorDirectSearchTool(
    //       `user_${userId}`,
    //       'user-document-search-tool',
    //     ),
    //   );
    // }

    const model = new ChatOpenAI({
      temperature: 0,
      maxConcurrency: 5,
      modelName: process.env.LLM_MODEL || 'gpt-3.5-turbo-1106',
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const agent = await createStructuredChatAgent({
      llm: model,
      tools,
      prompt: promptTemplate,
    });

    const agentExecutor = new AgentExecutor({
      agent,
      tools,
      returnIntermediateSteps: false,
      maxIterations: 20,
    });

    const result = await agentExecutor.invoke(
      {
        input: prompt,
        chat_history: history.map((h) =>
          h.role === Roles.ROLE_AI
            ? new AIMessage(h.content)
            : h.role === Roles.ROLE_SYSTEM
            ? new SystemMessage(h.content)
            : new HumanMessage(h.content),
        ),
      },
      {
        callbacks: [
          {
            handleAgentAction(action, runId) {
              logger.log('handleAgentAction', action, runId);
            },
            handleAgentEnd(action, runId) {
              logger.log('handleAgentEnd', action, runId);
            },
            handleToolEnd(output, runId) {
              logger.log('handleToolEnd', output, runId);
            },
          },
        ],
      },
    );

    // await this.chatService.releaseCollection('artifacts');
    return result;
  }
}
