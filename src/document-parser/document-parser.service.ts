import { Injectable } from '@nestjs/common';
import { readFileSync } from 'fs';
import { convertToHtml as mammothConvertToHtml } from 'mammoth';
import { parseOfficeAsync } from 'officeparser';
import pdf from 'pdf-parse';

@Injectable()
export class DocumentParserService {
  /**
   * Extract text form a word document, converted to HTML for better formatting
   * @param filePath Path to the word (.docx) file
   * @returns Parsed word document now in HTML format
   */
  async parseWord(filePath: string): Promise<string> {
    const converted = await mammothConvertToHtml({
      path: filePath,
    });

    return converted.value;
  }

  // async parsePdf(filePath: string) {
  //   const fileBuffer: Buffer = readFileSync(filePath);

  //   const parsed = await pdf(fileBuffer);

  //   return parsed.text;
  // }

  /**
   * Parse almost any common document format (docx, odt, pptx, odp, pdf)
   * @param filePath Path to the document
   * @returns Extracted text
   */
  async parseDocument(filePath: string): Promise<string> {
    const parsedDocument = await parseOfficeAsync(filePath);

    return parsedDocument;
  }
}
