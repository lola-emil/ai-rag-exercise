import { readFile, readdir, stat } from "node:fs/promises"
import { join, resolve, extname, basename } from "node:path"
import OpenAI from 'openai';
import { PDFParse } from "pdf-parse";
import pg from 'pg';
import { config } from 'dotenv'
import {
    formatVectorForPgvector,
    chunkText,
    chunkBySection,
} from "./utils.js"

config()

const __dirname = import.meta.dirname;
const docsPath = resolve(__dirname, './docs');

const client = new OpenAI({
    baseURL: process.env.AI_BASE_URL,
    apiKey: process.env.AI_API_KEY,
});

const EMBED_MODEL = process.env.EMBED_MODEL;

const pool = new pg.Pool({
    user: process.env.DB_USER,
    host: 'localhost',
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: 5432,
});

async function ingest() {
    let docs = await readdir(docsPath);
    docs = docs.filter(doc => extname(doc) == '.pdf');


    console.log('Chunking and Embedding text...');

    let count = 0;

    try {
        // Clear existing data for a fresh start
        await pool.query('DELETE FROM documents');
        console.log('Cleared existing documents.');

        for (const doc of docs) {
            const filePath = resolve(docsPath, doc);
            const stats = await stat(filePath);
            const fileName = basename(filePath);
            const fileSize = stats.size;

            const buffer = await readFile(filePath);
            const pdf = new PDFParse({ data: buffer });

            const result = await pdf.getText();
            const text = result.text;
            const chunks = chunkBySection(text);
            const totalChunks = chunks.length;

            let index = 0;

            for (const chunk of chunks) {
                if (chunk.trim().length < 10) continue;

                const response = await client.embeddings.create({
                    model: EMBED_MODEL,
                    input: chunk,
                });

                const embedding = response.data[0].embedding;
                const vectorString = formatVectorForPgvector(embedding);

                await pool.query(
                    `INSERT INTO documents (text, embedding, source_file, chunk_index, total_chunks, file_size_bytes, metadata) 
                        VALUES ($1, $2, $3, $4, $5, $6, $7)`,
                    [
                        chunk,
                        vectorString,
                        fileName,
                        index,
                        totalChunks,
                        fileSize,
                        JSON.stringify({
                            full_path: resolve(filePath),
                            extension: extname(filePath),
                        })
                    ]
                );

                index++;
                count++;
                console.log(`Stored chunk ${count}: "${chunk.substring(0, 50)}..."`);

            }

            index = 0;
        }

        await new Promise(resolve => setTimeout(resolve, 100));

        console.log(`\nIngestion complete! ${count} chunks stored in pgvector.`);
    } catch (err) {
        console.error(err);
        console.error('Error during ingestion:', err.message);
    } finally {
        await pool.end();
    }
}


ingest();
