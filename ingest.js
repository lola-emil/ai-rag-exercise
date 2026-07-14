import { readFile, readdir } from "node:fs/promises"
import { join, resolve } from "node:path"
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
    const docs = await readdir(docsPath);

    console.log('Chunking and Embedding text...');

    let count = 0;

    try {
        // Clear existing data for a fresh start
        await pool.query('DELETE FROM documents');
        console.log('Cleared existing documents.');

        for (const doc of docs) {
            const buffer = await readFile(resolve(docsPath, doc));
            const pdf = new PDFParse({ data: buffer });

            const result = await pdf.getText();
            const text = result.text;
            const chunks = chunkBySection(text);

            for (const chunk of chunks) {
                if (chunk.trim().length < 10) continue;

                const response = await client.embeddings.create({
                    model: EMBED_MODEL,
                    input: chunk,
                });

                const embedding = response.data[0].embedding;
                const vectorString = formatVectorForPgvector(embedding);
                await pool.query(
                    'INSERT INTO documents (text, embedding) VALUES ($1, $2)',
                    [chunk, vectorString]
                );

                count++;
                console.log(`Stored chunk ${count}: "${chunk.substring(0, 50)}..."`);

            }
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
