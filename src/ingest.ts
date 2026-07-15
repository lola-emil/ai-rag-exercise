import { readdir, stat } from "node:fs/promises";
import { resolve, extname, basename } from "node:path";
import OpenAI from 'openai';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import pg from 'pg';
import { config } from 'dotenv';
import {
    formatVectorForPgvector
} from "./utils.js";

config()

if (!process.env.AI_BASE_URL ||
    !process.env.AI_API_KEY ||
    !process.env.CHAT_MODEL ||
    !process.env.EMBED_MODEL
) {
    console.log(".env file might be missing")
    process.exit(1);
}

const docsPath = resolve(__dirname, '../docs');

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
    let files = await readdir(docsPath);
    files = files.filter(doc => extname(doc) == '.pdf');


    console.log('Chunking and Embedding text...');
    try {

        let count = 0;

        // Clear existing data for a fresh start
        await pool.query('DELETE FROM documents');
        console.log('Cleared existing documents.');

        for (const doc of files) {
            const filePath = resolve(docsPath, doc);
            const stats = await stat(filePath);
            const fileName = basename(filePath);
            const fileSize = stats.size;

            const loader = new PDFLoader(filePath);
            const documents = await loader.load();

            let index = 0;

            for (const docu of documents) {

                const response = await client.embeddings.create({
                    model: EMBED_MODEL,
                    input: docu.pageContent,
                });

                if (!response.data[0]) {
                    console.log("\n No embedding returned \n")
                    continue
                }
                const embedding = response.data[0].embedding;
                const vectorString = formatVectorForPgvector(embedding);


                await pool.query(
                    `INSERT INTO documents (text, embedding, source_file, chunk_index, total_chunks, file_size_bytes, metadata) 
                        VALUES ($1, $2, $3, $4, $5, $6, $7)`,
                    [
                        docu.pageContent,
                        vectorString,
                        fileName,
                        index,
                        documents.length,
                        fileSize,
                        JSON.stringify({
                            full_path: resolve(filePath),
                            extension: extname(filePath),
                        })
                    ]
                );

                index++;
                count++;
                console.log(`Stored chunk ${count}: "${docu.pageContent.substring(0, 50)}..."`);

            }

            index = 0;
        }

        await new Promise(resolve => setTimeout(resolve, 100));

        console.log(`\nIngestion complete! ${count} chunks stored in pgvector.`);
    } catch (err: any) {
        console.error(err);
        console.error('Error during ingestion:', err.message);
    } finally {
        await pool.end();
    }
}


ingest();
