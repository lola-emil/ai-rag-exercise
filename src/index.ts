import OpenAI from 'openai';
import readline from 'readline';
import pg from 'pg';
import { config } from 'dotenv';
import { formatVectorForPgvector } from './utils';
import { ChatCompletionMessageParam } from 'openai/resources';

interface Document {
    id: number;
    text: string;

    embedding: number[];
    source_file: string;
    chunk_index: number;

    total_chunks?: number | null;
    file_size_bytes?: number | null;

    created_at: Date | string;

    metadata?: Record<string, any> | unknown[] | null;
}


config()

if (!process.env.AI_BASE_URL ||
    !process.env.AI_API_KEY ||
    !process.env.CHAT_MODEL ||
    !process.env.EMBED_MODEL
) {
    console.log(".env file might be missing")
    process.exit(1);
}

const client = new OpenAI({
    baseURL: process.env.AI_BASE_URL,
    apiKey: process.env.AI_API_KEY,
});

const rerankClient = new OpenAI({
    baseURL: "https://dashscope-intl.aliyuncs.com/compatible-api/v1",
    apiKey: process.env.AI_API_KEY,
});

const CHAT_MODEL = process.env.CHAT_MODEL;
const EMBED_MODEL = process.env.EMBED_MODEL;

const pool = new pg.Pool({
    user: process.env.DB_USER,
    host: 'localhost',
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: 5432,
});

async function rerankDocuments(query: string, documentsWithMeta: Document[], topN = 3) {
    const documents = documentsWithMeta.map(d => d.text);

    const response = await rerankClient.post<{
        results: {
            index: number,
            source_file?: string,
            chunk_index?: number,
            relevance_score: number,

        }[]
    }>("/reranks", {
        body: {
            model: "qwen3-rerank",
            query: query,
            documents: documents,
            top_n: topN,
        },
    });

    return response.results.map(result => ({
        text: documents[result.index],
        source_file: documentsWithMeta[result.index]?.source_file,
        chunk_index: documentsWithMeta[result.index]?.chunk_index,
        score: result.relevance_score,
    }));
}

async function chat() {
    let conversationHistory: ChatCompletionMessageParam[] = [];

    console.log('RAG Chat ready. Type your question (or "exit").\n');

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const ask = async () => {
        rl.question('You: ', async (question) => {
            if (question.toLowerCase() === 'exit') {
                rl.close();
                await pool.end();
                return;
            }

            try {
                const queryRes = await client.embeddings.create({
                    model: EMBED_MODEL,
                    input: question
                });
                const queryVector = queryRes.data[0]?.embedding;

                if (!queryVector) {
                    console.log('\n No embedding returned.\n');
                    ask();
                    return;
                }

                const vectorString = formatVectorForPgvector(queryVector);

                const result = await pool.query(
                    `SELECT text, source_file, chunk_index, 1 - (embedding <=> $1) AS similarity 
                    FROM documents 
                    ORDER BY embedding <=> $1 
                    LIMIT 10`,
                    [vectorString]
                );


                if (result.rows.length === 0) {
                    console.log('\n No documents found in database. Run `node ingest.js` first.\n');
                    ask();
                    return;
                }


                console.log('Reranking documents...');

                const reranked = await rerankDocuments(question, result.rows, 5);

                const context = reranked.map((r, i) => `
                    ${i + 1}. ${r.source_file} 
                    (
                        chunk ${r.chunk_index}, 
                        score: ${r.score.toFixed(3)}
                    ) 
                    text: ${r.text}`).join('\n\n---\n\n');

                const messages: ChatCompletionMessageParam[] = [
                    {
                        role: 'system',
                        content: `
                            You are a helpful, knowledgeable assistant. Use the provided context as your primary source of information, but feel free to supplement it with your general knowledge when it adds value.

                            Guidelines:
                            - Prioritize information from the context when it's directly relevant
                            - You can elaborate, explain concepts, or provide additional context from your training data
                            - If the context doesn't contain relevant information, you can still answer based on your general knowledge
                            - Be conversational and helpful, not robotic
                            - If you're unsure about something specific to the context, acknowledge that uncertainty
                            - When making inferences, distinguish between what the policy explicitly states and what is reasonably inferred. Do not invent examples, scenarios, or exceptions that are not mentioned in the retrieved context.
                            - If response was supplemented with general knowledge, add a disclaimer or a note saying that the supplemented information was not in the provided context.

                            Context:
                            ${context}
                        `
                    },
                    ...conversationHistory,
                    {
                        role: 'user',
                        content: question
                    }
                ];

                process.stdout.write('\nQwen: ');
                let fullResponse = '';

                const stream = await client.chat.completions.create({
                    model: CHAT_MODEL,
                    messages: messages,
                    stream: true,
                });

                for await (const chunk of stream) {
                    const content = chunk.choices[0]?.delta?.content || '';
                    process.stdout.write(content);
                    fullResponse += content;
                }

                conversationHistory.push({ role: 'user', content: question });
                conversationHistory.push({ role: 'assistant', content: fullResponse });

                console.log('\n');
            } catch (err: any) {
                console.log(err);
                console.error('\nError:', err.message);
            }

            ask();
        });
    };

    ask();
}

chat();