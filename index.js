import OpenAI from 'openai';
import readline from 'readline';
import pg from 'pg';
import { config } from 'dotenv'
import { formatVectorForPgvector } from './utils.js';

config()

const client = new OpenAI({
    baseURL: process.env.AI_BASE_URL,
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

async function chat() {
    let conversationHistory = [];

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
                const queryVector = queryRes.data[0].embedding;
                const vectorString = formatVectorForPgvector(queryVector);

                const result = await pool.query(
                    `SELECT text, 1 - (embedding <=> $1) AS similarity 
                     FROM documents 
                     ORDER BY embedding <=> $1 
                     LIMIT 3`,
                    [vectorString]
                );

                if (result.rows.length === 0) {
                    console.log('\n No documents found in database. Run `node ingest.js` first.\n');
                    ask();
                    return;
                }

                const context = result.rows.map(r => r.text).join('\n\n');

                const messages = [
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
                            - If response was supplemented with general knowledge, add a disclaimer or a note.

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
            } catch (err) {
                console.error('\n❌ Error:', err.message);
            }

            ask();
        });
    };

    ask();
}

chat();