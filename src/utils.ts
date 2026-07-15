import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

function formatVectorForPgvector(embedding: number[]) {
    return '[' + embedding.join(',') + ']';
}

function chunkText(text: string, chunkSize = 500, overlap = 50) {
    const chunks = [];
    let start = 0;
    while (start < text.length) {
        chunks.push(text.slice(start, start + chunkSize));
        start += chunkSize - overlap;
    }
    return chunks;
}

function chunkBySection(text: string) {
    return text
        .split(/(?=^[a-z]\.\s+)/gm)
        .filter(Boolean);
}

function chunkBySentences(text: string, maxSentences = 5, overlap = 1) {
    // Split on sentence-ending punctuation
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const chunks = [];

    for (let i = 0; i < sentences.length; i += maxSentences - overlap) {
        const chunk = sentences.slice(i, i + maxSentences).join(' ').trim();
        if (chunk.length > 0) chunks.push(chunk);
    }

    return chunks;
}

async function performSemanticChunking(document: string, chunkSize = 500, chunkOverlap = 100) {
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: chunkSize,
        chunkOverlap: chunkOverlap,
        separators: ["\n\n", "\n", ". ", " ", ""],
    });

    const semanticChunks = await textSplitter.splitText(document);
    console.log(`Document split into ${semanticChunks.length} semantic chunks`);

    const documents = [];
    let currentSection = "Introduction";

    const stopwords = new Set([
        'the', 'and', 'is', 'of', 'to', 'a', 'in', 'that', 'it', 'with', 'as', 'for'
    ]);

    for (let i = 0; i < semanticChunks.length; i++) {
        const chunk = semanticChunks[i];

        if (!chunk) continue;

        const chunkLines = chunk.split('\n');

        for (const line of chunkLines) {
            if (/^#+\s+(.+)$/.test(line)) {
                currentSection = line;
                break;
            }
            if (/^[A-Z\s]+:$/.test(line)) {
                currentSection = line;
                break;
            }
        }

        const underlinedMatch = chunk.match(/^(.+)\n[=\-]{2,}$/m);
        if (underlinedMatch) {
            currentSection = underlinedMatch[1] ?? ''; // Extract just the title text
        }

        // Calculate semantic density ratio of non-stopwords to total words)
        const words = chunk.toLowerCase().match(/\b\w+\b/g) || [];
        const contentWords = words.filter(word => !stopwords.has(word));
        const semanticDensity = contentWords.length / Math.max(1, words.length);

        const doc = new Document({
            pageContent: chunk,
            metadata: {
                chunk_id: i,
                total_chunks: semanticChunks.length,
                chunk_size: chunk.length,
                chunk_type: "semantic",
                section: currentSection,
                semantic_density: Number(semanticDensity.toFixed(2))
            }
        });

        documents.push(doc);
    }

    return documents;
}

export {
    formatVectorForPgvector,
    chunkText,
    chunkBySection,
    chunkBySentences,
    performSemanticChunking,
}