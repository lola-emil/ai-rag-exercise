
function formatVectorForPgvector(embedding) {
    return '[' + embedding.join(',') + ']';
}

function chunkText(text, chunkSize = 500, overlap = 50) {
    const chunks = [];
    let start = 0;
    while (start < text.length) {
        chunks.push(text.slice(start, start + chunkSize));
        start += chunkSize - overlap;
    }
    return chunks;
}

function chunkBySection(text) {
    return text
        .split(/(?=^[a-z]\.\s+)/gm)
        .filter(Boolean);
}


export {
    formatVectorForPgvector,
    chunkText,
    chunkBySection,
}