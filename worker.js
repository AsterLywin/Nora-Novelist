// worker.js

// Import libraries from the local 'libs' folder
import { pipeline, env } from './libs/transformers.min.js';
import { ChromaClient } from './libs/chromadb.mjs';

// Configuration for the transformer pipeline
env.allowLocalModels = false;
env.useBrowserCache = true;

// --- State Management & Helper Functions ---
const log = (message) => self.postMessage({ type: 'log', payload: message });
const error = (message) => self.postMessage({ type: 'error', payload: message });

let embeddingPipeline = null;
let chromaClient = null;

class TransformerEmbeddingFunction {
    constructor(pipeline) { this.pipeline = pipeline; }
    async generate(documents) {
        try {
            let embeddings = await this.pipeline(documents, { pooling: 'mean', normalize: true });
            return embeddings.tolist();
        } catch (e) {
            error(`Embedding generation failed: ${e.message}`);
            return [];
        }
    }
}

async function initialize() {
    try {
        log('Initializing worker and loading embedding model...');
        embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { quantized: true });
        chromaClient = new ChromaClient();
        log('Worker initialized successfully.');
        
        // *** ADDED: Send a 'ready' message to the main thread when initialization is complete ***
        self.postMessage({ type: 'ready' });

    } catch (e) {
        error(`Initialization failed: ${e.message}`);
    }
}

function chunkText(text, chunkSize = 250, overlap = 50) {
    const words = text.split(/\s+/);
    const chunks = [];
    let i = 0;
    while (i < words.length) {
        const chunk = words.slice(i, i + chunkSize).join(' ');
        chunks.push(chunk);
        i += chunkSize - overlap;
    }
    return chunks;
}

// --- API Handlers for communication with the main thread ---
async function getOrCreateCollections(chatId) {
    const embedder = new TransformerEmbeddingFunction(embeddingPipeline);
    const activeCollectionName = `chat_${chatId}_active`;
    const archiveCollectionName = `chat_${chatId}_archive`;
    const activeCollection = await chromaClient.getOrCreateCollection({ name: activeCollectionName, embeddingFunction: embedder });
    const archiveCollection = await chromaClient.getOrCreateCollection({ name: archiveCollectionName, embeddingFunction: embedder });
    return { activeCollection, archiveCollection };
}

async function handleAddToMemory({ chatId, messageId, text }) {
    if (!embeddingPipeline || !chromaClient) { error('Worker is not initialized.'); return; }
    log(`Adding text from message ${messageId} to memory for chat ${chatId}...`);
    try {
        const { activeCollection } = await getOrCreateCollections(chatId);
        const chunks = chunkText(text);
        if (chunks.length === 0) return;
        const ids = chunks.map((_, i) => `msg_${messageId}_chunk_${i}`);
        const metadatas = chunks.map(() => ({ message_id: messageId, timestamp: Date.now() }));
        await activeCollection.add({ ids, documents: chunks, metadatas });
        log(`Successfully added ${chunks.length} chunks to active memory for chat ${chatId}.`);
        self.postMessage({ type: 'add-to-memory-done', payload: { messageId } });
    } catch (e) {
        error(`Failed to add to memory: ${e.message}`);
    }
}

async function handleGetContext({ chatId, queryText, messageId, regenerationInstruction }) {
    if (!embeddingPipeline || !chromaClient) { error('Worker is not initialized.'); return; }
    log(`Retrieving context for chat ${chatId} with query: "${queryText}"`);
    try {
        const { activeCollection, archiveCollection } = await getOrCreateCollections(chatId);
        let context = [];
        const activeResults = await activeCollection.query({ queryTexts: [queryText], nResults: 5 });
        if (activeResults.documents && activeResults.documents[0].length > 0) {
            context.push(...activeResults.documents[0]);
        }
        const archiveResults = await archiveCollection.query({ queryTexts: [queryText], nResults: 2 });
        if (archiveResults.documents && archiveResults.documents[0].length > 0) {
            context.unshift(...archiveResults.documents[0].map(summary => `[خلاصه از گذشته: ${summary}]`));
        }
        const uniqueContext = [...new Set(context)];
        const formattedContext = uniqueContext.join("\n---\n");
        log(`Retrieved context for chat ${chatId}: ${formattedContext.substring(0, 200)}...`);
        self.postMessage({ type: 'context-retrieved', payload: { context: formattedContext, messageId, regenerationInstruction } });
    } catch (e) {
        error(`Failed to get context: ${e.message}`);
    }
}

async function handleCreateCollection({ chatId }) {
    log(`Ensuring collections exist for new chat ${chatId}...`);
    try {
        await getOrCreateCollections(chatId);
        log(`Collections ready for chat ${chatId}.`);
    } catch (e) {
        error(`Failed to create collections for chat ${chatId}: ${e.message}`);
    }
}

async function handleClearCollection({ chatId }) {
    log(`Clearing memory for chat ${chatId}...`);
    try {
        const { activeCollection, archiveCollection } = await getOrCreateCollections(chatId);
        const activeItems = await activeCollection.get();
        if (activeItems.ids.length > 0) await activeCollection.delete({ ids: activeItems.ids });
        const archiveItems = await archiveCollection.get();
        if (archiveItems.ids.length > 0) await archiveCollection.delete({ ids: archiveItems.ids });
        log(`Memory cleared for chat ${chatId}.`);
    } catch (e) {
        error(`Failed to clear memory for chat ${chatId}: ${e.message}`);
    }
}

async function handleDeleteCollection({ chatId }) {
    log(`Deleting memory collections for chat ${chatId}...`);
    try {
        const activeCollectionName = `chat_${chatId}_active`;
        const archiveCollectionName = `chat_${chatId}_archive`;
        await chromaClient.deleteCollection({ name: activeCollectionName });
        await chromaClient.deleteCollection({ name: archiveCollectionName });
        log(`Collections for chat ${chatId} deleted.`);
    } catch (e) {
        log(`Could not delete collections for chat ${chatId} (they may not have existed): ${e.message}`);
    }
}

// Main message handler for the worker
self.onmessage = (e) => {
    const { type, payload } = e.data;
    switch (type) {
        case 'add-to-memory': handleAddToMemory(payload); break;
        case 'get-context': handleGetContext(payload); break;
        case 'create-collection': handleCreateCollection(payload); break;
        case 'clear-collection': handleClearCollection(payload); break;
        case 'delete-collection': handleDeleteCollection(payload); break;
        default: error(`Unknown message type: ${type}`); break;
    }
};

// Start the initialization process as soon as the worker is loaded
initialize();
