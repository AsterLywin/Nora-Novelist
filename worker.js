// worker.js

// Import libraries from the local 'libs' folder
import { pipeline, env } from './libs/transformers.min.js';
import { ChromaClient } from './libs/chromadb.mjs';

env.allowLocalModels = false;
env.useBrowserCache = true;

const log = (message) => self.postMessage({ type: 'log', payload: message });
const error = (message) => self.postMessage({ type: 'error', payload: message });

const ACTIVE_MEMORY_CHAPTER_LIMIT = 20;const ARCHIVE_BATCH_SIZE = 1;

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
        const metadatas = chunks.map(() => ({ chapter_id: messageId, timestamp: Date.now() }));
        await activeCollection.add({ ids, documents: chunks, metadatas });
        log(`Successfully added ${chunks.length} chunks to active memory for chat ${chatId}.`);
        self.postMessage({ type: 'add-to-memory-done', payload: { messageId } });

        runMaintenance(chatId);

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

async function runMaintenance(chatId) {
    log(`[Maintenance] Running check for chat ${chatId}...`);
    try {
        const { activeCollection } = await getOrCreateCollections(chatId);
        const allItems = await activeCollection.get({ include: ["metadatas"] });

        if (!allItems.metadatas || allItems.metadatas.length === 0) {
            log('[Maintenance] Active memory is empty. Nothing to do.');
            return;
        }

        const chapterIds = [...new Set(allItems.metadatas.map(m => m.chapter_id))].sort((a, b) => a - b);
        
        if (chapterIds.length > ACTIVE_MEMORY_CHAPTER_LIMIT) {
            const chaptersToArchive = chapterIds.slice(0, chapterIds.length - ACTIVE_MEMORY_CHAPTER_LIMIT).slice(0, ARCHIVE_BATCH_SIZE);
            log(`[Maintenance] Found ${chaptersToArchive.length} chapter(s) to archive: [${chaptersToArchive.join(', ')}]`);

            for (const chapterId of chaptersToArchive) {
                const chapterChunks = await activeCollection.get({ where: { chapter_id: chapterId } });
                const fullText = chapterChunks.documents.join(' ');
                
                self.postMessage({
                    type: 'summarize-and-archive',
                    payload: {
                        chatId,
                        chapterIdToArchive: chapterId,
                        fullText,
                    }
                });
            }
        } else {
            log(`[Maintenance] Active memory within limits (${chapterIds.length}/${ACTIVE_MEMORY_CHAPTER_LIMIT}). No archiving needed.`);
        }
    } catch (e) {
        error(`[Maintenance] Error during maintenance run for chat ${chatId}: ${e.message}`);
    }
}

async function handleArchiveData({ chatId, chapterIdToArchive, summary }) {
    log(`[Archive] Archiving summary for chapter ${chapterIdToArchive} in chat ${chatId}.`);
    try {
        const { activeCollection, archiveCollection } = await getOrCreateCollections(chatId);
        
        await archiveCollection.add({
            ids: [`summary_${chapterIdToArchive}`],
            documents: [summary],
            metadatas: [{ original_chapter_id: chapterIdToArchive }]
        });
        log(`[Archive] Summary for chapter ${chapterIdToArchive} added to archive.`);

        const chunksToDelete = await activeCollection.get({ where: { chapter_id: chapterIdToArchive } });
        if (chunksToDelete.ids && chunksToDelete.ids.length > 0) {
            await activeCollection.delete({ ids: chunksToDelete.ids });
            log(`[Archive] Deleted ${chunksToDelete.ids.length} chunks for chapter ${chapterIdToArchive} from active memory.`);
        }

    } catch (e) {
        error(`[Archive] Failed to finalize archiving for chapter ${chapterIdToArchive}: ${e.message}`);
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
        const activeCollectionName = `chat_${chatId}_active`;
        const archiveCollectionName = `chat_${chatId}_archive`;
        await chromaClient.deleteCollection({ name: activeCollectionName });
        await chromaClient.deleteCollection({ name: archiveCollectionName });
        await getOrCreateCollections(chatId); // Re-create them empty
        log(`Memory cleared for chat ${chatId}.`);
    } catch (e) {
        try { await getOrCreateCollections(chatId); } catch (e2) {
             error(`Failed to clear/re-create memory for chat ${chatId}: ${e.message}`);
        }
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

self.onmessage = (e) => {
    const { type, payload } = e.data;
    switch (type) {
        case 'add-to-memory': handleAddToMemory(payload); break;
        case 'get-context': handleGetContext(payload); break;
        case 'create-collection': handleCreateCollection(payload); break;
        case 'clear-collection': handleClearCollection(payload); break;
        case 'delete-collection': handleDeleteCollection(payload); break;
        case 'archive-data': handleArchiveData(payload); break;
        default: error(`Unknown message type: ${type}`); break;
    }
};

initialize();
