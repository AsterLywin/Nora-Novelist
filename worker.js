// worker.js

// Import necessary libraries from a CDN.
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1/dist/transformers.min.js';
import { ChromaClient } from 'https://cdn.jsdelivr.net/npm/chromadb@1.8.1/dist/index.min.js';

// Skip local model checks for a smoother, faster setup in the browser environment.
env.allowLocalModels = false;
env.useBrowserCache = true;

// --- State Management ---
let embeddingPipeline = null;
let chromaClient = null;

// Log messages back to the main thread for easier debugging.
const log = (message) => self.postMessage({ type: 'log', payload: message });
const error = (message) => self.postMessage({ type: 'error', payload: message });


// --- Core Classes & Functions ---

/**
 * Custom Embedding Function for Transformers.js
 * This class acts as a bridge between ChromaDB and the local sentence-transformer model.
 */
class TransformerEmbeddingFunction {
    constructor(pipeline) {
        this.pipeline = pipeline;
    }

    /**
     * Generates embeddings for a batch of documents.
     * @param {string[]} documents - An array of strings to be embedded.
     * @returns {Promise<number[][]>} A promise that resolves to an array of embeddings.
     */
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

/**
 * Initializes the embedding model and the ChromaDB client.
 */
async function initialize() {
    try {
        log('Initializing worker...');
        embeddingPipeline = await pipeline('feature-extraction', 'sentence-transformers/all-MiniLM-L6-v2', {
             quantized: true,
        });
        
        chromaClient = new ChromaClient();
        
        log('Worker initialized successfully.');
        self.postMessage({ type: 'ready' });
    } catch (e) {
        error(`Initialization failed: ${e.message}`);
    }
}

/**
 * Splits a long text into smaller, manageable chunks.
 * @param {string} text - The input text.
 * @param {number} [chunkSize=250] - The approximate size of each chunk in words.
 * @param {number} [overlap=50] - The number of words to overlap between chunks.
 * @returns {string[]} An array of text chunks.
 */
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


// --- API Handlers ---

/**
 * Creates or gets the required ChromaDB collections for a specific chat.
 */
async function getOrCreateCollections(chatId) {
    const embedder = new TransformerEmbeddingFunction(embeddingPipeline);
    const activeCollectionName = `chat_${chatId}_active`;
    const archiveCollectionName = `chat_${chatId}_archive`;

    const activeCollection = await chromaClient.getOrCreateCollection({ name: activeCollectionName, embeddingFunction: embedder });
    const archiveCollection = await chromaClient.getOrCreateCollection({ name: archiveCollectionName, embeddingFunction: embedder });
    
    return { activeCollection, archiveCollection };
}

/**
 * Adds a new piece of text to the chat's active memory.
 */
async function handleAddToMemory({ chatId, messageId, text }) {
    if (!embeddingPipeline || !chromaClient) {
        error('Worker is not initialized.');
        return;
    }
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

/**
 * Retrieves relevant context from memory to help the LLM continue the story.
 */
async function handleGetContext({ chatId, queryText, messageId, regenerationInstruction }) {
    if (!embeddingPipeline || !chromaClient) {
        error('Worker is not initialized.');
        return;
    }
    log(`Retrieving context for chat ${chatId} with query: "${queryText}"`);
    try {
        const { activeCollection, archiveCollection } = await getOrCreateCollections(chatId);
        
        let context = [];

        const activeResults = await activeCollection.query({
            queryTexts: [queryText],
            nResults: 5
        });

        if (activeResults.documents && activeResults.documents[0].length > 0) {
            context.push(...activeResults.documents[0]);
        }

        const archiveResults = await archiveCollection.query({
            queryTexts: [queryText],
            nResults: 2
        });

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


/**
 * Handles the creation of a new collection for a new chat.
 */
async function handleCreateCollection({ chatId }) {
    log(`Ensuring collections exist for new chat ${chatId}...`);
    try {
        await getOrCreateCollections(chatId);
        log(`Collections ready for chat ${chatId}.`);
    } catch (e) {
        error(`Failed to create collections for chat ${chatId}: ${e.message}`);
    }
}

/**
 * Handles clearing all data from a chat's collections.
 */
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


/**
 * Handles deleting the collections associated with a chat.
 */
async function handleDeleteCollection({ chatId }) {
    log(`Deleting memory collections for chat ${chatId}...`);
    try {
        const activeCollectionName = `chat_${chatId}_active`;
        const archiveCollectionName = `chat_${chatId}_archive`;
        await chromaClient.deleteCollection({ name: activeCollectionName });
        await chromaClient.deleteCollection({ name: archiveCollectionName });
        log(`Collections for chat ${chatId} deleted.`);
    } catch(e) {
        log(`Could not delete collections for chat ${chatId} (they may not have existed): ${e.message}`);
    }
}

// --- Event Listener ---

self.onmessage = (e) => {
    const { type, payload } = e.data;
    switch (type) {
        case 'add-to-memory':
            handleAddToMemory(payload);
            break;
        case 'get-context':
            handleGetContext(payload);
            break;
        case 'create-collection':
            handleCreateCollection(payload);
            break;
        case 'clear-collection':
            handleClearCollection(payload);
            break;
        case 'delete-collection':
            handleDeleteCollection(payload);
            break;
        default:
            error(`Unknown message type: ${type}`);
            break;
    }
};

// --- Initialization ---
initialize();     */
    async generate(documents) {
        try {
            // Use the provided pipeline to generate embeddings.
            // The model is 'sentence-transformers/all-MiniLM-L6-v2'.
            // The output is normalized to ensure consistent vector lengths.
            let embeddings = await this.pipeline(documents, { pooling: 'mean', normalize: true });
            // Convert the Float32Array embeddings to a standard nested array format.
            return embeddings.tolist();
        } catch (e) {
            error(`Embedding generation failed: ${e.message}`);
            // Return an empty array to prevent downstream errors.
            return [];
        }
    }
}

/**
 * Initializes the embedding model and the ChromaDB client.
 * This is the primary setup function for the worker.
 */
async function initialize() {
    try {
        log('Initializing worker...');
        // Load the sentence-transformer model. This is a one-time setup cost.
        // 'feature-extraction' is the task type for getting embeddings.
        embeddingPipeline = await pipeline('feature-extraction', 'sentence-transformers/all-MiniLM-L6-v2', {
             quantized: true, // Use quantized models for better performance in the browser.
        });
        
        // Initialize the ChromaDB client. It will use IndexedDB for storage by default.
        chromaClient = new ChromaClient();
        
        log('Worker initialized successfully.');
        // Signal to the main thread that the worker is ready to process requests.
        self.postMessage({ type: 'ready' });
    } catch (e) {
        error(`Initialization failed: ${e.message}`);
    }
}

/**
 * Splits a long text into smaller, manageable chunks.
 * This is crucial for creating meaningful embeddings and efficient retrieval.
 * @param {string} text - The input text.
 * @param {number} [chunkSize=250] - The approximate size of each chunk in words.
 * @param {number} [overlap=50] - The number of words to overlap between chunks to maintain context.
 * @returns {string[]} An array of text chunks.
 */
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


// --- API Handlers (Message-based interface) ---

/**
 * Creates or gets the required ChromaDB collections for a specific chat.
 * Each chat has two collections: one for active memory (detailed chunks) and one for archival memory (summaries).
 * @param {number} chatId - The unique identifier for the chat.
 * @returns {Promise<{activeCollection: any, archiveCollection: any}>} A promise resolving to the collections.
 */
async function getOrCreateCollections(chatId) {
    const embedder = new TransformerEmbeddingFunction(embeddingPipeline);
    const activeCollectionName = `chat_${chatId}_active`;
    const archiveCollectionName = `chat_${chatId}_archive`;

    // `getOrCreateCollection` is an idempotent operation.
    const activeCollection = await chromaClient.getOrCreateCollection({ name: activeCollectionName, embeddingFunction: embedder });
    const archiveCollection = await chromaClient.getOrCreateCollection({ name: archiveCollectionName, embeddingFunction: embedder });
    
    return { activeCollection, archiveCollection };
}

/**
 * Adds a new piece of text (e.g., a story chapter) to the chat's active memory.
 * @param {object} payload - The data for the operation.
 * @param {number} payload.chatId - The ID of the chat.
 * @param {number} payload.messageId - The ID of the message this text corresponds to.
 * @param {string} payload.text - The text to add.
 */
async function handleAddToMemory({ chatId, messageId, text }) {
    if (!embeddingPipeline || !chromaClient) {
        error('Worker is not initialized.');
        return;
    }
    log(`Adding text from message ${messageId} to memory for chat ${chatId}...`);
    try {
        const { activeCollection } = await getOrCreateCollections(chatId);
        const chunks = chunkText(text);
        
        if (chunks.length === 0) return;

        // Create unique IDs for each chunk to store in the database.
        const ids = chunks.map((_, i) => `msg_${messageId}_chunk_${i}`);
        // Associate metadata with each chunk for filtering and context.
        const metadatas = chunks.map(() => ({ message_id: messageId, timestamp: Date.now() }));
        
        await activeCollection.add({ ids, documents: chunks, metadatas });
        log(`Successfully added ${chunks.length} chunks to active memory for chat ${chatId}.`);
        self.postMessage({ type: 'add-to-memory-done', payload: { messageId } });

        // TODO: Implement the maintenance/archival logic here.
        // This would involve checking the age/number of items in active memory
        // and triggering a summarization and archival process if a threshold is met.

    } catch (e) {
        error(`Failed to add to memory: ${e.message}`);
    }
}

/**
 * Retrieves relevant context from memory to help the LLM continue the story.
 * It performs a multi-step search: first in active memory, then in archival memory.
 * @param {object} payload - The data for the operation.
 * @param {number} payload.chatId - The ID of the chat.
 * @param {string} payload.queryText - The user's instruction or the last part of the story.
 * @param {number} payload.messageId - The ID of the message requesting context.
 */
async function handleGetContext({ chatId, queryText, messageId, regenerationInstruction }) {
    if (!embeddingPipeline || !chromaClient) {
        error('Worker is not initialized.');
        return;
    }
    log(`Retrieving context for chat ${chatId} with query: "${queryText}"`);
    try {
        const { activeCollection, archiveCollection } = await getOrCreateCollections(chatId);
        
        let context = [];

        // Step 1: Search in Active Memory (for recent, detailed context)
        const activeResults = await activeCollection.query({
            queryTexts: [queryText],
            nResults: 5 // Retrieve the top 5 most relevant recent chunks.
        });

        if (activeResults.documents && activeResults.documents[0].length > 0) {
            context.push(...activeResults.documents[0]);
        }

        // Step 2: Search in Archival Memory (for high-level, older context)
        const archiveResults = await archiveCollection.query({
            queryTexts: [queryText],
            nResults: 2 // Retrieve 1 or 2 relevant summaries from the past.
        });

        if (archiveResults.documents && archiveResults.documents[0].length > 0) {
            // Prepend archive results to provide broad context first.
            context.unshift(...archiveResults.documents[0].map(summary => `[خلاصه از گذشته: ${summary}]`));
        }

        // Remove duplicates and join the context pieces.
        const uniqueContext = [...new Set(context)];
        const formattedContext = uniqueContext.join("\n---\n");
        
        log(`Retrieved context for chat ${chatId}: ${formattedContext.substring(0, 200)}...`);

        // Send the retrieved context back to the main thread.
        self.postMessage({ type: 'context-retrieved', payload: { context: formattedContext, messageId, regenerationInstruction } });

    } catch (e) {
        error(`Failed to get context: ${e.message}`);
    }
}


/**
 * Handles the creation of a new collection for a new chat.
 */
async function handleCreateCollection({ chatId }) {
    log(`Ensuring collections exist for new chat ${chatId}...`);
    try {
        await getOrCreateCollections(chatId);
        log(`Collections ready for chat ${chatId}.`);
    } catch (e) {
        error(`Failed to create collections for chat ${chatId}: ${e.message}`);
    }
}

/**
 * Handles clearing all data from a chat's collections.
 */
async function handleClearCollection({ chatId }) {
    log(`Clearing memory for chat ${chatId}...`);
    try {
        const { activeCollection, archiveCollection } = await getOrCreateCollections(chatId);
        // Clear by fetching all items and deleting them. ChromaDB client may not have a direct `clear` method.
        const activeItems = await activeCollection.get();
        if (activeItems.ids.length > 0) await activeCollection.delete({ ids: activeItems.ids });

        const archiveItems = await archiveCollection.get();
        if (archiveItems.ids.length > 0) await archiveCollection.delete({ ids: archiveItems.ids });

        log(`Memory cleared for chat ${chatId}.`);
    } catch (e) {
        error(`Failed to clear memory for chat ${chatId}: ${e.message}`);
    }
}


/**
 * Handles deleting the collections associated with a chat.
 */
async function handleDeleteCollection({ chatId }) {
    log(`Deleting memory collections for chat ${chatId}...`);
    try {
        const activeCollectionName = `chat_${chatId}_active`;
        const archiveCollectionName = `chat_${chatId}_archive`;
        await chromaClient.deleteCollection({ name: activeCollectionName });
        await chromaClient.deleteCollection({ name: archiveCollectionName });
        log(`Collections for chat ${chatId} deleted.`);
    } catch(e) {
        // Chroma might throw an error if the collection doesn't exist, which is fine.
        log(`Could not delete collections for chat ${chatId} (they may not have existed): ${e.message}`);
    }
}


// --- Event Listener ---

// Listen for messages from the main thread.
self.onmessage = (e) => {
    const { type, payload } = e.data;

    // A simple router to handle different command types.
    switch (type) {
        case 'add-to-memory':
            handleAddToMemory(payload);
            break;
        case 'get-context':
            handleGetContext(payload);
            break;
        case 'create-collection':
            handleCreateCollection(payload);
            break;
        case 'clear-collection':
            handleClearCollection(payload);
            break;
        case 'delete-collection':
            handleDeleteCollection(payload);
            break;
        default:
            error(`Unknown message type: ${type}`);
            break;
    }
};

// --- Initialization ---

// Start the initialization process as soon as the worker is loaded.
initialize();
