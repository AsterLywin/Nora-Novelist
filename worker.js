// worker.js (Final Corrected Version)

// Use the classic, more robust importScripts to load external libraries in a worker.
try {
    importScripts(
        'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1/dist/transformers.min.js',
        'https://cdn.jsdelivr.net/npm/chromadb@1.8.1/dist/index.min.js'
    );
} catch (e) {
    console.error('Error importing scripts:', e);
    self.postMessage({ type: 'error', payload: 'Failed to load core AI libraries.' });
}

// After importScripts, libraries are available on the global `self` object.
const { pipeline, env } = self.transformers;
const { ChromaClient } = self.chromadb;

// Skip local model checks
env.allowLocalModels = false;
env.useBrowserCache = true;

// --- State Management ---
let embeddingPipeline = null;
let chromaClient = null;

const log = (message) => self.postMessage({ type: 'log', payload: message });
const error = (message) => self.postMessage({ type: 'error', payload: message });

// --- Core Classes & Functions ---

class TransformerEmbeddingFunction {
    constructor(pipeline) {
        this.pipeline = pipeline;
    }
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
    if (typeof pipeline === 'undefined' || typeof ChromaClient === 'undefined') {
        error("Initialization failed because core libraries did not load.");
        return;
    }
    
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
async function getOrCreateCollections(chatId){const embedder=new TransformerEmbeddingFunction(embeddingPipeline);const activeCollectionName=`chat_${chatId}_active`;const archiveCollectionName=`chat_${chatId}_archive`;const activeCollection=await chromaClient.getOrCreateCollection({name:activeCollectionName,embeddingFunction:embedder});const archiveCollection=await chromaClient.getOrCreateCollection({name:archiveCollectionName,embeddingFunction:embedder});return{activeCollection,archiveCollection}}
async function handleAddToMemory({chatId,messageId,text}){if(!embeddingPipeline||!chromaClient){error('Worker is not initialized.');return}
log(`Adding text from message ${messageId} to memory for chat ${chatId}...`);try{const{activeCollection}=await getOrCreateCollections(chatId);const chunks=chunkText(text);if(chunks.length===0)return;const ids=chunks.map((_,i)=>`msg_${messageId}_chunk_${i}`);const metadatas=chunks.map(()=>({message_id:messageId,timestamp:Date.now()}));await activeCollection.add({ids,documents:chunks,metadatas});log(`Successfully added ${chunks.length} chunks to active memory for chat ${chatId}.`);self.postMessage({type:'add-to-memory-done',payload:{messageId}})}catch(e){error(`Failed to add to memory: ${e.message}`)}}
async function handleGetContext({chatId,queryText,messageId,regenerationInstruction}){if(!embeddingPipeline||!chromaClient){error('Worker is not initialized.');return}
log(`Retrieving context for chat ${chatId} with query: "${queryText}"`);try{const{activeCollection,archiveCollection}=await getOrCreateCollections(chatId);let context=[];const activeResults=await activeCollection.query({queryTexts:[queryText],nResults:5});if(activeResults.documents&&activeResults.documents[0].length>0){context.push(...activeResults.documents[0])}
const archiveResults=await archiveCollection.query({queryTexts:[queryText],nResults:2});if(archiveResults.documents&&archiveResults.documents[0].length>0){context.unshift(...archiveResults.documents[0].map(summary=>`[خلاصه از گذشته: ${summary}]`))}
const uniqueContext=[...new Set(context)];const formattedContext=uniqueContext.join("\n---\n");log(`Retrieved context for chat ${chatId}: ${formattedContext.substring(0,200)}...`);self.postMessage({type:'context-retrieved',payload:{context:formattedContext,messageId,regenerationInstruction}})}catch(e){error(`Failed to get context: ${e.message}`)}}
async function handleCreateCollection({chatId}){log(`Ensuring collections exist for new chat ${chatId}...`);try{await getOrCreateCollections(chatId);log(`Collections ready for chat ${chatId}.`)}catch(e){error(`Failed to create collections for chat ${chatId}: ${e.message}`)}}
async function handleClearCollection({chatId}){log(`Clearing memory for chat ${chatId}...`);try{const{activeCollection,archiveCollection}=await getOrCreateCollections(chatId);const activeItems=await activeCollection.get();if(activeItems.ids.length>0)await activeCollection.delete({ids:activeItems.ids});const archiveItems=await archiveCollection.get();if(archiveItems.ids.length>0)await archiveCollection.delete({ids:archiveItems.ids});log(`Memory cleared for chat ${chatId}.`)}catch(e){error(`Failed to clear memory for chat ${chatId}: ${e.message}`)}}
async function handleDeleteCollection({chatId}){log(`Deleting memory collections for chat ${chatId}...`);try{const activeCollectionName=`chat_${chatId}_active`;const archiveCollectionName=`chat_${chatId}_archive`;await chromaClient.deleteCollection({name:activeCollectionName});await chromaClient.deleteCollection({name:archiveCollectionName});log(`Collections for chat ${chatId} deleted.`)}catch(e){log(`Could not delete collections for chat ${chatId} (they may not have existed): ${e.message}`)}}

// --- Event Listener ---
self.onmessage=(e)=>{const{type,payload}=e.data;switch(type){case'add-to-memory':handleAddToMemory(payload);break;case'get-context':handleGetContext(payload);break;case'create-collection':handleCreateCollection(payload);break;case'clear-collection':handleClearCollection(payload);break;case'delete-collection':handleDeleteCollection(payload);break;default:error(`Unknown message type: ${type}`);break;}};

// --- Initialization ---
initialize();
