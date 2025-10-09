'use strict';

// *** CHANGED: Updated cache name for new version ***
const CACHE_NAME = 'nora-novelist-cache-v1.0.6';

// *** ADDED: List of essential model files to pre-cache for faster initial load ***
const modelUrlsToCache = [
  'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/config.json',
  'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json',
  'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json',
  'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json',
  'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/vocab.txt',
  // This is the main (quantized) model file, which is the largest.
  'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx' 
];

// App shell files
const appShellUrlsToCache = [
  './',
  './index.html',
  './worker.js',
  './manifest.json',
  './Icons/icon-192x192.png',
  './Icons/icon-512x512.png',
  './Fonts/Iran%20Yekan%20Medium.ttf',
  './Fonts/San%20Francisco%20bold.ttf',
  // It's good practice to also cache the libraries used by the worker
  './libs/transformers.min.js',
  './libs/chromadb.mjs'
];

// Combine both lists
const allUrlsToCache = [...appShellUrlsToCache, ...modelUrlsToCache];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('Opened cache. Caching app shell and AI model files for Nora Novelist.');
      // Use addAll to cache all essential files. If any request fails, the installation will fail.
      // This is important to ensure the app is fully functional offline.
      return cache.addAll(allUrlsToCache);
    }).then(() => {
      console.log('All essential files cached. Service worker installation complete.');
      return self.skipWaiting();
    }).catch(error => {
        // Log the error if caching fails. This helps in debugging.
        console.error('Failed to cache files during install:', error);
    })
  );
});

self.addEventListener('activate', event => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            console.log('Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim()) // Ensure the new service worker takes control immediately
  );
});

// *** CHANGED: Cleaned up duplicate fetch listeners and implemented a cache-first strategy ***
self.addEventListener('fetch', event => {
  // For model files from Hugging Face, use a "cache-first" strategy
  if (event.request.url.startsWith('https://huggingface.co/')) {
    event.respondWith(
      caches.match(event.request).then(cachedResponse => {
        // If the file is in the cache, return it immediately.
        if (cachedResponse) {
          return cachedResponse;
        }
        // Otherwise, fetch it from the network.
        return fetch(event.request);
      })
    );
  } else {
    // For all other requests (app shell, API calls, etc.), use a "network-first, fallback to cache" strategy
    event.respondWith(
      fetch(event.request).catch(() => {
        // If the network fails (e.g., offline), try to find a match in the cache.
        return caches.match(event.request);
      })
    );
  }
});
