'use strict';

const CACHE_NAME = 'nora-novelist-cache-v1.0.0';
const urlsToCache = [
  './',
  './index.html',
  './worker.js',
  './manifest.json',
  './Icons/icon-192x192.png',
  './Icons/icon-512x512.png',
  './Fonts/Iran%20Yekan%20Medium.ttf',
  './Fonts/San%20Francisco%20bold.ttf'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('Opened cache and caching files for Nora Novelist');
      return cache.addAll(urlsToCache);
    }).then(() => self.skipWaiting())
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
    })
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    fetch(event.request).catch(() => {
      return caches.match(event.request);
    })
  );
});});

self.addEventListener('fetch', event => {
    event.respondWith(
    fetch(event.request).catch(() => {
      return caches.match(event.request);
    })
  );
});
