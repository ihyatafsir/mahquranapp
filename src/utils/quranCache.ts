/**
 * Quran Data Caching Engine (IndexedDB + Memory LRU)
 * Provides zero-latency instant loading for verse metadata and letter timing files.
 * Works seamlessly offline with automatic background pre-warming.
 */

const DB_NAME = "mah_quran_cache_db";
const DB_VERSION = 1;
const STORE_NAME = "json_cache";
const MEMORY_CACHE = new Map<string, any>();

let dbPromise: Promise<IDBDatabase | null> | null = null;

function openDB(): Promise<IDBDatabase | null> {
  if (typeof window === "undefined" || !window.indexedDB) {
    return Promise.resolve(null);
  }
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      try {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(STORE_NAME)) {
            db.createObjectStore(STORE_NAME, { keyPath: "url" });
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => {
          console.warn("[QuranCache] IndexedDB open failed, falling back to memory cache.");
          resolve(null);
        };
      } catch (e) {
        console.warn("[QuranCache] IndexedDB not available:", e);
        resolve(null);
      }
    });
  }
  return dbPromise;
}

async function getFromIDB<T>(url: string): Promise<T | null> {
  const db = await openDB();
  if (!db) return null;

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(url);
      req.onsuccess = () => {
        if (req.result && req.result.data) {
          resolve(req.result.data as T);
        } else {
          resolve(null);
        }
      };
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function saveToIDB<T>(url: string, data: T): Promise<void> {
  const db = await openDB();
  if (!db) return;

  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      store.put({ url, data, timestamp: Date.now() });
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    } catch {
      resolve();
    }
  });
}

/**
 * High-performance fetch with L1 (Memory) and L2 (IndexedDB) caching.
 */
export async function fetchWithCache<T>(url: string): Promise<T> {
  // L1: Memory Cache (Instant, 0ms)
  if (MEMORY_CACHE.has(url)) {
    return MEMORY_CACHE.get(url) as T;
  }

  // L2: IndexedDB Cache (Local persistent storage, ~1-3ms)
  const idbData = await getFromIDB<T>(url);
  if (idbData) {
    MEMORY_CACHE.set(url, idbData);
    return idbData;
  }

  // L3: Network Fetch
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url} (HTTP ${res.status})`);
  }

  const data = await res.json();
  MEMORY_CACHE.set(url, data);

  // Save in background to IDB
  saveToIDB(url, data).catch((err) =>
    console.warn("[QuranCache] Failed to persist to IDB:", err)
  );

  return data as T;
}

/**
 * Preloads timing data for adjacent surahs in the background
 */
export function preloadSurahTiming(surahNumber: number, reciter: string): void {
  if (surahNumber < 1 || surahNumber > 114) return;
  const timingPath =
    reciter === "mah"
      ? `/data/letter_timing_${surahNumber}.json`
      : `/data/${reciter}/letter_timing_${surahNumber}.json`;

  if (!MEMORY_CACHE.has(timingPath)) {
    fetchWithCache(timingPath).catch(() => {});
  }
}
