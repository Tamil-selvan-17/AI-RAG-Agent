// ==========================================================================
// Shared auth helpers — loaded by both login.html and index.html.
// The backend issues a signed, expiring token at login; every API call sends
// it as `Authorization: Bearer <token>`. A 401 sends the user back to login.
// ==========================================================================

const API_BASE = window.location.origin.includes(':5500') || window.location.protocol === 'file:'
  ? 'http://localhost:8000'
  : window.location.origin;

const AUTH_STORAGE_KEY = 'archive.auth';

function readSession() {
  try {
    const session = JSON.parse(localStorage.getItem(AUTH_STORAGE_KEY) || 'null');
    if (!session || !session.token) return null;
    if (session.expiresAt && session.expiresAt * 1000 < Date.now()) return null;
    return session;
  } catch (e) {
    return null;
  }
}

function saveSession(token, username, expiresAt) {
  try {
    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify({ token, username, expiresAt }));
  } catch (e) {
    // Storage blocked (private mode etc.) — login still works for this page only.
  }
  window.__archiveSession = { token, username, expiresAt };
}

function clearSession() {
  try { localStorage.removeItem(AUTH_STORAGE_KEY); } catch (e) {}
  window.__archiveSession = null;
}

function currentSession() {
  return window.__archiveSession || readSession();
}

function authHeaders(extra = {}) {
  const session = currentSession();
  return session ? { ...extra, Authorization: `Bearer ${session.token}` } : extra;
}

function redirectToLogin() {
  clearSession();
  window.location.replace('login.html');
}

// fetch() wrapper for authenticated API calls.
async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: authHeaders(options.headers || {}),
  });
  if (res.status === 401) {
    redirectToLogin();
    throw new Error('Your session has expired. Please log in again.');
  }
  return res;
}
