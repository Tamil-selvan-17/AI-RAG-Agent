// ==========================================================================
// Archive — local document assistant frontend
// Talks to the FastAPI backend for upload, document listing, and chat.
// ==========================================================================

// API_BASE, apiFetch, authHeaders, etc. come from auth.js.

// Not signed in (or session expired)? Go to the login page before doing anything.
if (!readSession()) redirectToLogin();

const CONVERSATION_STORAGE_KEY = 'archive.conversationId';

const state = {
  conversationId: null,
  documents: [],
  streamController: null, // AbortController of the answer currently streaming
};

// ---- DOM refs ----
const appShell = document.querySelector('.app-shell');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadProgressFill = document.getElementById('uploadProgressFill');
const uploadProgressLabel = document.getElementById('uploadProgressLabel');
const libraryList = document.getElementById('libraryList');
const docCount = document.getElementById('docCount');
const statusDot = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const chatScroll = document.getElementById('chatScroll');
const composerForm = document.getElementById('composerForm');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const stopBtn = document.getElementById('stopBtn');
const languageSelect = document.getElementById('languageSelect');
const clearBtn = document.getElementById('clearBtn');
const conversationMeta = document.getElementById('conversationMeta');
const userChip = document.getElementById('userChip');
const logoutBtn = document.getElementById('logoutBtn');
const libraryToggle = document.getElementById('libraryToggle');
const libraryBackdrop = document.getElementById('libraryBackdrop');
const toastStack = document.getElementById('toastStack');

const WELCOME_HTML = `
  <div class="welcome-card">
    <p class="welcome-eyebrow">Ask anything about your documents</p>
    <h3>What would you like to know?</h3>
    <p class="welcome-sub">Upload a document, then ask a question. Every answer is grounded in your files, with the exact passages one click away.</p>
  </div>`;

// ==========================================================================
// Small helpers: storage, toasts, formatting
// ==========================================================================

function storageGet(key) {
  try { return localStorage.getItem(key); } catch (e) { return null; }
}

function storageSet(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch (e) {
    // Storage blocked (private mode etc.) — the app still works, it just won't remember.
  }
}

// Non-blocking notifications instead of alert(). kind: 'info' | 'success' | 'error'.
function showToast(message, kind = 'info', timeoutMs = kind === 'error' ? 7000 : 4000) {
  const toast = document.createElement('div');
  toast.className = `toast toast-${kind}`;
  toast.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  toast.innerHTML = `<span class="toast-message"></span><button type="button" class="toast-close" aria-label="Dismiss">×</button>`;
  toast.querySelector('.toast-message').textContent = message;

  const dismiss = () => {
    toast.classList.add('leaving');
    setTimeout(() => toast.remove(), 200);
  };
  toast.querySelector('.toast-close').addEventListener('click', dismiss);
  setTimeout(dismiss, timeoutMs);
  toastStack.appendChild(toast);
}

function formatBytes(bytes) {
  if (!bytes) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ==========================================================================
// Health check
// ==========================================================================

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === 'healthy') {
      statusDot.className = 'status-dot ok';
      statusLabel.textContent = 'All services online';
    } else {
      statusDot.className = 'status-dot warn';
      const down = Object.entries(data.services || {})
        .filter(([, v]) => v !== 'healthy')
        .map(([k]) => k);
      statusLabel.textContent = `Degraded: ${down.join(', ') || 'unknown'}`;
    }
  } catch (err) {
    statusDot.className = 'status-dot error';
    statusLabel.textContent = 'Backend unreachable';
  }
}

// ==========================================================================
// Document library
// ==========================================================================

async function loadDocuments() {
  const previous = new Map(state.documents.map(doc => [doc.document_id, doc.status]));
  try {
    const res = await apiFetch('/api/documents');
    if (!res.ok) throw new Error('Failed to load documents');
    const data = await res.json();
    state.documents = data.documents || [];
    renderLibrary();
    announceFinishedIndexing(previous);
  } catch (err) {
    console.error(err);
  }
  schedulePollIfProcessing();
}

// Tell the user when a background upload they were waiting on finishes.
function announceFinishedIndexing(previousStatuses) {
  for (const doc of state.documents) {
    if (previousStatuses.get(doc.document_id) !== 'processing') continue;
    if (doc.status === 'ready') showToast(`“${doc.filename}” is ready to search`, 'success');
    if (doc.status === 'failed') showToast(`“${doc.filename}” failed: ${doc.error_message || 'unknown error'}`, 'error');
  }
}

// Uploads are indexed in the background, so keep refreshing the list while any
// document is still "processing" until it turns "ready" (or "failed").
let documentPollTimer = null;
function schedulePollIfProcessing() {
  const busy = state.documents.some(doc => doc.status === 'processing');
  if (busy && !documentPollTimer) {
    documentPollTimer = setTimeout(() => {
      documentPollTimer = null;
      loadDocuments();
    }, 2000);
  }
}

function renderLibrary() {
  const count = state.documents.length;
  docCount.textContent = `${count} document${count === 1 ? '' : 's'}`;
  libraryToggle.querySelector('.library-toggle-count').textContent = count;

  if (count === 0) {
    libraryList.innerHTML = `
      <div class="empty-state">
        <p>No documents yet.</p>
        <p class="empty-state-sub">Uploaded files will appear here once indexed.</p>
      </div>`;
    return;
  }

  libraryList.innerHTML = state.documents.map(doc => `
    <div class="doc-card">
      <div class="doc-card-top">
        <p class="doc-name">${escapeHtml(doc.filename)}</p>
        <span class="doc-badge ${doc.status}">${doc.status}</span>
      </div>
      <div class="doc-card-bottom">
        <p class="doc-meta">${doc.status === 'processing' ? 'Indexing…' : `${doc.chunk_count} chunks`} · ${formatBytes(doc.file_size_bytes)}</p>
        <button class="doc-delete-btn" data-doc-id="${doc.document_id}" title="Delete this document">Delete</button>
      </div>
      ${doc.status === 'failed' && doc.error_message ? `<p class="doc-error">${escapeHtml(doc.error_message)}</p>` : ''}
    </div>
  `).join('');

  libraryList.querySelectorAll('.doc-delete-btn').forEach(btn => {
    btn.addEventListener('click', () => deleteDocument(btn.dataset.docId));
  });
}

async function deleteDocument(documentId) {
  if (!confirm('Delete this document? This removes it from the knowledge base permanently.')) return;
  try {
    const res = await apiFetch(`/api/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    await loadDocuments();
    showToast('Document deleted', 'success');
  } catch (err) {
    showToast(`Could not delete document: ${err.message}`, 'error');
  }
}

const clearAllDocsBtn = document.getElementById('clearAllDocsBtn');
clearAllDocsBtn.addEventListener('click', async () => {
  if (state.documents.length === 0) return;
  if (!confirm(`Delete all ${state.documents.length} document(s)? This cannot be undone.`)) return;
  try {
    const res = await apiFetch('/api/documents', { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    await loadDocuments();
    showToast('All documents deleted', 'success');
  } catch (err) {
    showToast(`Could not delete documents: ${err.message}`, 'error');
  }
});

// ---- Mobile: the library is a slide-in drawer ----

function setLibraryOpen(open) {
  appShell.classList.toggle('library-open', open);
  libraryToggle.setAttribute('aria-expanded', String(open));
}

libraryToggle.addEventListener('click', () => setLibraryOpen(!appShell.classList.contains('library-open')));
libraryBackdrop.addEventListener('click', () => setLibraryOpen(false));
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') setLibraryOpen(false);
});

// ==========================================================================
// Upload flow (drag & drop + click, one or many files)
// ==========================================================================

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  uploadFiles([...e.dataTransfer.files]);
});

fileInput.addEventListener('change', () => {
  uploadFiles([...fileInput.files]);
  fileInput.value = '';
});

let uploadInProgress = false;

// Uploads files one after another (each returns quickly; indexing continues
// in the background), then refreshes the library once.
async function uploadFiles(files) {
  if (files.length === 0) return;
  if (uploadInProgress) {
    showToast('Please wait for the current upload to finish', 'info');
    return;
  }
  uploadInProgress = true;
  let accepted = 0;
  try {
    for (let i = 0; i < files.length; i++) {
      const label = files.length > 1 ? `(${i + 1}/${files.length}) ` : '';
      try {
        await uploadFile(files[i], label);
        accepted++;
      } catch (err) {
        if (err.message === 'unauthorized') return;
        showToast(`${files[i].name}: ${err.message}`, 'error');
      }
    }
  } finally {
    uploadInProgress = false;
    uploadProgress.hidden = true;
  }
  if (accepted > 0) {
    showToast(
      accepted === 1 ? 'Upload received — indexing in the background' : `${accepted} files received — indexing in the background`,
      'info',
    );
    loadDocuments();
  }
}

function uploadFile(file, label) {
  return new Promise((resolve, reject) => {
    const formData = new FormData();
    formData.append('file', file);

    uploadProgress.hidden = false;
    uploadProgressFill.style.width = '0%';
    uploadProgressLabel.textContent = `${label}Uploading ${file.name}…`;

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}/api/documents/upload`);
    const { Authorization } = authHeaders();
    if (Authorization) xhr.setRequestHeader('Authorization', Authorization);

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        uploadProgressFill.style.width = `${pct}%`;
        if (pct >= 100) uploadProgressLabel.textContent = `${label}Saving ${file.name}…`;
      }
    });

    xhr.onload = () => {
      if (xhr.status === 401) {
        redirectToLogin();
        reject(new Error('unauthorized'));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve();
        return;
      }
      let detail = 'Upload failed';
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (e) {}
      reject(new Error(detail));
    };

    xhr.onerror = () => reject(new Error('could not reach the backend'));
    xhr.send(formData);
  });
}

// ==========================================================================
// Chat flow
// ==========================================================================

function setStreaming(active) {
  sendBtn.hidden = active;
  stopBtn.hidden = !active;
  questionInput.toggleAttribute('aria-busy', active);
}

stopBtn.addEventListener('click', () => {
  if (state.streamController) state.streamController.abort();
});

composerForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question || state.streamController) return;

  clearWelcomeCard();
  appendQuestionBubble(question);
  questionInput.value = '';
  autoGrow();

  const loadingEl = appendLoadingAnswer();
  state.streamController = new AbortController();
  setStreaming(true);

  try {
    await streamAnswer(question, loadingEl, state.streamController.signal);
  } catch (err) {
    if (err.name !== 'AbortError') {
      renderAnswerContent(loadingEl, `Something went wrong: ${err.message}`, []);
    } else if (loadingEl.querySelector('.msg-answer.loading')) {
      // Stopped before the first token arrived (the fetch itself was aborted).
      renderAnswerContent(loadingEl, '_Stopped before any answer was generated._', [], { stopped: true });
    }
  } finally {
    state.streamController = null;
    setStreaming(false);
    questionInput.focus();
  }
});

async function streamAnswer(question, wrapEl, signal) {
  const selectedLanguage = languageSelect.value || null;
  const res = await apiFetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      conversation_id: state.conversationId,
      response_language: selectedLanguage,
    }),
    signal,
  });

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(typeof err.detail === 'string' ? err.detail : 'Request failed');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answerText = '';
  let answerEl = null;
  let renderPending = false;

  // Render markdown as tokens arrive, at most once per animation frame.
  const scheduleRender = () => {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      if (answerEl) {
        answerEl.innerHTML = renderMarkdown(answerText);
        chatScroll.scrollTop = chatScroll.scrollHeight;
      }
    });
  };

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by a blank line; process each complete one as it arrives.
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);

        const line = rawEvent.split('\n').find(l => l.startsWith('data: '));
        if (!line) continue;

        const event = JSON.parse(line.slice('data: '.length));

        if (event.type === 'token') {
          if (!answerEl) {
            wrapEl.innerHTML = '<div class="msg-answer streaming"></div>';
            answerEl = wrapEl.querySelector('.msg-answer');
          }
          answerText += event.content;
          scheduleRender();
        } else if (event.type === 'done') {
          setConversation(event.conversation_id);
          renderAnswerContent(wrapEl, answerText, event.sources);
        } else if (event.type === 'error') {
          renderAnswerContent(wrapEl, `Something went wrong: ${event.detail}`, []);
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      // Keep whatever was generated so far, marked as stopped. (The backend
      // only saves completed answers, so this one isn't in the history.)
      renderAnswerContent(wrapEl, answerText || '_Stopped before any answer was generated._', [], { stopped: true });
    }
    throw err;
  }
}

questionInput.addEventListener('input', autoGrow);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    composerForm.requestSubmit();
  }
});

function autoGrow() {
  questionInput.style.height = 'auto';
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
  questionInput.style.overflowY = questionInput.scrollHeight > 160 ? 'auto' : 'hidden';
}

function clearWelcomeCard() {
  const welcome = chatScroll.querySelector('.welcome-card');
  if (welcome) welcome.remove();
}

function appendQuestionBubble(question) {
  const turn = document.createElement('div');
  turn.className = 'turn';
  turn.innerHTML = `<div class="msg-question">${escapeHtml(question)}</div>`;
  chatScroll.appendChild(turn);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return turn;
}

function appendLoadingAnswer() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-answer-wrap';
  wrap.innerHTML = `<div class="msg-answer loading">Thinking through your documents…</div>`;
  chatScroll.lastElementChild.appendChild(wrap);
  chatScroll.scrollTop = chatScroll.scrollHeight;
  return wrap;
}

function renderMarkdown(text) {
  // marked + DOMPurify are loaded via CDN in index.html. Fall back to plain
  // escaped text if either failed to load (e.g. offline), so the app never
  // breaks -- it just shows literal markdown syntax instead of rendered HTML.
  if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
    return escapeHtml(text).replace(/\n/g, '<br>');
  }
  const rawHtml = marked.parse(text, { breaks: true });
  return DOMPurify.sanitize(rawHtml);
}

const SOURCES_ICON = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="8" y1="13" x2="16" y2="13"></line><line x1="8" y1="17" x2="13" y2="17"></line></svg>`;

// Sources are hidden by default; the small icon button or a double-click on the
// answer shows/hides them (with match %).
function toggleSources(wrapEl) {
  const sourcesEl = wrapEl.querySelector('.sources');
  const toggleBtn = wrapEl.querySelector('.sources-toggle');
  if (!sourcesEl || !toggleBtn) return;
  const show = sourcesEl.hidden;
  sourcesEl.hidden = !show;
  toggleBtn.setAttribute('aria-expanded', String(show));
  toggleBtn.classList.toggle('active', show);
  if (show) sourcesEl.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function formatPages(source) {
  if (!source.page_start) return '';
  const range = source.page_end && source.page_end !== source.page_start
    ? `pp. ${source.page_start}–${source.page_end}`
    : `p. ${source.page_start}`;
  return ` <span class="source-page">· ${range}</span>`;
}

function renderAnswerContent(wrapEl, answer, sources, { stopped = false } = {}) {
  let sourcesHtml = '';
  const hasSources = sources && sources.length > 0;
  if (hasSources) {
    const count = sources.length;
    sourcesHtml = `
      <div class="answer-actions">
        <button type="button" class="sources-toggle" aria-expanded="false" title="Show sources (or double-click the answer)">
          ${SOURCES_ICON}<span>${count} source${count === 1 ? '' : 's'}</span>
        </button>
      </div>
      <div class="sources" hidden>
        <p class="sources-label">Sources</p>
        ${sources.map(s => `
          <div class="source-card">
            <div class="source-card-top">
              <span class="source-filename">${escapeHtml(s.filename)}${formatPages(s)}</span>
              <span class="source-score">match ${(s.score * 100).toFixed(0)}%</span>
            </div>
            <p class="source-excerpt">${escapeHtml(s.chunk_text)}</p>
          </div>
        `).join('')}
      </div>`;
  }

  wrapEl.innerHTML = `
    <div class="msg-answer${hasSources ? ' has-sources' : ''}">${renderMarkdown(answer)}</div>
    ${stopped ? '<p class="answer-note">Stopped</p>' : ''}
    ${sourcesHtml}
  `;

  if (hasSources) {
    wrapEl.querySelector('.sources-toggle').addEventListener('click', () => toggleSources(wrapEl));
    wrapEl.querySelector('.msg-answer').addEventListener('dblclick', () => {
      // Double-click normally selects a word; drop that selection since it's a toggle here.
      window.getSelection()?.removeAllRanges();
      toggleSources(wrapEl);
    });
  }
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

// ==========================================================================
// Conversation persistence (survives a page refresh)
// ==========================================================================

function setConversation(conversationId) {
  state.conversationId = conversationId;
  storageSet(CONVERSATION_STORAGE_KEY, conversationId);
  conversationMeta.textContent = conversationId ? `Conversation ${conversationId.slice(0, 8)}` : 'New conversation';
}

async function restoreConversation() {
  const conversationId = storageGet(CONVERSATION_STORAGE_KEY);
  if (!conversationId) return;
  try {
    const res = await apiFetch(`/api/chat/history/${encodeURIComponent(conversationId)}`);
    if (res.status === 404) {
      // Expired, or the server's in-memory history was reset by a restart.
      setConversation(null);
      return;
    }
    if (!res.ok) return;
    const data = await res.json();
    if (!data.turns || data.turns.length === 0) return;

    clearWelcomeCard();
    for (const turn of data.turns) {
      appendQuestionBubble(turn.question);
      const wrap = appendLoadingAnswer();
      renderAnswerContent(wrap, turn.answer, turn.sources);
    }
    setConversation(conversationId);
  } catch (err) {
    console.error('Could not restore conversation', err);
  }
}

clearBtn.addEventListener('click', () => {
  if (state.streamController) state.streamController.abort();
  setConversation(null);
  chatScroll.innerHTML = WELCOME_HTML;
});

// ==========================================================================
// Init
// ==========================================================================

const session = currentSession();
if (session) userChip.textContent = session.username;

logoutBtn.addEventListener('click', async () => {
  try {
    await apiFetch('/api/auth/logout', { method: 'POST' });
  } catch (e) {
    // Logging out locally is what matters; ignore network errors here.
  }
  storageSet(CONVERSATION_STORAGE_KEY, null);
  redirectToLogin();
});

if (session) {
  chatScroll.innerHTML = WELCOME_HTML;
  checkHealth();
  loadDocuments();
  restoreConversation();
  setInterval(checkHealth, 20000);
}
