// ==========================================================================
// Archive — local document assistant frontend
// Talks to the FastAPI backend for upload, document listing, and chat.
// ==========================================================================

const API_BASE = window.location.origin.includes(':5500') || window.location.protocol === 'file:'
  ? 'http://localhost:8000'
  : window.location.origin;

const state = {
  conversationId: null,
  documents: [],
};

// ---- DOM refs ----
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
const clearBtn = document.getElementById('clearBtn');
const conversationMeta = document.getElementById('conversationMeta');

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
  try {
    const res = await fetch(`${API_BASE}/api/documents`);
    if (!res.ok) throw new Error('Failed to load documents');
    const data = await res.json();
    state.documents = data.documents || [];
    renderLibrary();
  } catch (err) {
    console.error(err);
  }
}

function renderLibrary() {
  docCount.textContent = `${state.documents.length} document${state.documents.length === 1 ? '' : 's'}`;

  if (state.documents.length === 0) {
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
        <p class="doc-meta">${doc.chunk_count} chunks · ${formatBytes(doc.file_size_bytes)}</p>
        <button class="doc-delete-btn" data-doc-id="${doc.document_id}" title="Delete this document">Delete</button>
      </div>
    </div>
  `).join('');

  libraryList.querySelectorAll('.doc-delete-btn').forEach(btn => {
    btn.addEventListener('click', () => deleteDocument(btn.dataset.docId));
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

async function deleteDocument(documentId) {
  if (!confirm('Delete this document? This removes it from the knowledge base permanently.')) return;
  try {
    const res = await fetch(`${API_BASE}/api/documents/${documentId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    await loadDocuments();
  } catch (err) {
    alert(`Could not delete document: ${err.message}`);
  }
}

const clearAllDocsBtn = document.getElementById('clearAllDocsBtn');
clearAllDocsBtn.addEventListener('click', async () => {
  if (state.documents.length === 0) return;
  if (!confirm(`Delete all ${state.documents.length} document(s)? This cannot be undone.`)) return;
  try {
    const res = await fetch(`${API_BASE}/api/documents`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Delete failed');
    await loadDocuments();
  } catch (err) {
    alert(`Could not delete documents: ${err.message}`);
  }
});

// ==========================================================================
// Upload flow (drag & drop + click)
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
  if (e.dataTransfer.files.length > 0) {
    uploadFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    uploadFile(fileInput.files[0]);
    fileInput.value = '';
  }
});

function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  uploadProgress.hidden = false;
  uploadProgressFill.style.width = '0%';
  uploadProgressLabel.textContent = `Uploading ${file.name}…`;

  const xhr = new XMLHttpRequest();
  xhr.open('POST', `${API_BASE}/api/documents/upload`);

  xhr.upload.addEventListener('progress', (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      uploadProgressFill.style.width = `${pct}%`;
      if (pct >= 100) {
        uploadProgressLabel.textContent = 'Processing and indexing…';
      }
    }
  });

  xhr.onload = () => {
    uploadProgress.hidden = true;
    if (xhr.status >= 200 && xhr.status < 300) {
      loadDocuments();
    } else {
      let detail = 'Upload failed';
      try { detail = JSON.parse(xhr.responseText).detail || detail; } catch (e) {}
      alert(detail);
    }
  };

  xhr.onerror = () => {
    uploadProgress.hidden = true;
    alert('Upload failed: could not reach the backend.');
  };

  xhr.send(formData);
}

// ==========================================================================
// Chat flow
// ==========================================================================

composerForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  clearWelcomeCard();
  appendQuestionBubble(question);
  questionInput.value = '';
  autoGrow();

  const loadingEl = appendLoadingAnswer();
  sendBtn.disabled = true;

  try {
    await streamAnswer(question, loadingEl);
  } catch (err) {
    renderAnswerContent(loadingEl, `Something went wrong: ${err.message}`, []);
  } finally {
    sendBtn.disabled = false;
  }
});

async function streamAnswer(question, wrapEl) {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      conversation_id: state.conversationId,
    }),
  });

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(err.detail || 'Request failed');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answerText = '';
  let firstToken = true;

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
        if (firstToken) {
          firstToken = false;
          wrapEl.innerHTML = '<div class="msg-answer"></div>';
        }
        answerText += event.content;
        wrapEl.querySelector('.msg-answer').textContent = answerText;
        chatScroll.scrollTop = chatScroll.scrollHeight;
      } else if (event.type === 'done') {
        state.conversationId = event.conversation_id;
        conversationMeta.textContent = `Conversation ${event.conversation_id.slice(0, 8)}`;
        renderAnswerContent(wrapEl, answerText, event.sources);
      } else if (event.type === 'error') {
        renderAnswerContent(wrapEl, `Something went wrong: ${event.detail}`, []);
      }
    }
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

function renderAnswerContent(wrapEl, answer, sources) {
  let sourcesHtml = '';
  if (sources && sources.length > 0) {
    sourcesHtml = `
      <div class="sources">
        <p class="sources-label">Sources</p>
        ${sources.map(s => `
          <div class="source-card">
            <div class="source-card-top">
              <span class="source-filename">${escapeHtml(s.filename)}</span>
              <span class="source-score">match ${(s.score * 100).toFixed(0)}%</span>
            </div>
            <p class="source-excerpt">${escapeHtml(s.chunk_text)}</p>
          </div>
        `).join('')}
      </div>`;
  }

  wrapEl.innerHTML = `
    <div class="msg-answer">${escapeHtml(answer)}</div>
    ${sourcesHtml}
  `;
  chatScroll.scrollTop = chatScroll.scrollHeight;
}

clearBtn.addEventListener('click', () => {
  state.conversationId = null;
  conversationMeta.textContent = 'New conversation';
  chatScroll.innerHTML = `
    <div class="welcome-card">
      <p class="welcome-eyebrow">Ask anything about your documents</p>
      <h3>What would you like to know?</h3>
      <p class="welcome-sub">Upload a document on the left, then ask a question. Every answer is grounded in your files, with the exact passages cited below it.</p>
    </div>`;
});

// ==========================================================================
// Init
// ==========================================================================

checkHealth();
loadDocuments();
setInterval(checkHealth, 20000);
