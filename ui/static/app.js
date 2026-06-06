// ──────────────────────────────────────────────────────────────────────────
//  Cluster Persona Studio — frontend
// ──────────────────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/** Minimum interval between automatic full UI re-renders (watchdog + evidence). */
const UI_REFRESH_INTERVAL_MS = 5 * 60 * 1000;

const state = {
  personas: {},
  profiles: {},
  summary: {},
  selectedForMerge: new Set(),
  openId: null,
  draft: null,           // unsaved edits for the currently open cluster
  pipelineRunning: false,
  runId: null,
  runStartedTs: null,    // Unix seconds — files older than this belong to prior runs
  events: [],
};

function _tsToSeconds(iso) {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? Math.floor(t / 1000) : null;
}

// ── Agent timeline (declared up front; rendered when live panel is shown)
const AGENT_STEPS = [
  {key: 'UserInput',        label: '① User Intent',         desc: 'Capture clustering goal'},
  {key: 'DatasetExaminer',  label: '② Dataset Examiner',    desc: 'Profile schema & propose feature groups'},
  {key: 'FeatureEngineer',  label: '③ Feature Engineer',    desc: 'Build entity-level features from raw transactions'},
  {key: 'FeatureSelector',  label: '④ Feature Selector',    desc: 'PCA + AE + VIF; LLM picks the subset'},
  {key: 'Clusterer',        label: '⑤ Clusterer',           desc: 'Silhouette-optimised clustering + deepening loop'},
  {key: 'PersonaNamer',     label: '⑥ Persona Namer',       desc: 'LLM names every cluster; Clarity Gate validates'},
  {key: 'Classifier',       label: '⑦ Classifier',          desc: 'CV F1 validates cluster separability'},
];

// ── API helpers ─────────────────────────────────────────────────────────────

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({error: res.statusText}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

function toast(msg, kind = 'success', ms = 2500) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = `toast ${kind}`;
  setTimeout(() => t.classList.add('hidden'), ms);
}

// ── Browser notification for interactive-mode pauses ────────────────────────
async function notifyUser(title, body) {
  if (!('Notification' in window)) return;
  if (Notification.permission === 'granted') {
    new Notification(title, { body, icon: '' });
  } else if (Notification.permission !== 'denied') {
    const perm = await Notification.requestPermission();
    if (perm === 'granted') {
      new Notification(title, { body, icon: '' });
    }
  }
}

// ── State loading + rendering ───────────────────────────────────────────────

async function loadState() {
  const s = await api('GET', '/api/state');
  state.personas = s.personas || {};
  state.profiles = s.profiles || {};
  state.summary = s.summary || {};
  renderSummary();
  renderGrid();
}

function renderSummary() {
  const s = state.summary;
  const n = s.n_clusters || 0;
  if (!n) {
    $('#summary').innerHTML = '';   // hide entirely when there's nothing to summarise
    return;
  }
  const f1 = s.cv_f1_macro != null ? Number(s.cv_f1_macro).toFixed(3) : 'n/a';
  $('#summary').innerHTML = `
    <b>${n}</b> clusters · <b>${s.total_entities || 0}</b> entities ·
    CV F1 (macro) <b>${f1}</b>
  `;
}

function renderGrid() {
  const grid = $('#cluster-grid');
  grid.innerHTML = '';
  const ids = Object.keys(state.personas).sort((a, b) => {
    const an = Number(a), bn = Number(b);
    return (isNaN(an) || isNaN(bn)) ? a.localeCompare(b) : an - bn;
  });
  if (!ids.length) {
    const hasCompletion = state.events.some(e => e.event === 'pipeline_complete');
    const running = state.pipelineRunning && !hasCompletion;
    grid.innerHTML = `
      <div class="cluster-empty">
        ${running
          ? '<b>Pipeline running…</b><br/>Named clusters from this run will appear here when it finishes.'
          : '<b>No clusters yet.</b><br/>Run the pipeline to generate named personas.'}
      </div>`;
    updateMergeBtn();
    return;
  }
  ids.forEach((cid) => grid.appendChild(renderCard(cid)));
  updateMergeBtn();
  autoScrollForDemo();
}

function renderCard(cid) {
  const data = state.personas[cid];
  const stats = data.cluster_stats || {};
  const p = data.persona || {};
  const n = stats.n_entities ?? stats.n_customers ?? 0;
  const pct = (stats.pct_total ?? (stats.pct_of_total || 0) * 100);

  const card = document.createElement('div');
  card.className = 'card';
  if (state.selectedForMerge.has(cid)) card.classList.add('selected');
  card.dataset.cid = cid;

  const above = Object.entries(stats.top_above_average || {}).slice(0, 3);
  const below = Object.entries(stats.top_below_average || {}).slice(0, 2);
  const maxRatio = Math.max(2.0, ...above.map(([, r]) => r));

  card.innerHTML = `
    <div class="checkbox" title="Select to merge">${state.selectedForMerge.has(cid) ? '✓' : ''}</div>
    <div class="id-pill">Cluster ${cid}</div>
    <h3>${escapeHtml(p.name || 'Unnamed')}</h3>
    <div class="tagline">${escapeHtml(p.tagline || '')}</div>
    <div class="stats">
      <span class="pill">${n} entities</span>
      <span class="pill">${Number(pct).toFixed(1)}%</span>
      <span class="pill">conf ${p.confidence ?? '—'}</span>
    </div>
    <div class="top-feats">
      ${above.map(([f, r]) => `
        <div class="bar">
          <span class="name" title="${f}">${escapeHtml(shortFeat(f))}</span>
          <div class="meter"><span style="width:${Math.min(100, (r / maxRatio) * 100)}%"></span></div>
          <span class="val">${r.toFixed(2)}×</span>
        </div>
      `).join('')}
      ${below.map(([f, r]) => `
        <div class="bar below">
          <span class="name" title="${f}">${escapeHtml(shortFeat(f))}</span>
          <div class="meter"><span style="width:${Math.min(100, (1 - r) * 100)}%"></span></div>
          <span class="val">${r.toFixed(2)}×</span>
        </div>
      `).join('')}
    </div>
  `;

  // checkbox toggles merge selection (without opening detail)
  card.querySelector('.checkbox').addEventListener('click', (e) => {
    e.stopPropagation();
    toggleMerge(cid);
  });
  card.addEventListener('click', () => openDetail(cid));
  return card;
}

function shortFeat(f) {
  return f.length > 28 ? f.slice(0, 27) + '…' : f;
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ── Merge selection ─────────────────────────────────────────────────────────

function toggleMerge(cid) {
  if (state.selectedForMerge.has(cid)) state.selectedForMerge.delete(cid);
  else state.selectedForMerge.add(cid);
  renderGrid();
}

function updateMergeBtn() {
  const n = state.selectedForMerge.size;
  const btn = $('#merge-btn');
  btn.textContent = `Merge selected (${n})`;
  btn.disabled = n < 2;
}

$('#merge-btn').addEventListener('click', async () => {
  const ids = Array.from(state.selectedForMerge);
  if (ids.length < 2) return;
  const hint = prompt(
    `Merging clusters ${ids.join(', ')} into one.\n` +
    `Optional: hint for the Decision Maker to name the merged cluster.\n` +
    `(Leave empty to let the LLM decide from the data.)`,
    ''
  );
  if (hint === null) return; // cancelled
  const btn = $('#merge-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Merging…`;
  try {
    const r = await api('POST', '/api/clusters/merge', {
      cluster_ids: ids,
      hint: hint.trim(),
      priority: 'high',
    });
    state.selectedForMerge.clear();
    toast(`Merged → "${r.persona.name}" (cluster ${r.new_cluster_id})`);
    await loadState();
    openDetail(r.new_cluster_id);
  } catch (e) {
    toast(e.message, 'error', 4500);
  } finally {
    btn.disabled = false;
    updateMergeBtn();
  }
});

// ── Detail panel ────────────────────────────────────────────────────────────

function openDetail(cid) {
  state.openId = cid;
  state.draft = JSON.parse(JSON.stringify(state.personas[cid].persona));
  renderDetail();
  $('#detail-panel').classList.remove('hidden');
}

$('#close-detail').addEventListener('click', () => {
  $('#detail-panel').classList.add('hidden');
  state.openId = null;
});

function renderDetail() {
  const cid = state.openId;
  const data = state.personas[cid];
  const stats = data.cluster_stats || {};
  const p = state.draft;
  const n = stats.n_entities ?? stats.n_customers ?? 0;
  const pct = (stats.pct_total ?? (stats.pct_of_total || 0) * 100);

  const traitsHtml = (p.traits || []).map((t, i) => `
    <div class="trait-row">
      <input type="text" value="${escapeHtml(t)}" data-trait-idx="${i}" />
      <button data-remove-trait="${i}" title="Remove">×</button>
    </div>
  `).join('');

  const above = Object.entries(stats.top_above_average || {});
  const below = Object.entries(stats.top_below_average || {});
  const maxR = Math.max(2.0, ...above.map(([, r]) => r));

  $('#detail-content').innerHTML = `
    <h2>Cluster ${cid} · <span style="color:var(--muted); font-weight:400">${n} entities (${Number(pct).toFixed(1)}%)</span></h2>

    <div class="field">
      <label>Persona name</label>
      <input id="f-name" type="text" value="${escapeHtml(p.name || '')}" />
    </div>
    <div class="field">
      <label>Tagline</label>
      <input id="f-tagline" type="text" value="${escapeHtml(p.tagline || '')}" />
    </div>
    <div class="field">
      <label>Description</label>
      <textarea id="f-description" rows="4">${escapeHtml(p.description || '')}</textarea>
    </div>
    <div class="field">
      <label>Traits</label>
      <div class="traits-list" id="traits-list">${traitsHtml}</div>
      <button class="ghost" id="add-trait" style="margin-top:6px">+ Add trait</button>
    </div>
    <div class="field">
      <label>Confidence</label>
      <div class="confidence">
        <input id="f-confidence" type="range" min="1" max="10" value="${p.confidence ?? 7}" />
        <span id="f-confidence-val">${p.confidence ?? 7}</span>
      </div>
    </div>

    <div class="regenerate-box">
      <label style="color:var(--accent-2)">Ask the Decision Maker to re-name (one-shot)</label>
      <textarea id="hint-text" rows="3"
        placeholder="e.g. 'Focus on dining behavior. Don't mention groceries.'"></textarea>
      <div class="row">
        <select id="hint-priority">
          <option value="high" selected>High priority</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
        <button class="primary" id="regen-btn">Regenerate with hint</button>
      </div>
    </div>

    <div class="chat-box">
      <label style="color:var(--good)">Discuss with agent (multi-turn · naming ledger)</label>
      <p class="muted" style="margin:4px 0 8px;font-size:11.5px">
        Ask why a trait was chosen, challenge a feature interpretation, or
        compare this cluster with another (e.g. <i>"why are clusters 0 and 3
        both high-spend, but 0 leans X while 3 leans Y?"</i>) — the agent can
        see every cluster's numbers. Each reply lands in the
        <b>Naming discussions</b> ledger, separate from the pipeline cost.
      </p>
      <div class="chat-thread" id="chat-thread"></div>
      <div class="chat-input-row">
        <textarea id="chat-input" rows="2"
          placeholder='e.g. "why do clusters 0 and 3 both look high-value but differ? which features?"'></textarea>
        <button class="primary" id="chat-send">Send</button>
      </div>
      <div class="chat-actions">
        <button class="ghost" id="chat-conclude">Conclude → propose action</button>
        <button class="ghost" id="chat-clear">Clear chat</button>
      </div>
      <div id="chat-proposal" class="chat-proposal hidden"></div>
    </div>

    <div class="section-title">Features ABOVE average</div>
    <div class="feature-list">
      ${above.slice(0, 10).map(([f, r]) => `
        <div class="bar">
          <span class="name" title="${escapeHtml(f)}">${escapeHtml(f)}</span>
          <div class="meter"><span style="width:${Math.min(100, (r / maxR) * 100)}%"></span></div>
          <span class="val">${r.toFixed(2)}×</span>
        </div>
      `).join('')}
    </div>
    <div class="section-title">Features BELOW average</div>
    <div class="feature-list">
      ${below.slice(0, 8).map(([f, r]) => `
        <div class="bar below">
          <span class="name" title="${escapeHtml(f)}">${escapeHtml(f)}</span>
          <div class="meter"><span style="width:${Math.min(100, (1 - r) * 100)}%"></span></div>
          <span class="val">${r.toFixed(2)}×</span>
        </div>
      `).join('')}
    </div>

    <div class="save-row">
      <button class="primary" id="save-btn">Save edits</button>
      <button class="ghost" id="discard-btn">Discard</button>
    </div>
  `;

  // Wire up live draft updates
  $('#f-name').oninput = (e) => state.draft.name = e.target.value;
  $('#f-tagline').oninput = (e) => state.draft.tagline = e.target.value;
  $('#f-description').oninput = (e) => state.draft.description = e.target.value;
  $('#f-confidence').oninput = (e) => {
    state.draft.confidence = Number(e.target.value);
    $('#f-confidence-val').textContent = e.target.value;
  };
  $$('#traits-list input').forEach((inp) => {
    inp.oninput = (e) => {
      const i = Number(e.target.dataset.traitIdx);
      state.draft.traits[i] = e.target.value;
    };
  });
  $$('#traits-list [data-remove-trait]').forEach((btn) => {
    btn.onclick = (e) => {
      const i = Number(e.target.dataset.removeTrait);
      state.draft.traits.splice(i, 1);
      renderDetail();
    };
  });
  $('#add-trait').onclick = () => {
    state.draft.traits = state.draft.traits || [];
    state.draft.traits.push('');
    renderDetail();
  };

  $('#save-btn').onclick = saveEdits;
  $('#discard-btn').onclick = () => {
    state.draft = JSON.parse(JSON.stringify(state.personas[cid].persona));
    renderDetail();
    toast('Discarded edits', 'success', 1500);
  };
  $('#regen-btn').onclick = regenerate;
  wireClusterChat(cid);
}

// ── Per-cluster chat (naming ledger) ───────────────────────────────────────
// Stateful in JS only — refresh wipes it. Keyed per cluster so switching
// detail panels resumes the conversation.
const _clusterChats = {};   // cid -> [{role, content}, ...]

function wireClusterChat(cid) {
  _clusterChats[cid] = _clusterChats[cid] || [];
  renderChatThread(cid);
  $('#chat-send').onclick = () => sendChat(cid, 'discuss');
  $('#chat-conclude').onclick = () => sendChat(cid, 'conclude');
  $('#chat-clear').onclick = () => {
    _clusterChats[cid] = [];
    document.getElementById('chat-proposal').classList.add('hidden');
    renderChatThread(cid);
  };
  $('#chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      sendChat(cid, 'discuss');
    }
  });
}

function renderChatThread(cid) {
  const thread = document.getElementById('chat-thread');
  const msgs = _clusterChats[cid] || [];
  if (!msgs.length) {
    thread.innerHTML = `<div class="muted" style="padding:10px;font-size:12px">No messages yet. Ask anything about this cluster's naming.</div>`;
    return;
  }
  thread.innerHTML = msgs.map(m => `
    <div class="chat-msg ${m.role}">
      <div class="chat-role">${m.role === 'user' ? 'You' : 'Agent'}</div>
      <div class="chat-content">${escapeHtml(m.content)}</div>
    </div>`).join('');
  thread.scrollTop = thread.scrollHeight;
}

async function sendChat(cid, mode) {
  const inputEl = document.getElementById('chat-input');
  const message = mode === 'conclude'
    ? (inputEl.value.trim() || 'Based on our discussion, what should we do?')
    : inputEl.value.trim();
  if (!message) {
    toast('Type a question first', 'error', 2500);
    return;
  }
  _clusterChats[cid].push({role: 'user', content: message});
  inputEl.value = '';
  renderChatThread(cid);
  const sendBtn = document.getElementById('chat-send');
  const concBtn = document.getElementById('chat-conclude');
  sendBtn.disabled = true; concBtn.disabled = true;
  sendBtn.innerHTML = `<span class="spinner"></span>${mode === 'conclude' ? 'Asking for conclusion…' : 'Asking agent…'}`;
  try {
    const r = await api('POST', '/api/cluster-chat', {
      cluster_id: cid,
      message,
      history: _clusterChats[cid].slice(0, -1),  // exclude the just-pushed user message
      mode,
    });
    _clusterChats[cid].push({role: 'assistant', content: r.reply || '(empty reply)'});
    renderChatThread(cid);
    if (mode === 'conclude' && r.proposal) {
      renderChatProposal(cid, r.proposal);
    }
  } catch (e) {
    _clusterChats[cid].push({role: 'assistant', content: `[error] ${e.message}`});
    renderChatThread(cid);
  } finally {
    sendBtn.disabled = false; concBtn.disabled = false;
    sendBtn.textContent = 'Send';
    concBtn.textContent = 'Conclude → propose action';
  }
}

function renderChatProposal(cid, p) {
  const box = document.getElementById('chat-proposal');
  if (!box) return;
  const action = p.action || 'keep';
  let actionBtnHtml = '';
  if (action === 'rename' && p.new_name) {
    actionBtnHtml = `<button class="primary" data-conclude="rename" data-name="${escapeHtml(p.new_name)}">Apply rename → "${escapeHtml(p.new_name)}"</button>`;
  } else if (action === 'merge' && p.merge_with) {
    actionBtnHtml = `<button class="primary" data-conclude="merge" data-with="${escapeHtml(p.merge_with)}">Merge with cluster ${escapeHtml(p.merge_with)}</button>`;
  } else if (action === 'keep') {
    actionBtnHtml = `<button class="ghost" data-conclude="keep">Keep as-is, close discussion</button>`;
  } else if (action === 'recluster') {
    actionBtnHtml = `<button class="ghost" data-conclude="recluster">Save guidance — next pipeline run will pick it up</button>`;
  }
  box.classList.remove('hidden');
  box.innerHTML = `
    <div class="chat-proposal-head">Agent's proposed conclusion · <b>${escapeHtml(action)}</b></div>
    <div class="chat-proposal-body">${escapeHtml(p.summary || '')}</div>
    ${p.reason ? `<div class="chat-proposal-reason"><b>Why:</b> ${escapeHtml(p.reason)}</div>` : ''}
    <div class="chat-proposal-actions">
      ${actionBtnHtml}
      <button class="ghost" data-conclude="dismiss">Keep discussing</button>
    </div>`;

  box.querySelectorAll('button[data-conclude]').forEach(btn => {
    btn.onclick = () => applyChatConclusion(cid, btn.dataset, p);
  });
}

async function applyChatConclusion(cid, ds, p) {
  const action = ds.conclude;
  try {
    if (action === 'rename') {
      const newName = ds.name || p.new_name;
      const r = await api('PUT', `/api/personas/${cid}`, {name: newName, priority: 'high'});
      state.personas[cid].persona = r.persona;
      // CRITICAL: refresh draft so the form fields show the new name; otherwise
      // a subsequent Save edits would push the stale draft name back over the rename.
      state.draft = JSON.parse(JSON.stringify(r.persona));
      // Persist the chat key-learnings as a high-priority memory rule so the next
      // pipeline run (and every agent that reads user_feedback_log) sees WHY the
      // rename happened — not just the before/after diff.
      const learning = [p.summary || '', p.reason ? `Why: ${p.reason}` : '']
        .filter(Boolean).join(' · ');
      if (learning) {
        try {
          await api('POST', '/api/feedback/global', {
            rule: `Cluster ${cid} renamed → "${newName}". Key learning: ${learning}`,
            priority: 'high',
          });
        } catch (_) { /* non-fatal — rename already saved */ }
      }
      renderGrid();
      renderDetail();
      document.getElementById('chat-proposal').classList.add('hidden');
      toast(`Renamed → "${newName}" · learning saved to memory`);
    } else if (action === 'merge') {
      const otherCid = ds.with;
      const r = await api('POST', '/api/clusters/merge', {
        cluster_ids: [cid, otherCid],
        hint: p.summary || '',
        priority: 'high',
      });
      toast(`Merged → "${r.persona.name}" (cluster ${r.new_cluster_id})`);
      await loadState();
      openDetail(r.new_cluster_id);
    } else if (action === 'recluster' || action === 'keep' || action === 'dismiss') {
      // Save the proposal text as a global rule so the next run sees it
      if (p.summary || p.reason) {
        await api('POST', '/api/feedback/global', {
          rule: `From cluster-${cid} chat: ${p.summary || ''} ${p.reason || ''}`.trim(),
          priority: 'high',
        });
      }
      document.getElementById('chat-proposal').classList.add('hidden');
      if (action !== 'dismiss') toast('Saved as memory rule for the next run', 'success', 3000);
    }
  } catch (e) {
    toast(e.message, 'error', 4500);
  }
}

async function saveEdits() {
  const cid = state.openId;
  const btn = $('#save-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Saving…`;
  try {
    const r = await api('PUT', `/api/personas/${cid}`, state.draft);
    state.personas[cid].persona = r.persona;
    renderGrid();
    toast(`Saved cluster ${cid}`);
  } catch (e) {
    toast(e.message, 'error', 4000);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save edits';
  }
}

async function regenerate() {
  const cid = state.openId;
  const hint = $('#hint-text').value.trim();
  if (!hint) { toast('Write a hint first', 'error', 2000); return; }
  const priority = $('#hint-priority').value;
  const btn = $('#regen-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Calling Decision Maker…`;
  try {
    const r = await api('POST', `/api/personas/${cid}/regenerate`, {hint, priority});
    state.personas[cid].persona = r.persona;
    state.draft = JSON.parse(JSON.stringify(r.persona));
    renderGrid();
    renderDetail();
    toast(`New name: "${r.persona.name}"`);
  } catch (e) {
    toast(e.message, 'error', 5000);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Regenerate with hint';
  }
}

// ── Memory drawer ──────────────────────────────────────────────────────────

const memState = {
  entries: [],
  filter: 'all',
};

$('#memory-btn').addEventListener('click', openMemory);
$('#close-memory').addEventListener('click', () => $('#memory-drawer').classList.add('hidden'));
$('#memory-drawer').addEventListener('click', (e) => {
  if (e.target.id === 'memory-drawer') $('#memory-drawer').classList.add('hidden');
});

// Filter chips
$$('#mem-filters .chip').forEach((chip) => {
  chip.onclick = () => {
    memState.filter = chip.dataset.filter;
    $$('#mem-filters .chip').forEach((c) => c.classList.toggle('on', c === chip));
    renderFeedback();
  };
});

// Inline "+ Add a new memory rule" form
$('#mem-add-btn').addEventListener('click', async () => {
  const rule = $('#mem-add-text').value.trim();
  if (!rule) { toast('Write a rule first', 'error', 2000); return; }
  const priority = $('#mem-add-priority').value;
  const btn = $('#mem-add-btn');
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Saving…`;
  try {
    await api('POST', '/api/feedback/global', {rule, priority});
    $('#mem-add-text').value = '';
    await refreshMemory();
    toast('Rule saved — agents will see it next run');
  } catch (e) {
    toast(e.message, 'error', 4000);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save rule';
  }
});

async function openMemory() {
  $('#memory-drawer').classList.remove('hidden');
  await refreshMemory();
}

async function refreshMemory() {
  const list = $('#feedback-list');
  list.innerHTML = `<div class="muted" style="padding:12px">Loading…</div>`;
  try {
    const [{entries}, {text}] = await Promise.all([
      api('GET', '/api/feedback'),
      api('GET', '/api/preferences-preview'),
    ]);
    memState.entries = entries || [];
    renderFeedback();
    $('#prefs-preview').textContent = text ||
      '(no active rules yet — add one above or edit a persona to start training the agents)';
  } catch (e) {
    list.innerHTML = `<div class="muted" style="padding:12px">Error: ${escapeHtml(e.message)}</div>`;
  }
}

function renderFeedback() {
  const list = $('#feedback-list');
  let entries = memState.entries.slice().reverse();   // newest first
  const filter = memState.filter;
  if (filter === 'inactive') {
    entries = entries.filter((e) => e.active === false);
  } else if (filter !== 'all') {
    entries = entries.filter((e) => e.type === filter);
  }
  $('#mem-count').textContent = `(${memState.entries.length})`;

  if (!entries.length) {
    list.innerHTML = renderEmptyState(memState.entries.length === 0);
    wireEmptyStateActions();
    return;
  }

  list.innerHTML = entries.map(renderRow).join('');
  wireRowActions();
}

function renderEmptyState(everEmpty) {
  if (everEmpty) {
    return `
      <div class="mem-empty">
        <b>No memory rules yet.</b><br/>
        Memory is built from three sources — try one:
        <ul>
          <li>Use the <b>+ Add a new memory rule</b> form above to write a global rule (e.g. <em>"Never use the word shopper"</em>).</li>
          <li>Click any cluster card → edit its name/tagline/description → Save. That's logged as a <em>manual edit</em>.</li>
          <li>Open a cluster → use <b>Regenerate with hint</b> to ask the Decision Maker to re-name it. That's a <em>naming hint</em>.</li>
        </ul>
        Each saved item appears here with a date, priority, and on/off controls.
      </div>`;
  }
  return `<div class="mem-empty">No entries match this filter.</div>`;
}

function wireEmptyStateActions() {
  // (Empty state has no buttons currently — leave a hook in case we add some.)
}

function renderRow(e) {
  const date = (e.date || '').slice(0, 10);
  const priority = (e.priority || 'medium').toUpperCase();
  const inactive = e.active === false ? 'inactive' : '';
  return `
    <div class="feedback-row ${inactive}" data-id="${e.id}">
      <div class="row-top">
        <span class="stamp">user_change : ${escapeHtml(priority)} : ${escapeHtml(date)}</span>
        <span class="tag ${e.type}">${typeLabel(e.type)}</span>
      </div>
      <div class="body">${describeEntry(e)}</div>
      <div class="row-controls">
        <span class="lbl">priority</span>
        <select data-priority="${e.id}">
          ${['high', 'medium', 'low'].map((p) =>
            `<option value="${p}" ${p === (e.priority || 'medium') ? 'selected' : ''}>${p}</option>`
          ).join('')}
        </select>
        <button class="row-btn" data-toggle="${e.id}">${e.active === false ? 'Enable' : 'Disable'}</button>
        <button class="row-btn danger" data-delete="${e.id}">Delete</button>
      </div>
    </div>`;
}

function typeLabel(t) {
  return ({
    'global_rule': 'Global rule',
    'naming_hint': 'Naming hint',
    'manual_override': 'Manual edit',
    'merge': 'Merge',
  })[t] || t;
}

function wireRowActions() {
  $$('#feedback-list select[data-priority]').forEach((sel) => {
    sel.onchange = async (ev) => {
      const id = ev.target.dataset.priority;
      try {
        await api('PATCH', `/api/feedback/${id}`, {priority: ev.target.value});
        await refreshMemory();
        toast('Priority updated');
      } catch (e) { toast(e.message, 'error', 4000); }
    };
  });
  $$('#feedback-list button[data-toggle]').forEach((b) => {
    b.onclick = async (ev) => {
      const id = ev.target.dataset.toggle;
      const row = $('#feedback-list').querySelector(`[data-id="${id}"]`);
      const nowActive = row.classList.contains('inactive');
      try {
        await api('PATCH', `/api/feedback/${id}`, {active: nowActive});
        await refreshMemory();
      } catch (e) { toast(e.message, 'error', 4000); }
    };
  });
  $$('#feedback-list button[data-delete]').forEach((b) => {
    b.onclick = async (ev) => {
      const id = ev.target.dataset.delete;
      if (!confirm('Delete this memory rule? This cannot be undone.')) return;
      try {
        await api('DELETE', `/api/feedback/${id}`);
        await refreshMemory();
        toast('Rule deleted');
      } catch (e) { toast(e.message, 'error', 4000); }
    };
  });
}

function describeEntry(e) {
  const target = escapeHtml(e.target_cluster_name || e.target_cluster_id || 'global');
  if (e.type === 'manual_override') {
    const keys = Object.keys(e.after || {});
    return `<b>${target}</b>: edited ${keys.map(escapeHtml).join(', ')}`;
  }
  if (e.type === 'naming_hint') {
    return `<b>${target}</b>: "${escapeHtml((e.hint || '').slice(0, 200))}"`;
  }
  if (e.type === 'merge') {
    return `Merged ${(e.merged_ids || []).join(' + ')} → <b>${target}</b>`
      + (e.hint ? `<br/><span class="muted">"${escapeHtml(e.hint)}"</span>` : '');
  }
  if (e.type === 'global_rule') {
    return `"${escapeHtml(e.rule || '')}"`;
  }
  return escapeHtml(JSON.stringify(e));
}

// ── Global rule modal ──────────────────────────────────────────────────────

$('#global-rule-btn').addEventListener('click', () => {
  $('#global-rule-text').value = '';
  $('#global-modal').classList.remove('hidden');
});
$('#global-rule-cancel').addEventListener('click', () => $('#global-modal').classList.add('hidden'));
$('#global-rule-save').addEventListener('click', async () => {
  const rule = $('#global-rule-text').value.trim();
  if (!rule) { toast('Write a rule first', 'error', 2000); return; }
  const priority = $('#global-rule-priority').value;
  try {
    await api('POST', '/api/feedback/global', {rule, priority});
    $('#global-modal').classList.add('hidden');
    toast('Global rule saved — agents will see it next run');
  } catch (e) { toast(e.message, 'error', 4000); }
});

// ── Live pipeline panel ─────────────────────────────────────────────────────

function showLivePanel(show) {
  // Legacy helper preserved for the awaiting/live event paths — it now
  // just ensures the Live tab is selected when we want to show the live
  // panel. The Named-clusters tab can be reached at any time.
  if (show) {
    selectView('live');
  }
  if (show) renderArchGraph();
}

// ── View tabs ──────────────────────────────────────────────────────────
function selectView(view) {
  $$('.view-tab').forEach((t) => t.classList.toggle('on', t.dataset.view === view));
  $('#live-pipeline').classList.toggle('hidden', view !== 'live');
  $('#cluster-grid').classList.toggle('hidden', view !== 'clusters');
  $('#evidence-view').classList.toggle('hidden', view !== 'evidence');
  if (view === 'live') renderArchGraph();
  if (view === 'evidence') renderEvidence();
}
function wireTabs() {
  $$('.view-tab').forEach((t) => { t.onclick = () => selectView(t.dataset.view); });
}
function updateTabCount() {
  const el = $('#tab-cluster-count');
  if (!el) return;
  const n = Object.keys(state.personas || {}).length;
  el.textContent = n ? `(${n})` : '';
}

// ── Architecture graph (SVG) ────────────────────────────────────────────
// Orchestrator at the top, 7 agents spread across a wide row below, with
// curved edges connecting each agent to the orchestrator.
const ARCH_VIEW = {w: 1120, h: 360};
const ARCH_ORCH = {x: ARCH_VIEW.w / 2, y: 32, w: 260, h: 64};
const ARCH_NODE = {w: 138, h: 52};

const ARCH_POSITIONS = (() => {
  const slots = AGENT_STEPS.length; // 7
  // Distribute the agents across the full width with comfortable padding.
  // Each node is 138 wide; gap between centres is computed from inner width.
  const pad = 90;                           // outer padding
  const innerW = ARCH_VIEW.w - 2 * pad;
  const step = slots > 1 ? innerW / (slots - 1) : 0;
  const baseY = 280;                        // row baseline
  return AGENT_STEPS.map((s, i) => ({
    key: s.key,
    label: s.label.replace(/^[①-⑦]\s*/, ''),     // strip the circled-number prefix
    short: s.label.replace(/^[①-⑦]\s*/, ''),
    x: pad + step * i,
    // Gentle U: the middle agents sit slightly higher so the row reads as a
    // graph, not a list. Edges show the flow clearly either way.
    y: baseY - Math.sin((i / (slots - 1)) * Math.PI) * 18,
  }));
})();

function renderArchGraph() {
  const svg = document.getElementById('arch-graph');
  if (!svg) return;
  svg.setAttribute('viewBox', `0 0 ${ARCH_VIEW.w} ${ARCH_VIEW.h}`);
  const halfW = ARCH_NODE.w / 2;
  const halfH = ARCH_NODE.h / 2;

  // Edges: smooth quadratic curve from the orchestrator's bottom edge
  // down to the top of each agent node, bowed outward for visual breathing.
  const edges = ARCH_POSITIONS.map((p) => {
    const ox = ARCH_ORCH.x;
    const oy = ARCH_ORCH.y + ARCH_ORCH.h;     // bottom edge of orch
    const tx = p.x;
    const ty = p.y - halfH;                   // top edge of agent
    // Control point: midway down, slightly biased to the agent's x so the
    // curve flares toward its target.
    const cpx = (ox + tx) / 2;
    const cpy = oy + (ty - oy) * 0.55;
    return `<path class="arch-edge pending" data-edge="${p.key}"
      d="M ${ox} ${oy} Q ${cpx} ${cpy} ${tx} ${ty}" />`;
  }).join('');

  const agents = ARCH_POSITIONS.map((p, i) => `
    <g data-node="${p.key}" transform="translate(${p.x}, ${p.y})">
      <rect class="arch-node-bg pending"
        x="-${halfW}" y="-${halfH}" width="${ARCH_NODE.w}" height="${ARCH_NODE.h}" rx="12"/>
      <text class="arch-node-label pending" y="-3">${escapeSvg(p.short)}</text>
      <text class="arch-node-label pending" y="15"
        style="font-size:10px;font-weight:500;opacity:.8">step ${i + 1}</text>
    </g>
  `).join('');

  svg.innerHTML = `
    <defs>
      <linearGradient id="orch-grad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#7aa2ff"/>
        <stop offset="100%" stop-color="#b08bff"/>
      </linearGradient>
    </defs>
    ${edges}
    <g transform="translate(${ARCH_ORCH.x}, ${ARCH_ORCH.y})">
      <rect class="arch-orch-bg" x="-${ARCH_ORCH.w/2}" y="0"
        width="${ARCH_ORCH.w}" height="${ARCH_ORCH.h}" rx="16"/>
      <text class="arch-orch-label" y="26" style="font-size:14px">⚙ Orchestrator</text>
      <text class="arch-orch-label" y="46"
        style="font-size:11px;font-weight:500;opacity:.85">decision maker + LLM gateway</text>
    </g>
    ${agents}
  `;
}

function escapeSvg(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;',
  }[c]));
}

function setArchNodeStatus(agentKey, status, isActive) {
  const node = document.querySelector(`#arch-graph g[data-node="${agentKey}"]`);
  const edge = document.querySelector(`#arch-graph path[data-edge="${agentKey}"]`);
  if (!node || !edge) return;
  const bg = node.querySelector('rect');
  const labels = node.querySelectorAll('text');
  const effective = isActive ? 'running' : status;
  // Only re-apply the class if it actually changed — otherwise we kill the
  // CSS animation by restarting it from frame 0 every time an event arrives.
  if (bg.dataset.statusClass === effective) return;
  bg.dataset.statusClass = effective;
  ['pending', 'running', 'success', 'warning', 'blocked', 'failure'].forEach((c) => {
    bg.classList.remove(c);
    labels.forEach((l) => l.classList.remove(c));
    edge.classList.remove(c);
  });
  bg.classList.add(effective);
  labels.forEach((l) => l.classList.add(effective));
  edge.classList.add(effective);
}

// Drive the architecture graph from the full event stream. Two key signals:
//   - agent_report → that agent's terminal status (success/warning/blocked)
//   - llm_call_started/finished → that agent is in-flight talking to Decision Maker
// We also highlight the orchestrator node whenever ANY LLM call is in flight,
// and the next-in-sequence agent gets a "running" pulse while it's working
// (its agent_report hasn't fired yet but its predecessor has).
function applyArchFromEvents() {
  const reported = {};       // agent → latest reported status
  const llmInFlight = {};    // agent → count of started minus finished
  let lastReportedAgent = null;  // most-RECENT agent report (handles loops)
  let pipelineDone = false;

  for (const e of state.events) {
    if (e.event === 'agent_report') {
      reported[e.agent] = e.status || 'success';
      lastReportedAgent = e.agent;
    } else if (e.event === 'llm_call_started') {
      llmInFlight[e.agent] = (llmInFlight[e.agent] || 0) + 1;
    } else if (e.event === 'llm_call_finished') {
      llmInFlight[e.agent] = Math.max(0, (llmInFlight[e.agent] || 0) - 1);
    } else if (e.event === 'pipeline_complete') {
      pipelineDone = true;
    } else if (e.event === 'iteration_started') {
      // A new iteration starts → predict the FIRST agent of the loop will run next
      // (only relevant if no agent reports yet in this iteration window)
      lastReportedAgent = null;
    } else if (e.event === 'feature_re_engineering') {
      // Re-engineering escalation → FeatureEngineer is about to re-run
      lastReportedAgent = null;
    }
  }
  // Convert lastReportedAgent into the next-in-sequence agent index
  const lastIdx = lastReportedAgent != null
    ? AGENT_STEPS.findIndex(a => a.key === lastReportedAgent)
    : -1;
  // The standard pipeline loop is: FeatureSelector → Clusterer → PersonaNamer → Classifier
  // Step 0/1/2 (UserInput / DatasetExaminer / FeatureEngineer) run once at the start.
  // Within the loop (idx 3-6), after Classifier the next is FeatureSelector again.
  let nextIdx;
  if (lastIdx < 0) {
    nextIdx = 0;                    // beginning of pipeline → UserInput
  } else if (lastIdx >= 3 && lastIdx < 6) {
    nextIdx = lastIdx + 1;          // within main loop, next step
  } else if (lastIdx === 6) {
    nextIdx = 3;                    // after Classifier → back to FeatureSelector
  } else if (lastIdx < 3) {
    nextIdx = lastIdx + 1;          // intro steps
  } else {
    nextIdx = -1;
  }

  // Decision Maker is "thinking" when:
  //   (a) any agent has an in-flight LLM call (active reasoning), OR
  //   (b) the pipeline is alive between agents (about to route)
  const anyInFlight = Object.values(llmInFlight).some(c => c > 0);
  const orchThinking = state.pipelineRunning && !pipelineDone;
  const orchActive   = state.pipelineRunning && !pipelineDone && anyInFlight;

  AGENT_STEPS.forEach((s, idx) => {
    let st;
    if ((llmInFlight[s.key] || 0) > 0) {
      // Agent is awaiting an LLM response right now
      st = 'running';
    } else if (state.pipelineRunning && !pipelineDone && idx === nextIdx) {
      // The next-in-sequence agent (handles loops + escalations correctly)
      st = 'running';
    } else if (reported[s.key]) {
      st = reported[s.key];
    } else {
      st = 'pending';
    }

    const isActive = st === 'running';
    setArchNodeStatus(s.key, st, isActive);
  });

  setOrchestratorThinking(orchThinking, orchActive);
}

function setOrchestratorThinking(alive, activeInFlight) {
  const orchRect = document.querySelector('#arch-graph .arch-orch-bg');
  if (!orchRect) return;
  const want = `${alive ? 'T' : ''}${activeInFlight ? 'A' : ''}`;
  if (orchRect.dataset.orchClass === want) return;
  orchRect.dataset.orchClass = want;
  orchRect.classList.toggle('thinking', !!alive);
  orchRect.classList.toggle('llm-active', !!activeInFlight);
}

function renderTimeline() {
  const ol = $('#agent-timeline');
  ol.innerHTML = AGENT_STEPS.map((s, i) => `
    <li class="pending" data-agent="${s.key}">
      <div class="step-num">${i + 1}</div>
      <div class="step-body">
        <div class="step-name">${escapeHtml(s.label)}</div>
        <div class="step-detail">${escapeHtml(s.desc)}</div>
        <div class="step-gates"></div>
      </div>
      <div class="step-meta">waiting</div>
    </li>
  `).join('');
}

// Map an agent_report event into a list of gate chips for that agent.
function gatesForAgent(agentKey, status, metrics, issues) {
  const m = metrics || {};
  const out = [];
  const push = (label, kind) => out.push({label, kind});

  if (agentKey === 'FeatureSelector') {
    if (m.max_vif_remaining != null)
      push(`VIF max ${Number(m.max_vif_remaining).toFixed(1)}`,
           m.max_vif_remaining <= 10 ? 'pass' : 'warn');
    if (m.n_selected != null)
      push(`${m.n_selected} features picked`, 'pass');
  } else if (agentKey === 'Clusterer') {
    if (m.silhouette != null) {
      const sil = Number(m.silhouette);
      push(`silhouette ${sil.toFixed(3)}`,
           sil >= 0.25 ? 'pass' : sil >= 0.15 ? 'warn' : 'fail');
    }
    // AutoML composite score chip
    const candBest = m.candidate_search?.best;
    if (candBest?.composite_score != null) {
      const comp = Number(candBest.composite_score);
      push(`AutoML comp ${comp.toFixed(3)}`, comp >= 0.6 ? 'pass' : comp >= 0.4 ? 'warn' : 'fail');
    }
    if (m.n_leaf_clusters != null || m.n_clusters != null)
      push(`${m.n_leaf_clusters || m.n_clusters} clusters`, 'pass');
    // 40% guard
    const oversized = (issues || []).find((s) => /40%|threshold/i.test(s));
    if (oversized) push('40% size guard ⚠', 'fail');
    else push('40% size guard ✓', 'pass');
  } else if (agentKey === 'PersonaNamer') {
    if (m.avg_confidence != null) {
      const c = Number(m.avg_confidence);
      push(`avg confidence ${c.toFixed(1)}/10`,
           c >= 6 ? 'pass' : 'fail');
    }
    if (m.gate_passed != null)
      push(`Clarity Gate ${m.gate_passed ? '✓' : '✗'}`,
           m.gate_passed ? 'pass' : 'fail');
    if (m.names_unique != null)
      push(`unique names ${m.names_unique ? '✓' : '✗'}`,
           m.names_unique ? 'pass' : 'fail');
  } else if (agentKey === 'Classifier') {
    // F1 is the gate metric — show it as the headline number. Accuracy is misleading
    // on imbalanced clusters (could be 0.96 while small clusters score F1=0) so we
    // intentionally do NOT surface cv_accuracy as a separate "pass" chip.
    if (m.cv_f1_macro != null) {
      const f1 = Number(m.cv_f1_macro);
      push(`CV F1 (macro) ${f1.toFixed(3)} · gate ≥ 0.70`,
           f1 >= 0.70 ? 'pass' : 'fail');
    }
  } else if (agentKey === 'DatasetExaminer') {
    if (m.n_rows != null) push(`${m.n_rows.toLocaleString()} rows`, 'pass');
    if (m.mean_skewness != null)
      push(`skew ${Number(m.mean_skewness).toFixed(1)}`,
           m.mean_skewness > 3 ? 'warn' : 'pass');
  } else if (agentKey === 'FeatureEngineer') {
    if (m.n_features != null) push(`${m.n_features} features built`, 'pass');
    if (m.n_entities != null) push(`${m.n_entities} entities`, 'pass');
  } else if (agentKey === 'UserInput') {
    if (m.target_entity) push(`target: ${m.target_entity}`, 'pass');
    if (m.n_clusters_requested) push(`k=${m.n_clusters_requested}`, 'pass');
  }
  if (status === 'blocked' || status === 'failure') {
    push('blocked', 'fail');
  }
  return out;
}

function renderGateChips(container, gates) {
  container.innerHTML = gates.map(
    (g) => `<span class="gate-chip ${g.kind}">${escapeHtml(g.label)}</span>`
  ).join('');
}

function setActiveSpotlight(agentKey, gates) {
  const sp = $('#active-spotlight');
  if (!agentKey) {
    sp.classList.add('hidden');
    return;
  }
  const idx = AGENT_STEPS.findIndex((s) => s.key === agentKey);
  if (idx < 0) return;
  const step = AGENT_STEPS[idx];
  $('#spot-num').textContent = idx + 1;
  $('#spot-name').textContent = step.label;
  $('#spot-detail').textContent = step.desc;
  renderGateChips($('#spot-gates'), gates || []);
  sp.classList.remove('hidden');
}

function setStepStatus(agentKey, status, detail, iteration, metrics, issues) {
  const li = document.querySelector(`#agent-timeline li[data-agent="${agentKey}"]`);
  if (!li) return;
  li.classList.remove('pending', 'running', 'success', 'warning', 'blocked', 'failure');
  li.classList.add(status);
  if (detail) {
    li.querySelector('.step-detail').innerHTML = escapeHtml(detail);
  }
  const meta = li.querySelector('.step-meta');
  meta.textContent = iteration != null
    ? `iter ${iteration} · ${status}`
    : status;
  const gates = gatesForAgent(agentKey, status, metrics, issues);
  renderGateChips(li.querySelector('.step-gates'), gates);

  // If this is the most-recent in-flight or just-completed agent, mirror to spotlight.
  setActiveSpotlight(agentKey, gates);
}

function appendLogLine(line) {
  const pre = $('#live-log-pre');
  if (!pre) return;
  pre.textContent += line + '\n';
  pre.scrollTop = pre.scrollHeight;
}

// In demo-mode recordings (Playwright captures only the viewport, no real user
// to scroll), force the focused container to scroll so newly-appended content
// is always visible. Without this, long pipeline runs look frozen in the
// recording because all the action is happening below the fold.
function autoScrollForDemo() {
  const cls = document.body.classList;
  if (!cls.contains('demo-mode')) return;
  const TARGETS = {
    'demo-outputs':  '.col-right',
    'demo-evidence': '#evidence-view',
    'demo-convos':   '#convos',
    'demo-log':      '#live-log',
    'demo-named':    '#cluster-grid',
  };
  for (const [klass, sel] of Object.entries(TARGETS)) {
    if (!cls.contains(klass)) continue;
    const el = document.querySelector(sel);
    if (el) el.scrollTop = el.scrollHeight;
    // Some demo modes (#evidence-view, #cluster-grid) don't have their own
    // overflow:auto — the body scrolls instead. Push the window to the bottom
    // too so newly-appended content is captured by the recording either way.
    window.scrollTo(0, document.documentElement.scrollHeight);
  }
}

// ── Evidence tab ─────────────────────────────────────────────────────────
async function renderEvidence() {
  const wrap = document.getElementById('evidence-grid');
  if (!wrap) return;
  wrap.innerHTML = `<div class="ev-empty">Loading evidence…</div>`;
  let agg = {};
  try { agg = await api('GET', '/api/evidence'); } catch (_) { agg = {}; }

  const cards = [];

  // Final pipeline summary — pinned at the TOP when the run has completed.
  // Cross-iteration token usage, time, silhouette / F1 / cluster counts,
  // with the winning iteration highlighted.
  const completion = state.events.slice().reverse().find(e => e.event === 'pipeline_complete');
  if (completion) cards.push(buildFinalSummaryCard(completion));

  // Threshold decisions that were auto-applied in bypass mode — pinned high
  // so the user sees what was relaxed before scrolling through the rest.
  const autoCard = buildAutoAppliedDecisionsCard();
  if (autoCard) cards.push(autoCard);

  // Uploaded dataset preview (raw input, before any agent runs)
  if (agg.upload_preview) cards.push(buildUploadPreviewCard(agg.upload_preview));

  // Dataset state AFTER FeatureEngineer / FeatureSelector ran.
  // Always shown alongside the original upload so the user can compare.
  if (agg.pre_modelling_preview) cards.push(buildPreModellingPreviewCard(agg.pre_modelling_preview));

  // Dataset profile — populates immediately from upload preview, then gets
  // enriched (suggested feature groups, skewness, missing) when DatasetExaminer runs.
  const dse = outputsState.DatasetExaminer?.latest;
  cards.push(buildDatasetCard(dse, agg.upload_preview));
  const rowCountNote = buildDatasetRowCountNoteCard(dse, agg.upload_preview);
  if (rowCountNote) cards.push(rowCountNote);

  if (dse) cards.push(buildSkewCard(dse));
  if (outputsState.DatasetExaminer?.history?.length)
    cards.push(buildFeatureGroupsHistoryCard(outputsState.DatasetExaminer.history));

  // The literal evidence behind the skewness warning: per-column skew bars
  if (agg.upload_preview && agg.upload_preview.preview && agg.upload_preview.preview.col_stats) {
    cards.push(buildPerColumnSkewCard(agg.upload_preview.preview));
    cards.push(buildEvidenceHistogramsCard(agg.upload_preview.preview));
    cards.push(buildPerColumnMissingCard(agg.upload_preview.preview));
  }
  // Per-agent iteration history — every iteration's output is preserved
  if (outputsState.FeatureEngineer?.history?.length)
    cards.push(buildAgentHistoryCard('FeatureEngineer', 'Feature engineering', outputsState.FeatureEngineer.history));
  if (outputsState.FeatureSelector?.history?.length)
    cards.push(buildAgentHistoryCard('FeatureSelector', 'Feature selection (PCA + AE + VIF)', outputsState.FeatureSelector.history));
  if (outputsState.Clusterer?.history?.length) {
    // Pair each Clusterer iteration with its PCA snapshot (if available)
    cards.push(buildAgentHistoryCard('Clusterer', 'Clustering — per iteration (with PCA projection)',
                                     outputsState.Clusterer.history, agg.pca_iterations));
  }
  // PersonaNamer + Classifier — show ONLY the final/best iteration, not all attempts
  if (outputsState.PersonaNamer?.history?.length) {
    const namingHistory = outputsState.PersonaNamer.history;
    const passed = namingHistory.filter(h => h.metrics?.gate_passed);
    const best = passed.length
      ? passed.reduce((a, b) => (b.metrics?.avg_confidence || 0) > (a.metrics?.avg_confidence || 0) ? b : a)
      : namingHistory[namingHistory.length - 1];
    cards.push(buildAgentHistoryCard('PersonaNamer', 'Persona naming — final result', [best]));
  }
  if (outputsState.Classifier?.history?.length) {
    const clfHistory = outputsState.Classifier.history;
    const best = clfHistory.reduce((a, b) =>
      (b.metrics?.cv_f1_macro || 0) > (a.metrics?.cv_f1_macro || 0) ? b : a);
    cards.push(buildAgentHistoryCard('Classifier', 'Classifier validation — best result', [best]));
  }

  // Stale-output filter: only show saved JSON outputs (silhouette_curve etc)
  // if THIS run has completed. Otherwise they're left over from a prior run.
  const hasCurrentCompletion = state.events.some(e => e.event === 'pipeline_complete');
  // If we know when the current run started, also require the completion to
  // belong to it. This catches the case where a stale pipeline_complete from
  // before the current restart is still in the events log.
  const showSavedOutputs = (state.runStartedTs && hasCurrentCompletion) ||
                           (!state.pipelineRunning && !state.runStartedTs);

  if (showSavedOutputs) {
    if (agg.silhouette_curve)       cards.push(buildSilhouetteCard(agg.silhouette_curve));
    if (agg.cluster_sizes)          cards.push(buildClusterSizeCard(agg.cluster_sizes));
    // Cross-cluster comparison / contrasting analysis (LLM, evidence ledger).
    if (Object.keys(state.personas || {}).length >= 2)
      cards.push(buildClusterComparisonCard());
    if (agg.lineage)                cards.push(buildLineageCard(agg.lineage));
    if (agg.classifier)             cards.push(buildClassifierCard(agg.classifier));
    if (agg.classifier && agg.classifier.top20_features)
      cards.push(buildTopFeaturesCard(agg.classifier.top20_features));
  } else {
    cards.push(`<div class="ev-card span2 ev-pending">
      <h3>Cluster results <span class="iter">running…</span></h3>
      <p class="lead">Silhouette curve, cluster sizes, classifier F1 and lineage will appear here when the pipeline finishes.</p>
    </div>`);
  }

  // Raw output files for transparency
  cards.push(`<div class="ev-card span2" id="ev-outputs-list-card">
    <h3>All pipeline outputs <span class="iter">on disk in outputs/</span></h3>
    <p class="lead">Every file the pipeline writes is listed here. Click to view the raw JSON / text.</p>
    <div id="ev-outputs-list" class="ev-files-list muted"></div>
  </div>`);

  wrap.innerHTML = cards.join('');

  // Lazy-load the outputs file list (separate endpoint)
  loadOutputsFiles();
  wireExplainButtons();
  wireComparisonButton();
  autoScrollForDemo();
}

// In-flight state for explain calls, keyed by `agent::issue`. Same rationale
// as _comparisonState: renderEvidence rebuilds the panel on every agent_report,
// so closure-captured DOM refs go stale mid-call.
const _explainState = new Map();  // key -> {resultHTML, inFlight, outId}

function wireExplainButtons() {
  document.querySelectorAll('.explain-btn[data-explain-issue]').forEach((btn) => {
    const agent = btn.dataset.explainAgent || 'unknown';
    const issue = btn.dataset.explainIssue || '';
    const key = `${agent}::${issue}`;
    const outId = btn.nextElementSibling && btn.nextElementSibling.id;

    // Re-attach to any prior call after a panel rebuild.
    const prior = _explainState.get(key);
    if (prior && outId) {
      const freshOut = document.getElementById(outId);
      if (prior.resultHTML && freshOut) {
        freshOut.hidden = false;
        freshOut.innerHTML = prior.resultHTML;
        btn.textContent = 'Explain again';
      }
      if (prior.inFlight) {
        btn.disabled = true;
        btn.innerHTML = `<span class="spinner"></span>Asking LLM (evidence ledger)…`;
      }
    }

    btn.onclick = async () => {
      if (_explainState.get(key)?.inFlight) return;  // guard double-fires after rebuild
      let evidence = {};
      try { evidence = JSON.parse(btn.dataset.explainEvidence || '{}'); } catch (_) {}
      const out = btn.nextElementSibling;
      _explainState.set(key, {inFlight: true, resultHTML: null, outId});
      btn.disabled = true;
      btn.innerHTML = `<span class="spinner"></span>Asking LLM (evidence ledger)…`;
      let html;
      let isError = false;
      try {
        const r = await api('POST', '/api/explain', {agent, issue, evidence});
        // Backend may return either {decision, reasoning, visual_to_check}
        // (the structured contract) OR {explanation, visual_to_check} (the
        // fallback when LLM output isn't valid JSON). Render whichever
        // fields are present so the user always sees something.
        const decision = r.decision || '';
        const reasoning = r.reasoning || '';
        const fallback = r.explanation || '';
        const visual = r.visual_to_check || '';
        const error = r.error || '';
        const cached = r._cached ? ' <span class="muted">· cached</span>' : '';
        if (error) {
          html = `<div class="muted" style="color:var(--bad)">Explain failed: ${escapeHtml(error)}</div>`;
          isError = true;
        } else {
          const sections = [];
          if (decision) sections.push(`
            <div class="explain-section">
              <div class="muted">Agent's decision${cached}</div>
              <div>${escapeHtml(decision)}</div>
            </div>`);
          if (reasoning) sections.push(`
            <div class="explain-section">
              <div class="muted">Why</div>
              <div>${escapeHtml(reasoning)}</div>
            </div>`);
          if (!decision && !reasoning && fallback) sections.push(`
            <div class="explain-section">
              <div class="muted">Plain-English explanation${cached}</div>
              <div>${escapeHtml(fallback)}</div>
            </div>`);
          if (visual) sections.push(`
            <div class="explain-section">
              <div class="muted">Which visual confirms it</div>
              <div>${escapeHtml(visual)}</div>
            </div>`);
          if (!sections.length) sections.push(`
            <div class="muted" style="color:var(--bad)">
              LLM returned an empty response — try again or check the server log.
            </div>`);
          html = `
            <div class="explain-card">
              ${sections.join('')}
              <div class="muted" style="font-size:10.5px;margin-top:6px">
                ✓ This cost was added to the <b>Evidence</b> ledger (not the Pipeline ledger).
              </div>
            </div>`;
        }
      } catch (e) {
        html = `<div class="muted" style="color:var(--bad)">Explain failed: ${escapeHtml(e.message)}</div>`;
        isError = true;
      }
      // Cache successful results so a panel rebuild can repaint them; clear
      // the in-flight flag either way.
      _explainState.set(key, {
        inFlight: false,
        resultHTML: isError ? null : html,
        outId,
      });
      // Re-look-up by id: renderEvidence may have rebuilt the panel while we
      // awaited, leaving our closure-captured `out` / `btn` refs detached.
      // The out div has a stable id; the button is its previous sibling.
      const freshOut = outId ? document.getElementById(outId) : out;
      const freshBtn = (freshOut && freshOut.previousElementSibling
                       && freshOut.previousElementSibling.classList.contains('explain-btn'))
        ? freshOut.previousElementSibling
        : btn;
      if (freshOut) {
        freshOut.hidden = false;
        freshOut.innerHTML = html;
      }
      if (freshBtn) {
        freshBtn.disabled = false;
        freshBtn.textContent = 'Explain again';
      }
    };
  });
}

async function loadOutputsFiles() {
  const target = document.getElementById('ev-outputs-list');
  if (!target) return;
  try {
    const {files} = await api('GET', '/api/outputs-files');
    // Only show files produced by THIS pipeline run. Anything older is from
    // a previous run and is intentionally hidden until the next pipeline_complete.
    const cutoff = state.runStartedTs;
    let visible = files || [];
    if (cutoff) {
      visible = visible.filter(f => f.mtime >= cutoff);
    }
    // Always hide the live event log + pending-state files from this list
    // (they're consumed via SSE / the form, not for browsing)
    const HIDE = new Set([
      'pipeline_events.jsonl', 'pending_intent.json',
      'pending_decision.json', 'pending_target_change.json',
      'pending_case_recall.json',
      'pipeline_mode.json',
    ]);
    visible = visible.filter(f => !HIDE.has(f.name));

    if (!visible.length) {
      const hasCompletion = state.events.some(e => e.event === 'pipeline_complete');
      target.innerHTML = `<div class="muted">${
        cutoff && !hasCompletion
          ? 'Pipeline running — no files saved yet for this run. Outputs from previous runs are intentionally hidden.'
          : 'No files yet for this run.'
      }</div>`;
      return;
    }
    target.innerHTML = visible.map(f => {
      const d = new Date(f.mtime * 1000);
      const when = d.toLocaleString('en-GB', {hour12: false});
      return `
        <a class="ev-file" href="/api/outputs-file/${encodeURIComponent(f.name)}" target="_blank">
          <span class="fname">${escapeHtml(f.name)}</span>
          <span class="fmeta">${escapeHtml(f.size_human)} · ${escapeHtml(when)}</span>
        </a>`;
    }).join('');
  } catch (e) {
    target.innerHTML = `<div class="muted">Error loading: ${escapeHtml(e.message)}</div>`;
  }
}

function buildPerColumnSkewCard(preview) {
  const numericCols = (preview.col_stats || []).filter(c => c.numeric && c.skew != null);
  if (!numericCols.length) {
    return `<div class="ev-card">
      <h3>Per-column skewness <span class="iter">no numeric columns</span></h3>
      <p class="lead">The dataset has no numeric columns to measure skewness on.</p>
    </div>`;
  }
  const sorted = numericCols.slice().sort((a, b) => Math.abs(b.skew) - Math.abs(a.skew));
  const max = Math.max(3.5, ...sorted.map(c => Math.abs(c.skew)));
  return `
    <div class="ev-card span2">
      <h3>Per-column skewness <span class="iter">evidence behind the warning · sampled ${preview.stats_sample_size?.toLocaleString?.() || ''} rows</span></h3>
      <p class="lead">|skew| &gt; 3 is the "high skew" threshold. The thumbnails on the right are the actual distributions — heavy right-tail visible for the worst columns.</p>
      <div class="ev-bars">
        ${sorted.map(c => {
          const v = Math.abs(c.skew);
          const k = v > 3 ? 'bad' : v > 1.5 ? 'warn' : 'good';
          const sign = c.skew >= 0 ? '+' : '';
          const histSvg = c.histogram ? renderMiniHistogram(c.histogram, k, 120, 28) : '';
          return `
            <div class="bar bar-with-hist">
              <span class="name" title="${escapeHtml(c.name)} (${escapeHtml(c.dtype)})">${escapeHtml(c.name)}</span>
              <div class="meter ${k}"><span style="width:${(v / max) * 100}%"></span></div>
              <span class="val">${sign}${c.skew.toFixed(2)}</span>
              <span class="hist">${histSvg}</span>
            </div>`;
        }).join('')}
      </div>
      <p class="lead" style="margin-top:8px">Bars are colored: <b style="color:var(--good)">good</b> &lt; 1.5 · <b style="color:var(--warn)">warn</b> 1.5–3 · <b style="color:var(--bad)">bad</b> &gt; 3.</p>
    </div>`;
}

// Tiny inline histogram (no axes, just the shape) used in the skew bar list.
function renderMiniHistogram(hist, kind, W, H) {
  const counts = hist.counts || [];
  if (!counts.length) return '';
  const max = Math.max(...counts);
  const bw = W / counts.length;
  const color = kind === 'bad' ? '#ff7a8a'
              : kind === 'warn' ? '#ffba6b'
              : '#4fd1a1';
  const bars = counts.map((c, i) => {
    const h = max > 0 ? (c / max) * H : 0;
    const x = i * bw;
    return `<rect x="${x.toFixed(2)}" y="${(H - h).toFixed(2)}" width="${(bw - 0.5).toFixed(2)}" height="${h.toFixed(2)}" fill="${color}" opacity="0.85"/>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="display:block">${bars}</svg>`;
}

// Larger card: the top-3 worst-skew columns with full-size histograms
function buildEvidenceHistogramsCard(preview) {
  const cols = (preview.col_stats || [])
    .filter(c => c.numeric && c.skew != null && Math.abs(c.skew) > 3 && c.histogram);
  if (!cols.length) return '';
  cols.sort((a, b) => Math.abs(b.skew) - Math.abs(a.skew));
  const top = cols.slice(0, 3);
  return `
    <div class="ev-card span2">
      <h3>Distribution evidence for the warning <span class="iter">top ${top.length} worst-skew columns</span></h3>
      <p class="lead">These are the actual sample distributions. A long right tail = high positive skew = log-transform candidate.</p>
      <div class="dist-grid">
        ${top.map(c => {
          const s = c.stats || {};
          return `
            <div class="dist-cell">
              <div class="dist-head">
                <b>${escapeHtml(c.name)}</b>
                <span class="muted">skew=${c.skew.toFixed(2)} · dtype ${escapeHtml(c.dtype)}</span>
              </div>
              ${renderFullHistogram(c.histogram, s)}
              <div class="dist-stats">
                <span>min ${s.min ?? '—'}</span>
                <span>median ${s.median ?? '—'}</span>
                <span>mean ${s.mean ?? '—'}</span>
                <span>max ${s.max ?? '—'}</span>
              </div>
            </div>`;
        }).join('')}
      </div>
    </div>`;
}

// Larger histogram with x-axis labels and a median marker.
function renderFullHistogram(hist, stats) {
  const counts = hist.counts || [];
  const edges = hist.edges || [];
  if (!counts.length) return '';
  const W = 420, H = 130, PADL = 30, PADB = 22, PADR = 8, PADT = 8;
  const max = Math.max(...counts) || 1;
  const innerW = W - PADL - PADR;
  const innerH = H - PADT - PADB;
  const bw = innerW / counts.length;
  const bars = counts.map((c, i) => {
    const h = (c / max) * innerH;
    const x = PADL + i * bw;
    const y = PADT + (innerH - h);
    return `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${(bw - 0.5).toFixed(2)}" height="${h.toFixed(2)}" fill="var(--accent)" opacity="0.85"/>`;
  }).join('');
  // X tick labels — min, mid, max
  const xMin = edges[0] ?? 0;
  const xMax = edges[edges.length - 1] ?? 0;
  const xMid = (xMin + xMax) / 2;
  const fmt = (v) => Math.abs(v) >= 1000 ? Number(v).toExponential(1) : Number(v).toFixed(2);
  // Median marker
  let medianLine = '';
  if (stats && stats.median != null && xMax > xMin) {
    const mx = PADL + ((stats.median - xMin) / (xMax - xMin)) * innerW;
    medianLine = `<line x1="${mx}" y1="${PADT}" x2="${mx}" y2="${PADT+innerH}" stroke="var(--accent-2)" stroke-width="1.5" stroke-dasharray="3 2"/>
      <text x="${mx + 3}" y="${PADT + 10}" font-size="9" fill="var(--accent-2)">median</text>`;
  }
  return `<svg class="ev-svg" viewBox="0 0 ${W} ${H}">
    <line class="axis" x1="${PADL}" y1="${PADT+innerH}" x2="${PADL+innerW}" y2="${PADT+innerH}"/>
    <line class="axis" x1="${PADL}" y1="${PADT}"        x2="${PADL}"        y2="${PADT+innerH}"/>
    ${bars}
    ${medianLine}
    <text class="tick" x="${PADL}"            y="${PADT+innerH+13}" text-anchor="start">${fmt(xMin)}</text>
    <text class="tick" x="${PADL+innerW/2}"   y="${PADT+innerH+13}" text-anchor="middle">${fmt(xMid)}</text>
    <text class="tick" x="${PADL+innerW}"     y="${PADT+innerH+13}" text-anchor="end">${fmt(xMax)}</text>
    <text class="tick" x="${PADL-4}"          y="${PADT+8}"         text-anchor="end">${max}</text>
    <text class="tick" x="${PADL-4}"          y="${PADT+innerH}"    text-anchor="end">0</text>
  </svg>`;
}

function buildPerColumnMissingCard(preview) {
  const cols = (preview.col_stats || []).filter(c => (c.missing_pct || 0) > 0)
    .sort((a, b) => b.missing_pct - a.missing_pct);
  if (!cols.length) {
    return `<div class="ev-card">
      <h3>Missing values</h3>
      <p class="lead" style="color:var(--good)">✓ No missing values detected in the sampled rows.</p>
    </div>`;
  }
  const max = Math.max(...cols.map(c => c.missing_pct), 5);
  return `
    <div class="ev-card">
      <h3>Missing-value rate per column</h3>
      <p class="lead">Columns with any missing values, worst first.</p>
      <div class="ev-bars">
        ${cols.map(c => {
          const k = c.missing_pct > 30 ? 'bad' : c.missing_pct > 10 ? 'warn' : 'good';
          return `
            <div class="bar">
              <span class="name" title="${escapeHtml(c.name)}">${escapeHtml(c.name)}</span>
              <div class="meter ${k}"><span style="width:${(c.missing_pct / max) * 100}%"></span></div>
              <span class="val">${c.missing_pct.toFixed(2)}%</span>
            </div>`;
        }).join('')}
      </div>
    </div>`;
}

function buildLineageCard(lineage) {
  // lineage = {cid: {parent, depth, siblings, pct_total, pct_of_parent, split_into?}}
  const entries = Object.entries(lineage).sort((a, b) => Number(a[0]) - Number(b[0]));
  if (!entries.length) return '';
  const tops = entries.filter(([, v]) => v.parent === null || v.parent === undefined);
  const children = (parent) => entries.filter(([, v]) => String(v.parent) === String(parent));
  const renderNode = (cid, info) => {
    const pct = (info.pct_total != null ? info.pct_total * 100 : 0).toFixed(1);
    const kind = pct > 40 ? 'bad' : pct > 25 ? 'warn' : 'good';
    const subs = children(cid);
    return `
      <li>
        <span class="lin-node lin-${kind}">Cluster ${escapeHtml(cid)}</span>
        <span class="lin-pct">${pct}%</span>
        ${subs.length ? `<ul>${subs.map(([k, v]) => renderNode(k, v)).join('')}</ul>` : ''}
      </li>`;
  };
  return `
    <div class="ev-card">
      <h3>Cluster lineage <span class="iter">deepening tree</span></h3>
      <p class="lead">Any cluster &gt; 40% gets auto-split into sub-clusters (recursive).</p>
      <ul class="lin-tree">${tops.map(([k, v]) => renderNode(k, v)).join('')}</ul>
    </div>`;
}

// Cross-iteration summary card that appears once the pipeline finishes.
// Tabulates each iteration's silhouette, F1, cluster count, tokens, cost, and
// elapsed time, with the winning iteration (by Clarity Gate pass + max
// avg_confidence + F1) highlighted.
function buildFinalSummaryCard(completion) {
  const events = state.events || [];

  // Group events by iteration window. iteration_started events define windows.
  const iterStarts = events.filter(e => e.event === 'iteration_started');
  if (!iterStarts.length) return '';

  // Helper: walk events between [startEvent, nextStartEvent) and aggregate
  const between = (startEv, nextEv) => {
    const t0 = Date.parse(startEv.ts || 0);
    const t1 = nextEv ? Date.parse(nextEv.ts || 0) : Infinity;
    return events.filter(e => {
      const t = Date.parse(e.ts || 0);
      return t >= t0 && t < t1;
    });
  };

  const rows = iterStarts.map((startEv, i) => {
    const nextEv = iterStarts[i + 1];
    const win = between(startEv, nextEv);
    // tokens + cost across pipeline ledger only (evidence + naming excluded)
    let totIn = 0, totOut = 0, totTimeS = 0;
    for (const e of win) {
      if (e.event === 'llm_call_finished' && (e.category || 'pipeline') === 'pipeline') {
        totIn += Number(e.input_tokens || 0);
        totOut += Number(e.output_tokens || 0);
        totTimeS += Number(e.time_s || 0);
      }
    }
    const cost = costOf(totIn, totOut);
    // Wall-clock elapsed = next iter start - this iter start (last iter uses pipeline_complete ts)
    let elapsedS = 0;
    if (nextEv) {
      elapsedS = (Date.parse(nextEv.ts) - Date.parse(startEv.ts)) / 1000;
    } else if (completion) {
      elapsedS = (Date.parse(completion.ts) - Date.parse(startEv.ts)) / 1000;
    }
    // Per-iter agent reports
    const reportsByAgent = {};
    for (const e of win) {
      if (e.event === 'agent_report') reportsByAgent[e.agent] = e;
    }
    const cluster = reportsByAgent['Clusterer'];
    const naming = reportsByAgent['PersonaNamer'];
    const classifier = reportsByAgent['Classifier'];

    const sil = cluster?.metrics?.silhouette;
    const target = cluster?.metrics?.silhouette_target;
    const k = cluster?.metrics?.n_leaf_clusters || cluster?.metrics?.n_clusters;
    // Algorithm column — annotate whether it ran, was auto-selected, or was skipped.
    let algo = '—';
    let algoTip = 'No clustering ran in this iteration.';
    if (cluster) {
      const raw = cluster.metrics?.algorithm || '';
      const algoReason = cluster.context?.algo_reasoning || '';
      const isAuto = /auto-select|auto select|recommend|chose/i.test(algoReason);
      algo = raw ? (isAuto ? `${raw} (auto)` : raw) : '—';
      algoTip = algoReason
        ? `Selected: ${raw || '?'} — ${algoReason}`
        : `Algorithm: ${raw || 'unknown'}`;
    }
    const f1 = classifier?.metrics?.cv_f1_macro;
    const namingPassed = naming?.metrics?.gate_passed;
    const avgConf = naming?.metrics?.avg_confidence;

    // Status of this iteration overall + WHY it didn't continue (so the user
    // can see "iter 1 silhouette was high but Clarity Gate failed" instead of
    // wondering why we kept iterating).
    let status = 'pending';
    let reason = '';
    if (cluster?.status === 'blocked') {
      status = 'blocked';
      reason = (cluster.issues && cluster.issues[0]) || 'Clusterer blocked.';
    } else if (!cluster) {
      status = 'skipped';
      reason = 'Clustering did not run this iteration (re-engineering features).';
    } else if (cluster.status === 'warning' && sil != null && target != null && sil < target) {
      status = 'silhouette miss';
      reason = `silhouette ${Number(sil).toFixed(3)} < target ${Number(target).toFixed(2)} → reselect features`;
    } else if (naming && namingPassed === false) {
      status = 'clarity fail';
      reason = (naming.issues && naming.issues[0]) || 'Clarity Gate failed → re-cluster';
    } else if (classifier && classifier.status !== 'success') {
      status = 'F1 low';
      reason = `CV F1 ${f1 != null ? Number(f1).toFixed(3) : '?'} below gate → ${classifier.context?.action || 'recluster'}`;
    } else if (classifier?.status === 'success') {
      status = 'success';
      reason = 'all gates passed';
    } else if (naming?.metrics?.gate_passed === true) {
      status = 'naming ok';
      reason = 'naming passed; classifier did not run (max iter reached?)';
    }

    return {
      iter: startEv.iteration, status, reason,
      algo, algoTip, k, sil, target, f1, namingPassed, avgConf,
      tokensIn: totIn, tokensOut: totOut, cost,
      elapsedS,
    };
  });

  // Determine the winning iteration: prefer Classifier success + highest F1;
  // fall back to highest silhouette if no success
  const successful = rows.filter(r => r.f1 != null && r.namingPassed);
  let winnerIter = null;
  if (successful.length) {
    winnerIter = successful.reduce((a, b) => (b.f1 || 0) > (a.f1 || 0) ? b : a).iter;
  } else {
    const best = rows.reduce((a, b) =>
      (b.sil || -Infinity) > (a.sil || -Infinity) ? b : a, rows[0]);
    if (best) winnerIter = best.iter;
  }

  const totalCost   = rows.reduce((s, r) => s + r.cost, 0);
  const totalTokens = rows.reduce((s, r) => s + r.tokensIn + r.tokensOut, 0);
  const totalTime   = rows.reduce((s, r) => s + r.elapsedS, 0);
  const completionStatus = completion.status || 'success';
  const statusKind = completionStatus === 'success' ? 'good'
                   : completionStatus === 'max_iterations_reached' ? 'warn'
                   : 'bad';

  const fmtTime = (s) => {
    if (!s) return '—';
    if (s < 60) return `${s.toFixed(0)}s`;
    const m = Math.floor(s / 60), r = Math.round(s - m * 60);
    return `${m}m ${r}s`;
  };
  const fmtSil = (v) => v == null ? '—' : Number(v).toFixed(4);
  const fmtF1  = (v) => v == null ? '—' : Number(v).toFixed(3);
  const sumTitle = (
    completionStatus === 'success'        ? 'Pipeline approved by Clarity Gate + F1 gate'
    : completionStatus === 'best_effort'  ? 'Max iterations — best-effort result saved'
    : completionStatus === 'max_iterations_reached' ? 'Max iterations reached'
    : completionStatus
  );

  return `
    <div class="ev-card span2 ev-finalsum">
      <h3>Final pipeline summary
        <span class="iter ${statusKind === 'good' ? 'good' : statusKind === 'warn' ? 'warn' : 'bad'}">${escapeHtml(sumTitle)}</span>
      </h3>
      <div class="finalsum-totals">
        <div class="ft-cell"><b>${rows.length}</b><span>iterations</span></div>
        <div class="ft-cell"><b>${fmtNum(totalTokens)}</b><span>tokens (pipeline)</span></div>
        <div class="ft-cell"><b>${fmtUsd(totalCost)}</b><span>cost (pipeline)</span></div>
        <div class="ft-cell"><b>${fmtTime(totalTime)}</b><span>wall-clock time</span></div>
        <div class="ft-cell"><b>${winnerIter != null ? 'iter ' + winnerIter : '—'}</b><span>winning iteration</span></div>
      </div>
      <p class="lead">One row per iteration. The winning iteration is highlighted — chosen by Clarity Gate pass + highest F1, falling back to silhouette. The <b>Why</b> column explains what made the orchestrator continue past iterations whose silhouette already looked high.</p>
      <table class="finalsum-table">
        <thead>
          <tr>
            <th>Iter</th><th>Algo</th><th>k</th><th>Silhouette</th><th>CV F1</th>
            <th>Naming</th><th>Tokens</th><th>Cost</th><th>Time</th><th>Status</th><th>Why</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => {
            const silStr = r.sil != null && r.target != null
              ? `${fmtSil(r.sil)} <span class="muted" style="font-size:10.5px">/ tgt ${Number(r.target).toFixed(2)}</span>`
              : fmtSil(r.sil);
            return `
            <tr class="${r.status}${r.iter === winnerIter ? ' winner' : ''}">
              <td><b>${r.iter}${r.iter === winnerIter ? ' ★' : ''}</b></td>
              <td title="${escapeHtml(r.algoTip || '')}">${escapeHtml(r.algo || '—')}</td>
              <td>${r.k ?? '—'}</td>
              <td>${silStr}</td>
              <td>${fmtF1(r.f1)}</td>
              <td>${r.namingPassed === true ? '✓' : r.namingPassed === false ? '✗' : '—'}${r.avgConf != null ? ` (${Number(r.avgConf).toFixed(1)}/10)` : ''}</td>
              <td>${fmtNum(r.tokensIn + r.tokensOut)}</td>
              <td>${fmtUsd(r.cost)}</td>
              <td>${fmtTime(r.elapsedS)}</td>
              <td><span class="iter-badge ${r.status.replace(/\s+/g,'-')}">${escapeHtml(r.status)}</span></td>
              <td class="why-cell" title="${escapeHtml(r.reason || '')}">${escapeHtml(r.reason || '—')}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>`;
}

function buildUploadPreviewCard(up) {
  const p = up.preview || {};
  const cols = p.columns || [];
  const rows = p.rows || [];
  const nRows = p.n_rows != null ? Number(p.n_rows).toLocaleString() : '?';
  const head = '<thead><tr>' + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead>';
  const body = '<tbody>' + rows.map(r =>
    '<tr>' + r.map(c => `<td>${escapeHtml(c)}</td>`).join('') + '</tr>'
  ).join('') + '</tbody>';
  return `
    <div class="ev-card span2" id="ev-original-upload-card">
      <h3>Original uploaded dataset
        <span class="iter">${escapeHtml(up.name || '')} · ${nRows} × ${cols.length} · showing first ${rows.length} rows</span>
      </h3>
      <p class="lead">Raw data exactly as you uploaded it — pinned here so you can always compare against any later transformation. Saved to <code>${escapeHtml(up.path || '')}</code>.</p>
      <div class="dz-preview-table-wrap">
        <table class="dz-preview-table">${head}${body}</table>
      </div>
    </div>`;
}

function _preModellingStageLabel(p) {
  return p.stage === 'feature_selection'
    ? `After FeatureSelector (iter ${p.iteration})`
    : `After FeatureEngineer (iter ${p.iteration})`;
}

function _preModellingLead(p) {
  return p.stage === 'feature_selection'
    ? 'Dataset after feature selection trimmed the engineered table — this is what the Clusterer actually sees.'
    : 'Dataset after FeatureEngineer turned the raw rows into entity-level features.';
}

function _preModellingTable(p) {
  const cols = p.columns || [];
  const rows = p.rows || [];
  const head = '<thead><tr>' + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead>';
  const body = '<tbody>' + rows.map(r =>
    '<tr>' + r.map(c => `<td>${escapeHtml(c)}</td>`).join('') + '</tr>'
  ).join('') + '</tbody>';
  return `
    <div class="dz-preview-table-wrap">
      <table class="dz-preview-table">${head}${body}</table>
    </div>`;
}

function _diffPreModellingCols(prev, latest) {
  const prevCols = new Set(prev.columns || []);
  const latestCols = new Set(latest.columns || []);
  const added = [...latestCols].filter(c => !prevCols.has(c));
  const dropped = [...prevCols].filter(c => !latestCols.has(c));
  return {added, dropped};
}

function buildPreModellingPreviewCard(snapshots) {
  // Back-compat: older runs wrote a single dict, not a list.
  const list = Array.isArray(snapshots) ? snapshots
             : (snapshots ? [snapshots] : []);
  if (!list.length) return '';
  // Show most-recent at the top.
  const sorted = list.slice().sort((a, b) => {
    // Order by ts when present, else by iteration.
    const ta = a.ts || `${a.iteration}:${a.stage}`;
    const tb = b.ts || `${b.iteration}:${b.stage}`;
    return tb.localeCompare(ta);
  });
  const latest = sorted[0];
  const prior = sorted.slice(1);

  const fmt = (v) => v != null ? Number(v).toLocaleString() : '?';
  const latestRows = (latest.rows || []);
  const latestHeader = `${escapeHtml(_preModellingStageLabel(latest))} · ${fmt(latest.n_rows)} × ${fmt(latest.n_cols)} · showing first ${latestRows.length} rows`;

  const stageCount = sorted.length;
  const feCount = sorted.filter(s => s.stage === 'feature_engineering').length;
  const fsCount = sorted.filter(s => s.stage === 'feature_selection').length;
  const stageSummary = `${stageCount} snapshot${stageCount > 1 ? 's' : ''} · ${feCount} FeatureEngineer · ${fsCount} FeatureSelector`;

  const priorHtml = prior.map(snap => {
    const diff = _diffPreModellingCols(snap, latest);
    const diffBits = [];
    if (diff.added.length)   diffBits.push(`+${diff.added.length} new cols vs latest`);
    if (diff.dropped.length) diffBits.push(`−${diff.dropped.length} dropped vs latest`);
    if (!diffBits.length)    diffBits.push('column set unchanged vs latest');
    const subTitle = `${fmt(snap.n_rows)} × ${fmt(snap.n_cols)} · ${diffBits.join(' · ')}`;
    return `
      <details class="iter-row" style="margin-top:6px">
        <summary class="iter-row-head" style="cursor:pointer">
          <span class="iter-badge">${escapeHtml(_preModellingStageLabel(snap))}</span>
          <span class="iter-summary">${escapeHtml(subTitle)}</span>
        </summary>
        <div class="iter-row-body" style="padding-top:8px">
          <div class="muted" style="margin-bottom:6px">${escapeHtml(_preModellingLead(snap))}</div>
          ${(diff.added.length || diff.dropped.length) ? `
            <div class="muted" style="font-size:11.5px;margin-bottom:8px">
              ${diff.added.length   ? `<b>Added (${diff.added.length}):</b> ${escapeHtml(diff.added.slice(0,12).join(', '))}${diff.added.length > 12 ? ` …(+${diff.added.length - 12} more)` : ''}<br>` : ''}
              ${diff.dropped.length ? `<b>Dropped (${diff.dropped.length}):</b> ${escapeHtml(diff.dropped.slice(0,12).join(', '))}${diff.dropped.length > 12 ? ` …(+${diff.dropped.length - 12} more)` : ''}` : ''}
            </div>` : ''}
          ${_preModellingTable(snap)}
        </div>
      </details>`;
  }).join('');

  return `
    <div class="ev-card span2" id="ev-pre-modelling-card">
      <h3>Pre-modelling dataset
        <span class="iter">${latestHeader}</span>
      </h3>
      <p class="lead">${_preModellingLead(latest)} <span class="muted">· ${escapeHtml(stageSummary)}${prior.length ? ' — expand any earlier snapshot below to compare' : ''}.</span></p>
      ${_preModellingTable(latest)}
      ${prior.length ? `
        <div class="muted" style="margin-top:14px;font-size:11.5px;text-transform:uppercase;letter-spacing:0.5px">
          Earlier snapshots (newest first)
        </div>
        ${priorHtml}` : ''}
    </div>`;
}

function buildDatasetRowCountNoteCard(dse, upload) {
  const m = dse?.metrics || {};
  const up = upload?.preview || {};
  const profileRows = m.n_rows ?? up.stats_sample_size ?? null;
  const fullRows = m.n_rows_source ?? up.n_rows ?? null;
  const subsampled = fullRows != null && profileRows != null && fullRows > profileRows;
  if (!subsampled) return '';

  const fmt = (v) => (typeof v === 'number' ? Number(v).toLocaleString() : String(v));
  const fullLabel = fmt(fullRows);
  const profileLabel = fmt(profileRows);

  return `
    <div class="ev-card span2 ev-note">
      <h3>Why is the row count lower than my file?</h3>
      <p class="lead">
        When a raw CSV has more than 50,000 rows, <b>DatasetExaminer</b> profiles a random
        ${profileLabel}-row sample for speed — it only needs schema, column types, missing rates,
        and feature-group suggestions. The <b>Dataset profile</b> figure above (${profileLabel} rows)
        is that sample size, not your full file.
      </p>
      <p class="lead">
        Your full dataset (<b>${fullLabel} rows</b>) is still loaded and passed to
        <b>FeatureEngineer</b>, which uses every row. After that step, row count reflects
        <b>entities</b> (e.g. customers or cards), not raw transaction rows — so it will
        differ again from the source file.
      </p>
      <div class="ev-note-table-wrap">
        <table class="ev-note-table">
          <thead><tr><th>Step</th><th>Rows used</th></tr></thead>
          <tbody>
            <tr><td>DatasetExaminer (profile above)</td><td>${profileLabel} (sample)</td></tr>
            <tr><td>FeatureEngineer</td><td>${fullLabel} (full file)</td></tr>
            <tr><td>Clustering</td><td>Engineered entity matrix</td></tr>
          </tbody>
        </table>
      </div>
    </div>`;
}

function buildDatasetCard(dse, upload) {
  const m = dse?.metrics || {};
  const up = upload?.preview || {};
  const colStats = up.col_stats || [];
  // Prefer the agent's profile once it runs; otherwise use the upload directly
  const rows = (m.n_rows ?? up.n_rows);
  const totalCols = m.n_cols ?? up.n_cols ?? (colStats.length || null);
  const numericFromUpload = colStats.filter(c => c.numeric).length || null;
  const numericCols = m.n_numeric_cols ?? numericFromUpload;
  const highMissingFromUpload = colStats.filter(c => (c.missing_pct || 0) > 30).length;
  const highMissing = m.n_high_missing ?? (colStats.length ? highMissingFromUpload : null);

  const fmt = (v) => v == null ? '?' :
    (typeof v === 'number' ? Number(v).toLocaleString() : String(v));

  const lead = dse
    ? 'Profiled by DatasetExaminer.'
    : (upload
        ? 'Loaded from your upload — DatasetExaminer will enrich this in a moment.'
        : 'Upload a file or wait for DatasetExaminer to run.');

  return `
    <div class="ev-card span2">
      <h3>Dataset profile <span class="iter">${dse ? `iter ${dse.iteration}` : (upload ? 'from upload' : 'pending')}</span></h3>
      <p class="lead">${lead}</p>
      <div class="ev-flow">
        <div class="step"><b>${fmt(rows)}</b>rows</div>
        <div class="arrow">·</div>
        <div class="step"><b>${fmt(totalCols)}</b>columns total</div>
        <div class="arrow">·</div>
        <div class="step"><b>${fmt(numericCols)}</b>numeric</div>
        <div class="arrow">·</div>
        <div class="step"><b>${fmt(m.n_suggested_groups)}</b>feature groups suggested${dse ? '' : ' <span class="muted">(agent pending)</span>'}</div>
        <div class="arrow">·</div>
        <div class="step"><b>${fmt(highMissing)}</b>cols with high missing</div>
      </div>
    </div>`;
}

function buildSkewCard(dse) {
  const m = dse.metrics || {};
  const skew = Number(m.mean_skewness || 0);
  const max = Math.max(6, skew * 1.1);
  const fill = (skew / max) * 100;
  const thresholdPct = (3 / max) * 100;
  const kind = skew > 3 ? 'bad' : skew > 1.5 ? 'warn' : 'good';
  const issues = (dse.issues || []).join(' · ');
  const explainBtn = (kind !== 'good' && issues) ? `
    <button class="explain-btn" data-explain-agent="DatasetExaminer"
            data-explain-issue="${escapeHtml(issues)}"
            data-explain-evidence='${escapeHtml(JSON.stringify({mean_skewness: skew, threshold: 3}))}'>
      Explain this warning (LLM · evidence ledger)
    </button>
    <div class="explain-out" id="explain-out-skew" hidden></div>` : '';
  return `
    <div class="ev-card">
      <h3>Mean skewness <span class="iter">${kind === 'bad' ? 'high' : 'ok'}</span></h3>
      <p class="lead">${issues || 'Symmetric features tend to give better clusters.'}</p>
      <div class="ev-meter">
        <div class="track">
          <div class="fill ${kind}" style="width:${fill}%"></div>
          <div class="threshold" data-label="high-skew threshold (3.0)" style="left:${thresholdPct}%"></div>
        </div>
        <div class="scale"><span>0</span><span>${max.toFixed(1)}</span></div>
      </div>
      <div style="font-size:22px;font-weight:700;margin-top:4px;color:var(--text)">
        ${skew.toFixed(2)} <span style="font-size:12px;color:var(--muted);font-weight:400">measured</span>
      </div>
      <p class="lead" style="margin-top:8px">${skew > 3
        ? 'A log-transform on heavy-tailed columns typically halves skewness and improves cluster separation. The per-column chart below is the literal evidence.'
        : 'No transform needed.'}</p>
      ${explainBtn}
    </div>`;
}

// Render ALL iterations of one agent as a stacked list inside one Evidence card.
// Each iteration becomes its own row, keyed by iter number + status colour.
function buildAgentHistoryCard(agentKey, label, history, pcaSnapshots) {
  if (!history || !history.length) return '';
  const STATUS_GLYPH = {success: '✓', warning: '⚠', blocked: '✗', failure: '!!'};
  const sorted = history.slice().sort((a, b) =>
    (Number(a.iteration) || 0) - (Number(b.iteration) || 0));
  // Index PCA snapshots by iteration for fast lookup
  const pcaByIter = {};
  (pcaSnapshots || []).forEach(p => { pcaByIter[p.iteration] = p; });
  return `
    <div class="ev-card span2">
      <h3>${escapeHtml(label)}
        <span class="iter">${sorted.length} iteration${sorted.length > 1 ? 's' : ''}</span>
      </h3>
      <p class="lead">Every run of <code>${escapeHtml(agentKey)}</code> across all pipeline iterations. Newest at the bottom.</p>
      <div class="iter-list">
        ${sorted.map(iter => {
          const m = iter.metrics || {};
          const body = buildOutputBody(agentKey, m) ||
            `<div class="muted">${escapeHtml(iter.what_was_done || '')}</div>`;
          const status = iter.status || 'success';
          const issues = (iter.issues || []).filter(Boolean);
          const pca = pcaByIter[iter.iteration];
          const pcaSvg = pca ? renderPCAScatter(pca) : '';
          return `
            <div class="iter-row ${status}">
              <div class="iter-row-head">
                <span class="iter-badge ${status}">${STATUS_GLYPH[status] || ''} iter ${iter.iteration}</span>
                <span class="iter-summary">${escapeHtml(iter.what_was_done || '')}</span>
              </div>
              <div class="iter-row-body">${body}</div>
              ${pcaSvg}
              ${issues.length ? `
                <div class="iter-row-issues">
                  ${issues.map(i => `<div class="iter-issue">⚠ ${escapeHtml(i)}</div>`).join('')}
                </div>` : ''}
            </div>`;
        }).join('')}
      </div>
    </div>`;
}

// 2-D PCA scatter — one dot per (sampled) data point, coloured by cluster id.
function renderPCAScatter(pca) {
  const pts = pca.points || [];
  if (!pts.length) return '';
  const W = 420, H = 260, PADL = 30, PADB = 24, PADR = 12, PADT = 14;
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const p of pts) {
    if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
  }
  if (xMax === xMin) xMax = xMin + 1;
  if (yMax === yMin) yMax = yMin + 1;
  const innerW = W - PADL - PADR;
  const innerH = H - PADT - PADB;
  const sx = (x) => PADL + ((x - xMin) / (xMax - xMin)) * innerW;
  const sy = (y) => PADT + innerH - ((y - yMin) / (yMax - yMin)) * innerH;
  // 14 distinct cluster colours — falls back to hash if more
  const PALETTE = ['#7aa2ff','#b08bff','#4fd1a1','#ffba6b','#ff7a8a','#6ee7d2',
                   '#f59ee0','#a8e063','#ffd166','#90c8ff','#c3a3ff','#ffadc8',
                   '#80e1c5','#ffaf6b'];
  const colorOf = (c) => PALETTE[Math.abs(c) % PALETTE.length];
  const ev = pca.explained_variance_ratio || [];
  const evLabel = ev.length
    ? `PC1 ${(ev[0]*100).toFixed(1)}% · PC2 ${(ev[1]*100).toFixed(1)}%`
    : '';
  // Build the dots
  const dots = pts.map(p =>
    `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="2.2" fill="${colorOf(p.c)}" fill-opacity="0.72"/>`
  ).join('');
  // Legend — list unique cluster ids
  const seen = new Set();
  const ids = [];
  for (const p of pts) if (!seen.has(p.c)) { seen.add(p.c); ids.push(p.c); }
  const legend = ids.sort((a, b) => a - b).map(c =>
    `<span class="pca-leg"><span class="pca-leg-dot" style="background:${colorOf(c)}"></span>C${c}</span>`
  ).join('');
  return `
    <div class="pca-wrap">
      <div class="pca-head">
        <span class="muted">2-D PCA projection</span>
        <span class="muted" style="margin-left:8px">${escapeHtml(evLabel)} · ${pts.length} sampled points</span>
      </div>
      <svg class="ev-svg pca-svg" viewBox="0 0 ${W} ${H}">
        <line class="axis" x1="${PADL}" y1="${PADT+innerH}" x2="${PADL+innerW}" y2="${PADT+innerH}"/>
        <line class="axis" x1="${PADL}" y1="${PADT}"        x2="${PADL}"        y2="${PADT+innerH}"/>
        ${dots}
        <text class="tick" x="${PADL+innerW}" y="${PADT+innerH+14}" text-anchor="end">PC1</text>
        <text class="tick" x="${PADL-4}"      y="${PADT+10}"        text-anchor="end">PC2</text>
      </svg>
      <div class="pca-legend">${legend}</div>
    </div>`;
}

function _renderFeatureGroupsList(dse) {
  const groups = dse?.context?.suggested_feature_groups || [];
  const details = dse?.context?.group_details || {};
  const entries = groups.length ? groups : Object.keys(details);
  if (!entries.length) return '<div class="muted">No groups recorded for this iteration.</div>';
  return `
    <div class="feat-groups">
      ${entries.map(g => {
        const d = details[g] || {};
        const desc = d.description || '';
        const cols = (d.source_columns || []).join(', ');
        const why = d.rationale || '';
        return `
          <div class="feat-group">
            <div class="fg-name">${escapeHtml(g)}</div>
            ${desc ? `<div class="fg-desc">${escapeHtml(desc)}</div>` : ''}
            ${cols ? `<div class="fg-cols"><b>Built from:</b> ${escapeHtml(cols)}</div>` : ''}
            ${why ? `<div class="fg-why"><b>Why:</b> ${escapeHtml(why)}</div>` : ''}
          </div>`;
      }).join('')}
    </div>`;
}

function _groupSet(dse) {
  const groups = dse?.context?.suggested_feature_groups || [];
  const details = dse?.context?.group_details || {};
  const entries = groups.length ? groups : Object.keys(details);
  return new Set(entries.map(String));
}

function buildFeatureGroupsHistoryCard(history) {
  const sorted = history.slice().sort((a, b) =>
    (Number(a.iteration) || 0) - (Number(b.iteration) || 0));
  const latest = sorted[sorted.length - 1];
  const prior = sorted.slice(0, -1);
  const latestSet = _groupSet(latest);
  const latestCount = latestSet.size;

  const priorHtml = prior.slice().reverse().map(iter => {
    const set = _groupSet(iter);
    const added = [...latestSet].filter(g => !set.has(g)).length;
    const removed = [...set].filter(g => !latestSet.has(g)).length;
    let diff;
    if (!set.size) diff = `${latestSet.size} groups`;
    else if (!added && !removed) diff = `${set.size} groups · unchanged vs latest`;
    else diff = `${set.size} groups · vs latest: +${added} new, −${removed} dropped`;
    return `
      <details class="iter-row" style="margin-top:6px">
        <summary class="iter-row-head" style="cursor:pointer">
          <span class="iter-badge">iter ${iter.iteration}</span>
          <span class="iter-summary">${escapeHtml(diff)}</span>
        </summary>
        <div class="iter-row-body" style="padding-top:8px">
          ${_renderFeatureGroupsList(iter)}
        </div>
      </details>`;
  }).join('');

  return `
    <div class="ev-card span2">
      <h3>Suggested feature groups
        <span class="iter">DatasetExaminer · iter ${latest.iteration} (latest) · ${latestCount} groups${prior.length ? ` · ${prior.length} earlier iteration${prior.length > 1 ? 's' : ''}` : ''}</span>
      </h3>
      <p class="lead">These are the families of features DatasetExaminer asked FeatureEngineer to build.
        Each group is a behavioural lens on the entity (e.g. "spending_behaviour", "category_preferences").
        ${prior.length ? 'Previous iterations are kept below — expand any row to compare.' : ''}</p>
      ${_renderFeatureGroupsList(latest)}
      ${prior.length ? `
        <div class="muted" style="margin-top:14px;font-size:11.5px;text-transform:uppercase;letter-spacing:0.5px">
          Earlier iterations (newest first)
        </div>
        ${priorHtml}` : ''}
    </div>`;
}

function buildFeatureEngineerCard(fe) {
  const m = fe.metrics || {};
  return `
    <div class="ev-card">
      <h3>Feature engineering <span class="iter">iter ${fe.iteration}</span></h3>
      <p class="lead">${escapeHtml(fe.what_was_done || '')}</p>
      <div class="ev-flow">
        <div class="step"><b>${(m.n_entities||'?').toLocaleString?.() ?? m.n_entities}</b>entities</div>
        <div class="arrow">→</div>
        <div class="step"><b>${m.n_features ?? '?'}</b>features built</div>
        <div class="arrow">·</div>
        <div class="step"><b>${m.n_builders_run ?? '?'}</b>builders run</div>
      </div>
    </div>`;
}

function buildFeatureSelectorCard(fs) {
  const m = fs.metrics || {};
  const vifKind = (m.max_vif_remaining || 0) > 10 ? 'warn' : 'good';
  return `
    <div class="ev-card">
      <h3>Feature selection (PCA + AE + VIF) <span class="iter">iter ${fs.iteration}</span></h3>
      <p class="lead">${escapeHtml(fs.what_was_done || '')}</p>
      <div class="ev-flow">
        <div class="step"><b>${m.n_input_features ?? '?'}</b>input</div>
        <div class="arrow">−</div>
        <div class="step"><b>${m.n_removed_by_vif ?? '?'}</b>removed by VIF</div>
        <div class="arrow">→</div>
        <div class="step"><b>${m.n_selected ?? '?'}</b>selected by LLM</div>
      </div>
      <div class="ev-bars" style="margin-top:14px">
        <div class="bar">
          <span class="name">max VIF remaining</span>
          <div class="meter ${vifKind}"><span style="width:${Math.min(100, ((m.max_vif_remaining||0)/20)*100)}%"></span></div>
          <span class="val">${(m.max_vif_remaining||0).toFixed?.(2) ?? m.max_vif_remaining}</span>
        </div>
      </div>
      <p class="lead" style="margin-top:8px">VIF ≤ 10 means no remaining multicollinearity in the kept features.</p>
    </div>`;
}

function buildSilhouetteCard(sc) {
  // sc = {algorithm, best_k, scores: {k: v}, algo_reasoning}
  const scores = sc.scores || {};
  const ks = Object.keys(scores).map(Number).sort((a,b) => a - b);
  if (!ks.length) return '';
  const W = 380, H = 180, PADL = 36, PADB = 28, PADR = 16, PADT = 12;
  const xs = ks.map(k => PADL + ((k - ks[0]) / (ks[ks.length-1] - ks[0] || 1)) * (W - PADL - PADR));
  const vs = ks.map(k => Number(scores[k]));
  const vmax = Math.max(...vs, 0.05);
  const ys = vs.map(v => PADT + (1 - v / vmax) * (H - PADT - PADB));
  const path = xs.map((x, i) => (i === 0 ? `M ${x} ${ys[i]}` : `L ${x} ${ys[i]}`)).join(' ');
  const area = `M ${xs[0]} ${H-PADB} L ` + xs.map((x, i) => `${x} ${ys[i]}`).join(' L ') + ` L ${xs[xs.length-1]} ${H-PADB} Z`;
  const peakK = Number(sc.best_k);
  return `
    <div class="ev-card">
      <h3>Silhouette curve <span class="iter">${escapeHtml(sc.algorithm || '?')} · best k = ${sc.best_k}</span></h3>
      <p class="lead">Higher = clusters are more compact and separated. The peak is the algorithm's choice.</p>
      <svg class="ev-svg" viewBox="0 0 ${W} ${H}">
        <line class="axis" x1="${PADL}" y1="${H-PADB}" x2="${W-PADR}" y2="${H-PADB}"/>
        <line class="axis" x1="${PADL}" y1="${PADT}"    x2="${PADL}"    y2="${H-PADB}"/>
        ${[0, 0.25, 0.5, 0.75, 1].map(t => {
          const y = PADT + t * (H - PADT - PADB);
          const v = (vmax * (1 - t)).toFixed(2);
          return `<line class="grid" x1="${PADL}" y1="${y}" x2="${W-PADR}" y2="${y}"/>
                  <text class="tick" x="${PADL-6}" y="${y+3}" text-anchor="end">${v}</text>`;
        }).join('')}
        ${ks.map((k, i) => `<text class="tick" x="${xs[i]}" y="${H-PADB+14}" text-anchor="middle">k=${k}</text>`).join('')}
        <path class="area" d="${area}"/>
        <path class="line" d="${path}"/>
        ${ks.map((k, i) => `<circle class="pt ${k === peakK ? 'peak' : ''}" cx="${xs[i]}" cy="${ys[i]}" r="${k === peakK ? 5 : 3}"/>`).join('')}
      </svg>
      <p class="lead" style="margin-top:4px">${escapeHtml(sc.algo_reasoning || '')}</p>
    </div>`;
}

function buildClusterSizeCard(sizes) {
  const total = sizes.reduce((s, c) => s + c.n, 0) || 1;
  const max = Math.max(...sizes.map(s => s.pct));
  return `
    <div class="ev-card">
      <h3>Cluster sizes <span class="iter">${sizes.length} clusters · ${total.toLocaleString()} entities</span></h3>
      <p class="lead">Any cluster over the 40% guard (orange line) auto-triggers sub-clustering.</p>
      <div class="ev-bars">
        ${sizes.sort((a,b) => b.pct - a.pct).map(c => {
          const kind = c.pct > 40 ? 'bad' : c.pct > 25 ? 'warn' : 'good';
          return `
            <div class="bar">
              <span class="name">Cluster ${escapeHtml(c.cluster_id)}</span>
              <div class="meter ${kind}"><span style="width:${(c.pct / Math.max(40, max)) * 100}%"></span></div>
              <span class="val">${c.pct.toFixed(1)}%</span>
            </div>`;
        }).join('')}
      </div>
    </div>`;
}

// Cross-cluster comparison card — an LLM contrasting analysis across ALL
// clusters at once (the per-cluster chat handles single-cluster questions).
// The call is lazy (button-triggered) so it only bills the evidence ledger
// when the user actually asks for it.
function buildClusterComparisonCard() {
  const n = Object.keys(state.personas || {}).length;
  return `
    <div class="ev-card span2" id="ev-cluster-comparison">
      <h3>Cross-cluster comparison <span class="iter">${n} clusters · LLM · evidence ledger</span></h3>
      <p class="lead">A contrasting analysis across every cluster at once — which
        axes separate them, which pairs look alike on one trait but diverge on
        another, and which clusters overlap. Runs on demand.</p>
      <div class="row" style="gap:8px;align-items:flex-start;margin-bottom:8px">
        <input id="ev-compare-focus" type="text"
          placeholder="Optional focus, e.g. 'why C0 vs C3 both high-spend differ'"
          style="flex:1;min-width:220px" />
        <button class="primary" id="ev-compare-btn">Compare clusters</button>
      </div>
      <div id="ev-compare-out" class="explain-out" hidden></div>
    </div>`;
}

// In-flight state for the cross-cluster comparison call. Held at module level
// so a panel rebuild (renderEvidence fires on every agent_report SSE event)
// can re-attach to the same logical call instead of orphaning its result.
// Without this, the result would be written to a detached <div id="ev-compare-out">
// and the user would see the spinner vanish with no answer ("page refreshed").
const _comparisonState = {inFlight: false, focus: '', resultHTML: null};

function _renderComparisonResult(r) {
  return `
    <div class="explain-card">
      <div class="explain-section">
        <div class="muted">Contrasting analysis across ${r.n_clusters} clusters${r._cached ? ' · cached' : ''}</div>
        <div style="white-space:pre-wrap;word-break:break-word;line-height:1.5">${escapeHtml(r.comparison || '(no analysis)')}</div>
      </div>
      <div class="muted" style="font-size:10.5px;margin-top:6px">
        ✓ This cost was added to the <b>Evidence</b> ledger (not the Pipeline ledger).
      </div>
    </div>`;
}

function _renderComparisonError(e) {
  const msg = String(e.message || 'unknown error');
  const staleUi = /not found/i.test(msg);
  return staleUi
    ? `<div class="muted" style="color:var(--bad)">
        Comparison failed: the UI server is outdated (missing <code>/api/cluster-comparison</code>).
        Stop the old process on this port and restart <code>python run_pipeline.py</code>,
        or open the URL printed in the terminal if it moved to a new port (e.g. 5058).
      </div>`
    : `<div class="muted" style="color:var(--bad)">Comparison failed: ${escapeHtml(msg)}</div>`;
}

function wireComparisonButton() {
  const btn = document.getElementById('ev-compare-btn');
  if (!btn) return;

  // If a previous call finished while the panel was being rebuilt, the result
  // HTML was stashed on _comparisonState. Paint it into the fresh DOM.
  if (_comparisonState.resultHTML) {
    const freshOut = document.getElementById('ev-compare-out');
    if (freshOut) {
      freshOut.hidden = false;
      freshOut.innerHTML = _comparisonState.resultHTML;
    }
    btn.textContent = 'Compare again';
  }

  // If a call is still in flight when the panel rebuilds, restore the in-progress
  // UI on the new button + restore the focus input value the user typed.
  if (_comparisonState.inFlight) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>Comparing (evidence ledger)…`;
    const focusInput = document.getElementById('ev-compare-focus');
    if (focusInput && _comparisonState.focus) focusInput.value = _comparisonState.focus;
  }

  btn.onclick = async () => {
    if (_comparisonState.inFlight) return;  // guard double-fires after rebuild
    const focus = (document.getElementById('ev-compare-focus')?.value || '').trim();
    _comparisonState.inFlight = true;
    _comparisonState.focus = focus;
    _comparisonState.resultHTML = null;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>Comparing (evidence ledger)…`;
    let html;
    let isError = false;
    try {
      const r = await api('POST', '/api/cluster-comparison', {focus});
      html = _renderComparisonResult(r);
    } catch (e) {
      html = _renderComparisonError(e);
      isError = true;
    }
    _comparisonState.inFlight = false;
    // Only cache successful results — errors shouldn't survive a panel rebuild.
    _comparisonState.resultHTML = isError ? null : html;
    // Re-look-up by id: the original `btn` / `out` closure refs may now point
    // to detached DOM nodes if renderEvidence rebuilt the panel while we awaited.
    const freshOut = document.getElementById('ev-compare-out');
    const freshBtn = document.getElementById('ev-compare-btn');
    if (freshOut) {
      freshOut.hidden = false;
      freshOut.innerHTML = html;
    }
    if (freshBtn) {
      freshBtn.disabled = false;
      freshBtn.textContent = 'Compare again';
    }
  };
}

function buildClassifierCard(c) {
  const f1 = Number(c.cv_f1_macro || 0);
  const pcf = c.per_class_f1 || {};
  const f1Kind = f1 >= 0.7 ? 'good' : 'bad';
  return `
    <div class="ev-card span2">
      <h3>Classifier validation <span class="iter">CV F1 macro = ${f1.toFixed(3)}</span></h3>
      <p class="lead">Random Forest / XGBoost trained to predict cluster membership. F1 ≥ 0.70 means the clusters are crisp.</p>
      <div class="ev-meter">
        <div class="track">
          <div class="fill ${f1Kind}" style="width:${(f1 * 100).toFixed(1)}%"></div>
          <div class="threshold" data-label="gate (0.70)" style="left:70%"></div>
        </div>
        <div class="scale"><span>0</span><span>1</span></div>
      </div>
      <div class="ev-bars" style="margin-top:14px">
        ${Object.entries(pcf).sort((a,b) => b[1] - a[1]).map(([name, score]) => {
          const k = score >= 0.85 ? 'good' : score >= 0.7 ? 'warn' : 'bad';
          return `
            <div class="bar">
              <span class="name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
              <div class="meter ${k}"><span style="width:${(score * 100).toFixed(1)}%"></span></div>
              <span class="val">${Number(score).toFixed(3)}</span>
            </div>`;
        }).join('')}
      </div>
    </div>`;
}

function buildTopFeaturesCard(feats) {
  const entries = Object.entries(feats).slice(0, 15);
  if (!entries.length) return '';
  const max = Math.max(...entries.map(([, v]) => Number(v)));
  return `
    <div class="ev-card">
      <h3>Top features driving the clustering</h3>
      <p class="lead">By classifier feature-importance — what the model uses to tell clusters apart.</p>
      <div class="ev-bars">
        ${entries.map(([name, imp]) => `
          <div class="bar">
            <span class="name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
            <div class="meter"><span style="width:${(Number(imp) / max) * 100}%"></span></div>
            <span class="val">${Number(imp).toFixed(4)}</span>
          </div>`).join('')}
      </div>
    </div>`;
}

// ── Agent outputs panel (right column) ─────────────────────────────────────
// Per-agent record of what they produced. Keyed by agent name; most-recent
// iteration's data wins for that agent.
const outputsState = {};

function recordAgentOutput(e) {
  const cur = outputsState[e.agent];
  const inc = {
    iteration: e.iteration ?? (cur?.latest?.iteration ?? 0),
    metrics: e.metrics || {},
    context: e.context || {},
    status: e.status || 'success',
    what_was_done: e.what_was_done || '',
    issues: e.issues || [],
  };
  // Preserve full history per agent. SSE may replay events on reconnect; we
  // de-dupe by (iteration, status, what_was_done) so the history doesn't
  // grow on every page refresh.
  if (!cur) {
    outputsState[e.agent] = {latest: inc, history: [inc], autoDecision: null};
  } else {
    const dupKey = (x) => `${x.iteration}::${x.status}::${x.what_was_done}`;
    const exists = cur.history.some(x => dupKey(x) === dupKey(inc));
    if (!exists) cur.history.push(inc);
    if (inc.iteration >= (cur.latest?.iteration || 0)) cur.latest = inc;
  }
  renderOutputsPanel();

  if (state.currentMode === 'bypass' && inc.issues && inc.issues.length) {
    triggerBypassAutoDecision(e.agent, inc.issues);
  }
}

async function triggerBypassAutoDecision(agent, issues) {
  const issueText = issues.join(' · ');
  const cur = outputsState[agent];
  if (cur?.autoDecision && cur.autoDecision._key === issueText) return;
  const prior = cur?.autoDecision;
  const currentIter = cur?.latest?.iteration;
  // Only preserve prior decision if it's from the SAME iteration.
  // If iteration changed, old decisions are stale and must not be shown.
  const sameIter = prior && prior.iteration != null && currentIter != null
    ? prior.iteration === currentIter
    : true;  // if no iteration info, preserve for backwards compat
  outputsState[agent].autoDecision = {
    _key: issueText,
    iteration: currentIter,
    loading: true,
    // Keep prior fields visible while we wait ONLY if same iteration
    decision: sameIter ? prior?.decision : undefined,
    reasoning: sameIter ? prior?.reasoning : undefined,
    visual: sameIter ? prior?.visual : undefined,
  };
  renderOutputsPanel();
  try {
    const r = await api('POST', '/api/explain', {
      agent, issue: issueText,
      iteration: currentIter,
      evidence: cur?.latest?.metrics || {},
    });
    outputsState[agent].autoDecision = {
      _key: issueText,
      iteration: currentIter,
      decision: r.decision || r.explanation || '',
      reasoning: r.reasoning || '',
      visual: r.visual_to_check || '',
      loading: false,
    };
  } catch (err) {
    // On error, restore prior decision only if same iteration
    outputsState[agent].autoDecision = {
      _key: prior?._key || issueText,
      iteration: sameIter ? prior?.iteration : currentIter,
      decision: sameIter ? prior?.decision : undefined,
      reasoning: sameIter ? prior?.reasoning : undefined,
      visual: sameIter ? prior?.visual : undefined,
      error: err.message,
      loading: false,
    };
  }
  renderOutputsPanel();
}

function fmtKV(label, value, kind) {
  const k = `<span class="k">${escapeHtml(label)}</span>`;
  const v = `<span class="v ${kind || ''}">${escapeHtml(value)}</span>`;
  return `<div class="row">${k}${v}</div>`;
}

function buildOutputBody(agent, m) {
  const rows = [];
  const list = [];
  if (agent === 'UserInput') {
    if (m.target_entity)         rows.push(fmtKV('target', m.target_entity));
    if (m.purpose_length != null) rows.push(fmtKV('purpose', `${m.purpose_length} chars`));
    if (m.n_clusters_requested)  rows.push(fmtKV('k requested', m.n_clusters_requested));
    if (m.must_have_clusters && m.must_have_clusters.length)
      rows.push(fmtKV('must-have', m.must_have_clusters.join(', ')));
  } else if (agent === 'DatasetExaminer') {
    if (m.n_rows != null)            rows.push(fmtKV('rows', m.n_rows.toLocaleString()));
    if (m.n_numeric_cols != null)    rows.push(fmtKV('numeric cols', m.n_numeric_cols));
    if (m.mean_skewness != null) {
      const sk = Number(m.mean_skewness);
      rows.push(fmtKV('mean skew', sk.toFixed(2), sk > 3 ? 'warn' : 'good'));
    }
    if (m.n_suggested_groups != null) rows.push(fmtKV('feature groups suggested', m.n_suggested_groups));
    if (m.algo_hint)                  rows.push(fmtKV('algo hint', m.algo_hint));
  } else if (agent === 'FeatureEngineer') {
    if (m.n_entities != null)  rows.push(fmtKV('entities', m.n_entities.toLocaleString()));
    if (m.n_features != null)  rows.push(fmtKV('features built', m.n_features));
    if (m.n_builders_run != null) rows.push(fmtKV('builders run', m.n_builders_run));
  } else if (agent === 'FeatureSelector') {
    if (m.n_input_features != null)  rows.push(fmtKV('input', m.n_input_features));
    if (m.n_removed_by_vif != null)  rows.push(fmtKV('removed by VIF', m.n_removed_by_vif));
    if (m.n_selected != null)        rows.push(fmtKV('selected', m.n_selected, 'good'));
    if (m.max_vif_remaining != null) {
      const v = Number(m.max_vif_remaining);
      rows.push(fmtKV('max VIF remaining', v.toFixed(2), v > 10 ? 'warn' : 'good'));
    }
  } else if (agent === 'Clusterer') {
    if (m.algorithm)              rows.push(fmtKV('algorithm', m.algorithm));
    if (m.k_selected != null)     rows.push(fmtKV('k (auto-selected)', m.k_selected));
    if (m.n_leaf_clusters != null) rows.push(fmtKV('leaf clusters', m.n_leaf_clusters, 'good'));
    if (m.silhouette != null) {
      const s = Number(m.silhouette);
      const kind = s >= 0.25 ? 'good' : s >= 0.15 ? 'warn' : 'bad';
      rows.push(fmtKV('silhouette', s.toFixed(4), kind));
    }
    if (m.k_scores) {
      const ks = Object.entries(m.k_scores).slice(0, 5);
      list.push('<div class="k" style="margin-top:4px;font-size:10.5px">k-curve:</div>');
      list.push('<ul class="oc-list">' + ks.map(([k, v]) =>
        `<li>k=${k}: ${Number(v).toFixed(3)}</li>`).join('') + '</ul>');
    }
  } else if (agent === 'PersonaNamer') {
    if (m.n_clusters != null)   rows.push(fmtKV('clusters named', m.n_clusters));
    if (m.avg_confidence != null) {
      const c = Number(m.avg_confidence);
      rows.push(fmtKV('avg confidence', c.toFixed(1) + '/10', c >= 6 ? 'good' : 'bad'));
    }
    if (m.gate_passed != null)  rows.push(fmtKV('Clarity Gate', m.gate_passed ? 'PASSED' : 'FAILED', m.gate_passed ? 'good' : 'bad'));
    if (m.names_unique != null) rows.push(fmtKV('unique names', m.names_unique ? '✓' : '✗', m.names_unique ? 'good' : 'bad'));
    if (m.must_have_clusters && m.must_have_clusters.length)
      rows.push(fmtKV('must-have enforced', m.must_have_clusters.join(', ')));
  } else if (agent === 'Classifier') {
    if (m.model_name)            rows.push(fmtKV('model', m.model_name));
    if (m.cv_f1_macro != null) {
      const f = Number(m.cv_f1_macro);
      rows.push(fmtKV('CV F1 macro (gate ≥ 0.70)', f.toFixed(4), f >= 0.7 ? 'good' : 'bad'));
    }
    if (m.cv_f1_weighted != null) rows.push(fmtKV('CV F1 weighted', Number(m.cv_f1_weighted).toFixed(4)));
    if (m.n_classes != null)     rows.push(fmtKV('classes', m.n_classes));
  } else if (agent === 'Orchestrator') {
    if (m.silhouette_target_previous != null) {
      rows.push(fmtKV('target before', Number(m.silhouette_target_previous).toFixed(2)));
      rows.push(fmtKV('target after',  Number(m.silhouette_target_new).toFixed(2), 'good'));
      rows.push(fmtKV('mode', m.mode || ''));
    } else if (m.consecutive_failures != null) {
      rows.push(fmtKV('failures in a row', m.consecutive_failures));
      if (m.silhouette_target != null)
        rows.push(fmtKV('target at escalation', Number(m.silhouette_target).toFixed(2)));
    } else {
      rows.push(fmtKV('role', 'parameter tuning + escalation'));
    }
  }
  return rows.join('') + (list.length ? '\n' + list.join('') : '');
}

function renderOutputsPanel() {
  const wrap = document.getElementById('outputs-panel');
  if (!wrap) return;
  const order = ['UserInput', 'DatasetExaminer', 'FeatureEngineer',
                 'FeatureSelector', 'Clusterer', 'PersonaNamer',
                 'Classifier', 'Orchestrator'];
  const seen = order.filter((a) => outputsState[a]);
  if (!seen.length) {
    wrap.innerHTML = `<div class="outputs-empty muted">As each agent finishes, its computed result appears here.</div>`;
    return;
  }
  const STATUS_GLYPH = {success: '✓', warning: '⚠', blocked: '✗', failure: '!!'};
  wrap.innerHTML = seen.map((agent) => {
    const entry = outputsState[agent];
    const o = entry.latest;
    const history = entry.history || [];
    const body = buildOutputBody(agent, o.metrics) ||
      `<div class="muted">${escapeHtml(o.what_was_done || '')}</div>`;
    const issues = (o.issues || []).filter(Boolean);
    const warnHtml = issues.length ? `
      <div class="oc-warns">
        ${issues.map((iss, idx) => `
          <div class="oc-warn">
            <div class="text">⚠ ${escapeHtml(iss)}</div>
            <button data-warn-agent="${escapeHtml(agent)}" data-warn-idx="${idx}">Respond — tell agents how to handle this</button>
          </div>
        `).join('')}
      </div>` : '';
    const ad = entry.autoDecision;
    const hasDecision = ad && (ad.decision || ad.reasoning);
    const autoHtml = (ad && (ad.loading || hasDecision || ad.error)) ? `
      <div class="oc-auto">
        <div class="oc-auto-head">
          Pipeline decision
          <span class="muted">(bypass mode · evidence ledger${ad.iteration != null ? ` · iter ${ad.iteration}` : ''})</span>
          ${ad.loading ? `<span class="muted" style="color:var(--accent)"> · <span class="spinner"></span>updating…</span>` : ''}
        </div>
        ${hasDecision
          ? `<div class="oc-auto-text"><b>→</b> ${escapeHtml(ad.decision || '')}</div>
             ${ad.reasoning ? `<div class="oc-auto-reasoning"><b>Why:</b> ${escapeHtml(ad.reasoning)}</div>` : ''}
             ${ad.visual ? `<div class="oc-auto-visual"><b>Where to verify:</b> ${escapeHtml(ad.visual)}</div>` : ''}`
          : ad.error
            ? `<div class="muted" style="color:var(--bad)">${escapeHtml(ad.error)}</div>`
            : `<div class="muted">Asking LLM how the pipeline should handle this…</div>`}
      </div>` : '';
    // Per-iteration history strip — preserves the past so the user can see all attempts
    const historyHtml = history.length > 1 ? `
      <div class="oc-history" title="One pill per iteration this agent has run">
        ${history.map(h => `
          <span class="oc-pill ${h.status}" title="iter ${h.iteration} · ${h.status}: ${escapeHtml((h.what_was_done || '').slice(0, 120))}">
            iter ${h.iteration} ${STATUS_GLYPH[h.status] || ''}
          </span>
        `).join('')}
      </div>` : '';
    return `
      <div class="output-card ${o.status || 'success'}">
        <div class="oc-head">
          <div class="oc-name">${escapeHtml(agent)}</div>
          <div class="oc-iter">iter ${o.iteration} · ${o.status} · ${history.length} run${history.length > 1 ? 's' : ''}</div>
        </div>
        ${historyHtml}
        <div class="oc-body">${body}</div>
        ${warnHtml}
        ${autoHtml}
      </div>`;
  }).join('');

  // Wire up "Respond" buttons for warnings
  wrap.querySelectorAll('button[data-warn-agent]').forEach((btn) => {
    btn.onclick = () => {
      const agent = btn.dataset.warnAgent;
      const idx = Number(btn.dataset.warnIdx);
      const issue = (outputsState[agent]?.issues || [])[idx] || '';
      openWarnModal(agent, issue);
    };
  });
}

// ── Mode toggle (Bypass / Interactive) ─────────────────────────────────────
async function loadMode() {
  try {
    const r = await api('GET', '/api/mode');
    applyModeToButtons(r.mode || 'bypass');
  } catch (_) { applyModeToButtons('bypass'); }
  // If we just switched to bypass, fire auto-decisions on any existing
  // agent reports that have unresolved warnings (e.g. after a page refresh).
  if (state.currentMode === 'bypass') {
    Object.entries(outputsState).forEach(([agent, o]) => {
      if (o.issues && o.issues.length && !o.autoDecision) {
        triggerBypassAutoDecision(agent, o.issues);
      }
    });
  }
  autoScrollForDemo();
}
function applyModeToButtons(mode) {
  $$('#mode-toggle .mode-btn').forEach((b) => {
    b.classList.toggle('on', b.dataset.mode === mode);
  });
  state.currentMode = mode;
}
function wireModeToggle() {
  $$('#mode-toggle .mode-btn').forEach((btn) => {
    btn.onclick = async () => {
      const mode = btn.dataset.mode;
      try {
        await api('POST', '/api/mode', {mode});
        applyModeToButtons(mode);
        toast(mode === 'interactive'
          ? 'Interactive mode ON — pipeline will pause on warnings'
          : 'Bypass mode — agents auto-proceed past warnings',
          'success', 3500);
      } catch (e) { toast(e.message, 'error', 4000); }
    };
  });
}

// ── Mid-pipeline decision modal (interactive mode) ────────────────────────
let _pendingDecision = null;
function openDecisionModal(e) {
  _pendingDecision = e;
  $('#decision-agent').textContent = e.agent || 'An agent';
  $('#decision-context').innerHTML = `
    <div class="warn-from">From ${escapeHtml(e.agent || '?')} · iter ${e.iteration ?? '?'}</div>
    <div><b>What was done:</b> ${escapeHtml(e.what_was_done || '')}</div>
    ${(e.issues || []).map(i => `<div style="color:var(--warn);margin-top:4px">⚠ ${escapeHtml(i)}</div>`).join('')}
    ${e.doubts ? `<div style="margin-top:4px"><b>Doubts:</b> ${escapeHtml(e.doubts)}</div>` : ''}`;
  $('#decision-response').value = '';
  $('#decision-priority').value = 'high';
  $('#decision-modal').classList.remove('hidden');
}
function closeDecisionModal() {
  _pendingDecision = null;
  $('#decision-modal').classList.add('hidden');
}
async function submitDecision(action) {
  const response = $('#decision-response').value.trim();
  if (action === 'apply' && !response) {
    toast('Write your guidance, or click Bypass instead.', 'error', 3500);
    return;
  }
  const priority = $('#decision-priority').value;
  try {
    await api('POST', '/api/decision', {
      agent: _pendingDecision?.agent,
      response, action, priority,
    });
    closeDecisionModal();
    toast(action === 'apply'
      ? 'Decision sent — pipeline resuming with your guidance'
      : 'Bypassed — pipeline continuing',
      'success', 3000);
  } catch (e) { toast(e.message, 'error', 4000); }
}
// ── Silhouette-relax modal (interactive mode, 5 consecutive misses) ────────
function openRelaxModal(e) {
  const cur = Number(e.current_target || 0).toFixed(2);
  const sug = Number(e.suggested_target || 0).toFixed(2);
  $('#relax-context').innerHTML = `
    <div class="warn-from">After ${e.consecutive_failures || 5} consecutive iterations</div>
    <div>Current target: <b>${cur}</b> · suggested: <b>${sug}</b></div>`;
  $('#relax-target').value = sug;
  $('#relax-modal').classList.remove('hidden');
}
function closeRelaxModal() { $('#relax-modal').classList.add('hidden'); }
async function submitRelax(value) {
  try {
    await api('POST', '/api/silhouette-target', {target: Number(value)});
    closeRelaxModal();
    toast(`New silhouette_target = ${Number(value).toFixed(2)} — pipeline resuming`, 'success', 3500);
  } catch (e) { toast(e.message, 'error', 4500); }
}
function wireRelaxModal() {
  $('#relax-apply').onclick = () => {
    const v = parseFloat($('#relax-target').value);
    if (!(v > 0.05 && v < 1.0)) { toast('Target must be between 0.05 and 1.0', 'error', 3500); return; }
    submitRelax(v);
  };
  $('#relax-auto').onclick = () => {
    const v = parseFloat($('#relax-target').value);  // suggested is pre-filled
    submitRelax(v);
  };
  $('#relax-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'relax-modal') closeRelaxModal();
  });
}

function wireDecisionModal() {
  $('#decision-apply').onclick = () => submitDecision('apply');
  $('#decision-bypass').onclick = () => submitDecision('ignore');
  $('#decision-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'decision-modal') closeDecisionModal();
  });
}

// ── Threshold-decision modal (clusterer / orchestrator gates) ─────────────
// Triggered by `awaiting_threshold_decision` SSE events from
// skills/user_decisions.py. The event payload carries an option list +
// recommended key; we render one button per option and POST the chosen key
// back to /api/threshold-decision which unblocks the polling agent.
let _pendingThreshold = null;
let _thresholdTimeoutTimer = null;

function openThresholdModal(e) {
  _pendingThreshold = e;
  $('#threshold-title').textContent = `⏸ ${e.title || 'Threshold reached'}`;
  $('#threshold-summary').textContent = e.summary || '';
  const extra = e.extra || {};
  const fromLine = `<div class="warn-from">From ${escapeHtml(e.agent || '?')} · decision id <code>${escapeHtml(e.decision_id || '')}</code></div>`;
  // Render every non-empty extra field as a metric row so the user has
  // concrete evidence beside the prompt (silhouette value, cluster count, ...).
  const extraRows = Object.entries(extra)
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => `<div><b>${escapeHtml(k)}:</b> ${escapeHtml(String(v))}</div>`)
    .join('');
  $('#threshold-context').innerHTML = fromLine + extraRows;
  const optsHost = $('#threshold-options');
  optsHost.innerHTML = '';
  const options = Array.isArray(e.options) ? e.options : [];
  const recommended = e.recommended;
  options.forEach((opt) => {
    const btn = document.createElement('button');
    btn.className = opt.key === recommended ? 'primary' : 'ghost';
    btn.style.cssText = 'text-align:left;padding:10px 12px;display:flex;flex-direction:column;align-items:flex-start;gap:2px';
    btn.innerHTML = `
      <span><b>${escapeHtml(opt.label || opt.key)}</b>${opt.key === recommended ? ' <span class="muted" style="font-weight:normal">(recommended)</span>' : ''}</span>
      ${opt.description ? `<span class="muted" style="font-size:12px">${escapeHtml(opt.description)}</span>` : ''}`;
    btn.onclick = () => submitThresholdDecision(opt.key);
    optsHost.appendChild(btn);
  });
  $('#threshold-use-recommended').onclick = () => submitThresholdDecision(recommended);
  $('#threshold-modal').classList.remove('hidden');
  // Countdown until the backend auto-applies the recommended option.
  if (_thresholdTimeoutTimer) clearInterval(_thresholdTimeoutTimer);
  const deadline = Date.now() + (Number(e.timeout_s || 300) * 1000);
  const tickLabel = () => {
    const remaining = Math.max(0, Math.round((deadline - Date.now()) / 1000));
    const m = Math.floor(remaining / 60), s = remaining % 60;
    $('#threshold-timeout-label').textContent =
      remaining > 0
        ? `Auto-applies "${recommended}" in ${m}:${String(s).padStart(2, '0')} if no choice.`
        : 'Auto-applying recommended…';
  };
  tickLabel();
  _thresholdTimeoutTimer = setInterval(tickLabel, 1000);
}

function closeThresholdModal() {
  _pendingThreshold = null;
  if (_thresholdTimeoutTimer) { clearInterval(_thresholdTimeoutTimer); _thresholdTimeoutTimer = null; }
  $('#threshold-modal').classList.add('hidden');
}

async function submitThresholdDecision(chosenKey) {
  const decision = _pendingThreshold;
  if (!decision || !chosenKey) return;
  try {
    await api('POST', '/api/threshold-decision', {
      decision_id: decision.decision_id,
      chosen_key: chosenKey,
    });
    closeThresholdModal();
    toast(`Decision sent: ${chosenKey} — pipeline resuming`, 'success', 3000);
  } catch (e) {
    toast(`Could not submit decision: ${e.message}`, 'error', 4000);
  }
}

function wireThresholdModal() {
  $('#threshold-modal').addEventListener('click', (ev) => {
    // Don't close on background click — these are critical decisions.
    // The user must pick an option (or wait for the timeout).
    ev.stopPropagation();
  });
}

// ── Control-gates tuning modal (after dataset examination) ─────────────────
// Triggered by `awaiting_control_gates` SSE events. The user sets
// max_cluster_size_pct, sub_n_clusters, and max_depth before clustering.
let _pendingControlGates = null;
let _controlGatesTimeoutTimer = null;

function openControlGatesModal(e) {
  _pendingControlGates = e;
  const defaults = e.defaults || {};
  $('#cg-max-pct').value = defaults.max_cluster_size_pct ?? 0.40;
  $('#cg-sub-k').value = defaults.sub_n_clusters ?? 3;
  $('#cg-max-depth').value = defaults.max_depth ?? 2;

  // Build a friendly stats summary
  const stats = e.dataset_stats || {};
  const lines = [];
  if (stats.n_rows) lines.push(`${Number(stats.n_rows).toLocaleString()} rows`);
  if (stats.n_features) lines.push(`${stats.n_features} features`);
  if (stats.modality) lines.push(`modality=${stats.modality}`);
  if (stats.mean_abs_skewness != null) lines.push(`skew=${stats.mean_abs_skewness}`);
  $('#control-gates-context').innerHTML =
    lines.length ? `<b>Dataset:</b> ${lines.join(' · ')}` : '';

  $('#control-gates-modal').classList.remove('hidden');

  if (_controlGatesTimeoutTimer) clearInterval(_controlGatesTimeoutTimer);
  const deadline = Date.now() + (Number(e.timeout_s || 300) * 1000);
  const tickLabel = () => {
    const remaining = Math.max(0, Math.round((deadline - Date.now()) / 1000));
    const m = Math.floor(remaining / 60), s = remaining % 60;
    $('#control-gates-timeout-label').textContent =
      remaining > 0
        ? `Auto-continues with defaults in ${m}:${String(s).padStart(2, '0')}.`
        : 'Auto-continuing with defaults…';
  };
  tickLabel();
  _controlGatesTimeoutTimer = setInterval(tickLabel, 1000);
}

function closeControlGatesModal() {
  _pendingControlGates = null;
  if (_controlGatesTimeoutTimer) { clearInterval(_controlGatesTimeoutTimer); _controlGatesTimeoutTimer = null; }
  $('#control-gates-modal').classList.add('hidden');
}

async function submitControlGates(useDefaults) {
  const payload = useDefaults
    ? {
        max_cluster_size_pct: (_pendingControlGates?.defaults?.max_cluster_size_pct) ?? 0.40,
        sub_n_clusters: (_pendingControlGates?.defaults?.sub_n_clusters) ?? 3,
        max_depth: (_pendingControlGates?.defaults?.max_depth) ?? 2,
      }
    : {
        max_cluster_size_pct: Number($('#cg-max-pct').value),
        sub_n_clusters: Number($('#cg-sub-k').value),
        max_depth: Number($('#cg-max-depth').value),
      };
  try {
    await api('POST', '/api/control-gates', payload);
    closeControlGatesModal();
    toast(
      `Control gates set: max=${payload.max_cluster_size_pct}% · sub=${payload.sub_n_clusters} · depth=${payload.max_depth}`,
      'success', 3000
    );
  } catch (err) {
    toast(`Could not submit control gates: ${err.message}`, 'error', 4000);
  }
}

function wireControlGatesModal() {
  $('#cg-apply').onclick = () => submitControlGates(false);
  $('#cg-use-defaults').onclick = () => submitControlGates(true);
  $('#control-gates-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'control-gates-modal') closeControlGatesModal();
  });
}

// ── Column-resolution modal (after dataset examination when ambiguous) ─────
let _pendingColumnResolution = null;
let _columnResolutionTimeoutTimer = null;

function openColumnResolutionModal(e) {
  _pendingColumnResolution = e;
  const ambig = e.ambiguous_roles || {};
  const heur = e.heuristics || {};
  const allCols = (e.schema || '').split('\n').map(l => l.trim()).filter(l => l.startsWith('  '));
  const colNames = allCols.map(l => l.split(':')[0].trim());

  function _buildSelect(id, role, candidates, fallback) {
    const sel = $(id);
    sel.innerHTML = '';
    const optBlank = document.createElement('option');
    optBlank.value = '';
    optBlank.textContent = '-- none --';
    sel.appendChild(optBlank);
    const cands = candidates && candidates.length ? candidates : colNames;
    for (const c of cands) {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = c;
      if (c === fallback) opt.selected = true;
      sel.appendChild(opt);
    }
    // If no candidate matched the fallback, add it
    if (fallback && !Array.from(sel.options).some(o => o.value === fallback)) {
      const opt = document.createElement('option');
      opt.value = fallback;
      opt.textContent = fallback + ' (detected)';
      opt.selected = true;
      sel.insertBefore(opt, sel.options[1]);
    }
  }

  _buildSelect('#cr-entity-id', 'entity_id', ambig.entity_id, heur.entity_id);
  _buildSelect('#cr-timestamp', 'timestamp', ambig.timestamp, heur.timestamp);
  _buildSelect('#cr-amount', 'amount', ambig.amount, heur.amount);
  _buildSelect('#cr-category', 'category', ambig.category, heur.category);

  const stats = e.dataset_stats || {};
  const lines = [];
  if (stats.n_rows) lines.push(`${Number(stats.n_rows).toLocaleString()} rows`);
  if (stats.n_cols) lines.push(`${stats.n_cols} columns`);
  $('#column-resolution-context').innerHTML =
    lines.length ? `<b>Dataset:</b> ${lines.join(' · ')}` : '';

  $('#column-resolution-modal').classList.remove('hidden');

  if (_columnResolutionTimeoutTimer) clearInterval(_columnResolutionTimeoutTimer);
  const deadline = Date.now() + (Number(e.timeout_s || 300) * 1000);
  const tickLabel = () => {
    const remaining = Math.max(0, Math.round((deadline - Date.now()) / 1000));
    const m = Math.floor(remaining / 60), s = remaining % 60;
    $('#column-resolution-timeout-label').textContent =
      remaining > 0
        ? `Auto-continues with detected in ${m}:${String(s).padStart(2, '0')}.`
        : 'Auto-continuing with detected…';
  };
  tickLabel();
  _columnResolutionTimeoutTimer = setInterval(tickLabel, 1000);
}

function closeColumnResolutionModal() {
  _pendingColumnResolution = null;
  if (_columnResolutionTimeoutTimer) { clearInterval(_columnResolutionTimeoutTimer); _columnResolutionTimeoutTimer = null; }
  $('#column-resolution-modal').classList.add('hidden');
}

async function submitColumnResolution(useHeuristics) {
  const payload = useHeuristics
    ? {
        entity_id: (_pendingColumnResolution?.heuristics?.entity_id) || null,
        timestamp: (_pendingColumnResolution?.heuristics?.timestamp) || null,
        amount: (_pendingColumnResolution?.heuristics?.amount) || null,
        category: (_pendingColumnResolution?.heuristics?.category) || null,
      }
    : {
        entity_id: $('#cr-entity-id').value || null,
        timestamp: $('#cr-timestamp').value || null,
        amount: $('#cr-amount').value || null,
        category: $('#cr-category').value || null,
      };
  try {
    await api('POST', '/api/column-resolution', payload);
    closeColumnResolutionModal();
    toast('Column roles confirmed — pipeline continuing.', 'success', 3000);
  } catch (err) {
    toast(`Could not submit column resolution: ${err.message}`, 'error', 4000);
  }
}

function wireColumnResolutionModal() {
  $('#cr-apply').onclick = () => submitColumnResolution(false);
  $('#cr-use-heuristics').onclick = () => submitColumnResolution(true);
  $('#column-resolution-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'column-resolution-modal') closeColumnResolutionModal();
  });
}

// ── Human-checkpoint modal (end of every passing iteration) ────────────────
// Triggered by `awaiting_human_checkpoint` SSE events emitted by
// agents/orchestrator.py::_collect_human_decision. Shows the per-iteration
// stats + persona table; user clicks Approve / Re-cluster / Re-select / Quit;
// POST /api/human-checkpoint writes outputs/pending_human_checkpoint.json
// which the paused orchestrator consumes.
let _pendingCheckpoint = null;
let _checkpointTimeoutTimer = null;

function openCheckpointModal(e) {
  _pendingCheckpoint = e;
  const sil = (e.silhouette != null) ? Number(e.silhouette).toFixed(3) : 'n/a';
  const f1 = (e.cv_f1_macro != null) ? Number(e.cv_f1_macro).toFixed(3) : 'n/a';
  const acc = (e.cv_accuracy != null) ? Number(e.cv_accuracy).toFixed(3) : 'n/a';
  $('#checkpoint-stats').innerHTML = `
    <div><b>Algorithm:</b> ${escapeHtml(e.algorithm || '?')}
         &nbsp;·&nbsp; <b>Leaf clusters:</b> ${e.n_leaf ?? '?'}</div>
    <div><b>Silhouette:</b> ${sil}
         &nbsp;·&nbsp; <b>CV F1 (macro):</b> ${f1}
         &nbsp;·&nbsp; <b>CV accuracy:</b> ${acc}</div>`;
  const rows = (e.personas || []).map(p => {
    const f1Cell = (p.cv_f1 != null) ? Number(p.cv_f1).toFixed(3) : 'n/a';
    return `<tr>
      <td><b>C${escapeHtml(String(p.cid))}</b></td>
      <td>${escapeHtml(String(p.name || ''))}</td>
      <td class="muted">${escapeHtml(String(p.tagline || ''))}</td>
      <td style="text-align:right">${p.confidence ?? '?'}</td>
      <td style="text-align:right">${f1Cell}</td>
    </tr>`;
  }).join('');
  $('#checkpoint-personas').innerHTML = rows
    ? `<table style="width:100%;border-collapse:collapse;font-size:12.5px">
         <thead><tr style="text-align:left;border-bottom:1px solid var(--border)">
           <th>#</th><th>Persona</th><th>Tagline</th>
           <th style="text-align:right">Conf</th><th style="text-align:right">CV-F1</th>
         </tr></thead><tbody>${rows}</tbody></table>`
    : '<div class="muted">No persona table available for this iteration.</div>';
  $('#checkpoint-feedback').value = '';
  $('#checkpoint-modal').classList.remove('hidden');

  if (_checkpointTimeoutTimer) clearInterval(_checkpointTimeoutTimer);
  const deadline = Date.now() + (Number(e.timeout_s || 600) * 1000);
  const tickLabel = () => {
    const remaining = Math.max(0, Math.round((deadline - Date.now()) / 1000));
    const m = Math.floor(remaining / 60), s = remaining % 60;
    $('#checkpoint-timeout-label').textContent = remaining > 0
      ? `Auto-applies "Approve" in ${m}:${String(s).padStart(2, '0')}.`
      : 'Auto-applying approve…';
  };
  tickLabel();
  _checkpointTimeoutTimer = setInterval(tickLabel, 1000);
}

function closeCheckpointModal() {
  _pendingCheckpoint = null;
  if (_checkpointTimeoutTimer) { clearInterval(_checkpointTimeoutTimer); _checkpointTimeoutTimer = null; }
  $('#checkpoint-modal').classList.add('hidden');
}

async function submitCheckpoint(action) {
  if (!_pendingCheckpoint) return;
  const feedback = ($('#checkpoint-feedback')?.value || '').trim();
  try {
    await api('POST', '/api/human-checkpoint', {action, feedback});
    closeCheckpointModal();
    toast(`Checkpoint: ${action} — pipeline resuming`, 'success', 3000);
  } catch (e) {
    toast(`Could not submit decision: ${e.message}`, 'error', 4000);
  }
}

function wireCheckpointModal() {
  $('#chk-approve').onclick   = () => submitCheckpoint('approve');
  $('#chk-recluster').onclick = () => submitCheckpoint('recluster');
  $('#chk-reselect').onclick  = () => submitCheckpoint('reselect_features');
  $('#chk-quit').onclick      = () => submitCheckpoint('quit');
  // Background click closes — the orchestrator's terminal fallback will still
  // unblock the run if the user wants to handle it that way.
  $('#checkpoint-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'checkpoint-modal') closeCheckpointModal();
  });
}

// Bypass-mode auto-applied threshold decisions. Each event is appended to this
// list and surfaced as a persistent banner in the Evidence tab so the user
// can review what was relaxed during an unattended run.
const _autoAppliedDecisions = [];
function recordAutoAppliedDecision(e) {
  // De-duplicate by decision_id — the same decision can't be auto-applied twice
  // in a single run; if we see a repeat it's an SSE replay.
  if (_autoAppliedDecisions.some(d => d.decision_id === e.decision_id)) return;
  _autoAppliedDecisions.push({
    decision_id: e.decision_id,
    agent: e.agent,
    title: e.title,
    summary: e.summary,
    chosen: e.chosen,
    options: e.options || [],
    extra: e.extra || {},
    ts: e.ts,
  });
  // If Evidence is visible, refresh so the banner appears immediately.
  if (!document.getElementById('evidence-view').classList.contains('hidden')) {
    renderEvidence();
  }
}

function buildAutoAppliedDecisionsCard() {
  if (!_autoAppliedDecisions.length) return '';
  const rows = _autoAppliedDecisions.map((d) => {
    const chosenOpt = (d.options || []).find(o => o.key === d.chosen) || {};
    const extraBits = Object.entries(d.extra || {})
      .filter(([, v]) => v !== null && v !== undefined && v !== '')
      .map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(String(v))}`)
      .join(' · ');
    return `
      <div class="explain-section" style="border-left:3px solid var(--warn);padding-left:10px;margin-bottom:8px">
        <div><b>${escapeHtml(d.title || d.decision_id)}</b>
          <span class="muted" style="font-size:11px">· ${escapeHtml(d.agent || '?')}</span></div>
        <div class="muted" style="font-size:12px;margin:2px 0 4px">${escapeHtml(d.summary || '')}</div>
        <div>Applied: <b>${escapeHtml(chosenOpt.label || d.chosen)}</b>
          <span class="muted">— ${escapeHtml(chosenOpt.description || '(recommended fallback)')}</span></div>
        ${extraBits ? `<div class="muted" style="font-size:11px;margin-top:2px">${extraBits}</div>` : ''}
      </div>`;
  }).join('');
  return `
    <div class="ev-card span2">
      <h3>⚠ Auto-applied threshold decisions <span class="iter">${_autoAppliedDecisions.length} in bypass mode</span></h3>
      <p class="lead">The pipeline ran in <b>bypass mode</b>, so these decision
        points did not pause to ask. Each one was resolved by applying the
        recommended fallback. Review and re-run interactively if any choice
        was wrong for your use case.</p>
      ${rows}
    </div>`;
}

// ── Case-memory recall modal (DatasetExaminer matched a prior run) ─────────
function openRecallModal(e) {
  const matchType = (e.match_type || 'similar').toLowerCase();
  const isExact = matchType === 'exact';
  $('#recall-title').textContent = isExact
    ? '🧠 Exact match — reuse the prior winning recipe?'
    : '🧠 Similar prior run found — reuse, modify, or ignore?';
  const intro = isExact
    ? 'Same dataset fingerprint as a prior run. The recipe below converged before — strong candidate for reuse.'
    : 'Different dataset/goal but same family. The prior recipe is inspiration only — do not assume it transfers cleanly.';
  const fmt = (v) => (v == null || v === '') ? '<span class="muted">—</span>'
    : (typeof v === 'number' ? Number(v).toLocaleString() : String(v));
  const fmtFloat = (v, d = 3) => (v == null || v === '') ? '<span class="muted">—</span>'
    : Number(v).toFixed(d);
  $('#recall-context').innerHTML = `
    <div class="warn-from">${escapeHtml(intro)}</div>
    <div class="warn-from" style="margin-top:6px"><b>Match:</b> ${escapeHtml(e.notes || '')}</div>
    <div style="margin-top:10px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 18px;font-size:12.5px;">
      <div><b>Prior dataset:</b> ${fmt(e.prior_dataset)}</div>
      <div><b>Prior purpose:</b> ${fmt((e.prior_purpose || '').slice(0, 80))}</div>
      <div><b>Algorithm:</b> ${fmt(e.prior_algorithm)}</div>
      <div><b>k:</b> ${fmt(e.prior_k)}</div>
      <div><b>vif_threshold:</b> ${fmt(e.prior_vif_threshold)}</div>
      <div><b>min_silhouette:</b> ${fmtFloat(e.prior_min_silhouette, 2)}</div>
      <div><b>features kept:</b> ${fmt(e.prior_n_features_kept)}</div>
      <div><b>Prior outcome:</b> sil=${fmtFloat(e.prior_silhouette)} · F1=${fmtFloat(e.prior_cv_f1_macro)}</div>
    </div>
    <div class="muted" style="margin-top:10px;font-size:11.5px">
      Pipeline is paused — pick one within 5 minutes or it defaults to <b>Modify</b>.
    </div>`;
  $('#recall-modal').classList.remove('hidden');
}

function closeRecallModal() { $('#recall-modal').classList.add('hidden'); }

async function submitRecallDecision(decision) {
  try {
    await api('POST', '/api/case-recall-decision', {decision});
    closeRecallModal();
    const label = {reuse: 'Reusing prior recipe', modify: 'Recall kept as hint', ignore: 'Recall ignored'}[decision] || decision;
    toast(`${label} — pipeline resuming`, 'success', 3500);
  } catch (e) { toast(e.message, 'error', 4500); }
}

function wireRecallModal() {
  $('#recall-reuse').onclick  = () => submitRecallDecision('reuse');
  $('#recall-modify').onclick = () => submitRecallDecision('modify');
  $('#recall-ignore').onclick = () => submitRecallDecision('ignore');
  $('#recall-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'recall-modal') closeRecallModal();
  });
}

// ── Warning-respond modal ────────────────────────────────────────────────
let _pendingWarn = null;
function openWarnModal(agent, issue) {
  _pendingWarn = {agent, issue};
  const ctx = document.getElementById('warn-context');
  ctx.innerHTML = `
    <div class="warn-from">From ${escapeHtml(agent)}</div>
    <div>${escapeHtml(issue)}</div>`;
  document.getElementById('warn-response').value = '';
  document.getElementById('warn-priority').value = 'high';
  document.getElementById('warn-modal').classList.remove('hidden');
}
function closeWarnModal() {
  _pendingWarn = null;
  document.getElementById('warn-modal').classList.add('hidden');
}
function wireWarnModal() {
  document.getElementById('warn-cancel').onclick = closeWarnModal;
  document.getElementById('warn-modal').addEventListener('click', (ev) => {
    if (ev.target.id === 'warn-modal') closeWarnModal();
  });
  document.getElementById('warn-save').onclick = async () => {
    const text = document.getElementById('warn-response').value.trim();
    if (!text) { toast('Write your guidance first', 'error', 3000); return; }
    if (!_pendingWarn) return;
    const priority = document.getElementById('warn-priority').value;
    // Frame the rule with the warning context so agents understand why it exists
    const rule = `Guidance for ${_pendingWarn.agent} warning ("${_pendingWarn.issue}"): ${text}`;
    const btn = document.getElementById('warn-save');
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>Saving…`;
    try {
      await api('POST', '/api/feedback/global', {rule, priority});
      closeWarnModal();
      toast('Saved. The Decision Maker will see it on the next prompt.', 'success', 3500);
    } catch (e) {
      toast(e.message, 'error', 4000);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Save & apply';
    }
  };
}

// ── Token + cost accumulator ────────────────────────────────────────────────
// Sonnet-tier prices used for an illustrative running total.
const PRICE_IN_PER_M  = 3.0;
const PRICE_OUT_PER_M = 15.0;

const costStats = {};         // pipeline category — agent -> {calls, in, out, time, running}
const costEvidenceStats = {}; // evidence category — same shape
const costNamingStats   = {}; // naming-discussion category — per cluster chat

function costOf(inTok, outTok) {
  return (inTok * PRICE_IN_PER_M + outTok * PRICE_OUT_PER_M) / 1_000_000;
}

function fmtNum(n) {
  return Number(n || 0).toLocaleString();
}
function fmtUsd(n) {
  return '$' + Number(n || 0).toFixed(4);
}

function _ledger(category) {
  if (category === 'evidence') return costEvidenceStats;
  if (category === 'naming')   return costNamingStats;
  return costStats;
}
// Dedupe set so SSE replay doesn't double-count tokens / calls
const _countedCalls = new Set();

function _callKey(e) {
  // ts + agent + purpose is unique enough for our event stream
  return `${e.ts || ''}::${e.agent || ''}::${e.purpose || ''}`;
}
function noteCallStart(agent, category, e) {
  const ledger = _ledger(category);
  ledger[agent] = ledger[agent] || {calls: 0, in: 0, out: 0, time: 0, running: 0};
  ledger[agent].running = (ledger[agent].running || 0) + 1;
  renderCostPanel();
}
function noteCallFinish(agent, inTok, outTok, time_s, category, e) {
  const key = e ? _callKey(e) : null;
  if (key && _countedCalls.has(key)) {
    // Already counted (SSE replayed it); just clear the running flag below.
    const ledger = _ledger(category);
    if (ledger[agent]) ledger[agent].running = Math.max(0, (ledger[agent].running || 1) - 1);
    renderCostPanel();
    return;
  }
  if (key) _countedCalls.add(key);
  const ledger = _ledger(category);
  const s = ledger[agent] = ledger[agent] || {calls: 0, in: 0, out: 0, time: 0, running: 0};
  s.calls += 1;
  s.in += Number(inTok || 0);
  s.out += Number(outTok || 0);
  s.time += Number(time_s || 0);
  s.running = Math.max(0, (s.running || 1) - 1);
  renderCostPanel();
}

function _renderLedger(ledger, tbodyId, emptyMsg) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return {totIn: 0, totOut: 0, totCost: 0};
  const agents = Object.keys(ledger);
  if (!agents.length) {
    tbody.innerHTML = `<tr class="empty"><td colspan="6" class="muted">${emptyMsg}</td></tr>`;
    return {totIn: 0, totOut: 0, totCost: 0};
  }
  const ordered = AGENT_STEPS.map((s) => s.key).concat(['Orchestrator'])
    .filter((k) => ledger[k]);
  for (const k of agents) if (!ordered.includes(k)) ordered.push(k);

  let totIn = 0, totOut = 0, totCost = 0;
  tbody.innerHTML = ordered.map((agent) => {
    const s = ledger[agent];
    const c = costOf(s.in, s.out);
    totIn += s.in; totOut += s.out; totCost += c;
    return `
      <tr${s.running ? ' class="running"' : ''}>
        <td>${escapeHtml(agent)}</td>
        <td>${fmtNum(s.calls)}</td>
        <td>${fmtNum(s.in)}</td>
        <td>${fmtNum(s.out)}</td>
        <td>${(s.time || 0).toFixed(1)}s</td>
        <td>${fmtUsd(c)}</td>
      </tr>`;
  }).join('');
  return {totIn, totOut, totCost};
}

function renderCostPanel() {
  const pipe = _renderLedger(costStats, 'cost-tbody',
    'No pipeline LLM calls yet — waiting for the first agent to consult the Decision Maker.');
  const ev = _renderLedger(costEvidenceStats, 'cost-evidence-tbody',
    'No evidence LLM calls yet. Click "Explain this warning" on any warning in the Evidence tab.');
  const nm = _renderLedger(costNamingStats, 'cost-naming-tbody',
    'No naming chats yet. Open any cluster card and click "Discuss with agent".');

  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set('cost-total-tokens', `${fmtNum(pipe.totIn + pipe.totOut)} (${fmtNum(pipe.totIn)} in / ${fmtNum(pipe.totOut)} out)`);
  set('cost-total-usd', fmtUsd(pipe.totCost));
  set('cost-evidence-tokens', `${fmtNum(ev.totIn + ev.totOut)} (${fmtNum(ev.totIn)} in / ${fmtNum(ev.totOut)} out)`);
  set('cost-evidence-usd', fmtUsd(ev.totCost));
  set('cost-naming-tokens', `${fmtNum(nm.totIn + nm.totOut)} (${fmtNum(nm.totIn)} in / ${fmtNum(nm.totOut)} out)`);
  set('cost-naming-usd', fmtUsd(nm.totCost));
}

// ── Agent ↔ Decision Maker conversation bubbles (typed live) ───────────────
//
// For each `llm_call_started` we render an "ask" bubble that types out the
// prompt; for each `llm_call_finished` we render an "answer" bubble that
// types out the response. Typewriter speed scales with length so even big
// prompts finish in ~1.5s.

function _short(s, n) {
  s = String(s || '');
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function _initials(name) {
  const w = String(name || '').replace(/[^A-Za-z ]/g, '').split(/\s+/).filter(Boolean);
  return ((w[0] || '?')[0] + (w[1] ? w[1][0] : '')).toUpperCase();
}

// Type `text` into `elem` over ~`totalMs` ms. Returns a function to cancel.
function typewriter(elem, text, totalMs = 1400) {
  elem.classList.add('typing');
  const safe = String(text || '');
  // Chunk so very long prompts (5–15k chars) still feel snappy.
  const totalChars = safe.length || 1;
  const ticks = Math.min(80, Math.max(12, Math.ceil(totalMs / 30)));
  const charsPerTick = Math.max(8, Math.ceil(totalChars / ticks));
  const intervalMs = Math.max(15, Math.floor(totalMs / ticks));
  let i = 0;
  let cancelled = false;
  function step() {
    if (cancelled) return;
    const next = Math.min(totalChars, i + charsPerTick);
    elem.textContent = safe.slice(0, next);
    elem.parentElement.scrollTop = elem.parentElement.scrollHeight;
    const wrap = document.getElementById('convos');
    if (wrap) wrap.scrollTop = wrap.scrollHeight;
    i = next;
    if (i < totalChars) setTimeout(step, intervalMs);
    else elem.classList.remove('typing');
  }
  step();
  return () => { cancelled = true; elem.classList.remove('typing'); };
}

// id → {askBodyElem, askTextElem, answerBodyElem, answerTextElem}
const _convoNodes = new Map();
function _convoKey(e) {
  return `${e.agent}::${e.purpose}::${e.ts}`;
}
function _convoKeyForFinished(e) {
  // The finished event might not match the started ts exactly, so look up
  // the most recent ask-bubble for this (agent, purpose) pair.
  for (const k of Array.from(_convoNodes.keys()).reverse()) {
    if (k.startsWith(`${e.agent}::${e.purpose}::`)) return k;
  }
  return null;
}

function renderAskBubble(e) {
  const wrap = document.getElementById('convos');
  if (!wrap) return;
  const id = _convoKey(e);
  if (_convoNodes.has(id)) return;
  const bubble = document.createElement('div');
  bubble.className = 'convo-bubble ask';
  bubble.innerHTML = `
    <div class="convo-avatar">${_initials(e.agent)}</div>
    <div class="convo-body">
      <div class="convo-meta">
        <b>${escapeHtml(e.agent)}</b> asks <b>Decision Maker</b> for LLM reasoning
        <span>· "${escapeHtml(_short(e.purpose, 80))}"</span>
        <span>· ${e.prompt_chars || 0} chars</span>
      </div>
      <div class="convo-text"></div>
    </div>`;
  wrap.appendChild(bubble);
  const textElem = bubble.querySelector('.convo-text');
  _convoNodes.set(id, {bubble, textElem});
  const fullPrompt = e.prompt || e.prompt_preview || '';
  // Cap the visible prompt at ~3000 chars (preserves performance for huge prompts)
  const visible = fullPrompt.length > 3000
    ? fullPrompt.slice(0, 3000) + `\n\n…(+${fullPrompt.length - 3000} more chars hidden)`
    : fullPrompt;
  // Scale typing duration with size, but keep snappy
  const dur = Math.min(2500, 600 + visible.length * 0.6);
  typewriter(textElem, visible, dur);
}

function renderAnswerBubble(e) {
  const wrap = document.getElementById('convos');
  if (!wrap) return;
  const askKey = _convoKeyForFinished(e);
  // Build the answer bubble
  const bubble = document.createElement('div');
  bubble.className = 'convo-bubble answer';
  const tokenInfo = `in=${e.input_tokens || 0} · out=${e.output_tokens || 0} · ${e.time_s || 0}s`;
  bubble.innerHTML = `
    <div class="convo-avatar">🧠</div>
    <div class="convo-body">
      <div class="convo-meta">
        <b>Decision Maker</b> called the LLM, returns to <b>${escapeHtml(e.agent)}</b>
        <span>· ${escapeHtml(tokenInfo)}</span>
      </div>
      <div class="convo-text"></div>
    </div>`;
  wrap.appendChild(bubble);
  const textElem = bubble.querySelector('.convo-text');
  const fullResp = e.response || e.response_preview || '';
  const visible = fullResp.length > 3000
    ? fullResp.slice(0, 3000) + `\n\n…(+${fullResp.length - 3000} more chars hidden)`
    : fullResp;
  const dur = Math.min(2500, 600 + visible.length * 0.6);
  typewriter(textElem, visible, dur);
}

function handleEvent(e) {
  state.events.push(e);
  const ev = e.event;
  // Refresh the architecture graph on any event — agent reports AND LLM call
  // starts/finishes so the active-agent + thinking-orchestrator are live.
  setTimeout(applyArchFromEvents, 0);
  if (ev === 'silhouette_target_missed') {
    const best = e.candidate_best || {};
    const algoLabel = best.algorithm || 'N/A';
    const kLabel = best.k || 'N/A';
    const comp = best.composite_score != null ? Number(best.composite_score).toFixed(3) : 'N/A';
    const db = best.davies_bouldin != null ? Number(best.davies_bouldin).toFixed(2) : 'N/A';
    const ch = best.calinski_harabasz != null ? Number(best.calinski_harabasz).toFixed(1) : 'N/A';
    const ari = best.stability_ari != null ? Number(best.stability_ari).toFixed(3) : 'N/A';
    const algosStr = (e.algorithms || []).join(', ');
    const logMsg = (
      `[${(e.ts || '').slice(11,19)}] ESCALATION CHECK - ` +
      `silhouette ${Number(e.silhouette || 0).toFixed(3)} < target ${Number(e.target || 0).toFixed(2)} ` +
      `(re-eng ${e.consecutive_failures}/${e.max_failures} - relax ${e.relax_failures || 0}/${e.max_relax_failures || 5})` +
      ` | AutoML: ${e.n_candidates || 0} candidates (${algosStr}), best=${algoLabel}(k=${kLabel}, comp=${comp}, sil=${(best.silhouette != null ? Number(best.silhouette).toFixed(3) : 'N/A')}, DB=${db}, CH=${ch}, ARI=${ari})`
    );
    appendLogLine(logMsg);
    const toastMsg = (
      `Silhouette ${Number(e.silhouette).toFixed(3)} < ${Number(e.target).toFixed(2)} ` +
      `- re-eng ${e.consecutive_failures}/${e.max_failures} ` +
      `- AutoML best: ${algoLabel} k=${kLabel} (comp=${comp})`
    );
    toast(toastMsg, e.consecutive_failures >= e.max_failures ? 'error' : 'success', 5000);
    return;
  }
  if (ev === 'feature_re_engineering') {
    appendLogLine(`[${(e.ts || '').slice(11,19)}] ESCALATION — re-engineering features (${e.consecutive_failures} failures in a row)`);
    toast(`Escalating — ${e.consecutive_failures} failures, re-engineering features from raw data`, 'error', 5000);
    recordAgentOutput({
      event: 'agent_report',
      agent: 'Orchestrator',
      iteration: state.events.filter(x => x.event === 'iteration_started').slice(-1)[0]?.iteration ?? '?',
      status: 'warning',
      what_was_done: `ESCALATION — re-engineered features from raw data after ${e.consecutive_failures} silhouette misses · cleared algorithm pick so Decision Maker chooses fresh`,
      metrics: {
        consecutive_failures: e.consecutive_failures,
        silhouette_target: e.silhouette_target,
      },
      issues: [],
    });
    return;
  }
  if (ev === 'awaiting_silhouette_relaxation') {
    const alreadyAnswered = state.events.some(
      x => (x.event === 'silhouette_target_relaxed' || x.event === 'threshold_decision_made') && x.iteration === e.iteration
    );
    if (state.currentMode === 'bypass' || alreadyAnswered) {
      appendLogLine(`[${(e.ts || '').slice(11,19)}] RELAX SKIPPED — iter ${e.iteration} (bypass=${state.currentMode==='bypass'}, answered=${alreadyAnswered})`);
      return;
    }
    showLivePanel(true);
    openRelaxModal(e);
    toast('⏸ Pipeline paused — silhouette target needs your input', 'error', 6000);
    notifyUser('Pipeline paused', 'Silhouette target relaxation — the pipeline needs your decision.');
    appendLogLine(`[${(e.ts || '').slice(11,19)}] PAUSED — 5 silhouette misses, asking you to lower the target`);
    return;
  }
  if (ev === 'awaiting_control_gates') {
    const alreadyAnswered = state.events.some(
      x => x.event === 'control_gates_tuned' && x.iteration === e.iteration
    );
    if (state.currentMode === 'bypass' || alreadyAnswered) {
      appendLogLine(`[${(e.ts || '').slice(11,19)}] GATES SKIPPED — iter ${e.iteration} (bypass=${state.currentMode==='bypass'}, answered=${alreadyAnswered})`);
      return;
    }
    showLivePanel(true);
    openControlGatesModal(e);
    toast('⏸ Pipeline paused — control gates need your input', 'error', 6000);
    notifyUser('Pipeline paused', 'Control gates tuning — the pipeline needs your decision.');
    appendLogLine(`[${(e.ts || '').slice(11,19)}] PAUSED — dataset examined, asking you to tune control gates`);
    return;
  }
  if (ev === 'control_gates_tuned') {
    appendLogLine(`[${(e.ts || '').slice(11,19)}] Control gates tuned (${e.mode} · ${e.source}): max=${(e.max_cluster_size_pct * 100).toFixed(0)}% sub=${e.sub_n_clusters} depth=${e.max_depth}`);
    return;
  }
  if (ev === 'awaiting_column_resolution') {
    const alreadyAnswered = state.events.some(
      x => x.event === 'columns_resolved' && x.iteration === e.iteration
    );
    if (state.currentMode === 'bypass' || alreadyAnswered) {
      appendLogLine(`[${(e.ts || '').slice(11,19)}] COLUMNS SKIPPED — iter ${e.iteration} (bypass=${state.currentMode==='bypass'}, answered=${alreadyAnswered})`);
      return;
    }
    showLivePanel(true);
    openColumnResolutionModal(e);
    toast('⏸ Pipeline paused — column roles need your input', 'error', 6000);
    notifyUser('Pipeline paused', 'Column resolution — the pipeline needs your decision.');
    appendLogLine(`[${(e.ts || '').slice(11,19)}] PAUSED — ambiguous columns, asking you to confirm roles`);
    return;
  }
  if (ev === 'columns_resolved') {
    appendLogLine(`[${(e.ts || '').slice(11,19)}] Columns resolved (${e.mode} · ${e.source}): entity=${e.entity_id || 'N/A'} ts=${e.timestamp || 'N/A'} amount=${e.amount || 'N/A'} category=${e.category || 'N/A'}`);
    return;
  }
  if (ev === 'case_memory_recall') {
    // Passive chip — the awaiting_case_recall_decision event drives the modal.
    // In bypass mode we only get this event (no awaiting_*), so users still
    // see the match was found.
    appendLogLine(`[${(e.ts || '').slice(11,19)}] 🧠 case memory recall (${e.match_type}) — prior algo=${e.prior_algorithm || '?'} k=${e.prior_k ?? '?'}`);
    return;
  }
  if (ev === 'awaiting_case_recall_decision') {
    const alreadyAnswered = state.events.some(
      x => x.event === 'case_memory_decision' && x.case_id === e.case_id
    );
    if (state.currentMode === 'bypass' || alreadyAnswered) {
      appendLogLine(`[${(e.ts || '').slice(11,19)}] RECALL SKIPPED — ${e.case_id?.slice(0,8) || '?'} (bypass=${state.currentMode==='bypass'}, answered=${alreadyAnswered})`);
      return;
    }
    showLivePanel(true);
    openRecallModal(e);
    toast('⏸ Pipeline paused — case memory recall needs your input', 'error', 6000);
    notifyUser('Pipeline paused', 'Case memory recall — the pipeline needs your decision.');
    appendLogLine(`[${(e.ts || '').slice(11,19)}] PAUSED — case memory matched, asking Reuse/Modify/Ignore`);
    return;
  }
  if (ev === 'case_memory_decision') {
    appendLogLine(`[${(e.ts || '').slice(11,19)}] case recall decision: ${e.decision} (${e.match_type})`);
    return;
  }
  if (ev === 'case_memory_saved') {
    appendLogLine(`[${(e.ts || '').slice(11,19)}] 🧠 case saved to memory (case_id=${(e.case_id || '').slice(0,8)}…)`);
    return;
  }
  if (ev === 'silhouette_target_changed') {
    appendLogLine(`[${(e.ts || '').slice(11,19)}] silhouette target ${Number(e.previous).toFixed(2)} → ${Number(e.new).toFixed(2)} (${e.mode})`);
    toast(`silhouette target lowered: ${Number(e.previous).toFixed(2)} → ${Number(e.new).toFixed(2)} (${e.mode})`, 'success', 4000);
    // Record as an Orchestrator output so it appears in the right-column
    // history and is preserved alongside other agent activity.
    recordAgentOutput({
      event: 'agent_report',
      agent: 'Orchestrator',
      iteration: state.events.filter(x => x.event === 'iteration_started').slice(-1)[0]?.iteration ?? '?',
      status: 'success',
      what_was_done: `Relaxed silhouette_target ${Number(e.previous).toFixed(2)} → ${Number(e.new).toFixed(2)} (${e.mode})`,
      metrics: {
        silhouette_target_previous: e.previous,
        silhouette_target_new: e.new,
        mode: e.mode,
      },
      issues: [],
    });
    return;
  }
  if (ev === 'awaiting_user_decision') {
    // Skip stale decision events: already answered, or now in bypass mode.
    const alreadyAnswered = state.events.some(
      x => x.event === 'user_decision_received' && x.agent === e.agent && x.iteration === e.iteration
    );
    if (state.currentMode === 'bypass' || alreadyAnswered) {
      appendLogLine(`[${(e.ts || '').slice(11,19)}] DECISION SKIPPED — ${e.agent} (bypass=${state.currentMode==='bypass'}, answered=${alreadyAnswered})`);
      return;
    }
    showLivePanel(true);
    openDecisionModal(e);
    toast(`⏸ Pipeline paused — ${e.agent} needs your decision`, 'error', 6000);
    notifyUser('Pipeline paused', `${e.agent} — the pipeline needs your decision.`);
    appendLogLine(`[${(e.ts || '').slice(11,19)}] PAUSED — awaiting your decision (${e.agent})`);
    return;
  }
  if (ev === 'user_decision_received') {
    appendLogLine(`[${(e.ts || '').slice(11,19)}] RESUMED — ${e.action}: ${e.response || '(none)'}`);
    return;
  }
  if (ev === 'awaiting_threshold_decision') {
    const alreadyAnswered = state.events.some(
      x => x.event === 'threshold_decision_made' && x.decision_id === e.decision_id
    );
    if (state.currentMode === 'bypass' || alreadyAnswered) {
      appendLogLine(`[${(e.ts || '').slice(11,19)}] THRESHOLD SKIPPED — ${e.decision_id} (bypass=${state.currentMode==='bypass'}, answered=${alreadyAnswered})`);
      return;
    }
    showLivePanel(true);
    openThresholdModal(e);
    toast('⏸ Pipeline paused — threshold decision needs your input', 'error', 6000);
    notifyUser('Pipeline paused', 'Threshold decision — the pipeline needs your decision.');
    appendLogLine(`[${(e.ts || '').slice(11,19)}] PAUSED — threshold decision (${e.decision_id})`);
    return;
  }
  if (ev === 'awaiting_human_checkpoint') {
    const alreadyAnswered = state.events.some(
      x => x.event === 'human_checkpoint_resolved' && x.iteration === e.iteration
    );
    if (state.currentMode === 'bypass' || alreadyAnswered) {
      appendLogLine(`[${(e.ts || '').slice(11,19)}] CHECKPOINT SKIPPED — iter ${e.iteration} (bypass=${state.currentMode==='bypass'}, answered=${alreadyAnswered})`);
      return;
    }
    showLivePanel(true);
    openCheckpointModal(e);
    toast('⏸ Pipeline paused — human checkpoint needs your approval', 'error', 6000);
    notifyUser('Pipeline paused', 'Human checkpoint — the pipeline needs your approval.');
    appendLogLine(`[${(e.ts || '').slice(11,19)}] PAUSED — human checkpoint (algo=${e.algorithm}, F1=${e.cv_f1_macro})`);
    return;
  }
  if (ev === 'human_checkpoint_resolved') {
    if (_pendingCheckpoint) closeCheckpointModal();
    appendLogLine(`[${(e.ts || '').slice(11,19)}] RESUMED — checkpoint ${e.action} (${e.source})`);
    return;
  }
  if (ev === 'threshold_decision_resolved') {
    // Defensive: the modal usually closes itself when the user POSTs,
    // but timeout / cross-window submissions land here.
    if (_pendingThreshold && _pendingThreshold.decision_id === e.decision_id) {
      closeThresholdModal();
    }
    appendLogLine(`[${(e.ts || '').slice(11,19)}] RESUMED — ${e.decision_id}: ${e.chosen} (${e.source})`);
    return;
  }
  if (ev === 'threshold_decision_auto_applied') {
    // Bypass mode: pipeline never paused. Record it so the user sees what
    // was relaxed under their feet in the Evidence tab.
    recordAutoAppliedDecision(e);
    appendLogLine(`[${(e.ts || '').slice(11,19)}] BYPASS — auto-applied ${e.chosen} for ${e.decision_id}`);
    return;
  }
  if (ev === 'awaiting_intent') {
    showLivePanel(true);
    $('#intent-form-wrap').classList.remove('hidden');
    /* live-sub removed from HTML by user request */
    $('#live-dot').className = 'dot warning';
    // Pipeline is idle on the intent form — hide the abort button.
    const abortBtn = $('#abort-btn');
    if (abortBtn) { abortBtn.classList.add('hidden'); abortBtn.disabled = false; abortBtn.textContent = 'Abort & New Run'; }
    toast('⏸ Pipeline paused — waiting for your intent', 'error', 6000);
    notifyUser('Pipeline paused', 'The pipeline is waiting for your clustering intent.');
    appendLogLine(`[${(e.ts || '').slice(11,19)}] awaiting_intent — pipeline paused for UI`);
    return;
  }
  if (ev === 'agent_report' && e.agent === 'UserInput') {
    $('#intent-form-wrap').classList.add('hidden');
    // Pipeline just took intent and is starting work — show the abort button.
    const abortBtn = $('#abort-btn');
    if (abortBtn) abortBtn.classList.remove('hidden');
  }
  if (ev === 'pipeline_complete') {
    const abortBtn = $('#abort-btn');
    if (abortBtn) { abortBtn.classList.add('hidden'); abortBtn.disabled = false; abortBtn.textContent = 'Abort & New Run'; }
    // Re-open the intent form so the user can submit a fresh run without
    // restarting the script. Without this, run_pipeline.py's wait loop has
    // no way to receive a new pending_intent.json from the browser.
    const status = (e.status || '').toLowerCase();
    if (status === 'blocked' || status === 'aborted'
        || status === 'success' || status === 'max_iterations_reached'
        || status === 'best_effort') {
      const wrap = $('#intent-form-wrap');
      if (wrap) wrap.classList.remove('hidden');
      // Reset the submit button so it's clickable for the next run.
      const sBtn = document.getElementById('intent-submit');
      if (sBtn) { sBtn.disabled = false; sBtn.textContent = 'Start pipeline'; }
      // Hint the user with a toast tailored to the status.
      const msg = (status === 'blocked')
        ? 'Pipeline blocked — edit the intent (e.g. different dataset) and submit to retry.'
        : (status === 'aborted')
        ? 'Pipeline aborted — submit a new intent to start fresh.'
        : 'Pipeline finished — submit a new intent to start another run, or stay on this page to review the results.';
      try { toast(msg, status === 'blocked' ? 'warning' : 'info', 6000); } catch (_) {}
    }
  }
  if (ev === 'run_started' || ev === 'pipeline_started') {
    const newRunId = e.run_id || null;
    const isNewRun = newRunId && newRunId !== state.runId;
    state.pipelineRunning = true;
    state.runId = newRunId || state.runId;
    state.runStartedTs = _tsToSeconds(e.ts) || state.runStartedTs;
    // Only wipe accumulated state when this is a NEW pipeline run. SSE replays
    // re-send earlier run_started/pipeline_started events when the browser
    // reconnects — without this guard, those replays would wipe everything
    // we'd already built up for the active run.
    if (isNewRun) {
      Object.keys(costStats).forEach((k) => delete costStats[k]);
      Object.keys(costEvidenceStats).forEach((k) => delete costEvidenceStats[k]);
      Object.keys(costNamingStats).forEach((k) => delete costNamingStats[k]);
      Object.keys(outputsState).forEach((k) => delete outputsState[k]);
      _autoAppliedDecisions.length = 0;
      state.personas = {};
      state.profiles = {};
      state.summary = {};
      state.selectedForMerge = new Set();
      renderSummary();
      renderGrid();
      updateTabCount();
      if (!document.getElementById('evidence-view').classList.contains('hidden')) {
        renderEvidence();
      }
      renderCostPanel();
      renderOutputsPanel();
      const convos = document.getElementById('convos');
      if (convos) convos.innerHTML = '';
      _convoNodes.clear();
    }
    renderTimeline();
    showLivePanel(true);
    /* live-sub removed from HTML */
    $('#live-dot').className = 'dot running';
    appendLogLine(`[${(e.ts || '').slice(11,19)}] ${ev}`);
  } else if (ev === 'iteration_started') {
    $('#live-iteration').textContent =
      `iteration ${e.iteration}/${e.max_total_iterations || '?'}`;
    appendLogLine(`[${(e.ts || '').slice(11,19)}] iteration ${e.iteration} started`);
    // Reset feature/cluster/naming/classifier rows to pending each new iter
    ['FeatureSelector', 'Clusterer', 'PersonaNamer', 'Classifier'].forEach((k) => {
      const li = document.querySelector(`#agent-timeline li[data-agent="${k}"]`);
      if (li && !li.classList.contains('success')) {
        li.classList.remove('running', 'warning', 'blocked', 'failure');
        li.classList.add('pending');
      }
    });
  } else if (ev === 'agent_report') {
    const status = e.status || 'success';
    const detail = e.what_was_done || '';
    setStepStatus(e.agent, status, detail, e.iteration, e.metrics, e.issues);
    recordAgentOutput(e);
    if (e.issues && e.issues.length) {
      const li = document.querySelector(`#agent-timeline li[data-agent="${e.agent}"]`);
      if (li) {
        const det = li.querySelector('.step-detail');
        det.innerHTML += `<br/><span class="issue">⚠ ${e.issues.map(escapeHtml).join(' · ')}</span>`;
      }
    }
    appendLogLine(`[${(e.ts || '').slice(11,19)}] ${e.agent} ${status.toUpperCase()} — ${(detail || '').slice(0, 120)}`);
    if (!document.getElementById('evidence-view').classList.contains('hidden')) {
      renderEvidence();
    }
  } else if (ev === 'llm_call_started') {
    noteCallStart(e.agent || 'unknown', e.category || 'pipeline', e);
    renderAskBubble(e);
    appendLogLine(`[${(e.ts || '').slice(11,19)}] ${e.agent} → Decision Maker — ${e.purpose} (${e.category || 'pipeline'})`);
  } else if (ev === 'llm_call_finished') {
    noteCallFinish(e.agent || 'unknown', e.input_tokens, e.output_tokens, e.time_s, e.category || 'pipeline', e);
    renderAnswerBubble(e);
    appendLogLine(`[${(e.ts || '').slice(11,19)}] Decision Maker → ${e.agent} — in=${e.input_tokens||0} out=${e.output_tokens||0} ${e.time_s||0}s`);
  } else if (ev === 'pipeline_complete') {
    state.pipelineRunning = false;
    const status = e.status || 'success';
    const dot = $('#live-dot');
    dot.className = 'dot ' + (status === 'success' || status === 'best_effort' ? 'success'
      : status === 'max_iterations_reached' ? 'warning' : 'blocked');
    const finishedLabel = status === 'success'
      ? 'Pipeline finished'
      : `Pipeline finished (${status})`;
    const bits = [];
    if (e.n_clusters) bits.push(`${e.n_clusters} clusters`);
    if (e.silhouette != null) bits.push(`silhouette ${Number(e.silhouette).toFixed(3)}`);
    if (e.cv_f1_macro != null) bits.push(`CV F1 ${Number(e.cv_f1_macro).toFixed(3)}`);
    /* live-sub removed from HTML */ void finishedLabel;
    setActiveSpotlight(null);
    appendLogLine(`[${(e.ts || '').slice(11,19)}] pipeline_complete — ${status}`);
    // Pipeline finished — load the named clusters and switch to that tab.
    // The Live pipeline view stays available behind the tab for review.
    setTimeout(() => {
      loadState().then(() => {
        updateTabCount();
        selectView('clusters');
        toast(`Pipeline finished — ${bits.join(' · ') || status}. View tabs above to switch.`,
              'success', 5000);
      }).catch((err) => toast(`Failed to load results: ${err.message}`, 'error', 6000));
    }, 800);
  }
}

function subscribeToEvents() {
  try {
    const src = new EventSource('/api/events/stream');
    src.onmessage = (msg) => {
      try {
        const evt = JSON.parse(msg.data);
        handleEvent(evt);
      } catch (_) { /* skip non-JSON heartbeats */ }
    };
    src.onerror = () => {
      // Browser will auto-reconnect; nothing to do.
    };
  } catch (e) {
    console.warn('EventSource unsupported:', e);
  }
}

// ── Intent form ─────────────────────────────────────────────────────────────

// Holds the latest uploaded file's server-side path. Wins over the typed path.
let _uploadedPath = null;

function wireDropZone() {
  const zone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('intent-file');
  const browseBtn = document.getElementById('dz-browse');
  const clearBtn = document.getElementById('dz-clear');
  const pathInput = document.getElementById('intent-dataset');
  if (!zone) return;

  // Click-to-browse
  browseBtn.onclick = (e) => { e.preventDefault(); fileInput.click(); };
  fileInput.onchange = () => {
    if (fileInput.files && fileInput.files[0]) uploadFile(fileInput.files[0]);
  };

  // Drag and drop
  ['dragenter', 'dragover'].forEach((ev) => {
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      // Ignore drags that don't carry files (e.g. text drags within the form)
      if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files')) {
        zone.classList.add('dragover');
      }
    });
  });
  ['dragleave', 'drop'].forEach((ev) => {
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
    });
  });
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });

  clearBtn.onclick = async () => {
    _uploadedPath = null;
    pathInput.value = '';
    document.getElementById('dz-filled').classList.add('hidden');
    document.getElementById('dz-empty').classList.remove('hidden');
    document.getElementById('dz-preview').classList.add('hidden');
    fileInput.value = '';
    // Tell the server to drop the saved preview too, so the Evidence tab cleans up
    try { await api('DELETE', '/api/upload-preview'); } catch (_) {}
    // Force-refresh the Evidence tab if it's open
    if (!document.getElementById('evidence-view').classList.contains('hidden')) {
      renderEvidence();
    }
    toast('Upload cleared', 'success', 2000);
  };

  // When the user types a server path manually, that overrides any prior upload
  pathInput.addEventListener('input', () => {
    if (pathInput.value.trim()) _uploadedPath = null;
  });
}

function uploadFile(file) {
  const empty = document.getElementById('dz-empty');
  const filled = document.getElementById('dz-filled');
  const progress = document.getElementById('dz-progress');
  const progressFill = document.getElementById('dz-progress-fill');
  const progressLabel = document.getElementById('dz-progress-label');
  const fileName = document.getElementById('dz-file-name');
  const fileMeta = document.getElementById('dz-file-meta');

  empty.classList.add('hidden');
  filled.classList.add('hidden');
  progress.classList.remove('hidden');
  progressFill.style.width = '0%';
  progressLabel.textContent = `Uploading ${file.name}…`;

  const fd = new FormData();
  fd.append('file', file);
  const xhr = new XMLHttpRequest();
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = (e.loaded / e.total) * 100;
      progressFill.style.width = pct.toFixed(1) + '%';
      progressLabel.textContent = `Uploading ${file.name} — ${pct.toFixed(0)}%`;
    }
  };
  xhr.onload = () => {
    progress.classList.add('hidden');
    if (xhr.status >= 200 && xhr.status < 300) {
      let resp = {};
      try { resp = JSON.parse(xhr.responseText); } catch (_) {}
      _uploadedPath = resp.path || null;
      fileName.textContent = resp.name || file.name;
      fileMeta.textContent = `${resp.size_human || ''} · saved to ${resp.path}`;
      filled.classList.remove('hidden');
      document.getElementById('intent-dataset').value = '';
      renderDataPreview(resp.preview);
      toast(`Uploaded "${resp.name}"`, 'success', 2500);
    } else {
      empty.classList.remove('hidden');
      let msg = 'Upload failed';
      try { msg = JSON.parse(xhr.responseText).error || msg; } catch (_) {}
      toast(msg, 'error', 5000);
    }
  };
  xhr.onerror = () => {
    progress.classList.add('hidden');
    empty.classList.remove('hidden');
    toast('Upload failed (network error)', 'error', 5000);
  };
  xhr.open('POST', '/api/upload');
  xhr.send(fd);
}

function renderDataPreview(preview) {
  const wrap = document.getElementById('dz-preview');
  const shapeElem = document.getElementById('dz-preview-shape');
  const tbl = document.getElementById('dz-preview-table');
  if (!wrap || !preview || preview.error) {
    if (wrap) wrap.classList.add('hidden');
    return;
  }
  const nRows = preview.n_rows != null ? Number(preview.n_rows).toLocaleString() : '?';
  shapeElem.innerHTML =
    `<b>${nRows}</b> rows × <b>${preview.n_cols || 0}</b> columns ` +
    `<span class="muted">· see full preview in the <b>Data &amp; evidence</b> tab</span>`;

  // Compact: just list column names as chips, no row table
  const cols = preview.columns || [];
  tbl.outerHTML = `<div class="dz-cols" id="dz-preview-table">
    ${cols.map(c => `<span class="col-chip">${escapeHtml(c)}</span>`).join('')}
  </div>`;
  wrap.classList.remove('hidden');
}

function wireAbortButton() {
  const btn = document.getElementById('abort-btn');
  if (!btn) return;
  btn.onclick = async () => {
    if (!confirm('Abort the current run and open the intent form for a fresh start?')) return;
    btn.disabled = true;
    btn.textContent = 'Aborting…';
    try {
      await api('POST', '/api/abort', { reason: 'user_abort_from_ui', restart: true });
      toast('Abort signal sent. Current iteration will finish, then the intent form will reopen.', 'success', 5000);
    } catch (e) {
      btn.disabled = false;
      btn.textContent = 'Abort & New Run';
      toast(`Abort failed: ${e.message}`, 'error', 5000);
    }
  };
}

function wireIntentForm() {
  wireDropZone();
  const btn = document.getElementById('intent-submit');
  if (!btn) return;
  btn.onclick = async () => {
    const target = $('#intent-target').value.trim();
    const purpose = $('#intent-purpose').value.trim();
    if (!target) {
      toast('Please enter a target entity (what you want to cluster).', 'error', 4000);
      return;
    }
    if (purpose.length < 5) {
      toast('Please enter a business purpose (a sentence or two).', 'error', 4000);
      return;
    }
    const k = $('#intent-k').value.trim();
    const maxItersRaw = $('#intent-max-iters').value.trim();
    const musthave = $('#intent-musthave').value.split(',').map(s => s.trim()).filter(Boolean);
    const datasetPath = _uploadedPath || $('#intent-dataset').value.trim();
    let maxTotalIterations = null;
    if (maxItersRaw && /^\d+$/.test(maxItersRaw)) {
      const n = Number(maxItersRaw);
      if (n >= 1 && n <= 50) maxTotalIterations = n;
    }
    const colHint = (id) => {
      const v = $(id).value.trim();
      return v || null;
    };
    const payload = {
      target_entity: target,
      business_purpose: purpose,
      dataset_path: datasetPath,
      constraints: $('#intent-constraints').value.trim(),
      n_clusters_requested: k && /^\d+$/.test(k) ? Number(k) : null,
      max_total_iterations: maxTotalIterations,
      must_have_clusters: musthave,
    };
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>Submitting…`;
    try {
      await api('POST', '/api/intent', payload);
      $('#intent-form-wrap').classList.add('hidden');
      $('#live-title').textContent = 'Starting pipeline…';
      /* live-sub removed from HTML */
      toast('Intent sent. The pipeline will start in a few seconds.', 'success', 4000);
    } catch (e) {
      toast(e.message, 'error', 5000);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Start pipeline';
    }
  };
}

// ── Boot ──────────────────────────────────────────────────────────────────

async function boot() {
  let status;
  try {
    status = await api('GET', '/api/status');
  } catch (_) {
    status = {pipeline_running: false, has_personas: true};
  }
  state.pipelineRunning = !!status.pipeline_running;
  state.runId = status.run_id || null;

  // Pre-load the cluster grid only when personas truly belong to the LATEST
  // completed run. During a running pipeline that hasn't completed yet,
  // personas.json would be from a prior run (or already deleted by bus init).
  const showStaleClusters = status.has_personas && !status.pipeline_running;
  if (showStaleClusters) {
    try { await loadState(); updateTabCount(); } catch (_) {}
  } else {
    state.personas = {};
    state.profiles = {};
    state.summary = {};
    renderGrid();
    updateTabCount();
  }

  if (status.pipeline_running || !status.has_personas) {
    renderTimeline();
    selectView('live');
    if (!status.has_personas && !status.pipeline_running) {
      /* live-sub removed from HTML */
      $('#live-dot').className = 'dot';
    }
    // If we joined while the pipeline is waiting for intent, show the form.
    try {
      const {events} = await api('GET', '/api/events');
      state.events = events || [];
      const haveAwaiting = state.events.some(e => e.event === 'awaiting_intent');
      const haveUserInputReport = state.events.some(
        e => e.event === 'agent_report' && e.agent === 'UserInput'
      );
      if (haveAwaiting && !haveUserInputReport) {
        $('#intent-form-wrap').classList.remove('hidden');
        /* live-sub removed from HTML by user request */
        $('#live-dot').className = 'dot warning';
      }
      applyArchFromEvents();
      // Boot replay — populate cost / outputs / chat bubbles. The dedupe
      // sets (_countedCalls, _convoNodes, outputsState's history exists check)
      // ensure SSE's later replay-on-connect doesn't double-count.
      for (const ev of state.events) {
        if (ev.event === 'run_started' || ev.event === 'pipeline_started') {
          state.runStartedTs = _tsToSeconds(ev.ts) || state.runStartedTs;
          state.runId = ev.run_id || state.runId;
        }
        if (ev.event === 'llm_call_started') {
          renderAskBubble(ev);
        }
        if (ev.event === 'llm_call_finished') {
          noteCallFinish(ev.agent || 'unknown', ev.input_tokens, ev.output_tokens, ev.time_s, ev.category || 'pipeline', ev);
          renderAnswerBubble(ev);
        }
        if (ev.event === 'agent_report') {
          recordAgentOutput(ev);
        }
      }
    } catch (_) { /* best-effort */ }
  } else {
    selectView('clusters');
  }

  wireIntentForm();
  wireAbortButton();
  wireWarnModal();
  wireDecisionModal();
  wireRelaxModal();
  wireThresholdModal();
  wireControlGatesModal();
  wireColumnResolutionModal();
  wireCheckpointModal();
  wireRecallModal();
  wireModeToggle();
  wireTabs();
  loadMode();
  updateTabCount();
  subscribeToEvents();
  startGlobalRefreshWatchdog();
}

// Defensive watchdog that re-renders whichever view is currently visible.
// Works in BOTH the bare `full` recording and the focused demo recordings:
// even if the SSE stream goes silent, the user sees fresh content periodically.
// The render functions are idempotent — re-running them with identical state
// is a no-op cost-wise.
function startGlobalRefreshWatchdog() {
  if (window._globalRefreshTimer) return;
  window._globalRefreshTimer = setInterval(() => {
    try {
      // Live pipeline panels (graph + cost) are cheap and always present
      // in the DOM. Re-render to catch any state drift.
      renderArchGraph();
      renderCostPanel();
      // Evidence + Named tabs are conditional; only refresh if shown.
      if (!document.getElementById('evidence-view').classList.contains('hidden')) {
        renderEvidence();
      }
      if (!document.getElementById('cluster-grid').classList.contains('hidden')) {
        renderGrid();
      }
    } catch (_) { /* render fns are defensive */ }
  }, UI_REFRESH_INTERVAL_MS);
}

// ── Demo / recording focus mode ──────────────────────────────────────────
// Add ?demo=<graph|convos|outputs|evidence|tokens|named|intent|log> to the URL
// to isolate one part of the UI for a clean screen recording.
function applyDemoMode() {
  const params = new URLSearchParams(window.location.search);
  const area = params.get('demo');
  if (!area) return;
  document.body.classList.add('demo-mode', `demo-${area}`);
  if (area === 'evidence') selectView('evidence');
  if (area === 'named') {
    // The Named Clusters tab is the 'clusters' view. Switch to it now so the
    // recording captures the cluster grid the moment personas.json is written
    // (the SSE pipeline_complete handler already triggers a reload of personas).
    selectView('clusters');
  }
  // Defensive auto-refresh for `graph` and `tokens` recordings. The SSE
  // stream usually drives both, but on long static periods the connection
  // can stall (proxy timeouts, browser sleeping a background tab, etc.).
  // Polling every 2s guarantees the visible content keeps moving while the
  // recorder is running, so the resulting video never looks frozen.
  if (area === 'graph' || area === 'tokens') {
    setInterval(() => {
      try {
        if (area === 'graph')  renderArchGraph();
        if (area === 'tokens') renderCostPanel();
      } catch (_) { /* render fns are defensive, swallow */ }
    }, 2000);
  }
  // Render a small exit button so you can leave demo mode without editing the URL
  if (!document.getElementById('demo-exit')) {
    const btn = document.createElement('button');
    btn.id = 'demo-exit';
    btn.textContent = '× Exit demo mode';
    btn.onclick = () => {
      const u = new URL(window.location);
      u.searchParams.delete('demo');
      window.location.href = u.toString();
    };
    document.body.appendChild(btn);
  }
}

boot().then(() => applyDemoMode())
      .catch((e) => toast(`Failed to start: ${e.message}`, 'error', 6000));
