// ──────────────────────────────────────────────────────────────────────────
//  Cluster Persona Studio — frontend
// ──────────────────────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  personas: {},
  profiles: {},
  summary: {},
  selectedForMerge: new Set(),
  openId: null,
  draft: null,           // unsaved edits for the currently open cluster
};

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
  const f1 = s.cv_f1_macro != null ? Number(s.cv_f1_macro).toFixed(3) : 'n/a';
  $('#summary').innerHTML = `
    <b>${s.n_clusters || 0}</b> clusters · <b>${s.total_entities || 0}</b> entities ·
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
  ids.forEach((cid) => grid.appendChild(renderCard(cid)));
  updateMergeBtn();
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
      <label style="color:var(--accent-2)">Ask the Decision Maker to re-name</label>
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

$('#memory-btn').addEventListener('click', openMemory);
$('#close-memory').addEventListener('click', () => $('#memory-drawer').classList.add('hidden'));
$('#memory-drawer').addEventListener('click', (e) => {
  if (e.target.id === 'memory-drawer') $('#memory-drawer').classList.add('hidden');
});

async function openMemory() {
  $('#memory-drawer').classList.remove('hidden');
  const list = $('#feedback-list');
  list.innerHTML = `<div class="muted">Loading…</div>`;
  try {
    const [{entries}, {text}] = await Promise.all([
      api('GET', '/api/feedback'),
      api('GET', '/api/preferences-preview'),
    ]);
    renderFeedback(entries);
    $('#prefs-preview').textContent = text || '(no active feedback yet — make an edit to start training the agents)';
  } catch (e) {
    list.innerHTML = `<div class="muted">Error: ${escapeHtml(e.message)}</div>`;
  }
}

function renderFeedback(entries) {
  const list = $('#feedback-list');
  if (!entries.length) {
    list.innerHTML = `<div class="muted">No feedback yet. Edits, regenerations, merges, and global rules will appear here.</div>`;
    return;
  }
  // newest first
  entries = entries.slice().reverse();
  list.innerHTML = entries.map((e) => `
    <div class="feedback-row ${e.active === false ? 'inactive' : ''}" data-id="${e.id}">
      <div class="meta">
        <div>${(e.date || '').slice(0, 10)}</div>
        <span class="tag ${e.type}">${e.type}</span>
      </div>
      <div class="body">${describeEntry(e)}</div>
      <div class="controls">
        <select data-priority="${e.id}">
          ${['high', 'medium', 'low'].map((p) =>
            `<option value="${p}" ${p === (e.priority || 'medium') ? 'selected' : ''}>${p}</option>`
          ).join('')}
        </select>
        <button class="ghost" data-toggle="${e.id}">${e.active === false ? 'Enable' : 'Disable'}</button>
      </div>
    </div>
  `).join('');

  $$('select[data-priority]').forEach((sel) => {
    sel.onchange = async (ev) => {
      const id = ev.target.dataset.priority;
      try {
        await api('PATCH', `/api/feedback/${id}`, {priority: ev.target.value});
        await openMemory();
        toast('Priority updated');
      } catch (e) { toast(e.message, 'error', 4000); }
    };
  });
  $$('button[data-toggle]').forEach((b) => {
    b.onclick = async (ev) => {
      const id = ev.target.dataset.toggle;
      const row = list.querySelector(`[data-id="${id}"]`);
      const nowActive = row.classList.contains('inactive');
      try {
        await api('PATCH', `/api/feedback/${id}`, {active: nowActive});
        await openMemory();
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
    return `<b>Global rule:</b> "${escapeHtml(e.rule || '')}"`;
  }
  return JSON.stringify(e);
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

// ── Boot ──────────────────────────────────────────────────────────────────

loadState().catch((e) => toast(`Failed to load state: ${e.message}`, 'error', 6000));
