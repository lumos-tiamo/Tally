/* Tally console.
 *
 * No build step and no framework, deliberately: the rest of this project runs
 * offline with no credentials, and a dashboard that needs npm install to render
 * would be the one part nobody can actually start.
 *
 * The centrepiece is renderLedger(). Slot bidding with hard floors is the
 * hardest thing in the platform to explain in a sentence and the easiest to
 * show, so the console's job is to draw where every token in every prompt went
 * and which slot lost what.
 */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
};

/* Tables are built through this rather than by nesting el() calls seven deep.
 * Hand-counting closing parens is how the first draft of this file shipped three
 * separate syntax errors, and the nesting was unreadable besides. `columns` is a
 * list of {label, numeric} and `rows` a list of cell arrays. */
const table = (columns, rows, emptyText = 'Nothing to show.') => {
  if (!rows.length) return el('div', { class: 'empty' }, emptyText);
  const head = el('tr', {}, columns.map((c) =>
    el('th', c.numeric ? { class: 'num' } : {}, c.label || '')));
  const body = rows.map((cells) => el('tr', {}, cells.map((cell, i) => {
    const spec = columns[i] || {};
    const attrs = spec.numeric ? { class: 'num' } : (spec.cellClass ? { class: spec.cellClass } : {});
    return el('td', attrs, cell);
  })));
  return el('table', {}, el('thead', {}, head), el('tbody', {}, body));
};

const SLOTS = ['system', 'skill', 'tools', 'workspace', 'memory', 'history'];
const SLOT_WHY = {
  system: 'persona + operating contract — pinned, never evicted',
  skill: 'instructions for the skills matched this step',
  tools: 'tool signatures, retrieval-filtered to this step',
  workspace: 'file tree and artefact digests — never file bodies',
  memory: 'recalled long-term memory — first to be sacrificed',
  history: 'session turns: tail verbatim, middle compacted',
};

const nf = (n) => (n == null ? '—' : Number(n).toLocaleString());
const pct = (n) => (n == null ? '—' : `${(Number(n) * 100).toFixed(1)}%`);
const fx = (n, d = 3) => (n == null ? '—' : Number(n).toFixed(d));

let state = { view: 'overview', health: null, agents: [], editing: null, run: null };

/* ---------- transport ---------- */
async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: {
      'Content-Type': 'application/json',
      ...(localStorage.getItem('tally_token')
        ? { 'X-Tally-Token': localStorage.getItem('tally_token') }
        : {}),
      ...(opts.headers || {}),
    },
  });
  if (res.status === 204) return null;
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    // Pydantic returns a list of field errors; a raw JSON dump in a toast is
    // useless, so the first message is surfaced and the rest kept in the console.
    const detail = Array.isArray(body.detail)
      ? body.detail.map((d) => `${(d.loc || []).slice(-1)}: ${d.msg}`).join('; ')
      : body.detail || res.statusText;
    throw new Error(detail);
  }
  return body;
}

function toast(message, kind = '') {
  const node = el('div', { class: `toast ${kind}` }, message);
  document.body.append(node);
  setTimeout(() => node.remove(), kind === 'bad' ? 8000 : 4000);
}

/* ---------- shared pieces ---------- */
function statCard(label, value, foot, opts = {}) {
  return el('div', { class: 'card stat' },
    el('div', { class: 'label' }, label),
    el('div', { class: `value ${opts.small ? 'sm' : ''}` }, value),
    foot ? el('div', { class: 'foot' }, foot) : null);
}

function banner(warnings, kind = '') {
  if (!warnings || !warnings.length) return null;
  return el('div', { class: `banner ${kind}` },
    el('strong', {}, warnings.length === 1 ? 'Note' : `${warnings.length} notes`),
    el('ul', {}, warnings.map((w) => el('li', {}, w))));
}

function pageHead(title, hint, sub) {
  return [
    el('div', { class: 'page-head' }, el('h2', {}, title), hint ? el('span', { class: 'hint' }, hint) : null),
    sub ? el('p', { class: 'page-sub' }, sub) : null,
  ];
}

/* ---------- the ledger visualisation ---------- */
/* Two facts, two visual channels, and the split matters.
 *
 * A first version drew one bar per step with each slot sized against the whole
 * budget. It was arithmetically right and useless: at 4% utilisation the entire
 * composition was squeezed into a sliver four pixels wide, so the one thing the
 * chart exists to show — which slot got what — was invisible exactly when the
 * budget was comfortable.
 *
 * So composition is normalised to what the prompt actually used, and budget
 * pressure gets its own thin track underneath. Composition stays legible at any
 * utilisation; pressure is still visible when it matters. */
function renderLedger(steps, opts = {}) {
  if (!steps || !steps.length) {
    return el('div', { class: 'empty' },
      'No prompt has been built yet. The allocation appears as the run takes its first step.');
  }
  const wrap = el('div', {});
  for (const step of steps.slice(-(opts.limit || 40))) {
    const budget = step.budget || 1;
    const used = step.used || 0;
    const util = step.utilisation || 0;

    const present = SLOTS
      .map((name) => [name, (step.slots || []).find((s) => s.slot === name)])
      .filter(([, slot]) => slot && slot.used);

    const bar = el('div', { class: 'ledger-bar' });
    for (const [name, slot] of present) {
      // Normalised to `used`, not to `budget` — see the note above.
      const share = used ? (slot.used / used) * 100 : 0;
      bar.append(el('div', {
        class: 'ledger-seg', 'data-slot': name, style: `width:${Math.max(0.6, share)}%`,
      }, el('div', { class: 'tip' },
        el('div', {}, `${name} — ${nf(slot.used)} tok (${share.toFixed(1)}% of this prompt)`),
        el('div', { class: 'faint' }, `allowance ${nf(slot.allowance)} tok`),
        el('div', { class: 'faint' }, SLOT_WHY[name]),
        slot.evicted
          ? el('div', { class: 'faint' },
              `evicted ${slot.evicted} item(s), ${nf(slot.evicted_tokens)} tok` +
              (slot.degraded ? ' — replaced by a compact stand-in' : ''))
          : null,
        slot.overflowed ? el('div', {}, 'pinned content exceeded its allowance') : null)));
    }

    const pressureClass = util > 0.9 ? 'bad' : util > 0.7 ? 'warn' : 'good';
    wrap.append(el('div', { class: 'ledger-step' },
      el('div', { class: 'head' },
        el('span', { class: 'name' }, step.name || 'step'),
        el('span', {}, `${nf(used)} tok`),
        step.evicted_tokens ? el('span', { class: 'pill warn' }, `evicted ${nf(step.evicted_tokens)}`) : null,
        el('span', { class: 'util' },
          el('span', { class: 'faint' }, `${nf(budget)} budget · `),
          el('span', { class: `pill ${util > 0.9 ? 'bad' : util > 0.7 ? 'warn' : ''}` }, pct(util)))),
      bar,
      el('div', { class: 'bar-track', style: 'margin-top:3px', title: `${pct(util)} of the budget used` },
        el('div', { class: `bar-fill ${pressureClass}`, style: `width:${Math.min(100, util * 100)}%` })),
      (step.notes || []).length ? el('div', { class: 'faint mono', style: 'font-size:10.5px;margin-top:3px' },
        step.notes.join(' · ')) : null));
  }
  wrap.append(el('div', { class: 'legend' },
    SLOTS.map((s) => el('span', {}, el('i', { style: `background:var(--slot-${s})` }), s)),
    el('span', { class: 'faint' }, '— bar shows composition; the thin track below is budget used')));
  return wrap;
}

/* ---------- views ---------- */
const views = {};

views.overview = async (main) => {
  const [insight, health] = await Promise.all([
    api('/api/insight/overview'), api('/api/insight/health'),
  ]);
  const t = insight.totals || {};
  main.append(...pageHead('Overview', null,
    'What this deployment has run, and what the repository has measured. ' +
    'Measured figures are read from committed artefacts — the console never recomputes a score.'));
  main.append(banner([...(health.warnings || []), ...(health.config_warnings || [])], 'bad'));

  main.append(el('div', { class: 'grid c4' },
    statCard('Agents', nf(t.agents)),
    statCard('Runs', nf(t.runs), `${nf(t.finished)} finished · ${nf(t.failed)} failed`),
    statCard('Tokens', nf(t.tokens_total), `$${fx(t.cost_usd, 4)}`),
    statCard('Needs a human', nf(t.needs_human), t.needs_human ? 'waiting for an answer' : 'nothing pending')));

  const floor = insight.zero_model_floor;
  const corpus = insight.corpus;
  const sig = insight.l3_signals;
  main.append(el('div', { class: 'grid c2', style: 'margin-top:14px' },
    el('div', { class: 'card' },
      el('h3', {}, 'Zero-model floor'),
      el('p', { class: 'note' },
        'The document tools driven by a fixed caption table, with no model calls at all. ' +
        'This is the floor a language model has to beat.'),
      floor
        ? el('div', { class: 'grid c3' },
            statCard('Numeric @strict', fx(floor.numeric_strict), `${nf(floor.cases)} cases`, { small: true }),
            statCard('Citation', fx(floor.citation), 'quote locatable in source', { small: true }),
            statCard('Abstention', fx(floor.abstention), 'declined the undisclosed', { small: true }))
        : el('div', { class: 'empty' }, 'Not measured yet — run scripts/deterministic_baseline.py')),
    el('div', { class: 'card' },
      el('h3', {}, 'Ground truth'),
      el('p', { class: 'note' },
        'Built semi-automatically from SEC XBRL, which is what makes this many cases affordable. ' +
        'Selected by accounting difficulty rather than market cap.'),
      corpus
        ? el('div', { class: 'grid c3' },
            statCard('Cases', nf(corpus.cases), `${nf(corpus.held_out)} held out`, { small: true }),
            statCard('Quarantined', nf(corpus.quarantined), 'failed a validity check', { small: true }),
            statCard('L1 coverage', pct(corpus.mean_l1_coverage), 'mean per case', { small: true }))
        : el('div', { class: 'empty' }, 'Not built — run tally dataset build'))));

  if (sig) {
    // Built with intermediate names rather than one deeply nested expression:
    // hand-counting seven levels of closing parens is how the first version of
    // this block shipped a syntax error.
    const severityPill = (severity) => el('span', {
      class: `pill ${severity === 'red_flag' ? 'bad' : severity === 'concern' ? 'warn' : ''}`,
    }, severity);
    const signalRows = Object.entries(sig.by_signal || {}).map(([key, row]) => el('tr', {},
      el('td', { class: 'mono' }, key),
      el('td', {}, severityPill(row.severity)),
      el('td', { class: 'num' }, nf(row.fired)),
      el('td', { class: 'num' }, nf(row.detectable)),
      el('td', { class: 'num' }, pct(row.fire_rate))));
    const signalHead = el('tr', {},
      el('th', {}, 'Signal'), el('th', {}, 'Severity'),
      el('th', { class: 'num' }, 'Fired'), el('th', { class: 'num' }, 'Detectable'),
      el('th', { class: 'num' }, 'Rate'));
    const signalTable = el('table', {},
      el('thead', {}, signalHead),
      el('tbody', {}, signalRows));
    const note = `${nf(sig.pairs)} year-over-year pairs, ${nf(sig.signals_total)} signals. ` +
      `${nf(sig.quiet_pairs)} pairs (${pct(sig.quiet_share)}) show nothing material — those are the ` +
      'cases that test whether an agent avoids inventing findings, and they carry no recall denominator.';
    main.append(el('div', { class: 'card', style: 'margin-top:14px' },
      el('h3', {}, 'L3 judgement signals'),
      el('p', { class: 'note' }, note),
      signalTable));
  }
};

views.agents = async (main) => {
  const [{ agents }, { scenarios }] = await Promise.all([
    api('/api/agents'), api('/api/scenarios'),
  ]);
  state.agents = agents;
  main.append(...pageHead('Agents', `${agents.length} defined`,
    'An agent is a declaration: persona, which tools and skills it may use, and its token budget. ' +
    'Its working state is never stored here — that lives in the run workspace on disk.'));

  const agentRows = agents.map((a) => [
    el('div', {}, el('div', { class: 'mono' }, a.name),
      el('div', { class: 'faint', style: 'font-size:11px' }, a.display_name || '')),
    el('span', { class: 'pill accent' }, a.scenario),
    nf(a.window),
    a.max_steps,
    a.enabled ? el('span', { class: 'pill good' }, 'enabled') : el('span', { class: 'pill' }, 'disabled'),
    el('div', { class: 'row-actions' },
      el('button', { class: 'btn ghost', onclick: () => openEditor(a) }, 'Edit'),
      el('button', {
        class: 'btn ghost',
        onclick: () => { state.consoleAgent = a.id; go('console'); },
      }, 'Run')),
  ]);
  const agentTable = table([
    { label: 'Name' }, { label: 'Scenario' }, { label: 'Window', numeric: true },
    { label: 'Steps', numeric: true }, { label: '' }, { label: '' },
  ], agentRows, 'No agents yet.');

  const listCard = el('div', { class: 'card' },
    el('h3', {}, 'Defined agents'),
    el('p', { class: 'note' },
      'Click one to edit. Delegation permissions are per-pair and default to denied.'),
    agentTable);

  const editorCard = el('div', { class: 'card' },
    el('h3', {}, state.editing ? 'Edit agent' : 'New agent'),
    el('p', { class: 'note' },
      'The window bound is not a preference: the ledger reserves hard floors totalling ~1,000 ' +
      'tokens, so a smaller window cannot satisfy them and the run fails on construction.'),
    agentForm(scenarios));

  main.append(el('div', { class: 'split' }, listCard, editorCard));
};

function openEditor(agent) {
  state.editing = agent;
  render();
}

function agentForm(scenarios) {
  const a = state.editing || {};
  const form = el('div', {});
  const f = {};
  const input = (key, label, attrs = {}) => {
    const node = el('input', { value: a[key] ?? attrs.value ?? '', ...attrs });
    f[key] = node;
    return el('div', {}, el('label', {}, label), node);
  };

  const scenarioSelect = el('select', {},
    scenarios.map((s) => el('option', { value: s.name, selected: a.scenario === s.name }, s.display_name)));
  f.scenario = scenarioSelect;
  const persona = el('textarea', { rows: 9 }, a.persona || '');
  f.persona = persona;

  scenarioSelect.addEventListener('change', () => {
    const chosen = scenarios.find((s) => s.name === scenarioSelect.value);
    if (chosen && !state.editing) persona.value = chosen.persona || '';
  });
  if (!state.editing && !persona.value && scenarios.length) {
    persona.value = scenarios[0].persona || '';
  }

  form.append(
    input('name', 'name (lowercase, no spaces)', { placeholder: 'dd-analyst-2', disabled: !!state.editing }),
    input('display_name', 'display name', { placeholder: 'Diligence analyst' }),
    el('div', {}, el('label', {}, 'scenario'), scenarioSelect),
    el('div', {}, el('label', {}, 'persona'), persona),
    el('div', { class: 'field-row' },
      input('window', 'window', { type: 'number', value: a.window ?? 32000 }),
      input('max_output', 'max output', { type: 'number', value: a.max_output ?? 2048 }),
      input('max_steps', 'max steps', { type: 'number', value: a.max_steps ?? 14 })),
    el('div', { class: 'field-row' },
      input('tool_k', 'tools/step', { type: 'number', value: a.tool_k ?? 8 }),
      input('skill_k', 'skills/step', { type: 'number', value: a.skill_k ?? 2 }),
      input('memory_k', 'memories/step', { type: 'number', value: a.memory_k ?? 5 }),
      input('compaction_threshold', 'compact at', { type: 'number', step: '0.05', value: a.compaction_threshold ?? 0.7 })),
    el('div', { class: 'form-actions' },
      el('button', { class: 'btn', onclick: () => submitAgent(f) }, state.editing ? 'Save' : 'Create'),
      state.editing ? el('button', { class: 'btn ghost', onclick: () => { state.editing = null; render(); } }, 'Cancel') : null,
      state.editing ? el('button', {
        class: 'btn danger',
        onclick: async () => {
          if (!confirm(`Delete ${state.editing.name}? Its runs go with it.`)) return;
          try {
            await api(`/api/agents/${state.editing.id}`, { method: 'DELETE' });
            state.editing = null; toast('Agent deleted', 'good'); render();
          } catch (e) { toast(e.message, 'bad'); }
        },
      }, 'Delete') : null));

  if (state.editing) form.append(relationsPanel(state.editing));
  return form;
}

function relationsPanel(agent) {
  const panel = el('div', { style: 'margin-top:22px;border-top:1px solid var(--line);padding-top:16px' },
    el('h3', {}, 'Delegation (A2A)'),
    el('p', { class: 'note' },
      'Absence is denial: an unlisted pair is refused. The tenant comes from the stored ' +
      'relation, never from the request, so a cross-tenant delegation cannot be authorised by ' +
      'anything the caller asserts.'));
  const list = el('div', {});
  panel.append(list);

  api(`/api/agents/${agent.id}/relations`).then(({ relations }) => {
    list.replaceChildren(relations.length
      ? el('table', {},
          el('thead', {}, el('tr', {}, el('th', {}, 'Target'), el('th', {}, 'Tenant'), el('th', {}, 'Files'))),
          el('tbody', {}, relations.map((r) => {
            const target = state.agents.find((x) => x.id === r.target_id);
            return el('tr', {},
              el('td', { class: 'mono' }, target ? target.name : r.target_id),
              el('td', {}, el('span', { class: 'pill' }, r.tenant)),
              el('td', {}, r.may_share_files
                ? el('span', { class: 'pill warn' }, 'may share files')
                : el('span', { class: 'pill' }, 'no files')));
          })))
      : el('div', { class: 'empty' }, 'No delegation granted. This agent cannot call any other.'));
  });

  const targetSelect = el('select', {},
    state.agents.filter((x) => x.id !== agent.id).map((x) => el('option', { value: x.id }, x.name)));
  const tenant = el('input', { value: 'default' });
  const files = el('input', { type: 'checkbox' });
  panel.append(
    el('div', { class: 'field-row', style: 'margin-top:12px' },
      el('div', {}, el('label', {}, 'grant to'), targetSelect),
      el('div', {}, el('label', {}, 'tenant'), tenant),
      el('div', {}, el('label', {}, 'may share files'), files)),
    el('div', { class: 'form-actions' },
      el('button', {
        class: 'btn ghost',
        onclick: async () => {
          try {
            await api(`/api/agents/${agent.id}/relations`, {
              method: 'POST',
              body: JSON.stringify({ target_id: targetSelect.value, tenant: tenant.value, may_share_files: files.checked }),
            });
            toast('Delegation granted', 'good'); render();
          } catch (e) { toast(e.message, 'bad'); }
        },
      }, 'Grant')));
  return panel;
}

async function submitAgent(f) {
  const num = (k) => Number(f[k].value);
  const payload = {
    display_name: f.display_name.value.trim(),
    persona: f.persona.value.trim(),
    window: num('window'), max_output: num('max_output'), max_steps: num('max_steps'),
    tool_k: num('tool_k'), skill_k: num('skill_k'), memory_k: num('memory_k'),
    compaction_threshold: Number(f.compaction_threshold.value),
  };
  try {
    if (state.editing) {
      await api(`/api/agents/${state.editing.id}`, { method: 'PATCH', body: JSON.stringify(payload) });
      toast('Saved', 'good');
    } else {
      await api('/api/agents', {
        method: 'POST',
        body: JSON.stringify({ ...payload, name: f.name.value.trim(), scenario: f.scenario.value }),
      });
      toast('Agent created', 'good');
    }
    state.editing = null;
    render();
  } catch (e) { toast(e.message, 'bad'); }
}

views.console = async (main) => {
  const { agents } = await api('/api/agents');
  state.agents = agents;
  main.append(...pageHead('Console', null,
    'Give an agent an objective and watch it work. The allocation panel is the point: it shows ' +
    'where every token in every prompt went, updating as each step is built.'));

  if (!agents.length) { main.append(el('div', { class: 'empty' }, 'Define an agent first.')); return; }

  const select = el('select', {}, agents.map((a) =>
    el('option', { value: a.id, selected: state.consoleAgent === a.id }, `${a.name} — ${a.scenario}`)));
  const objective = el('textarea', { rows: 3, placeholder: 'Extract the twelve L1 fields for AAPL FY2024 from its annual report.' });
  const stream = el('div', { class: 'stream' });
  const ledgerBox = el('div', {});
  const statusBox = el('div', {});
  let socket = null;

  const startBtn = el('button', { class: 'btn' }, 'Run');
  const cancelBtn = el('button', { class: 'btn ghost', disabled: true }, 'Cancel');

  startBtn.addEventListener('click', async () => {
    if (!objective.value.trim()) { toast('Give it an objective first'); return; }
    stream.replaceChildren();
    ledgerBox.replaceChildren();
    try {
      const started = await api(`/api/agents/${select.value}/runs`, {
        method: 'POST', body: JSON.stringify({ objective: objective.value.trim() }),
      });
      state.run = started.run_id;
      startBtn.disabled = true; cancelBtn.disabled = false;
      statusBox.replaceChildren(el('span', { class: 'pill accent' }, `running · ${started.run_id}`));
      socket = openRunSocket(started.run_id, { stream, ledgerBox, statusBox, startBtn, cancelBtn });
    } catch (e) { toast(e.message, 'bad'); }
  });
  cancelBtn.addEventListener('click', async () => {
    if (!state.run) return;
    const out = await api(`/api/runs/${state.run}/cancel`, { method: 'POST' });
    toast(out.note || 'Cancelling', 'good');
  });

  main.append(el('div', { class: 'split' },
    el('div', {},
      el('div', { class: 'card' },
        el('h3', {}, 'Objective'),
        el('div', { class: 'field-row' }, el('div', {}, el('label', {}, 'agent'), select)),
        el('div', {}, el('label', {}, 'what should it do'), objective),
        el('div', { class: 'form-actions' }, startBtn, cancelBtn, statusBox)),
      el('div', { class: 'card', style: 'margin-top:14px' },
        el('h3', {}, 'Context allocation, per step'),
        el('p', { class: 'note' },
          'Each bar is one prompt. Hard floors are reserved first and surplus flows to the least ' +
          'elastic slot, which is why retrieved memory can never squeeze out the operating contract. ' +
          'Hover a segment for what it holds and what it dropped.'),
        ledgerBox)),
    el('div', { class: 'card' },
      el('h3', {}, 'Trace'),
      el('p', { class: 'note' }, 'Every span as it closes. Replayed spans are dimmed.'),
      stream)));

  window.addEventListener('hashchange', () => socket && socket.close(), { once: true });
};

function openRunSocket(runId, ui) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${proto}://${location.host}/api/ws/runs/${runId}`);
  const steps = [];

  socket.addEventListener('message', (event) => {
    const ev = JSON.parse(event.data);
    if (ev.type === 'ping') return;

    if (ev.type === 'span') {
      const kind = ev.kind || '';
      ui.stream.append(el('div', {
        class: `ev ${ev.error ? 'err' : ''} ${ev.replay ? 'replay' : ''}`, 'data-kind': kind,
      },
        el('span', { class: 'k' }, kind),
        el('span', { class: 'n' }, ev.error ? ev.error : (ev.name || '')),
        el('span', { class: 'd' }, `${fx(ev.duration_s, 2)}s`)));
      ui.stream.scrollTop = ui.stream.scrollHeight;

      if (kind === 'context_build') {
        const a = ev.attrs || {};
        steps.push({
          name: ev.name, budget: a.budget, used: a.used, utilisation: a.utilisation,
          evicted_tokens: a.evicted_tokens, slots: a.slots || [], notes: a.notes || [],
        });
        ui.ledgerBox.replaceChildren(renderLedger(steps));
      }
      return;
    }

    if (ev.type === 'done' || ev.type === 'error') {
      const ok = ev.state === 'finished';
      ui.statusBox.replaceChildren(el('span', {
        class: `pill ${ok ? 'good' : ev.state === 'needs_human' ? 'warn' : 'bad'}`,
      }, ev.state || 'error'));
      ui.startBtn.disabled = false; ui.cancelBtn.disabled = true;
      if (ev.result && ev.result.final_answer) {
        ui.stream.append(el('div', { class: 'ev' },
          el('span', { class: 'k' }, 'answer'),
          el('span', { class: 'n' }, ev.result.final_answer.slice(0, 400))));
      }
      if (ev.error) toast(ev.error, 'bad');
      socket.close();
    }
  });
  socket.addEventListener('error', () => toast('Lost the run feed; the run itself is unaffected', 'bad'));
  return socket;
}

views.runs = async (main) => {
  const { runs, live } = await api('/api/runs');
  main.append(...pageHead('Runs', `${runs.length} recorded`,
    'A run row is an index into its real artefacts: the workspace on disk and the trace file. ' +
    'It never duplicates them.'));

  if (live.length) {
    const liveRows = live.map((r) => [
      el('span', { class: 'mono' }, r.run_id.slice(-12)),
      r.objective,
      `${r.elapsed_s}s`,
      r.cancelling
        ? el('span', { class: 'pill warn' }, 'cancelling')
        : el('span', { class: 'pill accent' }, r.state),
    ]);
    main.append(el('div', { class: 'card', style: 'margin-bottom:14px' },
      el('h3', {}, `${live.length} live`),
      table([{ label: 'Run' }, { label: 'Objective' },
             { label: 'Elapsed', numeric: true }, { label: 'State' }], liveRows)));
  }

  const stateClass = (value) => ({
    finished: 'good', needs_human: 'warn', running: 'accent', failed: 'bad',
  }[value] || '');

  const runRows = runs.map((r) => [
    el('span', { class: 'mono faint' }, r.id.slice(-12)),
    (r.objective || '').slice(0, 70),
    el('span', { class: `pill ${stateClass(r.state)}` }, r.state),
    r.steps,
    nf(r.tokens_total),
    pct(r.peak_utilisation),
    el('span', { class: 'faint mono', style: 'font-size:11px' }, r.sandbox.backend || '—'),
    el('button', { class: 'btn ghost', onclick: () => showRun(r.id) }, 'Inspect'),
  ]);

  main.append(el('div', { class: 'card' }, table([
    { label: 'Run' }, { label: 'Objective' }, { label: 'State' },
    { label: 'Steps', numeric: true }, { label: 'Tokens', numeric: true },
    { label: 'Peak util', numeric: true }, { label: 'Sandbox' }, { label: '' },
  ], runRows, 'No runs yet.')));
  main.append(el('div', { id: 'run-detail', style: 'margin-top:14px' }));
};

async function showRun(runId) {
  const box = $('#run-detail');
  box.replaceChildren(el('div', { class: 'empty' }, 'Loading…'));
  const [run, ledger, workspace] = await Promise.all([
    api(`/api/runs/${runId}`), api(`/api/runs/${runId}/ledger`), api(`/api/runs/${runId}/workspace`),
  ]);
  const report = run.report || {};
  box.replaceChildren(
    el('div', { class: 'card' },
      el('h3', {}, `Run ${runId.slice(-12)}`),
      el('p', { class: 'note' }, run.objective),
      el('div', { class: 'grid c4' },
        statCard('State', run.state, run.stop_reason, { small: true }),
        statCard('Steps', run.steps, `${nf(run.tokens_total)} tok`, { small: true }),
        statCard('Peak util', pct(run.peak_utilisation), `${nf(ledger.total_evicted)} evicted`, { small: true }),
        statCard('Isolation', run.sandbox.isolation || '—', run.sandbox.backend, { small: true })),
      run.final_answer ? el('div', {},
        el('label', {}, 'final answer'),
        el('pre', { class: 'code' }, run.final_answer)) : null,
      run.error ? el('div', { class: 'banner bad', style: 'margin-top:12px' },
        el('strong', {}, 'Failed'), el('div', {}, run.error)) : null),
    el('div', { class: 'card', style: 'margin-top:14px' },
      el('h3', {}, 'Context allocation'),
      el('p', { class: 'note' },
        ledger.compactions && ledger.compactions.length
          ? `${ledger.compactions.length} compaction(s) fired at step boundaries above the threshold.`
          : 'No compaction was needed: utilisation stayed below the threshold at every step boundary.'),
      renderLedger(ledger.steps)),
    (() => {
      const viewArtifact = async (path) => {
        try {
          const got = await api(`/api/runs/${runId}/artifact?path=${encodeURIComponent(path)}`);
          box.append(el('div', { class: 'card', style: 'margin-top:14px' },
            el('h3', {}, path),
            el('pre', { class: 'code' }, got.text)));
        } catch (e) { toast(e.message, 'bad'); }
      };
      const artifactRows = workspace.artifacts.map((a) => [
        el('span', { class: 'mono' }, a.path),
        nf(a.size),
        el('span', { class: 'faint' }, a.note || ''),
        el('button', { class: 'btn ghost', onclick: () => viewArtifact(a.path) }, 'View'),
      ]);
      return el('div', { class: 'card', style: 'margin-top:14px' },
        el('h3', {}, 'Workspace'),
        el('p', { class: 'note' },
          'The artefacts the run produced. Listing and digests only — serving a 40MB parquet ' +
          'through this console would be the browser making the mistake the platform exists to avoid.'),
        table([{ label: 'Path' }, { label: 'Bytes', numeric: true }, { label: 'Note' },
               { label: '' }], artifactRows, 'This run produced no artefacts.'));
    })(),
    Object.keys(report).length ? el('div', { class: 'card', style: 'margin-top:14px' },
      el('h3', {}, 'Agent report'),
      el('pre', { class: 'code' }, JSON.stringify(report, null, 1))) : null);
  box.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

views.hitl = async (main) => {
  const { pending } = await api('/api/hitl');
  main.append(...pageHead('Needs a human', `${pending.length} waiting`,
    'A run reaches here when it repeated the same failure, hit a policy gate, or ran out of steps. ' +
    'The default for an unattended run is to decline — a run that would have needed a person is a ' +
    'run that did not succeed autonomously, and letting it proceed would inflate the result.'));
  if (!pending.length) { main.append(el('div', { class: 'empty' }, 'Nothing is waiting.')); return; }

  main.append(el('div', { class: 'grid c2' }, pending.map((p) => {
    const message = el('input', { placeholder: 'guidance, if you are giving any' });
    const answer = async (decision) => {
      try {
        await api(`/api/hitl/${p.run_id}`, {
          method: 'POST', body: JSON.stringify({ decision, message: message.value }),
        });
        toast(`Answered ${decision}`, 'good'); render();
      } catch (e) { toast(e.message, 'bad'); }
    };
    return el('div', { class: 'card' },
      el('h3', {}, el('span', { class: 'pill warn' }, p.reason), ` step ${p.step}`),
      el('p', { class: 'note' }, p.question),
      p.context ? el('pre', { class: 'code' }, p.context.slice(0, 1200)) : null,
      el('div', {}, el('label', {}, 'message'), message),
      el('div', { class: 'form-actions' },
        el('button', { class: 'btn', onclick: () => answer('guidance') }, 'Send guidance'),
        el('button', { class: 'btn ghost', onclick: () => answer('approve') }, 'Approve'),
        el('button', { class: 'btn ghost', onclick: () => answer('deny') }, 'Deny'),
        el('button', { class: 'btn danger', onclick: () => answer('abort') }, 'Abort')));
  })));
};

views.insight = async (main) => {
  const [toolCost, arms] = await Promise.all([
    api('/api/insight/tool-cost'), api('/api/insight/arms'),
  ]);
  main.append(...pageHead('Measurements', null,
    'Everything here is read from committed artefacts. A figure shown next to a claim is the same ' +
    'figure the repository can be diffed against.'));

  const rows = toolCost.rows || [];
  if (rows.length) {
    const worst = Math.max(...rows.map((r) => r.full_json_schemas));
    const costRows = rows.map((r) => [
      nf(r.tools),
      el('strong', {}, nf(r.retrieved_signatures)),
      nf(r.all_signatures),
      nf(r.full_json_schemas),
      el('div', { class: 'bar-track' }, el('div', {
        class: 'bar-fill bad', style: `width:${(r.full_json_schemas / worst) * 100}%`,
      })),
      `${fx(r.saving_vs_schemas_pct, 1)}%`,
    ]);
    main.append(el('div', { class: 'card' },
      el('h3', {}, 'Tool representation cost vs registry size'),
      el('p', { class: 'note' },
        'Retrieval cost is flat in registry size because k is fixed; both alternatives are linear. ' +
        'At the largest size, injecting schemas costs more than a 32k window holds at all.'),
      table([
        { label: 'Tools', numeric: true }, { label: 'Retrieved', numeric: true },
        { label: 'All signatures', numeric: true }, { label: 'All schemas', numeric: true },
        { label: '' }, { label: 'Saving', numeric: true },
      ], costRows)));
  }

  const gap = arms.gap_decomposition;
  if (gap) {
    const total = gap.gradable_slots || 1;
    const seg = (slot, count, tip) => el('div', {
      class: 'ledger-seg', 'data-slot': slot, style: `width:${(count / total) * 100}%`,
    }, el('div', { class: 'tip' }, `${tip} — ${nf(count)}`));

    main.append(el('div', { class: 'card', style: 'margin-top:14px' },
      el('h3', {}, "Where the deterministic arm's gap actually lives"),
      el('p', { class: 'note' },
        'The model node fires only on fields the pipeline failed to answer, so a field it answered ' +
        'confidently and wrongly is never revisited. That is a ceiling the arm cannot pass however ' +
        'good the model is — a fixed pipeline can fix its misses but not its mistakes.'),
      el('div', { class: 'grid c3' },
        statCard('Measured', fx(gap.measured_accuracy), `${nf(gap.correct)} correct`, { small: true }),
        statCard('Arm ceiling', fx(gap.arm_ceiling),
          `+${nf(gap.unresolved_routed_to_model)} addressable`, { small: true }),
        statCard('Unreachable', pct(gap.structurally_unreachable),
          `${nf(gap.resolved_but_wrong)} answered wrongly`, { small: true })),
      el('div', { style: 'margin-top:14px' },
        el('div', { class: 'ledger-bar', style: 'height:26px' },
          seg('tools', gap.correct, 'correct'),
          seg('memory', gap.resolved_but_wrong, 'answered wrongly, never revisited'),
          seg('workspace', gap.unresolved_routed_to_model, 'routed to the model')),
        el('div', { class: 'legend' },
          el('span', {}, el('i', { style: 'background:var(--slot-tools)' }), 'correct'),
          el('span', {}, el('i', { style: 'background:var(--slot-memory)' }), 'wrong, unreachable'),
          el('span', {}, el('i', { style: 'background:var(--slot-workspace)' }),
            'addressable by the model')))));
  }

  const armRows = (arms.arms || []).map((a) => {
    const result = a.result || {};
    const score = result.headline_rate != null
      ? el('span', {}, fx(result.headline_rate),
          result.measurement_kind === 'deterministic'
            ? el('span', { class: 'pill warn', style: 'margin-left:6px' }, 'no model')
            : null)
      : el('span', { class: 'faint' }, 'not run');
    return [
      el('span', { class: 'mono' }, a.arm),
      el('span', { class: 'dim' }, a.description),
      el('span', { class: `pill ${a.deterministic_workflow ? 'warn' : 'accent'}` },
        a.deterministic_workflow ? 'task graph' : 'agent loop'),
      score,
    ];
  });
  main.append(el('div', { class: 'card', style: 'margin-top:14px' },
    el('h3', {}, 'Ablation arms'),
    el('p', { class: 'note' },
      'An arm differs from `full` in one component. A result appears once the arm has run; an arm ' +
      'with no result has not been run, and the harness refuses to publish a score no model produced.'),
    table([{ label: 'Arm' }, { label: 'What it removes' }, { label: 'Mode' },
           { label: 'Result', numeric: true }], armRows)));

  const ctx = arms.context_cost;
  if (ctx && ctx.arms && ctx.arms.length) {
    const tightest = Math.min(...ctx.arms.map((a) => a.window));
    const tight = ctx.arms.filter((a) => a.window === tightest);
    const ctxRows = tight.map((a) => [
      el('span', { class: 'mono' }, a.arm),
      nf(a.prompt_tokens_mean_per_step),
      nf(a.evicted_tokens_total),
      a.compactions,
      `${a.delta_vs_full_pct > 0 ? '+' : ''}${fx(a.delta_vs_full_pct, 1)}%`,
    ]);
    main.append(el('div', { class: 'card', style: 'margin-top:14px' },
      el('h3', {}, `Context cost per arm at a ${nf(tightest)} window`),
      el('p', { class: 'note' },
        'Measured with an identical trajectory in every arm and no accuracy claim — accuracy needs a ' +
        'model, context cost does not. The window is swept because at a roomy window every arm looks ' +
        'alike: the mechanisms only engage once the budget binds.'),
      table([{ label: 'Arm' }, { label: 'Tokens/step', numeric: true },
             { label: 'Evicted', numeric: true }, { label: 'Compactions', numeric: true },
             { label: 'vs full', numeric: true }], ctxRows)));
  }
};

/* ---------- shell ---------- */
async function refreshHealth() {
  try {
    const [health, version, hitl] = await Promise.all([
      api('/api/insight/health'), api('/api/version'), api('/api/hitl'),
    ]);
    state.health = health;
    $('#version').textContent = `v${version.version}`;

    const sandbox = health.sandbox || {};
    const container = sandbox.active === 'docker';
    $('#dot-sandbox').className = `health-dot ${container ? 'ok' : 'warn'}`;
    $('#foot-sandbox').textContent = `sandbox ${sandbox.active || '—'}`;
    $('#foot-sandbox').title = container
      ? sandbox.isolation
      : sandbox.unavailable_reason || 'container backend unavailable';

    const session = health.session || {};
    $('#dot-session').className = `health-dot ${session.backend === 'redis' ? 'ok' : 'warn'}`;
    $('#foot-session').textContent = `session ${session.backend || '—'}`;
    $('#foot-session').title = session.degraded_reason || '';

    const usable = (health.usable_providers || []).length;
    $('#dot-provider').className = `health-dot ${usable ? 'ok' : 'bad'}`;
    $('#foot-provider').textContent = usable ? `${usable} provider(s)` : 'no provider';
    $('#foot-provider').title = usable
      ? (health.usable_providers || []).join(', ')
      : 'Runs will fail on NoProviderAvailable. Set a free-tier key or start ollama.';

    const badge = $('#hitl-badge');
    badge.textContent = (hitl.pending || []).length;
    badge.className = `badge ${(hitl.pending || []).length ? 'alert' : ''}`;
  } catch { /* the shell must render even when the API is unhappy */ }
}

function go(view) {
  state.view = view;
  if (location.hash !== `#${view}`) location.hash = view;
  render();
}

async function render() {
  const main = $('#main');
  main.replaceChildren(el('div', { class: 'empty' }, 'Loading…'));
  for (const button of document.querySelectorAll('#nav button')) {
    button.setAttribute('aria-current', String(button.dataset.view === state.view));
  }
  const view = views[state.view] || views.overview;
  const fresh = el('div', {});
  try {
    await view(fresh);
    main.replaceChildren(...fresh.childNodes);
  } catch (e) {
    main.replaceChildren(el('div', { class: 'banner bad' },
      el('strong', {}, 'Could not load this view'), el('div', {}, e.message)));
  }
}

$('#nav').addEventListener('click', (event) => {
  const button = event.target.closest('button[data-view]');
  if (button) go(button.dataset.view);
});
window.addEventListener('hashchange', () => {
  const view = location.hash.slice(1);
  if (view && views[view] && view !== state.view) { state.view = view; render(); }
});

state.view = (location.hash.slice(1) && views[location.hash.slice(1)]) ? location.hash.slice(1) : 'overview';
refreshHealth();
render();
setInterval(refreshHealth, 15000);
