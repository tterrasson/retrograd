// Viewer for a retrograd `[observe]` export. Reads `feed/` through <script>
// tags so it works from file://, or an `observe.jsonl` opened by hand.
// Every model or tool output is rendered through textContent only.
"use strict";

(() => {
  const POLL_MS = 2000;
  const LIVE_MS = 30000;
  const MEMORY_RECORDS = 200000;
  const FOLD_LINES = 20;
  const SCHEMA_VERSION = 1;

  const CHARTS = [
    { key: "reward/mean", band: "reward/std", title: "reward/mean ± std" },
    { key: "batch/trained_fraction", title: "batch/trained_fraction" },
    { key: "completions/length_mean", title: "completions/length_mean" },
    { key: "policy/kl", title: "policy/kl" },
    { key: "agent/turns_per_traj_mean", title: "agent/turns_per_traj_mean" },
    { key: "agent/tool_calls_per_traj", title: "agent/tool_calls_per_traj" },
    { key: "agent/failed_fraction", title: "agent/failed_fraction" },
  ];

  const $ = (id) => document.getElementById(id);

  const feed = {
    generation: null,
    chunks: 0,
    loaded: 0,
    pending: null,
    manifestStamp: null,
    changedAt: 0,
    dropped: 0,
    fromFile: false,
    importNote: "",
  };

  let data;
  const view = {
    selected: null,
    follow: true,
    keep: 200,
    compare: [],
    // `<details>` the reader unfolded, kept across re-renders; see `detailsKey`.
    open: new Set(),
    renderQueued: false,
  };

  function resetData() {
    data = {
      run: null,
      active: [],
      updates: new Map(),
      prompts: new Map(),
      tools: new Set(),
      records: 0,
    };
    view.selected = null;
    view.compare = [];
    view.open.clear();
  }
  resetData();

  // ---------------------------------------------------------------- feed

  window.RG_FEED = {
    manifest(manifest) {
      if (feed.fromFile) return;
      if (manifest.updated_at !== feed.manifestStamp) {
        feed.manifestStamp = manifest.updated_at;
        feed.changedAt = Date.now();
      }
      if (manifest.generation !== feed.generation) {
        resetData();
        feed.generation = manifest.generation;
        feed.loaded = 0;
        feed.pending = null;
      }
      feed.chunks = manifest.chunks;
      feed.dropped = manifest.dropped_batches || 0;
      requestNextChunk();
      scheduleRender();
    },
    chunk(index, records) {
      const script = document.currentScript;
      if (feed.fromFile || !script || script.dataset.generation !== feed.generation) return;
      if (index !== feed.loaded) return;
      feed.pending = null;
      feed.loaded += 1;
      ingestAll(records);
      requestNextChunk();
    },
  };

  function inject(src, generation) {
    const script = document.createElement("script");
    script.src = src;
    if (generation !== undefined) script.dataset.generation = generation;
    const requested = feed.pending;
    const finished = () => {
      script.remove();
      // A loaded script without its callback must be retried too. An old
      // request must not clear the pending chunk of a newer generation.
      if (generation !== undefined && generation === feed.generation && feed.pending === requested) {
        feed.pending = null;
      }
    };
    script.onload = finished;
    script.onerror = finished;
    document.head.appendChild(script);
  }

  function requestNextChunk() {
    if (feed.pending !== null || feed.loaded >= feed.chunks) return;
    feed.pending = feed.loaded;
    const name = String(feed.loaded).padStart(6, "0");
    inject(`feed/${feed.generation}/${name}.js`, feed.generation);
  }

  function poll() {
    if (!feed.fromFile) inject(`feed/manifest.js?t=${Date.now()}`);
    renderHeader();
  }

  // --------------------------------------------------------------- model

  const updateKey = (segment, update) => `${segment}:${update}`;
  const memberKey = (group, member) => `${group ?? "-"}:${member}`;

  function entryFor(segment, update) {
    const key = updateKey(segment, update);
    let entry = data.updates.get(key);
    if (!entry) {
      entry = {
        key,
        segment,
        update,
        rollouts: new Map(),
        selection: new Map(),
        outcome: new Map(),
        summary: null,
        evicted: false,
      };
      data.updates.set(key, entry);
    }
    return entry;
  }

  function ingestAll(records) {
    for (const record of records) ingest(record);
    enforceMemory();
    scheduleRender();
  }

  function ingest(record) {
    if (!record || record.v !== SCHEMA_VERSION) return;
    data.records += 1;
    switch (record.type) {
      case "run":
        startSegment(record);
        break;
      case "prompt":
        data.prompts.set(`${record.segment}|${record.key}`, record);
        break;
      case "rollout": {
        entryFor(record.segment, record.update).rollouts.set(
          memberKey(record.group, record.member),
          record,
        );
        for (const message of record.messages || []) {
          for (const call of message.tool_calls || []) data.tools.add(call.name);
        }
        break;
      }
      case "selection":
      case "outcome": {
        const entry = entryFor(record.segment, record.update);
        const target = record.type === "selection" ? entry.selection : entry.outcome;
        for (const item of record.entries || []) {
          target.set(memberKey(item.group, item.member), item);
        }
        break;
      }
      case "update":
        entryFor(record.segment, record.update).summary = record;
        break;
      default:
        break;
    }
  }

  // A run without a checkpoint starts an independent view. A resume from K
  // hides, at once, every update above K the earlier segments produced.
  function startSegment(run) {
    data.run = run;
    if (run.resumed_from_update === null || run.resumed_from_update === undefined) {
      data.active = [];
    } else {
      for (const segment of data.active) {
        segment.upto = Math.min(segment.upto, run.resumed_from_update);
      }
    }
    data.active.push({ segment: run.segment, upto: Infinity });
  }

  function isVisible(entry) {
    return data.active.some(
      (segment) => segment.segment === entry.segment && entry.update <= segment.upto,
    );
  }

  function visibleUpdates() {
    return [...data.updates.values()]
      .filter(isVisible)
      .sort((a, b) => a.update - b.update || a.segment - b.segment);
  }

  function enforceMemory() {
    if (data.records <= MEMORY_RECORDS) return;
    const visible = new Set(visibleUpdates().slice(-view.keep).map((entry) => entry.key));
    const candidates = [...data.updates.values()]
      .filter((entry) => !visible.has(entry.key) && entry.rollouts.size > 0)
      .sort((a, b) => a.segment - b.segment || a.update - b.update);
    for (const entry of candidates) {
      if (data.records <= MEMORY_RECORDS) break;
      data.records -= entry.rollouts.size;
      entry.rollouts.clear();
      entry.selection.clear();
      entry.outcome.clear();
      entry.evicted = true;
    }
  }

  function effective(entry, rollout) {
    const key = memberKey(rollout.group, rollout.member);
    const selection = entry.selection.get(key);
    const outcome = entry.outcome.get(key);
    const eligible = selection ? selection.eligible : rollout.eligible;
    let trained = null;
    if (outcome) trained = outcome.trained;
    else if (rollout.trained === false || eligible === false) trained = false;
    return {
      eligible,
      trained,
      advantage: rollout.advantage ?? (selection ? selection.advantage : null),
      skipReason: rollout.skip_reason ?? (selection ? selection.skip_reason : null),
    };
  }

  // -------------------------------------------------------------- render

  function el(tag, options = {}, children = []) {
    const node = document.createElement(tag);
    if (options.class) node.className = options.class;
    if (options.text !== undefined && options.text !== null) node.textContent = String(options.text);
    if (options.title) node.title = options.title;
    for (const child of children) if (child) node.appendChild(child);
    return node;
  }

  function svg(tag, attributes = {}) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
    return node;
  }

  function fmt(value, digits = 3) {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    const number = Number(value);
    if (Number.isInteger(number)) return String(number);
    return number.toFixed(digits);
  }

  function scheduleRender() {
    if (view.renderQueued) return;
    view.renderQueued = true;
    requestAnimationFrame(() => {
      view.renderQueued = false;
      render();
    });
  }

  function render() {
    const updates = visibleUpdates();
    if (view.follow && updates.length) view.selected = updates[updates.length - 1].key;
    renderHeader();
    renderCharts(updates);
    renderSidebar(updates);
    renderToolFilter();
    renderUpdate();
  }

  function renderHeader() {
    const run = data.run;
    $("algorithm").textContent = run ? run.algorithm : "no run loaded";
    $("model").textContent = run ? run.model : "";
    $("segment").textContent = run ? `segment ${run.segment}` : "";
    $("live").classList.toggle(
      "on",
      !feed.fromFile && feed.changedAt > 0 && Date.now() - feed.changedAt < LIVE_MS,
    );
    const updates = visibleUpdates().filter((entry) => entry.summary);
    const last = updates[updates.length - 1];
    $("last-update").textContent = last
      ? `update ${last.update} at ${new Date(last.summary.time).toLocaleTimeString()}`
      : "";
    $("record-count").textContent = `${data.records} records`;
    const notes = [];
    if (feed.dropped) notes.push(`${feed.dropped} export batches were dropped: some updates are missing data.`);
    if (feed.importNote) notes.push(feed.importNote);
    const banner = $("banner");
    banner.hidden = notes.length === 0;
    banner.textContent = notes.join(" ");
  }

  function renderCharts(updates) {
    const container = $("charts");
    container.replaceChildren();
    for (const chart of CHARTS) {
      const points = updates
        .filter((entry) => entry.summary && Number.isFinite(entry.summary.metrics[chart.key]))
        .map((entry) => ({
          entry,
          x: entry.update,
          y: entry.summary.metrics[chart.key],
          spread: chart.band ? entry.summary.metrics[chart.band] : null,
        }));
      if (!points.length) continue;
      container.appendChild(lineChart(chart.title, points));
    }
  }

  function lineChart(title, points) {
    const width = 260;
    const height = 90;
    const pad = { left: 30, right: 6, top: 6, bottom: 14 };
    const low = Math.min(...points.map((p) => p.y - (Number.isFinite(p.spread) ? p.spread : 0)));
    const high = Math.max(...points.map((p) => p.y + (Number.isFinite(p.spread) ? p.spread : 0)));
    const span = high - low || 1;
    const first = points[0].x;
    const last = points[points.length - 1].x;
    const sx = (x) => pad.left + ((x - first) / (last - first || 1)) * (width - pad.left - pad.right);
    const sy = (y) => height - pad.bottom - ((y - low) / span) * (height - pad.top - pad.bottom);
    const graph = svg("svg", { viewBox: `0 0 ${width} ${height}`, preserveAspectRatio: "none" });
    const banded = points.filter((p) => Number.isFinite(p.spread));
    if (banded.length > 1) {
      const upper = banded.map((p) => `${sx(p.x)},${sy(p.y + p.spread)}`);
      const lower = banded.map((p) => `${sx(p.x)},${sy(p.y - p.spread)}`).reverse();
      graph.appendChild(svg("polygon", { class: "band", points: upper.concat(lower).join(" ") }));
    }
    graph.appendChild(
      svg("polyline", { class: "line", points: points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ") }),
    );
    for (const point of points) {
      const dot = svg("circle", {
        class: point.entry.key === view.selected ? "point current" : "point",
        cx: sx(point.x),
        cy: sy(point.y),
        r: 2.5,
      });
      const tip = svg("title");
      tip.textContent = `update ${point.x}: ${fmt(point.y, 4)}`;
      dot.appendChild(tip);
      dot.addEventListener("click", () => select(point.entry.key));
      graph.appendChild(dot);
    }
    for (const [text, x, y, anchor] of [
      [fmt(high), 2, pad.top + 6, "start"],
      [fmt(low), 2, height - pad.bottom, "start"],
      [String(first), pad.left, height - 2, "start"],
      [String(last), width - pad.right, height - 2, "end"],
    ]) {
      const label = svg("text", { class: "axis", x, y, "text-anchor": anchor });
      label.textContent = text;
      graph.appendChild(label);
    }
    return el("div", { class: "chart" }, [el("h3", { text: title }), graph]);
  }

  function renderSidebar(updates) {
    const list = $("updates");
    list.replaceChildren();
    const rewards = updates
      .map((entry) => entry.summary && entry.summary.metrics["reward/mean"])
      .filter(Number.isFinite);
    const low = Math.min(...rewards);
    const high = Math.max(...rewards);
    for (const entry of updates) {
      const reward = entry.summary ? entry.summary.metrics["reward/mean"] : undefined;
      const bar = el("span");
      if (Number.isFinite(reward)) {
        bar.style.width = `${high > low ? ((reward - low) / (high - low)) * 100 : 100}%`;
      }
      const item = el(
        "li",
        {
          title: entry.summary
            ? `reward/mean ${fmt(reward)}`
            : "incomplete update, or incomplete export",
        },
        [el("span", { class: "label", text: `#${entry.update}` }), el("span", { class: "minibar" }, [bar])],
      );
      if (!entry.summary) item.classList.add("incomplete");
      if (entry.summary && entry.summary.status === "skipped") item.classList.add("skipped");
      if (entry.key === view.selected) item.classList.add("selected");
      item.addEventListener("click", () => select(entry.key));
      list.appendChild(item);
    }
    const selected = list.querySelector(".selected");
    if (selected && view.follow) selected.scrollIntoView({ block: "nearest" });
  }

  function renderToolFilter() {
    const picker = $("tool-filter");
    const current = picker.value;
    const names = [...data.tools].sort();
    if (names.every((name, index) => picker.options[index + 1]?.value === name) &&
        picker.options.length === names.length + 1) return;
    const any = el("option", { text: "any tool" });
    any.value = "";
    picker.replaceChildren(any);
    for (const name of names) {
      const option = el("option", { text: name });
      option.value = name;
      picker.appendChild(option);
    }
    picker.value = current;
  }

  function select(key) {
    view.selected = key;
    view.follow = false;
    $("follow").checked = false;
    scheduleRender();
  }

  function readFilters() {
    const number = (id) => {
      const value = $(id).value;
      return value === "" ? null : Number(value);
    };
    return {
      text: $("search").value.trim().toLowerCase(),
      min: number("reward-min"),
      max: number("reward-max"),
      trainedOnly: $("trained-only").checked,
      skippedOnly: $("skipped-only").checked,
      tool: $("tool-filter").value,
      compact: $("compact").checked,
    };
  }

  function textOf(rollout) {
    if (rollout.completion !== undefined) return rollout.completion;
    return (rollout.messages || [])
      .map((message) =>
        [message.content]
          .concat((message.tool_calls || []).map((call) => `${call.name} ${JSON.stringify(call.arguments)}`))
          .join("\n"),
      )
      .join("\n");
  }

  function accepts(filters, rollout, state) {
    if (filters.trainedOnly && state.trained !== true) return false;
    if (
      filters.skippedOnly &&
      !(rollout.truncated || state.skipReason || state.eligible === false)
    ) {
      return false;
    }
    if ((filters.min !== null || filters.max !== null) && !Number.isFinite(rollout.reward)) return false;
    if (filters.min !== null && !(rollout.reward >= filters.min)) return false;
    if (filters.max !== null && !(rollout.reward <= filters.max)) return false;
    if (
      filters.tool &&
      !(rollout.messages || []).some((message) =>
        (message.tool_calls || []).some((call) => call.name === filters.tool),
      )
    ) {
      return false;
    }
    if (filters.text && !textOf(rollout).toLowerCase().includes(filters.text)) return false;
    return true;
  }

  function renderUpdate() {
    const container = $("update-view");
    const entry = data.updates.get(view.selected);
    if (!entry || !isVisible(entry)) {
      container.replaceChildren(
        el("p", { class: "muted", text: data.records ? "Select an update." : "Waiting for data." }),
      );
      renderCompareButton();
      return;
    }
    const filters = readFilters();
    const head = el("div", { class: "update-head" }, [
      el("h2", { text: `Update ${entry.update}` }),
      el("span", { class: "muted", text: `segment ${entry.segment}` }),
    ]);
    if (!entry.summary) {
      head.appendChild(el("span", { class: "notice", text: "incomplete update, or incomplete export" }));
    } else {
      head.appendChild(el("span", { class: "badge", text: entry.summary.status }));
      for (const name of ["reward/mean", "reward/std", "batch/trained_fraction", "policy/kl"]) {
        if (name in entry.summary.metrics) {
          head.appendChild(el("span", { class: "muted", text: `${name} ${fmt(entry.summary.metrics[name])}` }));
        }
      }
    }
    if (entry.evicted) {
      head.appendChild(el("span", { class: "notice", text: "rollouts evicted from memory (see keep)" }));
    } else if (!entry.rollouts.size) {
      head.appendChild(el("span", { class: "muted", text: "no rollouts exported for this update" }));
    }
    const groups = new Map();
    for (const rollout of entry.rollouts.values()) {
      const key = rollout.group ?? "all";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(rollout);
    }
    const cards = [...groups.entries()]
      .sort(([a], [b]) => (a === "all" ? 0 : a) - (b === "all" ? 0 : b))
      .map(([group, rollouts]) => renderGroup(entry, group, rollouts, filters));
    container.replaceChildren(head, ...cards);
    for (const details of container.querySelectorAll("details")) {
      if (view.open.has(detailsKey(entry, details))) details.open = true;
    }
    renderCompareButton();
  }

  // Identifies a `<details>` by its update, the keyed group/member around it and
  // the summaries of its enclosing `<details>`: stable while records stream in.
  function detailsKey(entry, details) {
    const parts = [];
    for (let node = details; node && node.id !== "update-view"; node = node.parentElement) {
      if (node.dataset.key) parts.push(node.dataset.key);
      else if (node.tagName === "DETAILS") parts.push(node.querySelector(":scope > summary")?.textContent ?? "");
    }
    return `${entry.key}|${parts.reverse().join("|")}`;
  }

  function renderGroup(entry, group, rollouts, filters) {
    const texts = new Map();
    for (const rollout of rollouts) {
      const text = textOf(rollout);
      texts.set(text, (texts.get(text) || 0) + 1);
    }
    const promptKeys = [...new Set(rollouts.map((rollout) => rollout.prompt))];
    const card = el("div", { class: "group" }, [
      el("div", { class: "group-head" }, [
        el("strong", { text: group === "all" ? "rollouts" : `group ${group}` }),
        el("span", { class: "muted mono", text: promptKeys.join(", ") }),
      ]),
    ]);
    card.dataset.key = `group ${group}`;
    // PPO has no groups: each member carries its own prompt.
    const flat = group === "all";
    if (!flat) {
      for (const key of promptKeys) {
        card.appendChild(renderPrompt(key, data.prompts.get(`${entry.segment}|${key}`)));
      }
    }
    const members = rollouts
      .map((rollout) => ({ rollout, state: effective(entry, rollout) }))
      .filter(({ rollout, state }) => accepts(filters, rollout, state))
      .sort((a, b) => (b.rollout.reward ?? -Infinity) - (a.rollout.reward ?? -Infinity));
    for (const { rollout, state } of members) {
      const duplicate = texts.get(textOf(rollout)) > 1;
      const member = renderMember(entry, rollout, state, duplicate, filters.compact, true);
      if (flat) {
        member.insertBefore(
          renderPrompt(rollout.prompt, data.prompts.get(`${entry.segment}|${rollout.prompt}`)),
          member.children[1] || null,
        );
      }
      card.appendChild(member);
    }
    if (!members.length) card.appendChild(el("div", { class: "member muted", text: "no member matches the filters" }));
    return card;
  }

  function renderPrompt(key, prompt) {
    const details = el("details", { class: "prompt" }, [el("summary", { text: `prompt ${key}` })]);
    if (!prompt) {
      details.appendChild(el("p", { class: "muted", text: "prompt record not received" }));
      return details;
    }
    for (const message of prompt.messages) {
      if (message.role === "system") {
        details.appendChild(
          el("details", {}, [el("summary", { text: "system" }), el("pre", { class: "text", text: message.content })]),
        );
      } else {
        details.appendChild(bubble(message, null, []));
      }
    }
    if (prompt.reward_text !== undefined && prompt.reward_text !== lastUserText(prompt)) {
      details.appendChild(el("div", { class: "muted", text: "reward text" }));
      details.appendChild(el("pre", { class: "text", text: prompt.reward_text }));
    }
    if (prompt.metadata && Object.keys(prompt.metadata).length) {
      details.appendChild(
        el("details", {}, [
          el("summary", { text: "scenario metadata" }),
          el("pre", { class: "text", text: JSON.stringify(prompt.metadata, null, 2) }),
        ]),
      );
    }
    return details;
  }

  function lastUserText(prompt) {
    const users = prompt.messages.filter((message) => message.role === "user");
    return users.length ? users[users.length - 1].content : undefined;
  }

  function badge(text, tone) {
    return el("span", { class: tone ? `badge ${tone}` : "badge", text });
  }

  function advantageBar(value, scale) {
    const bar = el("span", { title: `advantage ${fmt(value, 4)}` });
    const holder = el("span", { class: "advantage", title: `advantage ${fmt(value, 4)}` }, [bar]);
    if (Number.isFinite(value)) {
      bar.className = value >= 0 ? "pos" : "neg";
      bar.style.width = `${Math.min(Math.abs(value) / scale, 1) * 50}%`;
    }
    return holder;
  }

  function renderMember(entry, rollout, state, duplicate, compact, selectable) {
    const identity = { key: entry.key, group: rollout.group, member: rollout.member };
    const head = el("div", { class: "member-head" });
    if (selectable) {
      const box = el("input");
      box.type = "checkbox";
      box.checked = view.compare.some((item) => sameMember(item, identity));
      box.addEventListener("change", () => toggleCompare(identity, box.checked));
      head.appendChild(box);
    }
    head.appendChild(el("span", { class: "mono", text: `#${rollout.member}` }));
    head.appendChild(el("span", { class: "reward", text: fmt(rollout.reward) }));
    if (rollout.reward_raw !== null && rollout.reward_raw !== rollout.reward) {
      head.appendChild(el("span", { class: "muted", text: `raw ${fmt(rollout.reward_raw)}` }));
    }
    if (rollout.judge_term) {
      head.appendChild(el("span", { class: "muted", text: `judge ${fmt(rollout.judge_term)}` }));
    }
    head.appendChild(advantageBar(state.advantage, 2));
    head.appendChild(el("span", { class: "muted", text: `adv ${fmt(state.advantage)}` }));
    if (rollout.advantage_min !== undefined) {
      head.appendChild(
        el("span", { class: "muted", text: `[${fmt(rollout.advantage_min)}, ${fmt(rollout.advantage_max)}]` }),
      );
    }
    head.appendChild(el("span", { class: "muted", text: `${rollout.tokens} tok` }));
    head.appendChild(el("span", { class: "muted", text: `seed ${rollout.seed}` }));
    if (rollout.truncated) head.appendChild(badge("truncated", "warn"));
    if (state.skipReason) head.appendChild(badge(`skipped: ${state.skipReason}`, "bad"));
    else if (state.eligible === false) head.appendChild(badge("skipped", "bad"));
    if (state.trained === true) head.appendChild(badge("trained", "good"));
    if (state.trained === null) head.appendChild(badge("unknown execution"));
    if (duplicate) head.appendChild(badge("duplicate"));
    const node = el("div", { class: "member" }, [head]);
    node.dataset.key = `member ${rollout.member}`;
    if (rollout.completion !== undefined) {
      node.appendChild(el("pre", { class: "text", text: rollout.completion }));
    } else if (compact) {
      node.appendChild(el("div", { class: "compact-line", text: compactLine(rollout.messages || []) }));
    } else {
      node.appendChild(conversation(rollout));
    }
    return node;
  }

  function compactLine(messages) {
    const names = new Map();
    const parts = [];
    for (const message of messages) {
      if (message.role === "tool") {
        const name = names.get(message.tool_call_id) || "?";
        parts.push(`tool(${name})${message.is_error ? "!" : ""}`);
      } else if (message.role === "assistant") {
        for (const call of message.tool_calls || []) names.set(call.id, call.name);
        const calls = (message.tool_calls || []).map((call) => call.name);
        parts.push(calls.length ? `assistant[${calls.join(", ")}]` : "assistant");
      } else {
        parts.push(message.role);
      }
    }
    return parts.join(" → ");
  }

  function conversation(rollout) {
    const messages = rollout.messages || [];
    const thread = el("div", { class: "thread" });
    if (rollout.prefix === false) {
      thread.appendChild(el("div", { class: "muted", text: "full conversation (the scenario prefix did not match)" }));
    }
    const calls = new Map();
    for (const message of messages) {
      for (const call of message.tool_calls || []) calls.set(call.id, call.name);
    }
    const rewardsAt = new Map();
    const unattributed = [];
    for (const step of rollout.step_rewards || []) {
      if (!step.message_indices.length) {
        unattributed.push(step);
        continue;
      }
      for (const index of step.message_indices) {
        if (!rewardsAt.has(index)) rewardsAt.set(index, []);
        rewardsAt.get(index).push(step);
      }
    }
    messages.forEach((message, index) => {
      thread.appendChild(bubble(message, calls, rewardsAt.get(index) || []));
    });
    if (unattributed.length) {
      thread.appendChild(
        el("div", {
          class: "unattributed",
          text: `step rewards without a message: ${unattributed
            .map((step) => `step ${step.step_index} (${step.kind}) ${fmt(step.reward)}`)
            .join(", ")}`,
        }),
      );
    }
    if (rollout.terminal_reward_raw !== null && rollout.terminal_reward_raw !== undefined) {
      thread.appendChild(el("div", { class: "muted", text: `terminal reward ${fmt(rollout.terminal_reward_raw)}` }));
    }
    if (rollout.judge_explanation) {
      thread.appendChild(
        el("div", { class: "judge" }, [
          el("div", { class: "muted", text: "judge" }),
          el("pre", { class: "text", text: rollout.judge_explanation }),
        ]),
      );
    }
    if (rollout.metadata && Object.keys(rollout.metadata).length) {
      thread.appendChild(
        el("details", {}, [
          el("summary", { text: "trajectory metadata" }),
          el("pre", { class: "text", text: JSON.stringify(rollout.metadata, null, 2) }),
        ]),
      );
    }
    return thread;
  }

  function foldable(text) {
    const lines = text.split("\n");
    if (lines.length <= FOLD_LINES) return el("pre", { class: "text", text });
    return el("details", {}, [
      el("summary", { text: `${lines.slice(0, 2).join(" ⏎ ")} … (${lines.length} lines)` }),
      el("pre", { class: "text", text }),
    ]);
  }

  function bubble(message, calls, steps) {
    const role = el("div", { class: "role" }, [el("span", { text: message.role })]);
    const node = el("div", { class: `bubble ${message.role}` }, [role]);
    if (message.role === "tool") {
      const name = calls && calls.get(message.tool_call_id);
      role.appendChild(
        el("span", {
          class: "mono",
          text: `${name || "unknown call"}${message.tool_call_id ? ` (${message.tool_call_id})` : ""}`,
        }),
      );
      if (message.is_error) {
        node.classList.add("error");
        role.appendChild(badge("error", "bad"));
      }
    }
    for (const step of steps) {
      role.appendChild(
        el("span", {
          class: step.reward < 0 ? "step-reward negative" : "step-reward",
          text: `step ${step.step_index} ${step.reward >= 0 ? "+" : ""}${fmt(step.reward)}`,
        }),
      );
    }
    if (message.content) {
      node.appendChild(message.role === "tool" ? foldable(message.content) : el("pre", { class: "text", text: message.content }));
    }
    for (const call of message.tool_calls || []) {
      node.appendChild(
        el("div", { class: "call" }, [
          el("span", { class: "name", text: call.name }),
          el("span", { class: "muted mono", text: ` ${call.id}` }),
          el("pre", { class: "text", text: JSON.stringify(call.arguments, null, 2) }),
        ]),
      );
    }
    return node;
  }

  // ------------------------------------------------------------- compare

  function sameMember(a, b) {
    return a.key === b.key && a.group === b.group && a.member === b.member;
  }

  function toggleCompare(identity, on) {
    view.compare = view.compare.filter((item) => !sameMember(item, identity));
    if (on) view.compare.push(identity);
    if (view.compare.length > 2) view.compare.shift();
    scheduleRender();
  }

  function renderCompareButton() {
    const button = $("compare");
    button.textContent = `compare ${view.compare.length}/2`;
    button.disabled = view.compare.length !== 2;
  }

  function openCompare() {
    const body = $("compare-body");
    body.replaceChildren();
    for (const identity of view.compare) {
      const entry = data.updates.get(identity.key);
      const rollout = entry && entry.rollouts.get(memberKey(identity.group, identity.member));
      if (!rollout) {
        body.appendChild(el("p", { class: "muted", text: "no longer in memory" }));
        continue;
      }
      body.appendChild(
        el("div", {}, [
          el("h3", { text: `update ${entry.update} · group ${identity.group ?? "–"} · #${identity.member}` }),
          renderMember(entry, rollout, effective(entry, rollout), false, false, false),
        ]),
      );
    }
    $("compare-dialog").showModal();
  }

  // ---------------------------------------------------------- file import

  function importText(text) {
    feed.fromFile = true;
    feed.dropped = 0;
    feed.importNote = "";
    resetData();
    const batches = new Map();
    const order = [];
    let unreadable = 0;
    for (const line of text.split("\n")) {
      if (!line.trim()) continue;
      let record;
      try {
        record = JSON.parse(line);
      } catch {
        unreadable += 1;
        continue;
      }
      if (!record || record.v !== SCHEMA_VERSION ||
          !Number.isSafeInteger(record.segment) || record.segment < 0 ||
          !Number.isSafeInteger(record.batch_id) || record.batch_id < 0 ||
          !Number.isSafeInteger(record.batch_index) || record.batch_index < 0 ||
          !Number.isSafeInteger(record.batch_len) || record.batch_len < 1) {
        unreadable += 1;
        continue;
      }
      const id = `${record.segment}:${record.batch_id}`;
      if (!batches.has(id)) {
        batches.set(id, []);
        order.push(id);
      }
      batches.get(id).push(record);
    }
    const complete = [];
    let skipped = 0;
    for (const id of order) {
      const records = batches.get(id);
      if (records.length === records[0].batch_len && records.every((record, index) =>
        record.batch_index === index && record.batch_len === records.length)) {
        for (const record of records) complete.push(record);
      }
      else skipped += 1;
    }
    if (unreadable || skipped) {
      feed.importNote = `${unreadable} unreadable lines and ${skipped} incomplete batches were ignored.`;
    }
    view.follow = true;
    $("follow").checked = true;
    ingestAll(complete);
  }

  function readFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => importText(String(reader.result));
    reader.readAsText(file);
  }

  // -------------------------------------------------------------- wiring

  $("open-file").addEventListener("change", (event) => readFile(event.target.files[0]));
  document.addEventListener("dragover", (event) => {
    event.preventDefault();
    document.body.classList.add("drop-target");
  });
  document.addEventListener("dragleave", () => document.body.classList.remove("drop-target"));
  document.addEventListener("drop", (event) => {
    event.preventDefault();
    document.body.classList.remove("drop-target");
    readFile(event.dataTransfer.files[0]);
  });
  $("follow").addEventListener("change", (event) => {
    view.follow = event.target.checked;
    scheduleRender();
  });
  $("keep").addEventListener("change", (event) => {
    view.keep = Math.max(1, Number(event.target.value) || 1);
    enforceMemory();
    scheduleRender();
  });
  $("filters").addEventListener("input", scheduleRender);
  $("update-view").addEventListener(
    "toggle",
    (event) => {
      const entry = data.updates.get(view.selected);
      if (!entry || event.target.tagName !== "DETAILS") return;
      const key = detailsKey(entry, event.target);
      if (event.target.open) view.open.add(key);
      else view.open.delete(key);
    },
    true,
  );
  $("filters").addEventListener("submit", (event) => event.preventDefault());
  $("compare").addEventListener("click", openCompare);
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    const step = { j: 1, ArrowDown: 1, k: -1, ArrowUp: -1 }[event.key];
    if (!step) return;
    const updates = visibleUpdates();
    if (!updates.length) return;
    event.preventDefault();
    const index = updates.findIndex((entry) => entry.key === view.selected);
    const next = Math.min(updates.length - 1, Math.max(0, (index < 0 ? updates.length : index) + step));
    select(updates[next].key);
  });

  poll();
  setInterval(poll, POLL_MS);
})();
