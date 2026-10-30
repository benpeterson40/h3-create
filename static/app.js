/* H3 Create - Create Video page (Motion transfer / Objects swap). Vanilla JS, no build step. */
"use strict";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const ICONS = {
  motion: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="8.5" stroke-dasharray="2.2 2.6"/><path d="M15.5 12a3.5 3.5 0 1 1-3.5-3.5"/></svg>',
  swap: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><rect x="3.5" y="8.5" width="12" height="12" rx="2.5"/><path d="M8.5 8.5V6a2.5 2.5 0 0 1 2.5-2.5h7A2.5 2.5 0 0 1 20.5 6v7a2.5 2.5 0 0 1-2.5 2.5h-2.5"/></svg>',
  upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V4m0 0-4 4m4-4 4 4"/><path d="M4 15v2.5A2.5 2.5 0 0 0 6.5 20h11a2.5 2.5 0 0 0 2.5-2.5V15"/></svg>',
  plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>',
  x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  chev: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>',
  spark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"><path d="M12 3.5 13.9 9l5.6 1.9-5.6 1.9L12 18.5l-1.9-5.7L4.5 10.9 10.1 9z"/><path d="M19 3v3m-1.5-1.5h3" stroke-linecap="round"/></svg>',
  spark4: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M13 2.5c.5 4.6 3.3 7.6 8.5 8.4-5.2.9-8 3.9-8.5 8.6-.5-4.7-3.3-7.7-8.5-8.6 5.2-.8 8-3.8 8.5-8.4Z"/><path d="M5.2 15.5c.2 1.6 1.1 2.6 2.8 2.9-1.7.3-2.6 1.3-2.8 2.9-.2-1.6-1.1-2.6-2.8-2.9 1.7-.3 2.6-1.3 2.8-2.9Z"/></svg>',
  folder: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M3.5 6.5A2 2 0 0 1 5.5 4.5h4l2 2.5h7a2 2 0 0 1 2 2v8.5a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/></svg>',
  aperture: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/><path d="M14.3 3.8 9.6 12M20.1 9.9h-9.4M17.9 18.1 13.2 10M9.7 20.2l4.7-8.2M3.9 14.1h9.4M6.1 5.9l4.7 8.1"/></svg>',
  wand: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="m4 20 11-11m-2-2 2 2M15 4v2m4 3h2m-3-5-1.4 1.4M20 14l-1.4-1.4"/><path d="M8.5 5.5v1.8m-.9-.9h1.8" stroke-width="1.4"/></svg>',
  book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="M12 6.5c-1.8-1.4-4.6-2-8-1.8v13c3.4-.2 6.2.4 8 1.8 1.8-1.4 4.6-2 8-1.8v-13c-3.4-.2-6.2.4-8 1.8Zm0 0v13"/></svg>',
  clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/></svg>',
  download: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11m0 0-4-4m4 4 4-4M5 19.5h14"/></svg>',
  retry: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3M19.5 4.5v4h-4"/></svg>',
  trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 7h14M10 7V5h4v2m-7 0 .8 12a1.5 1.5 0 0 0 1.5 1.4h5.4a1.5 1.5 0 0 0 1.5-1.4L17 7"/></svg>',
  info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8h.01"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>',
  alert: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4 2.8 19.5h18.4zM12 10v4.5M12 17.2h.01"/></svg>',
  stop: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6.5" y="6.5" width="11" height="11" rx="2"/></svg>',
  use: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12h12m0 0-4-4m4 4-4 4M20 5v14"/></svg>',
  star: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><path d="m12 3.8 2.5 5.1 5.6.8-4 3.9.9 5.6-5-2.6-5 2.6.9-5.6-4-3.9 5.6-.8z"/></svg>',
};
function paintIcons(root = document) { $$(".ico[data-ico]", root).forEach(el => { if (!el.firstChild) el.innerHTML = ICONS[el.dataset.ico] || ""; }); }
const ico = n => `<span class="ico">${ICONS[n] || ""}</span>`;

const MODES = {
  motion: {
    kicker: "Motion transfer", title: "One video. Any cast.",
    sub: "Upload any video, add pictures of your characters, objects or a new place, and we re-shoot it with them — same moves, same camera, same timing.",
    steps: ["Upload the video whose motion you want", "Add pictures: people, outfits, objects, a location", "Optional: tell it who replaces whom with @image1, @image2…", "Generate — Qwen3.8 writes the shot-by-shot prompt and checks the result"],
    placeholder: "Optional — e.g. '@image1 replaces the dancer, @image2 is the new location'. Turn the switch off and Qwen3.8 writes the whole prompt from your video and pictures.",
    hint: "Add pictures of the <b>new characters, objects or place</b>. Each one gets a tag — @image1, @image2 … — to use in the prompt.",
  },
  swap: {
    kicker: "Objects swap", title: "Change one thing. Keep the rest.",
    sub: "Swap a person, an object or the whole location in your video. Everything you don't mention stays exactly as filmed.",
    steps: ["Upload the video", "Add a picture of what should go in (optional for a location)", "Say what to change — 'replace the man with the cat from @image1'", "Generate — the result is checked against your request"],
    placeholder: "Describe what to change — e.g. 'replace the man with the cat from @image1' or 'change the location to mountains'. Mention attached images or videos using @…",
    hint: "Add a picture of <b>what goes into the video</b> — a person, an object or a place. Refer to it as @image1 in the prompt.",
  },
};

const state = { mode: "motion", video: null, images: [], promptOn: true, prompts: { motion: "", swap: "" }, style: "none", duration: "5s", quality: "standard", audit: true, keepAudio: true };
let currentTab = "create";
let styles = [], jobs = [], focusJob = null, llmName = "Qwen3.8", pollTimer = null, libraryCache = null;

function save() { try { localStorage.setItem("h3c.form", JSON.stringify(state)); } catch (e) { } }
function load() { try { Object.assign(state, JSON.parse(localStorage.getItem("h3c.form") || "{}")); } catch (e) { } }

async function api(path, opt = {}) {
  const r = await fetch(path, opt);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) { } throw new Error(typeof m === "string" ? m : JSON.stringify(m)); }
  return r.json();
}
function toast(msg, err = false) {
  const t = $("#toast"); t.textContent = msg; t.className = "toast" + (err ? " err" : ""); t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => t.hidden = true, err ? 6000 : 3200);
}
function ago(ts) {
  const s = Date.now() / 1000 - ts; if (s < 60) return "just now"; if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400) return Math.round(s / 3600) + " h ago"; return new Date(ts * 1000).toLocaleDateString();
}
const fmtDur = s => s >= 60 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}` : `${(+s).toFixed(1)} s`;

/* ------------------------------------------------------------------ mode, hero */
function setMode(m) {
  state.mode = m; $$(".seg-btn").forEach(b => b.classList.toggle("active", b.dataset.mode === m));
  const M = MODES[m];
  $("#heroKicker").textContent = M.kicker; $("#heroTitle").textContent = M.title; $("#heroSub").textContent = M.sub;
  $("#heroSteps").innerHTML = M.steps.map(s => `<li>${esc(s)}</li>`).join("");
  $("#prompt").placeholder = M.placeholder; $("#prompt").value = state.prompts[m] || "";
  $("#refsHint").innerHTML = M.hint; save(); refreshGen();
}

/* ------------------------------------------------------------------ source video */
function showVideo() {
  const v = state.video, vid = $("#srcVideo"), frame = $("#frame");
  $("#frameEmpty").hidden = !!v; vid.hidden = !v; $("#removeVideo").hidden = !v; $("#srcMeta").hidden = !v;
  frame.classList.remove("portrait", "square");
  if (!v) { vid.removeAttribute("src"); vid.load(); refreshGen(); return; }
  const r = v.meta.w / v.meta.h; if (r < 0.8) frame.classList.add("portrait"); else if (r < 1.3) frame.classList.add("square");
  if (vid.getAttribute("src") !== v.url) { vid.src = v.url; vid.poster = v.thumb || ""; }
  vid.play().catch(() => { });
  const long = v.meta.duration > 30.05;
  $("#srcMeta").innerHTML = `<span class="chip">${esc(v.name || "video")}</span><span class="chip">${fmtDur(v.meta.duration)}</span><span class="chip">${v.meta.w}×${v.meta.h}</span>` +
    (long && state.duration === "full" ? `<span class="chip warn">Full video uses the first 30 s</span>` : "");
  refreshGen();
}
async function uploadFile(file) {
  const fd = new FormData(); fd.append("file", file);
  return api("/api/upload", { method: "POST", body: fd });
}
async function setVideoFile(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|webm|mkv|m4v|avi)$/i.test(file.name)) return toast("That isn't a video file.", true);
  $("#frameEmpty .fe-title").textContent = "Uploading…";
  try { state.video = await uploadFile(file); save(); showVideo(); }
  catch (e) { toast("Upload failed: " + e.message, true); }
  finally { $("#frameEmpty .fe-title").textContent = "Upload a video"; }
}

/* ------------------------------------------------------------------ reference pictures */
function renderRefs() {
  const g = $("#refGrid"); $$(".ref", g).forEach(e => e.remove());
  state.images.forEach((im, i) => {
    const el = document.createElement("div"); el.className = "ref" + (im.uploading ? " uploading" : ""); el.draggable = !im.uploading; el.dataset.i = i;
    el.innerHTML = `<img src="${esc(im.thumb || im.preview)}" alt=""><span class="tag">@image${i + 1}</span>${im.uploading ? "" : `<button class="rm" title="Remove">${ico("x")}</button>`}`;
    g.appendChild(el);
  });
  $("#addRef").style.display = state.images.length >= 8 ? "none" : "";
  refreshGen();
}
async function addImages(files) {
  files = [...files].filter(f => f.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
  const room = 8 - state.images.length; if (files.length > room) toast(`Up to 8 pictures — added the first ${room}.`);
  for (const f of files.slice(0, room)) {
    const tmp = { uploading: true, preview: URL.createObjectURL(f) }; state.images.push(tmp); renderRefs();
    try { const rec = await uploadFile(f); Object.assign(tmp, rec, { uploading: false }); }
    catch (e) { state.images.splice(state.images.indexOf(tmp), 1); toast("Picture upload failed: " + e.message, true); }
    renderRefs(); save();
  }
}
let dragFrom = null;
function bindRefs() {
  const g = $("#refGrid");
  g.addEventListener("click", e => {
    const rm = e.target.closest(".rm"); if (rm) { state.images.splice(+rm.closest(".ref").dataset.i, 1); renderRefs(); save(); }
  });
  g.addEventListener("dragstart", e => { const r = e.target.closest(".ref"); if (!r) return; dragFrom = +r.dataset.i; r.classList.add("dragging"); e.dataTransfer.effectAllowed = "move"; });
  g.addEventListener("dragend", e => { dragFrom = null; $$(".ref", g).forEach(r => r.classList.remove("dragging")); });
  g.addEventListener("dragover", e => { if (dragFrom !== null) e.preventDefault(); });
  g.addEventListener("drop", e => {
    if (dragFrom === null) return; e.preventDefault(); e.stopPropagation();
    const r = e.target.closest(".ref"); const to = r ? +r.dataset.i : state.images.length - 1;
    const [m] = state.images.splice(dragFrom, 1); state.images.splice(to, 0, m); dragFrom = null; renderRefs(); save();
  });
  $("#addRef").addEventListener("click", () => $("#imageInput").click());
  $("#imageInput").addEventListener("change", e => { addImages(e.target.files); e.target.value = ""; });
  dropZone($("#refsCard"), files => addImages(files), () => dragFrom === null);
}
function dropZone(el, onFiles, ok = () => true) {
  el.addEventListener("dragover", e => { if (!ok() || !e.dataTransfer.types.includes("Files")) return; e.preventDefault(); el.classList.add("drag"); });
  el.addEventListener("dragleave", () => el.classList.remove("drag"));
  el.addEventListener("drop", e => { el.classList.remove("drag"); if (!ok() || !e.dataTransfer.files.length) return; e.preventDefault(); onFiles(e.dataTransfer.files); });
}

/* ------------------------------------------------------------------ prompt + @mentions */
let mention = null;
function mentionItems(q) {
  const items = [];
  if (state.video) items.push({ tag: "video1", label: "Your video", img: state.video.thumb });
  state.images.forEach((im, i) => !im.uploading && items.push({ tag: `image${i + 1}`, label: `Picture ${i + 1}`, img: im.thumb }));
  return items.filter(it => it.tag.startsWith(q.toLowerCase()));
}
function updateMention() {
  const ta = $("#prompt"), pos = ta.selectionStart, before = ta.value.slice(0, pos), m = before.match(/@([a-z0-9]*)$/i);
  const pop = $("#mentionPop");
  if (!m) { pop.hidden = true; mention = null; return; }
  const items = mentionItems(m[1]); if (!items.length) { pop.hidden = true; mention = null; return; }
  mention = { start: pos - m[0].length, end: pos, items, sel: Math.min(mention?.sel || 0, items.length - 1) };
  pop.innerHTML = items.map((it, i) => `<button data-i="${i}" class="${i === mention.sel ? "sel" : ""}">${it.img ? `<img src="${esc(it.img)}" alt="">` : ""}@${it.tag}<small>${esc(it.label)}</small></button>`).join("");
  const lines = before.split("\n").length; pop.style.top = Math.min(ta.offsetHeight - 10, 14 + lines * 22.5) + "px"; pop.hidden = false;
}
function pickMention(i) {
  const ta = $("#prompt"), it = mention.items[i]; const ins = "@" + it.tag + " ";
  ta.value = ta.value.slice(0, mention.start) + ins + ta.value.slice(mention.end);
  const p = mention.start + ins.length; ta.setSelectionRange(p, p); ta.focus(); $("#mentionPop").hidden = true; mention = null;
  state.prompts[state.mode] = ta.value; save(); refreshGen();
}
function bindPrompt() {
  const ta = $("#prompt");
  ta.addEventListener("input", () => { state.prompts[state.mode] = ta.value; save(); updateMention(); refreshGen(); });
  ta.addEventListener("click", updateMention);
  ta.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); generate(); return; }
    if (!mention) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); mention.sel = (mention.sel + (e.key === "ArrowDown" ? 1 : -1) + mention.items.length) % mention.items.length; updateMention(); }
    else if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); pickMention(mention.sel); }
    else if (e.key === "Escape") { $("#mentionPop").hidden = true; mention = null; }
  });
  ta.addEventListener("blur", () => setTimeout(() => { $("#mentionPop").hidden = true; mention = null; }, 150));
  $("#mentionPop").addEventListener("mousedown", e => { const b = e.target.closest("button"); if (b) { e.preventDefault(); pickMention(+b.dataset.i); } });
  $("#promptOn").addEventListener("change", e => { state.promptOn = e.target.checked; showPromptOn(); save(); refreshGen(); });
}
function showPromptOn() { $("#promptOn").checked = state.promptOn; $("#promptBody").hidden = !state.promptOn; $("#promptAuto").hidden = state.promptOn; }

/* ------------------------------------------------------------------ settings, cost, generate */
function bindSettings() {
  $$(".pills").forEach(p => p.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return; state[p.dataset.setting] = b.dataset.v; paintSettings(); save(); showVideo();
  }));
  $("#auditOn").addEventListener("change", e => { state.audit = e.target.checked; save(); refreshGen(); });
  $("#audioOn").addEventListener("change", e => { state.keepAudio = e.target.checked; save(); });
}
function paintSettings() {
  $$(".pills").forEach(p => $$("button", p).forEach(b => b.classList.toggle("active", b.dataset.v === state[p.dataset.setting])));
  $("#auditOn").checked = state.audit; $("#audioOn").checked = state.keepAudio;
}
function estimate() {
  const dur = state.video ? Math.min(30, state.video.meta.duration) : 5.2;
  const segs = state.duration === "full" && dur > 5.2 ? 1 + Math.ceil((dur - 5.2) / 4.96) : 1;
  const per = state.quality === "fast" ? 3.5 : 6.5;
  return Math.round(1.5 + segs * (per + (state.audit ? 1.2 : 0)));
}
function refreshGen() {
  if (currentTab !== "create") return;
  $("#cost").textContent = `~${estimate()} min`;
  let why = "";
  const uploading = state.images.some(i => i.uploading);
  if (!state.video) why = "Upload a video to start";
  else if (uploading) why = "Pictures are still uploading…";
  else if (state.mode === "motion" && !state.images.length) why = "Add at least one picture of the new character, object or place";
  else if (state.mode === "swap" && !state.images.length && !(state.promptOn && (state.prompts.swap || "").trim())) why = "Say what to change, or add a picture";
  else if (state.mode === "swap" && state.promptOn && !(state.prompts.swap || "").trim() && state.images.length) why = "";
  $("#generate").disabled = !!why; $("#genWhy").textContent = why;
}
async function generate() {
  if ($("#generate").disabled) return;
  const body = {
    mode: state.mode, video: state.video.id, images: state.images.map(i => i.id), prompt_on: state.promptOn, prompt: state.promptOn ? (state.prompts[state.mode] || "") : "",
    style: state.style, duration: state.duration, quality: state.quality, audit: state.audit, keep_audio: state.keepAudio,
  };
  $("#generate").disabled = true;
  try {
    const j = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    toast("Queued — you can keep working, it runs in the background."); focusJob = j.id; await refreshJobs(); $("#canvas").scrollTo({ top: $("#feed").offsetTop - 20, behavior: "smooth" });
  } catch (e) { toast(e.message, true); }
  finally { refreshGen(); }
}

/* ------------------------------------------------------------------ style picker */
const SWATCH = {
  none: "linear-gradient(135deg,#3a3b41,#1f2024)", photoreal: "linear-gradient(160deg,#9cc6e8,#4c6d8f 55%,#1f2a35)", anime: "linear-gradient(135deg,#ff9ad5,#7b8cff 60%,#5ad1ff)",
  "3d": "linear-gradient(135deg,#ffd27a,#ff8a5c 50%,#7b5cff)", clay: "linear-gradient(135deg,#e8b48a,#c46f4a 55%,#6b3b2a)", comic: "linear-gradient(135deg,#ffe14a,#ff4a4a 55%,#1a1a1a)",
  watercolor: "linear-gradient(135deg,#bfe6ff,#f6d6ff 50%,#fff2c2)", film: "linear-gradient(135deg,#d8a36b,#8a5a3b 55%,#2d221b)",
};
function paintStyle() {
  const s = styles.find(x => x.id === state.style); const a = $("#styleApply");
  if (s && s.id !== "none") { a.innerHTML = `${esc(s.name)} ${ico("chev")}`; a.classList.add("set"); }
  else { a.innerHTML = `Apply ${ico("chev")}`; a.classList.remove("set"); }
}
function openStyles() {
  openModal("Styles", `<div class="style-grid">${styles.map(s => `<button class="style-card ${s.id === state.style ? "sel" : ""}" data-id="${s.id}">
    <div class="style-swatch" style="background:${SWATCH[s.id] || SWATCH.none}"></div><b>${esc(s.name)}</b><span>${esc(s.text || "Keep the look of your video.")}</span></button>`).join("")}</div>
    <p class="how-note">A style re-renders everything — people, objects and the place — in that look, while the motion still comes from your video.</p>`);
  $("#modalBody").onclick = e => { const c = e.target.closest(".style-card"); if (!c) return; state.style = c.dataset.id; save(); paintStyle(); closeModal(); toast(state.style === "none" ? "Style cleared" : "Style applied"); };
}

/* ------------------------------------------------------------------ modal helpers */
function openModal(title, html, drawer = false) {
  $("#modalTitle").textContent = title; $("#modalBody").innerHTML = html; $("#modalBody").onclick = null;
  $("#modal").classList.toggle("drawer", drawer); $("#modal").hidden = false; $("#scrim").hidden = false; paintIcons($("#modal"));
}
function closeModal() { $("#modal").hidden = true; $("#scrim").hidden = true; $$("#modalBody video").forEach(v => v.pause()); $("#modal").dataset.job = ""; }

/* ------------------------------------------------------------------ motion library */
async function getLibrary() { if (!libraryCache) libraryCache = await api("/api/library"); return libraryCache; }
async function openLibrary() {
  openModal("Motion Library", `<div class="lib-sec">Loading…</div>`);
  let lib = [], recent = [];
  try { [lib, recent] = await Promise.all([getLibrary(), api("/api/uploads?kind=video&limit=12")]); } catch (e) { $("#modalBody").innerHTML = `<p>${esc(e.message)}</p>`; return; }
  const card = (it, kind) => `<button class="lib-item" data-kind="${kind}" data-id="${esc(it.id)}"><div class="lib-thumb"><img src="${esc(it.thumb)}" alt="" loading="lazy">
    <video muted loop playsinline preload="none" data-src="${esc(kind === "lib" ? it.video : it.url)}"></video></div>
    <div class="lib-cap"><span>${esc(it.title || it.name)}</span><small>${it.meta ? fmtDur(it.meta.duration) : ""}</small></div></button>`;
  $("#modalBody").innerHTML = (recent.length ? `<div class="lib-sec">Your uploads</div><div class="lib-grid">${recent.map(r => card(r, "up")).join("")}</div><div style="height:18px"></div>` : "") +
    `<div class="lib-sec">Library</div><div class="lib-grid">${lib.map(r => card(r, "lib")).join("")}</div>`;
  $$("#modalBody .lib-item").forEach(el => {
    const v = $("video", el);
    el.addEventListener("mouseenter", () => { if (!v.src) v.src = v.dataset.src; v.play().catch(() => { }); });
    el.addEventListener("mouseleave", () => v.pause());
  });
  $("#modalBody").onclick = async e => {
    const c = e.target.closest(".lib-item"); if (!c) return;
    try {
      const rec = c.dataset.kind === "lib" ? await api(`/api/library/${c.dataset.id}/use`, { method: "POST" }) : recent.find(r => r.id === c.dataset.id);
      if (currentTab === "control") { MC.motion = rec; mcsave(); renderControl(); }
      else if (currentTab === "edit" && T()) { ES.media[ES.tool] = rec; ES.maskData = null; esave(); renderEdit(); }
      else { state.video = rec; save(); showVideo(); }
      closeModal(); toast("Video set");
    } catch (err) { toast(err.message, true); }
  };
}

/* ------------------------------------------------------------------ how it works */
function openHow() {
  openModal("How it works", `<div class="how">
    <div class="how-step"><div class="n">STEP 1</div><h3>Your video drives the motion</h3><p>Every step, gesture, camera move and cut comes from the video you upload. Up to 30 s; the first 5 s by default.</p></div>
    <div class="how-step"><div class="n">STEP 2</div><h3>Pictures define the look</h3><p>Add pictures of people, outfits, objects or a place. They're tagged @image1, @image2 … so you can say exactly who replaces whom.</p></div>
    <div class="how-step"><div class="n">STEP 3</div><h3>${esc(llmName)} writes the prompt</h3><p>The vision model watches your video and pictures and writes a shot-by-shot edit plan, segment by segment.</p></div>
    <div class="how-step"><div class="n">STEP 4</div><h3>Rendered, then checked</h3><p>MiniMax H3 re-shoots each ≤5 s segment. ${esc(llmName)} compares it with your video and pictures and retakes once if something's off.</p></div>
  </div>
  <p class="how-note"><b>Motion transfer</b> replaces the cast (and the place, if you add a location picture). <b>Objects swap</b> changes only what you ask for — a person, an object or the location — and keeps everything else as filmed.<br><br>
  Longer videos are made in segments of 5 seconds or less; each new segment continues from the last frames of the previous one, so the joins are seamless. Everything runs on this machine's GPU.</p>`);
}

/* ------------------------------------------------------------------ jobs feed */
function statusText(j) {
  if (j.status === "queued") return j.queue_pos ? `Waiting in line (#${j.queue_pos})` : "Waiting in line";
  if (j.status === "running") return j.stage || "Working";
  if (j.status === "failed") return "Failed — " + (j.error || "");
  if (j.status === "cancelled") return "Cancelled";
  return "";
}
function auditBadge(j) {
  if (j.audit_score == null) return "";
  return j.audit_pass ? `<span class="audit ok" title="Checked by ${esc(llmName)}">${ico("check")}Checked ${j.audit_score}/10</span>`
    : `<span class="audit warn" title="${esc(llmName)} found issues — open Details">${ico("alert")}${j.audit_score}/10</span>`;
}
function jobChips(j) {
  const P = j.params || {};
  if (!j.tool || j.tool === "create")
    return `<span class="chip tool">${j.mode === "swap" ? "Objects swap" : "Motion transfer"}</span><span class="chip">${j.duration === "full" ? "Full video" : "5 s"} · ${j.quality === "fast" ? "480p" : "768p"}</span>`;
  if (j.tool === "improve") return `<span class="chip tool">Improved</span><span class="chip">${j.mode === "swap" ? "Objects swap" : "Motion transfer"} · ${j.duration === "full" ? "full video" : "5 s"}</span>`;
  const extra = { edit: j.quality === "fast" ? "480p" : "768p", extend: `${P.direction === "prequel" ? "Prequel" : "Sequel"} · ${P.duration || 5}s`, reframe: `${P.aspect || "9:16"} · ${P.quality || "720p"}`,
    upscale: `${(P.resolution || "2k").toUpperCase()} · ${P.creativity || "precise"}`, rmbg: P.output === "mask" ? "alpha mask" : "cutout video",
    control: `${j.scene_used || P.scene || "auto"} scene · ${j.duration === "full" ? "full video" : "5 s"} · ${j.quality === "fast" ? "480p" : "768p"}`,
    hdr: P.format === "hlg" ? "H.265 10-bit HLG" : "H.265 10-bit HDR10", fps: `${P.fps || 60} FPS`, depth: `${P.colormap || "gray"}${P.sbs === "1" ? " · side by side" : ""}` }[j.tool] || "";
  return `<span class="chip tool">${esc(j.tool_name || j.tool)}</span>${extra ? `<span class="chip">${esc(extra)}</span>` : ""}`;
}
function jobCard(j) {
  const portrait = j.canvas && j.canvas[1] > j.canvas[0];
  let media;
  if (j.status === "done" && j.result) media = j.result_kind === "image" ? `<img src="${esc(j.result)}" alt="">` : `<video src="${esc(j.result)}" poster="${esc(j.poster || "")}" controls loop playsinline preload="metadata"></video>`;
  else {
    const busy = j.status === "running" || j.status === "queued";
    media = `<img class="dim" src="${esc(j.source_thumb || "")}" alt=""><div class="job-over">
      <div class="job-stage ${j.status === "failed" ? "st-failed" : j.status === "cancelled" ? "st-cancelled" : ""}">${busy ? '<span class="spinner"></span>' : ""}${esc(statusText(j))}</div>
      ${busy ? `<div class="bar"><i style="width:${Math.max(2, Math.round((j.progress || 0) * 100))}%"></i></div>` : ""}</div>`;
  }
  const busy = j.status === "running" || j.status === "queued";
  const prompt = j.prompt_on && j.prompt ? `<div class="job-prompt">${esc(j.prompt)}</div>` : `<div class="job-prompt auto">Auto prompt by ${esc(llmName)}</div>`;
  return `<article class="job ${j.id === focusJob ? "focus" : ""}" data-id="${esc(j.id)}">
    <div class="job-media ${portrait ? "portrait" : ""}">${media}</div>
    <div class="job-body">
      <div class="job-meta">${jobChips(j)}
        ${j.style && j.style !== "none" ? `<span class="chip">${esc((styles.find(s => s.id === j.style) || {}).name || j.style)}</span>` : ""}
        ${auditBadge(j)}<span class="when">${ago(j.created)}</span></div>
      ${["create", "edit", "extend", "control"].includes(j.tool) ? prompt : j.tool === "rmbg" && j.prompt ? `<div class="job-prompt">${esc(j.prompt)}</div>` : ""}
      ${j.image_thumbs && j.image_thumbs.length ? `<div class="job-refs">${j.image_thumbs.map(t => `<img src="${esc(t)}" alt="">`).join("")}</div>` : ""}
      ${j.note ? `<div class="job-note">${esc(j.note)}</div>` : ""}
      ${j.extras && j.extras.length && j.status === "done" ? `<div class="job-extras">${j.extras.map(x => `<a class="btn" href="${esc(x.url)}" download>${ico("download")}${esc(x.label)}</a>`).join("")}</div>` : ""}
      <div class="job-actions">
        ${j.status === "done" ? `<a class="btn primary" href="${esc(j.result)}" download>${ico("download")}Download</a>` : ""}
        <button class="btn" data-act="details">${ico("info")}Details</button>
        ${busy ? `<button class="btn" data-act="cancel">${ico("stop")}Cancel</button>` : `<button class="btn" data-act="retry">${ico("retry")}Again</button>`}
        ${j.status === "done" && ["create", "edit", "control", "improve"].includes(j.tool) && j.audit_score != null && j.audit_score < 10 ? `<button class="btn" data-act="improve" title="${esc(llmName)} re-checks every shot and re-renders only the flawed ones">${ico("spark")}Improve</button>` : ""}
        ${j.status === "done" ? `<button class="btn" data-act="use">${ico("use")}Use as source</button>` : ""}
        ${busy ? "" : `<button class="btn danger" data-act="remove" title="Remove from history">${ico("trash")}</button>`}
      </div>
    </div></article>`;
}
function renderFeed() {
  const feed = $("#feed"); const show = jobs.slice(0, 12);
  $("#hero").classList.toggle("compact", jobs.length > 0);
  if (!show.length) { feed.innerHTML = ""; return; }
  // keep playing videos alive: only replace cards whose content changed
  const html = `<div class="feed-title"><b>Your videos</b>${jobs.length > 12 ? `<span>latest 12 of ${jobs.length} — see History</span>` : ""}</div>` + show.map(jobCard).join("");
  if (feed.dataset.sig === html) return; feed.dataset.sig = html;
  const old = new Map($$(".job", feed).map(el => [el.dataset.id, el]));
  const tmp = document.createElement("div"); tmp.innerHTML = html;
  [...tmp.children].forEach(nu => {
    const id = nu.dataset?.id, o = id && old.get(id);
    if (o && o.outerHTML === nu.outerHTML) tmp.replaceChild(o, nu);
  });
  feed.replaceChildren(...tmp.children);
}
async function refreshJobs() {
  try { jobs = await api("/api/jobs"); } catch (e) { return; }
  renderFeed();
  const active = jobs.some(j => j.status === "queued" || j.status === "running");
  clearTimeout(pollTimer); pollTimer = setTimeout(refreshJobs, active ? 2000 : 15000);
  const open = $("#modal").dataset.job; if (open && !$("#modal").hidden) refreshDetail(open, true);
}
async function jobAction(id, act) {
  try {
    if (act === "cancel") { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); toast("Cancelling…"); }
    else if (act === "retry") { const j = await api(`/api/jobs/${id}/retry`, { method: "POST" }); focusJob = j.id; toast("Queued again with a new seed"); }
    else if (act === "improve") { const j = await api(`/api/jobs/${id}/improve`, { method: "POST" }); focusJob = j.id; toast(`${llmName} will re-check every shot and repair the flawed ones`); }
    else if (act === "remove") { if (!confirm("Remove this video from your history? (It's moved to the app's trash folder.)")) return; await api(`/api/jobs/${id}`, { method: "DELETE" }); }
    else if (act === "details") { openDetail(id); return; }
    else if (act === "use") {
      const j = jobs.find(x => x.id === id); const b = await (await fetch(j.result)).blob();
      const ext = j.result_kind === "image" ? "png" : "mp4";
      const file = new File([b], `result-${id}.${ext}`, { type: j.result_kind === "image" ? "image/png" : "video/mp4" });
      if (currentTab === "edit" && typeof T === "function" && T()) {   // feed the open tool (e.g. upscale an edit)
        toast("Loading the result into " + T().title + "…"); ES.media[ES.tool] = await uploadFile(file); ES.maskData = null; esave(); renderEdit();
      } else { await setVideoFile(file); }
      $$(".panel-scroll").forEach(p => p.scrollTo({ top: 0, behavior: "smooth" })); return;
    }
    await refreshJobs();
  } catch (e) { toast(e.message, true); }
}

/* ------------------------------------------------------------------ job details drawer */
async function openDetail(id) { openModal("Details", `<div class="detail">Loading…</div>`, true); $("#modal").dataset.job = id; refreshDetail(id); }
async function refreshDetail(id, quiet = false) {
  let j; try { j = await api(`/api/jobs/${id}`); } catch (e) { return; }
  if ($("#modal").dataset.job !== id) return;
  const sig = JSON.stringify([j.status, j.stage, j.segments, j.log?.length]); if (quiet && $("#modalBody").dataset.sig === sig) return; $("#modalBody").dataset.sig = sig;
  const segs = (j.segments || []).map(s => {
    const att = s.attempts.map(a => {
      const au = a.audit || {};
      const issues = (au.issues || []).map(x => `<div class="issue"><b class="${esc(x.severity)}">${esc(x.severity)}</b>${x.time != null ? `@${esc(x.time)} s · ` : ""}${esc(x.what)}</div>`).join("");
      return `<h3>Segment ${s.k + 1}${j.segments.length > 1 ? ` (${(s.start / 24).toFixed(1)}–${((s.start + s.length) / 24).toFixed(1)} s)` : ""} · ${a.attempt >= 10 ? `repair ${a.attempt - 10}${a.repair ? " (" + a.repair.join(", ") + ")" : ""}` : `take ${a.attempt + 1}`}${s.chosen === a.attempt ? " · used" : ""}</h3>
        ${a.video ? `<video src="${esc(a.video)}" controls muted loop playsinline preload="metadata"></video>` : ""}
        ${a.audit ? (au.error ? `<p class="issue">Check failed: ${esc(au.error)}</p>` : `<p>${au.pass ? `<span class="audit ok">${ico("check")}Passed ${au.score}/10</span>` : `<span class="audit warn">${ico("alert")}${au.score}/10</span>`}</p>
          ${issues}${au.fix ? `<p class="issue"><b class="minor">fix</b>${esc(au.fix)}</p>` : ""}`) : ""}
        <details><summary style="cursor:pointer;color:var(--muted);font-size:13px;margin-top:8px">Prompt sent to H3 (seed ${a.seed})</summary><pre>${esc(a.prompt)}</pre></details>`;
    }).join("");
    return att;
  }).join("");
  $("#modalBody").innerHTML = `<div class="detail">
    <div class="job-meta"><span class="chip">${j.mode === "swap" ? "Objects swap" : "Motion transfer"}</span><span class="chip">${esc(j.status)}</span>
      ${j.seconds ? `<span class="chip">took ${Math.round(j.seconds / 60)} min</span>` : ""}${j.spec_source ? `<span class="chip">prompt: ${esc(j.spec_source)}</span>` : ""}</div>
    ${j.status === "running" || j.status === "queued" ? `<p style="margin-top:12px">${esc(statusText(j))}</p><div class="bar"><i style="width:${Math.round((j.progress || 0) * 100)}%"></i></div>` : ""}
    ${j.error ? `<p class="st-failed">${esc(j.error)}</p>` : ""}
    ${j.result ? `<h3>Result</h3><video src="${esc(j.result)}" controls loop playsinline preload="metadata"></video>` : ""}
    ${j.first_frame ? `<h3>First frame (placed by Qwen-Image)</h3><img src="${esc(j.first_frame)}" alt="" style="width:100%;border-radius:12px">` : ""}
    ${j.prompt_on && j.prompt ? `<h3>Your prompt</h3><pre>${esc(j.prompt)}</pre>` : ""}
    ${segs}
    <h3>Log</h3><pre>${esc((j.log || []).join("\n") || "—")}</pre></div>`;
  paintIcons($("#modalBody"));
}

/* ------------------------------------------------------------------ history drawer */
function openHistory() {
  openModal("History", jobs.length ? jobs.map(j => `<button class="hist-row" data-id="${esc(j.id)}"><img src="${esc(j.poster || j.source_thumb || "")}" alt="">
    <span><div class="t">${j.mode === "swap" ? "Objects swap" : "Motion transfer"} · ${j.duration === "full" ? "full video" : "5 s"}</div>
    <div class="s">${esc(j.status === "done" ? "Done" : statusText(j))} · ${ago(j.created)}</div></span></button>`).join("") : `<p style="color:var(--muted)">Nothing yet — your videos will show up here.</p>`, true);
  $("#modalBody").onclick = e => { const r = e.target.closest(".hist-row"); if (r) openDetail(r.dataset.id); };
}

/* ------------------------------------------------------------------ hero cards, status */
async function heroCards() {
  try {
    const lib = await getLibrary(); const pick = [lib.find(x => x.id === "mt12"), lib.find(x => x.id === "os1"), lib.find(x => x.id === "mt4")].filter(Boolean);
    $("#heroCards").innerHTML = pick.map(it => `<div class="hero-card"><img src="${esc(it.thumb)}" alt=""><span class="lbl">${esc(it.title)}</span></div>`).join("");
  } catch (e) { }
}
async function status() {
  try {
    const s = await api("/api/status"); llmName = s.llm.name.replace("-27B", ""); $("#llmName").textContent = s.llm.name;
    const d = $("#statusDot"); d.className = "status-dot " + (s.comfy.ok ? "ok" : "bad");
    d.title = `Video engine: ${s.comfy.ok ? "ready" : "offline"} · ${s.llm.name}: ${s.llm.available ? "installed" : "not installed (template prompts)"}`;
  } catch (e) { $("#statusDot").className = "status-dot bad"; }
  setTimeout(status, 20000);
}

function showTab(tab) {
  currentTab = tab; try { localStorage.setItem("h3c.tab", tab); } catch (e) { }
  $$(".tab").forEach(x => x.classList.toggle("active", x.dataset.tab === tab));
  $("#createPanel").hidden = tab !== "create"; $("#editPanel").hidden = tab !== "edit"; $("#controlPanel").hidden = tab !== "control"; $("#soonPanel").hidden = true;
  $(".gen-wrap").hidden = false;
  if (tab === "create") { setMode(state.mode); refreshGen(); }
  else if (tab === "edit") renderEdit();
  else renderControl();
}

/* ------------------------------------------------------------------ boot */
async function boot() {
  load(); paintIcons();
  $$(".seg-btn").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $$(".tab").forEach(t => t.addEventListener("click", () => showTab(t.dataset.tab)));
  $("#uploadCard").addEventListener("click", e => { if (e.target.closest("#removeVideo, #openLibraryFromCard")) return; if (!state.video) $("#videoInput").click(); });
  $("#uploadCard").addEventListener("keydown", e => { if ((e.key === "Enter" || e.key === " ") && !state.video) { e.preventDefault(); $("#videoInput").click(); } });
  $("#videoInput").addEventListener("change", e => { setVideoFile(e.target.files[0]); e.target.value = ""; });
  $("#removeVideo").addEventListener("click", () => { state.video = null; save(); showVideo(); });
  $("#openLibraryFromCard").addEventListener("click", openLibrary);
  dropZone($("#uploadCard"), files => setVideoFile(files[0]));
  bindRefs(); bindPrompt(); bindSettings();
  $("#styleRow").addEventListener("click", openStyles);
  $$(".nav-btn").forEach(b => b.addEventListener("click", () => ({ history: openHistory, library: openLibrary, styles: openStyles, how: openHow, ratings: () => openRatings() })[b.dataset.open]()));
  $("#generate").addEventListener("click", () => currentTab === "edit" ? editGenerate() : currentTab === "control" ? controlGenerate() : generate());
  $("#modalClose").addEventListener("click", closeModal); $("#scrim").addEventListener("click", closeModal);
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("#modal").hidden) closeModal(); });
  $("#feed").addEventListener("click", e => { const b = e.target.closest("[data-act]"); if (b) jobAction(b.closest(".job").dataset.id, b.dataset.act); });
  state.images = state.images.filter(i => !i.uploading);
  setMode(state.mode); showPromptOn(); paintSettings(); showVideo(); renderRefs();
  try { styles = await api("/api/styles"); } catch (e) { } paintStyle();
  status(); refreshJobs(); heroCards();
  let tab = "create"; try { tab = localStorage.getItem("h3c.tab") || "create"; } catch (e) { }
  const hash = location.hash.slice(1).split("/");   // deep links: #create, #edit, #edit/upscale ...
  if (["create", "edit", "control"].includes(hash[0])) { tab = hash[0]; if (tab === "edit") ES.tool = ETOOLS.some(t => t.id === hash[1]) ? hash[1] : null; }
  if (tab !== "create") showTab(tab);
}
boot();
