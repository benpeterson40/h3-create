/* H3 Create - Edit Video tab: tool list + one panel per tool. Uses the helpers from app.js ($, api, toast, uploadFile, ico, ...). */
"use strict";
Object.assign(ICONS, {
  magic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21v-6.5a3 3 0 0 1 6 0V21M8 21h8"/><circle cx="12" cy="8.5" r="2.6"/><path d="M4 4.5l.6 1.4L6 6.5l-1.4.6L4 8.5l-.6-1.4L2 6.5l1.4-.6zM19.5 3l.5 1.2 1.2.5-1.2.5-.5 1.3-.5-1.3-1.2-.5 1.2-.5zM20 10.5l.4.9.9.4-.9.4-.4.9-.4-.9-.9-.4.9-.4z"/></svg>',
  extend: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3" y="5" width="13" height="14" rx="2"/><path d="m8.5 9.5 3.5 2.5-3.5 2.5z" fill="currentColor"/><path d="M19 5.5v13M21.5 7.5v9" stroke-linecap="round"/></svg>',
  crop: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2.5V18h15.5M2.5 6H18v15.5"/></svg>',
  expand: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 4h6v6M20 4l-7 7M10 20H4v-6M4 20l7-7"/></svg>',
  checker: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M4 8h4V4M8 12h4V8M4 16h4v-4M8 20v-4h4M12 12v4h4" fill="currentColor"/><path d="M12 4v16" /></svg>',
  hdr: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="9" cy="12" r="4.5"/><circle cx="15" cy="12" r="4.5"/><path d="M12 2.5v1.5M12 20v1.5M4.5 5l1 1M18.5 18l1 1M2.5 12H4M20 12h1.5" stroke-dasharray="1 2"/></svg>',
  gauge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M4.3 17.5a8.5 8.5 0 1 1 15.4 0"/><path d="m12 13 3.5-3.5"/><circle cx="12" cy="13.5" r="1.4" fill="currentColor"/></svg>',
  depth: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3" y="9" width="12" height="12" rx="1.5"/><path d="M9 3h12v12h-3M9 3v3M15 3h0" stroke-dasharray="2 2"/></svg>',
  pencil: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="m13.5 6.5 4 4"/></svg>',
  cam: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="6" width="13" height="12" rx="2.5"/><path d="M17 10.5 21 8v8l-4-2.5z"/></svg>',
  pic: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"><rect x="3.5" y="4.5" width="17" height="15" rx="2"/><circle cx="9" cy="9.5" r="1.6" fill="currentColor"/><path d="m4 18 5.5-5 4 3.5 2.5-2 4 3.5"/></svg>',
  at: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="3.6"/><path d="M15.6 12v1.3a2.4 2.4 0 0 0 4.8 0V12a8.4 8.4 0 1 0-3.3 6.7"/></svg>',
  speaker: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/></svg>',
  mute: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9.5v5h3.5L12 18.5v-13L7.5 9.5z"/><path d="m16 9.5 5 5m0-5-5 5"/></svg>',
  bars: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="11" width="3.4" height="9" rx="1"/><rect x="8.6" y="6" width="3.4" height="14" rx="1"/><rect x="14.2" y="13" width="3.4" height="7" rx="1"/><rect x="19.4" y="4" width="2.6" height="16" rx="1"/></svg>',
  lines: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 7h10M4 12h7M4 17h5"/><path d="m17 11 .9 2.1 2.1.9-2.1.9L17 17l-.9-2.1-2.1-.9 2.1-.9z" fill="currentColor" stroke-width="1"/></svg>',
  draw: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 18c2.5-6 5-9.5 7-9.5 2.5 0-1 8 1.5 8 1.6 0 3.5-3 4.5-5"/><path d="m17 6 2.5 2.5L21 7l-2.5-2.5z"/></svg>',
});

const ETOOLS = [
  { id: "edit", title: "Edit Video", sub: "Make targeted changes", banner: "Make targeted changes with a prompt", icon: "magic", limit: 30, drop: "Add a video to edit", dropSub: "Up to 30s", model: "MiniMax H3", hero: "Describe what to change — a person, an object, the place, the weather — and H3 changes just that, keeping the motion, camera and everything else." },
  { id: "extend", title: "Extend Video", sub: "Generate a continuation", banner: "Generate a continuation", icon: "extend", limit: 30, min: 1, drop: "Add video to extend", dropSub: "Duration 1 - 30s", model: "MiniMax H3", hero: "H3 continues your clip from its last frame — or leads into its first one — with the same people, place and look." },
  { id: "reframe", title: "Reframe", sub: "Resize and reposition for any format", banner: "Resize and reposition for any format", icon: "crop", limit: 60, drop: "Add video to reframe", dropSub: "Duration up to 60s", model: "SAM 3 subject tracking", hero: "Turn a landscape video into 9:16, 1:1 or any shape. The crop follows your subject smoothly so nobody gets cut off." },
  { id: "rmbg", title: "Remove Background", sub: "Cut out the object", banner: "Remove background or make a mask from video", icon: "checker", isNew: true, limit: 75, maxMB: 200, drop: "Add a video", dropSub: "Up to 200 MB", model: "SAM 3", hero: "Keep the person — or any object you name — and drop everything else. Get a transparent video, a green-screen version and a matte." },
  { id: "hdr", title: "SDR to HDR", sub: "Convert SDR footage to HDR", banner: "Convert SDR footage to HDR", icon: "hdr", isNew: true, limit: 60, drop: "Add video to make it HDR", dropSub: "Up to 60s", model: "Inverse tone mapping", hero: "Expands highlights into real HDR (HDR10, BT.2020) while keeping skin and mid-tones natural." },
  { id: "depth", title: "Depth Map", sub: "Depth maps from media", banner: "Make depth map from video", icon: "depth", isNew: true, limit: 75, maxMB: 200, allowImage: true, drop: "Add a video to create a depth map", dropSub: "Up to 200 MB · a picture works too", model: "MoGe-2", hero: "A steady depth map for every frame — near is bright, far is dark — for compositing, relighting or 3D." },
];
const EOPTS = {
  edit: [{ key: "quality", label: "Quality", def: "standard", choices: [["standard", "768p"], ["fast", "480p", "about twice as fast"]] }],
  extend: [{ key: "direction", label: "Direction", def: "sequel", wide: true, choices: [["sequel", "Sequel", "continue after the last frame"], ["prequel", "Prequel", "lead into the first frame"]] },
    { key: "duration", label: "Duration", def: "5", choices: [["2", "2s"], ["5", "5s"], ["10", "10s", "two parts"]] },
    { key: "quality", label: "Quality", def: "standard", choices: [["standard", "768p"], ["fast", "480p", "about twice as fast"]] }],
  reframe: [{ key: "aspect", label: "Aspect ratio", def: "9:16", choices: [["9:16", "9:16", "Reels, Shorts, TikTok"], ["1:1", "1:1", "square"], ["4:5", "4:5", "feed"], ["16:9", "16:9", "widescreen"], ["4:3", "4:3"], ["3:4", "3:4"], ["21:9", "21:9", "cinema"]] },
    { key: "quality", label: "Quality", def: "720p", choices: [["480p", "480p"], ["720p", "720p"], ["1080p", "1080p"]] },
    { key: "subject", label: "Keep in frame", def: "person", text: true, wide: true }],
  rmbg: [],
  hdr: [{ key: "format", label: "Output format", def: "hdr10", wide: true, choices: [["hdr10", "H.265 10-bit", "HDR10 (PQ) — phones, TVs, YouTube"], ["hlg", "H.265 10-bit HLG", "broadcast HDR; also fine on normal screens"]] }],
  depth: [{ key: "resolution", label: "Resolution", def: "auto", choices: [["auto", "Auto", "same as the video"], ["720", "720p"], ["1080", "1080p"]] },
    { key: "maxframes", label: "Max frames", def: "1800", choices: [["300", "300"], ["600", "600"], ["1200", "1200"], ["1800", "1800"]] }],
};
const TILES = {
  rmbg: { key: "output", title: "Output format", def: "cutout", cols: 2, items: [["cutout", "Cutout video", "/static/img/tile_cutout.jpg"], ["mask", "Alpha mask", "/static/img/tile_mask.jpg"]] },
  depth: { key: "colormap", title: "Color map", def: "gray", cols: 3, items: [["gray", "Grayscale"], ["turbo", "Turbo"], ["inferno", "Inferno"], ["magma", "Magma"], ["viridis", "Viridis"]].map(x => [x[0], x[1], `/static/img/tile_depth_${x[0]}.jpg`]) },
};
const ES = { tool: null, media: {}, images: [], prompt: {}, params: {}, draw: false, sound: true };
function esave() { try { localStorage.setItem("h3c.edit", JSON.stringify({ ...ES, images: ES.images.filter(i => !i.uploading) })); } catch (e) { } }
function eload() { try { Object.assign(ES, JSON.parse(localStorage.getItem("h3c.edit") || "{}")); } catch (e) { } ES.images = (ES.images || []).filter(i => !i.uploading); }
const T = () => ETOOLS.find(t => t.id === ES.tool);
const P = () => (ES.params[ES.tool] = ES.params[ES.tool] || {});
const optVal = o => (P()[o.key] ?? o.def);
let bannerImg = "";

function editHero() {
  const t = T();
  $("#heroKicker").textContent = t ? "Edit video" : "Editing tools";
  $("#heroTitle").textContent = t ? t.title : "Modify, restyle, transform.";
  $("#heroSub").textContent = t ? t.hero : "Make targeted changes, extend a clip, reframe for any format, cut out the background, convert to HDR or get depth maps — all on this machine.";
  $("#heroSteps").innerHTML = "";
}

function renderEdit() {
  const el = $("#editPanel"); const t = T();
  if (!t) {
    el.innerHTML = `<section class="tool-banner" style="background-image:url('/static/img/banner_list.jpg')"><h2>Editing tools</h2><p>Modify, restyle, change angles, transform</p></section>
      <div class="tool-list">${ETOOLS.map(x => `<button class="tool-row" data-tool="${x.id}"><span class="ti">${ICONS[x.icon]}</span>
        <span class="tt"><b>${esc(x.title)}</b><span>${esc(x.sub)}</span></span>${x.isNew ? '<span class="badge-hot">NEW</span>' : ""}<span class="ico chev">${ICONS.chev}</span></button>`).join("")}</div><div class="panel-pad"></div>`;
    $(".gen-wrap").hidden = true; editHero(); return;
  }
  $(".gen-wrap").hidden = false;
  const m = ES.media[t.id];
  let html = `<section class="tool-banner" style="background-image:url('/static/img/banner_${t.id}.jpg')"><button class="tool-change" id="toolChange">${ico("pencil")}Change</button>
    <h2>${esc(t.title)}</h2><p>${esc(t.banner)}</p></section>`;
  if (t.id === "edit") html += `<div class="seg"><button class="seg-btn ${ES.draw ? "" : "active"}" data-draw="0">${ico("lines")}Prompt</button><button class="seg-btn ${ES.draw ? "active" : ""}" data-draw="1">${ico("draw")}Draw</button></div>`;
  html += mediaBox(t, m);
  if (t.id === "edit" || t.id === "extend") html += elementsBox();
  const opts = EOPTS[t.id] || [];
  const wide = opts.filter(o => o.wide && t.id === "extend"); const rest = opts.filter(o => !(o.wide && t.id === "extend"));
  if (wide.length) html += wide.map(optCard).join("");
  if (t.id === "edit" || t.id === "extend" || t.id === "rmbg") html += promptBox(t);
  if (TILES[t.id]) html += tilesBox(TILES[t.id]);
  if (rest.length) html += rest.length === 1 && rest[0].wide ? optCard(rest[0]) : `<div class="opt-grid">${rest.map(optCard).join("")}</div>`;
  if (t.id === "depth") html += `<label class="sbs-row"><span>Side by side <span class="info" title="Put the original next to the depth map in one video">i</span></span>
    <span class="toggle"><input type="checkbox" id="eSbs" ${String(P().sbs || "0") === "1" ? "checked" : ""}><span></span></span></label>`;
  if (t.id === "upscale") html += `<div class="note-card"><b>No prompt needed.</b> DLSS rebuilds detail from the video itself — Creativity sets how much new, natural-looking detail it adds.</div>`;
  if (t.id === "hdr") html += `<div class="note-card">HDR looks right on an HDR screen; on a normal screen it may look flat or dim.</div>`;
  if (t.id === "edit" || t.id === "extend") html += `<label class="set-row check card" style="padding:12px 16px"><span class="set-label">Check &amp; auto-fix<small>${esc(llmName)} reviews the result and retakes once if it's off</small></span>
    <span class="toggle"><input type="checkbox" id="eAudit" ${ES.audit === false ? "" : "checked"}><span></span></span></label>`;
  html += `<button class="model-card" id="modelCard"><span class="ml"><small>Model</small><b>${esc(t.model)} <span class="ico bars">${ICONS.bars}</span><span class="local">local</span></b></span><span class="ico">${ICONS.chev}</span></button><div class="panel-pad"></div>`;
  el.innerHTML = html; paintIcons(el); bindEdit(); editHero(); refreshEditGen();
}
function mediaBox(t, m) {
  if (!m) return `<section class="drop" id="eDrop"><span class="circle">${ico("cam")}</span><b>${esc(t.drop)}</b><small>${esc(t.dropSub)}</small>
    <input type="file" id="eFile" accept="${t.allowImage ? "video/*,image/*" : "video/*"}" hidden></section>`;
  const meta = m.kind === "video" ? `<span class="chip">${fmtDur(m.meta.duration)}</span><span class="chip">${m.meta.w}×${m.meta.h}</span>${m.meta.duration > t.limit + 0.05 ? `<span class="chip warn">uses the first ${t.limit} s</span>` : ""}`
    : `<span class="chip">${m.meta.w}×${m.meta.h} picture</span>`;
  if (t.id === "edit" && ES.draw && m.kind === "video") {
    return `<section class="drop filled" id="eDrop"><div class="paint-wrap" id="paintWrap"><img src="${esc(m.thumb)}" alt="" id="paintImg"><canvas id="paintCanvas"></canvas></div>
      <div class="paint-tools"><div class="pills" id="brushSize"><button data-b="14">S</button><button data-b="28" class="active">M</button><button data-b="56">L</button></div>
      <div class="pills" id="brushMode"><button data-m="paint" class="active">Paint</button><button data-m="erase">Erase</button></div><button class="btn" id="paintClear">Clear</button></div>
      <div class="paint-hint">Paint over what should change. Everything else stays exactly as filmed (works best with a steady camera).</div>
      <button class="rm-media" id="eRemove" title="Remove">${ico("x")}</button></section>`;
  }
  return `<section class="drop filled" id="eDrop">${m.kind === "video" ? `<video src="${esc(m.url)}" poster="${esc(m.thumb || "")}" muted loop playsinline autoplay></video>` : `<img class="still" src="${esc(m.url)}" alt="">`}
    <div class="meta"><span class="chip">${esc(m.name || "")}</span>${meta}</div><button class="rm-media" id="eRemove" title="Remove">${ico("x")}</button></section>`;
}
function elementsBox() {
  if (!ES.images.length) return `<section class="drop small" id="eElems"><span class="circles"><span class="circle">${ico("pic")}</span></span><b style="font-weight:500;color:var(--muted)">Add elements or references</b>
    <small><b style="font-size:13.5px;color:#c9cbd0">Up to 8 pictures</b> — people, objects, places</small><input type="file" id="eImgs" accept="image/*" multiple hidden></section>`;
  return `<section class="drop small filled" id="eElems"><div class="elem-grid">${ES.images.map((im, i) => `<div class="ref${im.uploading ? " uploading" : ""}" data-i="${i}"><img src="${esc(im.thumb || im.preview)}" alt=""><span class="tag">@image${i + 1}</span>${im.uploading ? "" : `<button class="rm" title="Remove">${ico("x")}</button>`}</div>`).join("")}
    ${ES.images.length < 8 ? `<button class="ref-add" id="eAddImg"><span class="plus">${ico("plus")}</span></button>` : ""}</div><input type="file" id="eImgs" accept="image/*" multiple hidden></section>`;
}
function tilesBox(g) {
  const cur = P()[g.key] ?? g.def;
  return `<section class="tiles-card"><h4>${esc(g.title)}</h4><div class="tiles ${g.cols === 3 ? "three" : ""}">${g.items.map(([v, label, img]) =>
    `<button class="tile ${cur === v ? "sel" : ""}" data-tile="${g.key}" data-v="${v}"><img src="${img}" alt=""><span>${esc(label)}</span></button>`).join("")}</div></section>`;
}
function promptBox(t) {
  if (t.id === "rmbg") return `<section class="eprompt"><small>Prompt</small><textarea id="ePrompt" style="min-height:84px" placeholder="Describe which elements should be removed or saved">${esc(ES.prompt.rmbg || "")}</textarea></section>`;
  const ph = t.id === "edit" ? (ES.draw ? "Describe what goes in the painted area — e.g. 'a golden retriever sitting' or 'the person from @image1'." : "Describe what to change in the video. Add reference images or elements using @…")
    : "Describe what happens next — e.g. \"she turns and walks away\" or \"the camera pulls back to reveal the city\".";
  return `<section class="eprompt"><small>Prompt</small><div style="position:relative"><textarea id="ePrompt" placeholder="${esc(ph)}">${esc(ES.prompt[t.id] || "")}</textarea><div class="mention-pop" id="eMention" hidden></div></div>
    <div class="chips"><button class="pchip" id="eAt">${ico("at")}Elements</button>${t.id === "edit" ? `<button class="pchip ${ES.sound ? "" : "off"}" id="eSound">${ico(ES.sound ? "speaker" : "mute")}${ES.sound ? "On" : "Off"}</button>` : ""}</div></section>`;
}
function optCard(o) {
  if (o.text) return `<label class="opt-card ${o.wide ? "wide" : ""}"><span class="ol"><small>${esc(o.label)}</small><input class="opt-text" data-key="${o.key}" value="${esc(optVal(o))}" maxlength="60"></span></label>`;
  const c = o.choices.find(x => x[0] === String(optVal(o))) || o.choices[0];
  return `<button class="opt-card ${o.wide ? "wide" : ""}" data-key="${o.key}"><span class="ol"><small>${esc(o.label)}</small><b>${esc(c[1])}</b></span><span class="ico">${ICONS.chev}</span></button>`;
}
function openMenu(anchor, o, onPick) {
  closeMenu(); const m = document.createElement("div"); m.className = "menu"; m.id = "optMenu";
  const cur = onPick ? String(o.cur ?? o.def) : String(optVal(o));
  m.innerHTML = o.choices.map(c => `<button data-v="${esc(c[0])}" class="${cur === c[0] ? "sel" : ""}">${esc(c[1])}${c[2] ? `<small>${esc(c[2])}</small>` : ""}</button>`).join("");
  document.body.appendChild(m); const r = anchor.getBoundingClientRect();
  m.style.left = Math.min(r.left, innerWidth - m.offsetWidth - 10) + "px"; m.style.top = (r.bottom + 6 + m.offsetHeight > innerHeight ? r.top - m.offsetHeight - 6 : r.bottom + 6) + "px";
  m.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    if (onPick) { onPick(b.dataset.v); closeMenu(); if (currentTab === "control") { mcsave(); renderControl(); } return; }
    P()[o.key] = b.dataset.v; esave(); closeMenu(); renderEdit();
  });
  setTimeout(() => document.addEventListener("mousedown", outsideMenu), 0);
}
function outsideMenu(e) { if (!e.target.closest("#optMenu")) closeMenu(); }
function closeMenu() { const m = $("#optMenu"); if (m) m.remove(); document.removeEventListener("mousedown", outsideMenu); }

function bindEdit() {
  const t = T();
  $("#toolChange").onclick = () => { ES.tool = null; esave(); renderEdit(); };
  $$("[data-draw]").forEach(b => b.onclick = () => { ES.draw = b.dataset.draw === "1"; esave(); renderEdit(); });
  const drop = $("#eDrop");
  if (!ES.media[t.id]) {
    drop.onclick = () => $("#eFile").click();
    $("#eFile").onchange = e => { setEditMedia(e.target.files[0]); e.target.value = ""; };
    dropZone(drop, files => setEditMedia(files[0]));
  } else {
    $("#eRemove").onclick = () => { delete ES.media[t.id]; ES.mask = null; esave(); renderEdit(); };
    if ($("#paintCanvas")) initPaint();
  }
  const el = $("#eElems");
  if (el) {
    const pick = () => $("#eImgs").click();
    if (!ES.images.length) el.onclick = pick; else { const a = $("#eAddImg"); if (a) a.onclick = pick; }
    $("#eImgs").onchange = e => { addEditImages(e.target.files); e.target.value = ""; };
    el.addEventListener("click", e => { const rm = e.target.closest(".rm"); if (rm) { e.stopPropagation(); ES.images.splice(+rm.closest(".ref").dataset.i, 1); esave(); renderEdit(); } });
    dropZone(el, files => addEditImages(files));
  }
  $$(".opt-card[data-key]").forEach(b => b.onclick = () => openMenu(b, (EOPTS[t.id] || []).find(o => o.key === b.dataset.key)));
  $$(".opt-text").forEach(inp => inp.oninput = () => { P()[inp.dataset.key] = inp.value; esave(); });
  const ta = $("#ePrompt");
  if (ta) {
    ta.oninput = () => { ES.prompt[t.id] = ta.value; esave(); eMention(); refreshEditGen(); };
    ta.onkeydown = e => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); editGenerate(); return; }
      if (!emention) return;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); emention.sel = (emention.sel + (e.key === "ArrowDown" ? 1 : -1) + emention.items.length) % emention.items.length; eMention(); }
      else if (e.key === "Enter" || e.key === "Tab") { e.preventDefault(); ePick(emention.sel); }
      else if (e.key === "Escape") { $("#eMention").hidden = true; emention = null; }
    };
    ta.onblur = () => setTimeout(() => { const p = $("#eMention"); if (p) p.hidden = true; emention = null; }, 150);
    const mp = $("#eMention"); if (mp) mp.onmousedown = e => { const b = e.target.closest("button"); if (b) { e.preventDefault(); ePick(+b.dataset.i); } };
    if ($("#eAt")) $("#eAt").onclick = () => { const p = ta.selectionStart ?? ta.value.length; ta.value = ta.value.slice(0, p) + "@" + ta.value.slice(p); ta.focus(); ta.setSelectionRange(p + 1, p + 1); ES.prompt[t.id] = ta.value; eMention(); };
  }
  $$("[data-tile]").forEach(b => b.onclick = () => { P()[b.dataset.tile] = b.dataset.v; esave(); renderEdit(); });
  const sbs = $("#eSbs"); if (sbs) sbs.onchange = () => { P().sbs = sbs.checked ? "1" : "0"; esave(); };
  const snd = $("#eSound"); if (snd) snd.onclick = () => { ES.sound = !ES.sound; esave(); renderEdit(); };
  const au = $("#eAudit"); if (au) au.onchange = () => { ES.audit = au.checked; esave(); refreshEditGen(); };
  $("#modelCard").onclick = () => openModal("Model", `<p class="how-note" style="margin-top:0"><b>${esc(t.model)}</b> runs locally on this machine's GPU.${t.id === "edit" || t.id === "extend" ? ` ${esc(llmName)} writes the detailed prompt from your video, pictures and text, and checks the result.` : ""}<br><br>No cloud service and no credits — the only cost is time on the GPU.</p>`);
}
async function setEditMedia(file) {
  const t = T(); if (!file) return;
  const isImg = file.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp)$/i.test(file.name);
  if (isImg && !t.allowImage) return toast("This tool needs a video.", true);
  if (t.maxMB && file.size > t.maxMB * 1e6) return toast(`That file is ${(file.size / 1e6).toFixed(0)} MB — this tool takes up to ${t.maxMB} MB.`, true);
  const d = $("#eDrop"); if (d) d.innerHTML = `<span class="spinner"></span><b>Uploading…</b>`;
  try { ES.media[t.id] = await uploadFile(file); ES.mask = null; esave(); }
  catch (e) { toast("Upload failed: " + e.message, true); }
  renderEdit();
}
async function addEditImages(files) {
  files = [...files].filter(f => f.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp)$/i.test(f.name));
  const room = 8 - ES.images.length; if (files.length > room) toast(`Up to 8 pictures — added the first ${room}.`);
  for (const f of files.slice(0, room)) {
    const tmp = { uploading: true, preview: URL.createObjectURL(f) }; ES.images.push(tmp); renderEdit();
    try { Object.assign(tmp, await uploadFile(f), { uploading: false }); } catch (e) { ES.images.splice(ES.images.indexOf(tmp), 1); toast("Picture upload failed: " + e.message, true); }
    esave(); renderEdit();
  }
}
let emention = null;
function eMention() {
  const ta = $("#ePrompt"), pop = $("#eMention"); if (!ta || !pop) return;
  const pos = ta.selectionStart, before = ta.value.slice(0, pos), m = before.match(/@([a-z0-9]*)$/i);
  if (!m) { pop.hidden = true; emention = null; return; }
  const items = []; const v = ES.media[ES.tool];
  if (v) items.push({ tag: "video1", label: "Your video", img: v.thumb });
  ES.images.forEach((im, i) => !im.uploading && items.push({ tag: `image${i + 1}`, label: `Picture ${i + 1}`, img: im.thumb }));
  const f = items.filter(it => it.tag.startsWith(m[1].toLowerCase())); if (!f.length) { pop.hidden = true; emention = null; return; }
  emention = { start: pos - m[0].length, end: pos, items: f, sel: Math.min(emention?.sel || 0, f.length - 1) };
  pop.innerHTML = f.map((it, i) => `<button data-i="${i}" class="${i === emention.sel ? "sel" : ""}">${it.img ? `<img src="${esc(it.img)}" alt="">` : ""}@${it.tag}<small>${esc(it.label)}</small></button>`).join("");
  pop.style.top = Math.min(ta.offsetHeight - 10, 14 + before.split("\n").length * 22.5) + "px"; pop.hidden = false;
}
function ePick(i) {
  const ta = $("#ePrompt"), it = emention.items[i], ins = "@" + it.tag + " ";
  ta.value = ta.value.slice(0, emention.start) + ins + ta.value.slice(emention.end); const p = emention.start + ins.length;
  ta.setSelectionRange(p, p); ta.focus(); $("#eMention").hidden = true; emention = null; ES.prompt[ES.tool] = ta.value; esave(); refreshEditGen();
}

/* ---- Draw mode: paint the area to change on the first frame */
let paint = null;
function initPaint() {
  const img = $("#paintImg"), cv = $("#paintCanvas");
  const setup = () => {
    cv.width = img.naturalWidth; cv.height = img.naturalHeight; const ctx = cv.getContext("2d");
    if (ES.maskData) { const im = new Image(); im.onload = () => ctx.drawImage(im, 0, 0, cv.width, cv.height); im.src = ES.maskData; }
    paint = { ctx, size: 28, mode: "paint", down: false };
    const pt = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height, cv.width / r.width]; };
    const dot = (x, y, s) => { ctx.globalCompositeOperation = paint.mode === "erase" ? "destination-out" : "source-over"; ctx.fillStyle = "rgba(217,255,46,0.55)";
      ctx.beginPath(); ctx.arc(x, y, paint.size * s / 2, 0, Math.PI * 2); ctx.fill(); };
    let last = null;
    cv.onpointerdown = e => { cv.setPointerCapture(e.pointerId); paint.down = true; const [x, y, s] = pt(e); dot(x, y, s); last = [x, y]; };
    cv.onpointermove = e => { if (!paint.down) return; const [x, y, s] = pt(e); const n = Math.ceil(Math.hypot(x - last[0], y - last[1]) / Math.max(2, paint.size * s / 4));
      for (let i = 1; i <= n; i++) dot(last[0] + (x - last[0]) * i / n, last[1] + (y - last[1]) * i / n, s); last = [x, y]; };
    cv.onpointerup = cv.onpointercancel = () => { if (!paint.down) return; paint.down = false; ES.maskData = cv.toDataURL("image/png"); ES.mask = null; esave(); refreshEditGen(); };
  };
  if (img.complete && img.naturalWidth) setup(); else img.onload = setup;
  $$("#brushSize button").forEach(b => b.onclick = () => { paint.size = +b.dataset.b; $$("#brushSize button").forEach(x => x.classList.toggle("active", x === b)); });
  $$("#brushMode button").forEach(b => b.onclick = () => { paint.mode = b.dataset.m; $$("#brushMode button").forEach(x => x.classList.toggle("active", x === b)); });
  $("#paintClear").onclick = () => { paint.ctx.clearRect(0, 0, cv.width, cv.height); ES.maskData = null; ES.mask = null; esave(); refreshEditGen(); };
}
async function uploadMask() {
  // the painted strokes -> a white-on-black PNG at the frame's size
  const cv = $("#paintCanvas"); const out = document.createElement("canvas"); out.width = cv.width; out.height = cv.height; const c = out.getContext("2d");
  c.fillStyle = "#000"; c.fillRect(0, 0, out.width, out.height);
  const src = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height); const dst = c.getImageData(0, 0, out.width, out.height);
  let n = 0; for (let i = 3; i < src.data.length; i += 4) if (src.data[i] > 20) { dst.data[i - 3] = dst.data[i - 2] = dst.data[i - 1] = 255; n++; }
  if (n < out.width * out.height * 0.002) throw new Error("Paint over what should change first.");
  c.putImageData(dst, 0, 0);
  const blob = await new Promise(r => out.toBlob(r, "image/png"));
  return (await uploadFile(new File([blob], "mask.png", { type: "image/png" }))).id;
}

/* ---- generate */
function editEstimate() {
  const t = T(), m = ES.media[t?.id]; if (!t) return 0;
  const dur = m && m.kind === "video" ? Math.min(t.limit, m.meta.duration) : 5; const p = P();
  const segs = Math.max(1, 1 + Math.ceil(Math.max(0, dur - 5.2) / 4.96)); const aud = ES.audit === false ? 0 : 1.2;
  const fast = (p.quality || "standard") === "fast";
  switch (t.id) {
    case "edit": return Math.round(1.5 + segs * ((fast ? 3.5 : 6.5) + aud));
    case "extend": { const s = +(p.duration || 5); return Math.round(1 + (s <= 2.5 ? 3 : s <= 5.3 ? (fast ? 3.5 : 6) : (fast ? 7 : 12)) + aud * (s > 5.3 ? 2 : 1)); }
    case "reframe": return Math.max(1, Math.round(1 + dur / 30));
    case "upscale": return Math.max(1, Math.round(1 + dur / 8 * ((p.resolution || "2k") === "4k" ? 2 : 1)));
    case "rmbg": return Math.max(1, Math.round(1 + dur / 10));
    case "hdr": return Math.max(1, Math.round(0.5 + dur / 15));
    case "fps": return Math.max(1, Math.round(0.7 + dur / 10));
    case "depth": return Math.max(1, Math.round(0.8 + dur / 15));
  }
  return 1;
}
function refreshEditGen() {
  if (currentTab !== "edit") return;
  const t = T(); if (!t) return;
  $("#cost").textContent = `~${editEstimate()} min`;
  const m = ES.media[t.id]; let why = "";
  if (!m) why = t.allowImage ? "Add a video or a picture to start" : "Add a video to start";
  else if (ES.images.some(i => i.uploading)) why = "Pictures are still uploading…";
  else if (t.id === "edit" && !(ES.prompt.edit || "").trim()) why = "Describe what to change";
  else if (t.id === "edit" && ES.draw && !ES.maskData) why = "Paint over what should change";
  else if (t.id === "extend" && m.meta.duration < 1) why = "The video must be at least 1 second long";
  $("#generate").disabled = !!why; $("#genWhy").textContent = why;
}
async function editGenerate() {
  const t = T(); if (!t || $("#generate").disabled) return;
  $("#generate").disabled = true;
  try {
    let mask = null;
    if (t.id === "edit" && ES.draw) mask = await uploadMask();
    const params = { ...P() }; (EOPTS[t.id] || []).forEach(o => params[o.key] = optVal(o));
    if (TILES[t.id]) params[TILES[t.id].key] = P()[TILES[t.id].key] ?? TILES[t.id].def;
    const body = { tool: t.id, video: ES.media[t.id].id, images: (t.id === "edit" || t.id === "extend") ? ES.images.map(i => i.id) : [], prompt: ["edit", "extend", "rmbg"].includes(t.id) ? (ES.prompt[t.id] || "") : "",
      params, audit: ES.audit !== false, keep_audio: t.id !== "edit" || ES.sound, mask };
    const j = await api("/api/tools", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    toast(`${t.title} queued — it runs in the background.`); focusJob = j.id; await refreshJobs(); $("#canvas").scrollTo({ top: $("#feed").offsetTop - 20, behavior: "smooth" });
  } catch (e) { toast(e.message, true); }
  finally { refreshEditGen(); }
}
async function editInit() {
  eload();
  try { const lib = await getLibrary(); const b = lib.find(x => x.id === "os5") || lib[0]; bannerImg = b ? b.thumb : ""; } catch (e) { }
  if (currentTab === "edit") renderEdit();
  $("#editPanel").addEventListener("click", e => { const r = e.target.closest(".tool-row"); if (r) { ES.tool = r.dataset.tool; esave(); renderEdit(); $("#editPanel").scrollTo({ top: 0 }); } });
}
editInit();
