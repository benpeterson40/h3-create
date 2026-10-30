/* H3 Create - Ratings view: watch the source and the result side by side, score each finished video 1-10 and leave a comment.
   Scores and comments are saved with each job (POST /api/jobs/<id>/rating). Uses the helpers from app.js ($, $$, esc, api, toast, ico, jobChips, auditBadge). */
"use strict";
const RSETS = [
  { id: "tests", label: "Test videos" }, { id: "people", label: "People swaps" }, { id: "testfiles", label: "Test files" }, { id: "challenge", label: "Challenge swaps" }, { id: "motion", label: "Motion transfers" },
  { id: "batch2", label: "Second batch" }, { id: "overnight", label: "Overnight batch" }, { id: "hard", label: "Harder batch" }, { id: "third", label: "Third batch" }, { id: "eighth", label: "Eighth batch" }, { id: "seventh", label: "Seventh batch" }, { id: "sixth", label: "Sixth batch" }, { id: "fifth", label: "Fifth batch" }, { id: "fourth", label: "Fourth batch" }, { id: "all", label: "All finished" },
];
const RV = { items: [], filter: "tests", hideRated: false, open: false, saving: {}, again: {}, timers: {}, typing: {} };
try { const s = JSON.parse(localStorage.getItem("h3c.ratings") || "{}"); if (RSETS.some(x => x.id === s.filter)) RV.filter = s.filter; RV.hideRated = !!s.hideRated; } catch (e) { }
const rvKeep = () => { try { localStorage.setItem("h3c.ratings", JSON.stringify({ filter: RV.filter, hideRated: RV.hideRated })); } catch (e) { } };
const rvClock = ts => new Date(ts * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
const rvNum = x => String(Math.round(x * 10) / 10);
const rvSetOrder = { people: -10, testfiles: -9, eighth: -8, seventh: -7, sixth: -6, fifth: -5, fourth: -4, third: -3, hard: -2, overnight: -1, challenge: 0, motion: 1, batch2: 2 };

function rvScore(it) { return it.rating && Number.isInteger(it.rating.score) ? it.rating.score : null; }
function rvTitle(it) {
  if (it.project && it.project.title) return it.project.title;
  if (it.prompt_on && it.prompt) return it.prompt.length > 90 ? it.prompt.slice(0, 88) + "…" : it.prompt;
  return `${it.tool_name || "Video"} · ${it.video_name || ""}`.trim();
}
function rvInSet(it, f) {
  if (f === "all") return true;
  if (f === "tests") return !!it.project;
  return !!it.project && it.project.set === f;
}
function rvSorted(list) {
  const tests = list.filter(x => x.project).sort((a, b) => (rvSetOrder[a.project.set] ?? 9) - (rvSetOrder[b.project.set] ?? 9) || (a.project.order || 0) - (b.project.order || 0) || (b.project.version || 1) - (a.project.version || 1));
  return tests.concat(list.filter(x => !x.project));   // the API already sends the rest newest first
}
function rvAvg(list) { const s = list.map(rvScore).filter(v => v != null); return s.length ? rvNum(s.reduce((a, b) => a + b, 0) / s.length) : "–"; }

function rvCard(it) {
  const portrait = it.canvas && it.canvas[1] > it.canvas[0];
  const ar = it.canvas ? `${it.canvas[0]} / ${it.canvas[1]}` : "16 / 9";
  const sc = rvScore(it), r = it.rating || {};
  const segs = (it.seg_scores || []).map((s, i) => `<span class="rv-seg ${s == null ? "" : s >= 10 ? "ok" : s >= 7 ? "warn" : "bad"}" title="Segment ${i + 1}">${s == null ? "–" : rvNum(s)}</span>`).join("");
  const len = it.frames ? `${rvNum(it.frames / 24)} s` : "";
  const buttons = Array.from({ length: 10 }, (_, k) => `<button type="button" class="rv-sc${sc === k + 1 ? " on" : ""}" data-v="${k + 1}" aria-pressed="${sc === k + 1}" aria-label="${k + 1} out of 10">${k + 1}</button>`).join("");
  return `<article class="rv-card" data-id="${esc(it.id)}">
    <header class="rv-card-head">
      ${it.project ? `<span class="rv-code">${esc(it.project.code)}${(it.project.version || 1) > 1 ? `<small>v${it.project.version}</small>` : ""}</span>` : ""}
      <div class="rv-t"><b>${esc(rvTitle(it))}</b>
        <div class="job-meta">${jobChips(it)}${len ? `<span class="chip">${len}</span>` : ""}${auditBadge(it)}<span class="when">${ago(it.created)}</span></div></div>
      <span class="rv-mine${sc ? "" : " none"}">${sc ? `Your score <b>${sc}</b>` : "Not rated"}</span>
    </header>
    <div class="rv-media${portrait ? " portrait" : ""}" style="--ar:${ar}">
      <figure><video class="rv-src" muted playsinline preload="none" src="${esc(it.source || "")}" poster="${esc(it.source_thumb || "")}"></video><figcaption>Source</figcaption></figure>
      <figure><video class="rv-res" controls playsinline preload="none" src="${esc(it.result || "")}" poster="${esc(it.poster || "")}"></video><figcaption>Result</figcaption></figure>
    </div>
    <div class="rv-body">
      <div class="rv-info">
        ${it.prompt_on && it.prompt ? `<p class="rv-prompt"><span class="rv-lbl">Prompt</span>${esc(it.prompt)}</p>` : `<p class="rv-prompt auto"><span class="rv-lbl">Prompt</span>Written by ${esc(llmName)}</p>`}
        ${it.image_thumbs && it.image_thumbs.length ? `<div><span class="rv-lbl">Pictures</span><div class="rv-pics">${it.image_thumbs.map(t => `<img src="${esc(t)}" alt="" loading="lazy">`).join("")}</div></div>` : ""}
        ${segs ? `<div><span class="rv-lbl">App check per segment</span><div class="rv-segs">${segs}</div></div>` : ""}
        ${it.project && it.project.note ? `<p class="rv-note"><span class="rv-lbl">Noted by Claude</span>${esc(it.project.note)}</p>` : ""}
      </div>
      <div class="rv-rate">
        <span class="rv-lbl" id="rvl-${esc(it.id)}">Your score</span>
        <div class="rv-scores" role="group" aria-labelledby="rvl-${esc(it.id)}">${buttons}</div>
        <div class="rv-scale" aria-hidden="true"><span>1 unusable</span><span>10 perfect</span></div>
        <label class="rv-lbl" for="rvc-${esc(it.id)}">Comment</label>
        <textarea id="rvc-${esc(it.id)}" rows="3" placeholder="What works, what looks wrong, what to fix">${esc(r.comment || "")}</textarea>
        <div class="rv-saved" aria-live="polite">${r.updated ? "Saved " + rvClock(r.updated) : ""}</div>
      </div>
    </div></article>`;
}

function rvSummary() {
  const tests = RV.items.filter(x => x.project), shown = RV.items.filter(x => rvInSet(x, RV.filter));
  const rated = shown.filter(x => rvScore(x) != null).length;
  const sets = RSETS.slice(1, 6).map(s => { const l = RV.items.filter(x => rvInSet(x, s.id)); return l.length ? `<span>${s.label} <b>${rvAvg(l)}</b> (${l.filter(x => rvScore(x) != null).length} of ${l.length})</span>` : ""; }).join("");
  $("#rvSum").innerHTML = `<div class="rv-stat"><b>${rated}<small>/ ${shown.length}</small></b><span>rated</span></div>
    <div class="rv-stat"><b>${rvAvg(shown)}</b><span>your average</span></div>${tests.length ? `<div class="rv-sets">${sets}</div>` : ""}`;
  $$("#rvFilters button").forEach(b => { const n = RV.items.filter(x => rvInSet(x, b.dataset.f)).length; $("i", b).textContent = n; b.hidden = !n && b.dataset.f !== "all"; });
}

function rvRender() {
  const list = rvSorted(RV.items.filter(x => rvInSet(x, RV.filter) && !(RV.hideRated && rvScore(x) != null)));
  $$("#rvList video").forEach(v => v.pause());
  $("#rvList").innerHTML = list.length ? list.map(rvCard).join("") :
    `<p class="rv-empty">${RV.hideRated ? "Everything here is rated. Turn off “Hide rated” to see your scores." : "No finished videos here yet."}</p>`;
  $$("#rvFilters button").forEach(b => b.classList.toggle("active", b.dataset.f === RV.filter));
  $("#rvHide").checked = RV.hideRated;
  $$("#rvList .rv-card").forEach(rvSync);
  rvSummary();
}

/* the source plays along with the result: the result is the master (its controls and sound); click the source to play or pause both */
function rvSync(card) {
  const src = $(".rv-src", card), res = $(".rv-res", card);
  if (!src || !res) return;
  const align = () => { if (Math.abs(src.currentTime - res.currentTime) > 0.12) { try { src.currentTime = res.currentTime; } catch (e) { } } };
  res.addEventListener("play", () => { align(); src.play().catch(() => { }); });
  res.addEventListener("pause", () => src.pause());
  res.addEventListener("seeked", align);
  res.addEventListener("timeupdate", align);
  res.addEventListener("ended", () => src.pause());
  src.addEventListener("click", () => res.paused ? res.play().catch(() => { }) : res.pause());
}

function rvPaintCard(id) {
  const it = RV.items.find(x => x.id === id), card = $(`#rvList .rv-card[data-id="${CSS.escape(id)}"]`);
  if (!it || !card) return;
  const sc = rvScore(it);
  $$(".rv-sc", card).forEach(b => { const on = Number(b.dataset.v) === sc; b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on)); });
  const m = $(".rv-mine", card); m.classList.toggle("none", !sc); m.innerHTML = sc ? `Your score <b>${sc}</b>` : "Not rated";
}
function rvSaved(id, text, err = false) {
  const el = $(`#rvList .rv-card[data-id="${CSS.escape(id)}"] .rv-saved`);
  if (el) { el.textContent = text; el.classList.toggle("err", err); }
}
async function rvSave(id) {
  if (RV.saving[id]) { RV.again[id] = true; return; }
  const it = RV.items.find(x => x.id === id); if (!it) return;
  RV.saving[id] = true; rvSaved(id, "Saving…");
  const r = it.rating || {};
  try {
    const j = await api(`/api/jobs/${encodeURIComponent(id)}/rating`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ score: Number.isInteger(r.score) ? r.score : null, comment: r.comment || "" }) });
    if (j.rating) { it.rating.updated = j.rating.updated; rvSaved(id, "Saved " + rvClock(j.rating.updated)); }
  } catch (e) { rvSaved(id, "Not saved: " + e.message + ". Change it again to retry.", true); }
  finally { RV.saving[id] = false; if (RV.again[id]) { RV.again[id] = false; rvSave(id); } }
}

async function rvLoad() {
  $("#rvList").innerHTML = `<p class="rv-empty">Loading your videos…</p>`;
  try { RV.items = await api("/api/ratings"); }
  catch (e) {
    $("#rvList").innerHTML = `<p class="rv-empty">${/not found/i.test(e.message) ? "Ratings turn on after the app restarts. It restarts as soon as the videos that are rendering now are finished." : "Couldn't load your videos: " + esc(e.message)}</p>`;
    return;
  }
  RV.items.forEach(it => { it.rating = it.rating || { score: null, comment: "", updated: 0 }; });
  rvRender();
}

function openRatings() {
  if (RV.open) return;
  RV.open = true; closeModal();
  $("#hero").hidden = true; $("#feed").hidden = true; $("#ratings").hidden = false;
  $$(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.open === "ratings"));
  history.replaceState(null, "", "#ratings"); $("#canvas").scrollTo({ top: 0 });
  rvLoad();
}
function closeRatings() {
  if (!RV.open) return;
  RV.open = false; $$("#rvList video").forEach(v => v.pause()); $("#rvList").innerHTML = "";
  $("#ratings").hidden = true; $("#hero").hidden = false; $("#feed").hidden = false;
  $$(".nav-btn").forEach(b => b.classList.remove("active"));
  history.replaceState(null, "", location.pathname + location.search);
}

(function bindRatings() {
  $("#ratings").innerHTML = `
    <div class="rv-head">
      <div class="rv-titles">
        <button class="rv-back" id="rvBack">${ico("chev")}Back to the studio</button>
        <div class="hero-kicker">Ratings</div>
        <h1>Score your videos</h1>
        <p>Watch the source and the result side by side, then give each video a score from 1 to 10 and a comment. Everything saves as you go.</p>
      </div>
      <div class="rv-sum" id="rvSum" aria-live="polite"></div>
    </div>
    <div class="rv-bar">
      <div class="rv-filters" id="rvFilters" role="group" aria-label="Show">${RSETS.map(s => `<button type="button" data-f="${s.id}">${s.label}<i></i></button>`).join("")}</div>
      <label class="rv-hide"><span class="toggle"><input type="checkbox" id="rvHide"><span></span></span>Hide rated</label>
    </div>
    <div class="rv-list" id="rvList"></div>`;
  $("#rvBack").addEventListener("click", closeRatings);
  $("#rvFilters").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; RV.filter = b.dataset.f; rvKeep(); rvRender(); });
  $("#rvHide").addEventListener("change", e => { RV.hideRated = e.target.checked; rvKeep(); rvRender(); });
  const list = $("#rvList");
  list.addEventListener("click", e => {
    const b = e.target.closest(".rv-sc"); if (!b) return;
    const id = b.closest(".rv-card").dataset.id, it = RV.items.find(x => x.id === id); if (!it) return;
    const v = Number(b.dataset.v); it.rating.score = it.rating.score === v ? null : v;
    rvPaintCard(id); rvSummary(); rvSave(id);
  });
  list.addEventListener("input", e => {
    const ta = e.target.closest("textarea"); if (!ta) return;
    const id = ta.closest(".rv-card").dataset.id, it = RV.items.find(x => x.id === id); if (!it) return;
    it.rating.comment = ta.value; RV.typing[id] = true; rvSaved(id, "Typing…");
    clearTimeout(RV.timers[id]); RV.timers[id] = setTimeout(() => { RV.typing[id] = false; rvSave(id); }, 900);
  });
  list.addEventListener("focusout", e => {
    const ta = e.target.closest("textarea"); if (!ta) return;
    const id = ta.closest(".rv-card").dataset.id;
    if (RV.typing[id]) { clearTimeout(RV.timers[id]); RV.typing[id] = false; rvSave(id); }
  });
  window.addEventListener("pagehide", () => Object.keys(RV.typing).forEach(id => { if (RV.typing[id]) { RV.typing[id] = false; rvSave(id); } }));
  if (location.hash === "#ratings") openRatings();
})();
