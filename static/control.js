/* H3 Create - Motion Control tab: a motion video + a character picture -> the character performs the motion (local MiniMax H3). */
"use strict";
const MC = { motion: null, character: null, quality: "standard", sceneOn: true, scene: "image", adv: false, length: "full", anchor: true, audit: true,
             keepAudio: true, prompt: "", seed: "" };
function mcsave() { try { localStorage.setItem("h3c.control", JSON.stringify(MC)); } catch (e) { } }
function mcload() { try { Object.assign(MC, JSON.parse(localStorage.getItem("h3c.control") || "{}")); } catch (e) { } if (MC.character && MC.character.uploading) MC.character = null; }

function controlHero() {
  $("#heroKicker").textContent = "Motion control";
  $("#heroTitle").textContent = "Your character. Any moves.";
  $("#heroSub").textContent = "Add a video with the motion you want and a picture of your character — H3 makes your character perform it, step for step, with the same camera and timing.";
  $("#heroSteps").innerHTML = "";
}

function renderControl() {
  const el = $("#controlPanel"); const m = MC.motion, c = MC.character;
  const motionBox = m
    ? `<div class="drop pair filled" id="mcMotion"><video src="${esc(m.url)}" poster="${esc(m.thumb || "")}" muted loop playsinline autoplay></video>
        <div class="meta"><span class="chip">${fmtDur(m.meta.duration)}</span>${m.meta.duration > 30.05 ? '<span class="chip warn">first 30 s</span>' : ""}</div>
        <button class="rm-media" data-rm="motion" title="Remove">${ico("x")}</button></div>`
    : `<div class="drop pair" id="mcMotion"><span class="circle">${ico("cam")}</span><b>Add motion to copy</b><small>Video duration:<br>3–30 seconds</small>
        <button class="link" id="mcLib">or pick from the library</button><input type="file" id="mcMotionFile" accept="video/*" hidden></div>`;
  const charBox = c
    ? `<div class="drop pair filled" id="mcChar">${c.uploading ? '<span class="spinner"></span>' : `<img class="still" src="${esc(c.thumb || c.url)}" alt="">`}
        <div class="meta"><span class="chip">${c.meta ? `${c.meta.w}×${c.meta.h}` : "uploading…"}</span></div>
        ${c.uploading ? "" : `<button class="rm-media" data-rm="character" title="Remove">${ico("x")}</button>`}</div>`
    : `<div class="drop pair" id="mcChar"><span class="circle">${ico("plus")}</span><b>Add your character</b><small>Image with visible<br>face and body</small>
        <input type="file" id="mcCharFile" accept="image/*" hidden></div>`;
  el.innerHTML = `
    <section class="tool-banner" style="background-image:url('/static/img/banner_control.jpg')"><button class="tool-change" id="mcHow">${ico("book")}How it works</button>
      <h2>Motion Control</h2><p>Control motion with video references</p></section>
    <section class="pair-card">${motionBox}${charBox}</section>
    <button class="model-card" id="mcModel"><span class="ml"><small>Model</small><b>MiniMax H3 Motion Control <span class="ico bars">${ICONS.bars}</span><span class="local">local</span></b></span><span class="ico">${ICONS.chev}</span></button>
    <button class="opt-card" id="mcQuality"><span class="ol"><small>Quality</small><b>${MC.quality === "fast" ? "480p" : "768p"}</b></span><span class="ico">${ICONS.chev}</span></button>
    <section class="scene-card">
      <div class="scene-head"><b>Scene control mode</b><label class="toggle green"><input type="checkbox" id="mcSceneOn" ${MC.sceneOn ? "checked" : ""}><span></span></label></div>
      <div class="seg ${MC.sceneOn ? "" : "off"}">
        <button class="seg-btn ${MC.sceneOn && MC.scene === "video" ? "active" : ""}" data-scene="video" ${MC.sceneOn ? "" : "disabled"}>${ico("cam")}Video</button>
        <button class="seg-btn ${MC.sceneOn && MC.scene === "image" ? "active" : ""}" data-scene="image" ${MC.sceneOn ? "" : "disabled"}>${ico("pic")}Image</button></div>
      <p>${MC.sceneOn ? "Choose where the background should come from: the character image or the motion video"
        : "Off: automatic — the character image's scene if it has one, otherwise the motion video's"}</p>
    </section>
    <button class="adv-row" id="mcAdv"><b>Advanced settings</b><span class="ico ${MC.adv ? "up" : ""}">${ICONS.chev}</span></button>
    ${MC.adv ? `<section class="adv-body">
      <div class="set-row"><span class="set-label">Length</span><div class="pills" id="mcLen"><button data-v="full" class="${MC.length === "full" ? "active" : ""}">Full video</button><button data-v="5s" class="${MC.length === "5s" ? "active" : ""}">First 5 s</button></div></div>
      <label class="set-row check"><span class="set-label">First-frame anchor<small>Qwen places your character in the video's first frame and H3 starts from it — best likeness</small></span>
        <span class="toggle"><input type="checkbox" id="mcAnchor" ${MC.anchor ? "checked" : ""}><span></span></span></label>
      <label class="set-row check"><span class="set-label">Check &amp; auto-fix<small>${esc(llmName)} compares the result with your video and picture, retakes once if it's off</small></span>
        <span class="toggle"><input type="checkbox" id="mcAudit" ${MC.audit ? "checked" : ""}><span></span></span></label>
      <label class="set-row check"><span class="set-label">Keep the motion video's audio</span>
        <span class="toggle"><input type="checkbox" id="mcAudio" ${MC.keepAudio ? "checked" : ""}><span></span></span></label>
      <div class="adv-field"><small>Prompt (optional)</small><textarea id="mcPrompt" rows="3" placeholder="Extra instructions — e.g. 'she keeps her sunglasses on' or 'the dog stays in the scene'">${esc(MC.prompt)}</textarea></div>
      <div class="adv-field"><small>Seed (optional)</small><input id="mcSeed" inputmode="numeric" placeholder="random" value="${esc(MC.seed)}"></div>
    </section>` : ""}
    <div class="panel-pad"></div>`;
  paintIcons(el); bindControl(); controlHero(); refreshControlGen();
}

function bindControl() {
  const mBox = $("#mcMotion"), cBox = $("#mcChar");
  if (!MC.motion) {
    mBox.onclick = e => { if (e.target.closest("#mcLib")) return; $("#mcMotionFile").click(); };
    $("#mcMotionFile").onchange = e => { setControlMotion(e.target.files[0]); e.target.value = ""; };
    $("#mcLib").onclick = e => { e.stopPropagation(); openLibrary(); };
    dropZone(mBox, files => setControlMotion(files[0]));
  }
  if (!MC.character) {
    cBox.onclick = () => $("#mcCharFile").click();
    $("#mcCharFile").onchange = e => { setControlChar(e.target.files[0]); e.target.value = ""; };
    dropZone(cBox, files => setControlChar(files[0]));
  }
  $$("[data-rm]").forEach(b => b.onclick = e => { e.stopPropagation(); MC[b.dataset.rm] = null; mcsave(); renderControl(); });
  $("#mcHow").onclick = openControlHow;
  $("#mcModel").onclick = () => openModal("Model", `<p class="how-note" style="margin-top:0"><b>MiniMax H3</b> with the Character Swap LoRA runs locally on this machine's GPU.
    Qwen-Image places your character in the first frame and ${esc(llmName)} writes the shot-by-shot prompt and checks the result.<br><br>No cloud service and no credits — the only cost is time on the GPU.</p>`);
  $("#mcQuality").onclick = () => openMenu($("#mcQuality"), { key: "quality", cur: MC.quality, def: "standard", choices: [["standard", "768p"], ["fast", "480p", "about twice as fast"]] }, v => { MC.quality = v; });
  $("#mcSceneOn").onchange = e => { MC.sceneOn = e.target.checked; mcsave(); renderControl(); };
  $$("[data-scene]").forEach(b => b.onclick = () => { MC.scene = b.dataset.scene; mcsave(); renderControl(); });
  $("#mcAdv").onclick = () => { MC.adv = !MC.adv; mcsave(); renderControl(); };
  if (MC.adv) {
    $$("#mcLen button").forEach(b => b.onclick = () => { MC.length = b.dataset.v; mcsave(); renderControl(); });
    $("#mcAnchor").onchange = e => { MC.anchor = e.target.checked; mcsave(); };
    $("#mcAudit").onchange = e => { MC.audit = e.target.checked; mcsave(); refreshControlGen(); };
    $("#mcAudio").onchange = e => { MC.keepAudio = e.target.checked; mcsave(); };
    $("#mcPrompt").oninput = e => { MC.prompt = e.target.value; mcsave(); };
    $("#mcSeed").oninput = e => { MC.seed = e.target.value.replace(/[^0-9]/g, ""); e.target.value = MC.seed; mcsave(); };
  }
}

async function setControlMotion(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|webm|mkv|m4v|avi)$/i.test(file.name)) return toast("The motion needs to be a video.", true);
  const b = $("#mcMotion"); if (b) b.innerHTML = `<span class="spinner"></span><b>Uploading…</b>`;
  try {
    const rec = await uploadFile(file);
    if (rec.meta.duration < 1) toast("That video is very short — motion needs at least 1 second (3+ is best).", true);
    else MC.motion = rec;
    mcsave();
  } catch (e) { toast("Upload failed: " + e.message, true); }
  renderControl();
}
async function setControlChar(file) {
  if (!file) return;
  if (!file.type.startsWith("image/") && !/\.(png|jpe?g|webp|bmp)$/i.test(file.name)) return toast("The character needs to be a picture.", true);
  MC.character = { uploading: true }; renderControl();
  try { MC.character = await uploadFile(file); mcsave(); } catch (e) { MC.character = null; toast("Upload failed: " + e.message, true); }
  renderControl();
}

function controlEstimate() {
  const dur = MC.motion ? Math.min(30, MC.motion.meta.duration) : 5.2;
  const n = MC.length === "full" && dur > 5.2 ? 1 + Math.ceil((dur - 5.2) / 4.96) : 1;
  return Math.round(1.5 + (MC.anchor ? 0.6 : 0) + n * ((MC.quality === "fast" ? 3.5 : 6.5) + (MC.audit ? 1.2 : 0)));
}
function refreshControlGen() {
  if (currentTab !== "control") return;
  $("#cost").textContent = `~${controlEstimate()} min`;
  let why = "";
  if (!MC.motion) why = "Add the motion video to copy";
  else if (!MC.character) why = "Add a picture of your character";
  else if (MC.character.uploading) why = "The picture is still uploading…";
  $("#generate").disabled = !!why; $("#genWhy").textContent = why;
}
async function controlGenerate() {
  if ($("#generate").disabled) return;
  $("#generate").disabled = true;
  try {
    const body = { tool: "control", video: MC.motion.id, images: [MC.character.id], prompt: MC.prompt || "", audit: MC.audit, keep_audio: MC.keepAudio,
                   params: { quality: MC.quality, scene: MC.sceneOn ? MC.scene : "auto", length: MC.length, anchor: MC.anchor, seed: MC.seed } };
    const j = await api("/api/tools", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    toast("Motion Control queued — it runs in the background."); focusJob = j.id; await refreshJobs(); $("#canvas").scrollTo({ top: $("#feed").offsetTop - 20, behavior: "smooth" });
  } catch (e) { toast(e.message, true); }
  finally { refreshControlGen(); }
}

function openControlHow() {
  openModal("How Motion Control works", `<div class="how">
    <div class="how-step"><div class="n">STEP 1</div><h3>The video gives the motion</h3><p>Every step, gesture, turn and camera move comes from the video you add — 3 to 30 seconds.</p></div>
    <div class="how-step"><div class="n">STEP 2</div><h3>The picture gives the character</h3><p>Face, hair, body and outfit come from your picture. A full-body photo with a clear face works best.</p></div>
    <div class="how-step"><div class="n">STEP 3</div><h3>Pick the scene</h3><p><b>Video</b>: your character appears in the video's own place. <b>Image</b>: the place behind your character in the picture becomes the scene. Off: decided for you.</p></div>
    <div class="how-step"><div class="n">STEP 4</div><h3>Rendered and checked</h3><p>Qwen-Image places your character in the first frame, H3 animates it in ≤5 s segments, and ${esc(llmName)} checks each one against your video and picture.</p></div>
  </div>
  <p class="how-note">The output keeps the motion video's shape and length. Everything runs on this machine — MiniMax H3 with the Character Swap LoRA, Qwen-Image and ${esc(llmName)}.</p>`);
}

mcload();
