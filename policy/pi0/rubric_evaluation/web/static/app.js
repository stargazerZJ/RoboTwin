const state = {
  status: null,
  versions: [],
  episodes: [],
  selectedIdx: -1,
  showVersions: false,
  loop: false,
  autoplay: true,
};

function $(id){ return document.getElementById(id); }

function fmtPct(x){ return (100*x).toFixed(1) + "%"; }

function unixToLocal(ts){
  if(!ts) return "";
  const d = new Date(ts*1000);
  return d.toLocaleString();
}

async function apiGet(path){
  const r = await fetch(path);
  if(!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function apiPost(path){
  const r = await fetch(path, {method:"POST"});
  if(!r.ok) throw new Error(await r.text());
  return await r.json();
}

function renderStatus(){
  const s = state.status;
  if(!s){ $("statusline").textContent = "Loading…"; return; }
  const rate = s.evaluated ? fmtPct(s.success_rate) : "—";
  $("statusline").textContent =
    `model=${s.model_name} | task=${s.task_name}-${s.task_config} | version=${s.current_version_id ?? "—"} | ` +
    `evaluated=${s.evaluated}/${s.num_episodes} | success=${s.success} | rate=${rate} | updated=${unixToLocal(s.last_update_unix)}`;
}

function renderEpisodes(){
  const list = $("episodeList");
  list.innerHTML = "";
  state.episodes.forEach((ep, idx) => {
    const div = document.createElement("div");
    div.className = "episode" + (idx === state.selectedIdx ? " active" : "");
    const ok = !!ep.success;
    const badge = document.createElement("div");
    badge.className = "badge " + (ok ? "good" : "bad");
    badge.textContent = ok ? "SUCCESS" : "FAIL";

    const main = document.createElement("div");
    main.className = "ep-main";
    const title = document.createElement("div");
    title.className = "ep-title";
    title.textContent = `ep=${ep.episode_id} seed=${ep.seed} steps=${ep.steps ?? "?"} sub=${ep.subtask_state_final ?? "?"}`;
    const sub = document.createElement("div");
    sub.className = "ep-sub";
    if(ep.error){
      sub.textContent = `ERROR: ${ep.error}`;
    } else {
      const dbg = ep.debug_last ? JSON.stringify(ep.debug_last) : "";
      sub.textContent = dbg;
    }

    main.appendChild(title);
    main.appendChild(sub);

    div.appendChild(main);
    div.appendChild(badge);

    div.onclick = () => selectEpisode(idx);
    list.appendChild(div);
  });
}

function videoUrlForEpisode(ep){
  const s = state.status;
  if(!s || !s.current_version_id) return null;
  const name = `episode_${String(ep.episode_id).padStart(4,"0")}.mp4`;
  // Directory structure: runs/{model_name}/{task_name}-{task_config}/{version_id}/videos/
  return `/runs/${s.model_name}/${s.task_name}-${s.task_config}/${s.current_version_id}/videos/${name}`;
}

function renderOverlay(ep){
  const overlay = $("overlay");
  if(!ep){ overlay.textContent = ""; return; }
  const lines = [];
  lines.push(`episode_id: ${ep.episode_id}`);
  lines.push(`seed: ${ep.seed}`);
  lines.push(`success: ${ep.success}`);
  if(ep.backend) lines.push(`backend: ${ep.backend.host}:${ep.backend.port}`);
  if(ep.subtask_state_final !== undefined) lines.push(`subtask_final: ${ep.subtask_state_final}`);
  if(ep.time_sec !== undefined) lines.push(`time_sec: ${ep.time_sec.toFixed(2)}`);
  if(ep.error){
    lines.push("");
    lines.push("error:");
    lines.push(String(ep.error));
  }
  if(ep.debug_last){
    lines.push("");
    lines.push("debug_last:");
    lines.push(JSON.stringify(ep.debug_last, null, 2));
  }
  overlay.textContent = lines.join("\n");
}

function selectEpisode(idx){
  state.selectedIdx = idx;
  renderEpisodes();

  const ep = state.episodes[idx];
  const url = videoUrlForEpisode(ep);
  const video = $("video");
  if(url){
    $("videoTitle").textContent = `Video: ep=${ep.episode_id}`;
    if(video.src !== location.origin + url){
      video.src = url;
    }
    video.loop = state.loop;
    video.playbackRate = parseFloat($("speed").value);
    if(state.autoplay){
      video.play().catch(()=>{});
    }
  } else {
    $("videoTitle").textContent = "Video";
  }
  renderOverlay(ep);
}

function renderVersions(){
  const panel = $("versionsPanel");
  panel.classList.toggle("hidden", !state.showVersions);
  const root = $("versions");
  root.innerHTML = "";
  state.versions.forEach(v => {
    const div = document.createElement("div");
    div.className = "version";
    const row = document.createElement("div");
    row.className = "row";
    const id = document.createElement("div");
    id.className = "id";
    id.textContent = `${v.version_id} (eval=${v.evaluated})`;
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = "Rollback";
    btn.onclick = async () => {
      await apiPost(`/api/rollback/${v.version_id}`);
      await refreshAll();
    };
    row.appendChild(id);
    row.appendChild(btn);

    const summary = document.createElement("div");
    summary.className = "summary";
    summary.textContent = v.summary || "";

    div.appendChild(row);
    div.appendChild(summary);
    root.appendChild(div);
  });
}

async function refreshAll(){
  state.status = await apiGet("/api/status");
  state.versions = (await apiGet("/api/versions")).versions;
  state.episodes = (await apiGet("/api/episodes?limit=80")).episodes;
  renderStatus();
  renderVersions();

  if(state.episodes.length === 0){
    state.selectedIdx = -1;
    renderEpisodes();
    renderOverlay(null);
    return;
  }
  if(state.selectedIdx < 0) state.selectedIdx = state.episodes.length - 1;
  if(state.selectedIdx >= state.episodes.length) state.selectedIdx = state.episodes.length - 1;
  renderEpisodes();
  selectEpisode(state.selectedIdx);
}

function setupControls(){
  $("btnReload").onclick = async () => {
    await apiPost("/api/reload");
    await refreshAll();
  };
  $("btnVersions").onclick = () => {
    state.showVersions = !state.showVersions;
    renderVersions();
  };
  $("speed").onchange = () => {
    const v = $("video");
    v.playbackRate = parseFloat($("speed").value);
  };
  $("loop").onchange = () => {
    state.loop = $("loop").checked;
    $("video").loop = state.loop;
  };
  $("autoplay").onchange = () => {
    state.autoplay = $("autoplay").checked;
  };
  $("btnPrev").onclick = () => {
    if(state.selectedIdx > 0) selectEpisode(state.selectedIdx - 1);
  };
  $("btnNext").onclick = () => {
    if(state.selectedIdx + 1 < state.episodes.length) selectEpisode(state.selectedIdx + 1);
  };

  window.addEventListener("keydown", (e) => {
    const video = $("video");
    if(e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.isContentEditable)) return;

    if(e.key === "r"){
      e.preventDefault();
      $("btnReload").click();
    } else if(e.key === "v"){
      e.preventDefault();
      $("btnVersions").click();
    } else if(e.key === " "){
      e.preventDefault();
      if(video.paused) video.play().catch(()=>{});
      else video.pause();
    } else if(e.key === "l"){
      e.preventDefault();
      $("loop").checked = !$("loop").checked;
      $("loop").dispatchEvent(new Event("change"));
    } else if(e.key === "ArrowUp"){
      e.preventDefault();
      if(state.selectedIdx > 0) selectEpisode(state.selectedIdx - 1);
    } else if(e.key === "ArrowDown"){
      e.preventDefault();
      if(state.selectedIdx + 1 < state.episodes.length) selectEpisode(state.selectedIdx + 1);
    } else if(e.key === "j"){
      e.preventDefault();
      video.currentTime = Math.max(0, video.currentTime - 1.0);
    } else if(e.key === "k"){
      e.preventDefault();
      video.currentTime = Math.min(video.duration || 1e9, video.currentTime + 1.0);
    } else if(e.key === "f"){
      e.preventDefault();
      video.pause();
      video.currentTime = Math.min(video.duration || 1e9, video.currentTime + 0.1);
    } else if(e.key === "<" || e.key === ","){
      e.preventDefault();
      const speeds = ["0.25","0.5","1","2","5","10"];
      const cur = $("speed").value;
      const i = Math.max(0, speeds.indexOf(cur) - 1);
      $("speed").value = speeds[i];
      $("speed").dispatchEvent(new Event("change"));
    } else if(e.key === ">" || e.key === "."){
      e.preventDefault();
      const speeds = ["0.25","0.5","1","2","5","10"];
      const cur = $("speed").value;
      const i = Math.min(speeds.length-1, speeds.indexOf(cur) + 1);
      $("speed").value = speeds[i];
      $("speed").dispatchEvent(new Event("change"));
    }
  });
}

async function main(){
  setupControls();
  await refreshAll();
  setInterval(async () => {
    try { await refreshAll(); } catch(e) { /* ignore */ }
  }, 2000);
}

main().catch(err => {
  console.error(err);
  $("statusline").textContent = "Error: " + err;
});