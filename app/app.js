/* Le Quinté du Jour — renders data/today.json. No build step, no framework.
   The data pipeline writes today.json; this file only displays it.
   Format of today.json: see README.md. */
"use strict";

const DATA_URL = "data/today.json";
const el = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const kg = n => String(n).replace(".", ",");
const store = {
  get(k, fallback) { try { const v = localStorage.getItem(k); return v == null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage blocked: keep going */ } }
};

let D, BY_NUM = {};

/* ---------- small builders ---------- */
const cloth = (n, cls = "", btn = false) => btn
  ? `<button class="cloth ${cls}" data-h="${n}" aria-label="Pourquoi le ${n} ${esc(BY_NUM[n]?.name)}">${n}</button>`
  : `<span class="cloth ${cls}">${n}</span>`;
const minis = (list, ref) => `<div class="minis">${list.map(n => cloth(n, "sm " + (ref ? (ref.includes(n) ? "hit" : "miss") : ""))).join("")}</div>`;
const common = (a, b) => a.slice(0, 5).filter(n => b.includes(n)).length;

function ticketHTML(t, payouts) {
  const lines = t.lines.map(l => `<div class="line"><div><div class="bet">${esc(l.bet)}</div><div class="nums">${l.nums.join(" · ")}</div></div><div class="price">${l.price_eur} €</div></div>`).join("");
  const win = payouts ? `<div class="win"><div class="wt">${esc(payouts.title)}</div>
      ${payouts.rows.map(r => `<div class="wr"><span>${esc(r.label)}</span><b>${esc(r.range)}</b></div>`).join("")}
      ${payouts.note ? `<div class="note">${esc(payouts.note)}</div>` : ""}</div>` : "";
  return `<div class="card ticket"><div class="eyebrow">${esc(t.title)}</div>${lines}
    <div class="total"><span>Total du jour</span><span>${t.total_eur} €</span></div>
    ${t.note ? `<div class="note">${esc(t.note)}</div>` : ""}${win}</div>`;
}

function horseRows(list) {
  const max = Math.max(...list.map(n => BY_NUM[n].win_chance_pct || 0)) || 1;
  return list.map(n => {
    const h = BY_NUM[n];
    return `<button class="horse" data-h="${n}">${cloth(n)}<div>
      <div class="name">${esc(h.name)}</div><div class="meta">${esc(h.jockey)}${h.weight_kg ? ` · ${kg(h.weight_kg)} kg` : ""}</div>
      <div class="one">${esc(h.verdict)}</div>
      <div class="bar"><i style="width:${Math.round(100 * (h.win_chance_pct || 0) / max)}%"></i></div>
      <div class="chance">Chance de gagner estimée : environ ${h.win_chance_pct} %</div></div><span class="go" aria-hidden="true">›</span></button>`;
  }).join("");
}

function modeHTML(m, hint, extra = "") {
  return `<div class="card">
      <div class="eyebrow">${esc(m.title)}</div>
      ${m.intro ? `<div class="small">${esc(m.intro)}</div>` : ""}
      <div class="chips">${m.picks.map(n => cloth(n, "", true)).join("")}</div>
      ${extra}
      <div class="tap-hint">${hint}</div></div>
    ${(m.notices || []).map(t => `<div class="preview">${esc(t)}</div>`).join("")}
    ${ticketHTML(m.ticket, m.payouts)}
    <div class="card"><h2>Pourquoi ces chevaux</h2><div>${horseRows(m.picks)}</div></div>`;
}

/* ---------- latest odds ---------- */
function raceStarted() {
  const [h, m] = D.race.start_time.split("h");
  const paris = new Date().toLocaleString("sv-SE", { timeZone: "Europe/Paris" });  // "YYYY-MM-DD HH:MM:SS" on any device
  return paris.slice(0, 16) >= `${D.race.date} ${h.padStart(2, "0")}:${m}`;
}
function oddsLabel() { return D.odds_updated ? `Cotes de ${D.odds_updated}` : "Cotes pas encore publiées"; }
function refreshHTML() {
  if (raceStarted()) return D.odds_updated ? `<div class="small muted">${oddsLabel()}</div>` : "";
  return `<div class="refresh"><button class="btn" data-refresh>↻ Actualiser les cotes</button>
    <span class="small muted" id="refresh-msg">${oddsLabel()}</span></div>`;
}
async function refreshOdds(btn) {
  const before = D.modes.prono.picks.join("-"), label = btn.textContent;
  btn.disabled = true; btn.textContent = "Actualisation…";
  try {
    const r = await fetch("refresh", { method: "POST" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const res = await fetch(DATA_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    D = await res.json();
  } catch (e) {
    btn.disabled = false; btn.textContent = label;
    const where = btn.closest("#odds-nudge") ? el("odds-nudge").querySelector(".small") : el("refresh-msg");
    if (where) where.textContent = "Actualisation impossible pour le moment, réessayez plus tard.";
    return;
  }
  el("odds-nudge").hidden = true;
  BY_NUM = {}; D.runners.forEach(r => BY_NUM[r.num] = r);
  renderRace(); renderCompare();
  const msg = el("refresh-msg");
  if (msg) msg.textContent = `${oddsLabel()} · ${D.modes.prono.picks.join("-") === before ? "classement inchangé" : "classement mis à jour"}`;
}
/* On load: the server says whether newer odds are published (the prono is more precise with them). */
async function checkOdds() {
  if (!D || raceStarted()) return;
  let c;
  try {
    const r = await fetch("odds-check", { cache: "no-store" });
    if (!r.ok) return;
    c = await r.json();
  } catch (e) { return; }  // no check: the refresh button stays available
  if (!c.offer) return;
  el("odds-nudge").innerHTML = `<div><b>${c.odds_shown ? "Nouvelles cotes disponibles" : "Les cotes sont publiées"}</b>
    <div class="small">${c.odds_shown ? `Le prono utilise les cotes de ${esc(c.odds_shown)}.` : "Le prono du matin a été fait sans les cotes."}
    Actualisez pour un prono plus précis avant de jouer.</div></div>
    <button class="btn primary" data-refresh>↻ Actualiser</button>`;
  el("odds-nudge").hidden = false;
}

/* ---------- horse sheet ---------- */
function musiqueHTML(mu) {
  return esc(mu).split(" ").map(t => /^\(\d+\)$/.test(t) ? `<span class="yr">20${t.slice(1, -1)} →</span>` : (parseInt(t) <= 3 ? `<span class="top">${t}</span>` : t)).join(" ");
}
const ICON = { plus: "+", minus: "–", eq: "=", q: "?" };
function openSheet(n) {
  const h = BY_NUM[n];
  const live = D.runners.filter(r => !r.non_runner).length;
  const head = h.non_runner ? "Non-partant" : `Rang IA : ${h.ai_rank}e sur ${live} · chance de gagner estimée : environ ${h.win_chance_pct} %`;
  el("sheet-in").innerHTML = `<div class="grab"></div>
    <div class="head">${cloth(n)}<div><h2 id="sh-name">${esc(h.name)}</h2><div class="small muted">${esc(h.sex_age)} · ${esc(h.jockey)}${h.draw ? ` · corde ${h.draw}` : ""}</div></div></div>
    <div class="small muted">${head}</div>
    <div class="verdict">${esc(h.verdict)}</div>
    <div><div class="eyebrow">Ses dernières courses</div><div class="musique">${musiqueHTML(h.musique)}</div>
      <div class="small muted">Le chiffre = sa place à l'arrivée, la plus récente à gauche. En vert : sur le podium.</div></div>
    <div class="factors">${h.factors.map(f => `<div class="f"><span class="ic ${f.sign}" aria-hidden="true">${ICON[f.sign] || "?"}</span><div><div class="t">${esc(f.title)}</div><div class="d">${esc(f.text)}</div></div></div>`).join("")}</div>
    <button class="btn primary" id="sh-close">Fermer</button>`;
  el("scrim").hidden = false; el("sheet").hidden = false; el("sheet").scrollTop = 0;
  el("sh-close").onclick = closeSheet; el("sh-close").focus({ preventScroll: true });
}
function closeSheet() { el("scrim").hidden = true; el("sheet").hidden = true; }

/* ---------- screens ---------- */
function renderRace() {
  const r = D.race;
  el("race-eyebrow").textContent = `Quinté+ · ${r.date_label}`;
  el("race-title").textContent = `${r.hippodrome} · ${r.prix}`;
  el("race-facts").innerHTML = `<span><b>${esc(r.start_time)}</b></span><span>${r.distance_m.toLocaleString("fr-FR")} m ${esc(r.surface)}</span><span>${esc(r.race_type)}</span><span><b>${r.runners_count} partants</b></span>`;
  el("saisie-eyebrow").textContent = `${r.date_label} · ${r.hippodrome}`;
  document.title = `Quinté · ${r.hippodrome}`;

  el("mode-prono").innerHTML = modeHTML(D.modes.prono, "Touchez un numéro pour savoir pourquoi.", refreshHTML());
  el("mode-value").innerHTML = modeHTML(D.modes.value, "Touchez un numéro pour le détail.");
  el("runners").innerHTML = D.runners.map(h => h.non_runner ? `<span class="cloth np" title="Non-partant">${h.num}</span>` : cloth(h.num, "", true)).join("");
  el("cmp-note").textContent = D.other_tips.note || "";
}

let picks, who = "veinard";
const picksKey = () => `q-picks-${D.race.date}`;

function renderCompare() {
  const AI = D.modes.prono.picks;
  const rows = D.other_tips.sources.map(s => [s.source, s.picks]);
  if (picks.veinard.length) rows.push(["Le Veinard", picks.veinard.slice(0, 5)]);
  if (picks.moi.length) rows.push(["Mon jeu", picks.moi.slice(0, 5)]);
  let html = rows.map(([s, l]) => `<tr><td class="src">${esc(s)}<br><span class="pill">${common(l, AI)} en commun avec l'IA</span></td><td>${minis(l, AI)}</td></tr>`).join("");
  if (!picks.veinard.length) html += `<tr><td class="src">Le Veinard</td><td><span class="pill grey">À saisir dans « Mes numéros »</span></td></tr>`;
  el("cmp").innerHTML = html;
}

function renderPad() {
  const l = picks[who];
  el("pad").querySelectorAll("button").forEach(b => {
    const n = +b.dataset.n, i = l.indexOf(n);
    b.setAttribute("aria-pressed", i >= 0); b.querySelector(".rank").textContent = i >= 0 ? i + 1 : "";
  });
  el("pick-chips").innerHTML = l.map(n => cloth(n, "sm")).join("") || `<span class="small muted">Aucun cheval choisi</span>`;
  el("who-veinard").setAttribute("aria-pressed", who === "veinard");
  el("who-moi").setAttribute("aria-pressed", who === "moi");
  el("pad-help").textContent = who === "veinard" ? "Touchez les chevaux dans l'ordre du journal (jusqu'à 8)." : "Touchez vos chevaux dans votre ordre (jusqu'à 8).";
}

function renderInput() {
  picks = store.get(picksKey(), { veinard: [], moi: [] });
  el("pad").innerHTML = D.runners.map(h => `<button id="p${h.num}" data-n="${h.num}" aria-pressed="false" ${h.non_runner ? "disabled" : ""} aria-label="Cheval ${h.num} ${esc(h.name)}">${h.num}<span class="rank"></span></button>`).join("");
  el("pad").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b || b.disabled) return;
    const n = +b.dataset.n, l = picks[who], i = l.indexOf(n);
    if (i >= 0) l.splice(i, 1); else if (l.length < 8) l.push(n);
    el("saved").textContent = ""; renderPad();
  });
  el("who-veinard").onclick = () => { who = "veinard"; el("saved").textContent = ""; renderPad(); };
  el("who-moi").onclick = () => { who = "moi"; el("saved").textContent = ""; renderPad(); };
  el("clear").onclick = () => { picks[who] = []; el("saved").textContent = ""; renderPad(); };
  el("save").onclick = () => {
    store.set(picksKey(), picks); renderCompare();
    el("saved").textContent = picks[who].length ? "Enregistré. Comparaison mise à jour dans « Le Quinté »." : "Enregistré.";
  };
  renderPad();
}

const euros = v => `${Number(v).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} €`;
function ticketRows(sources, arrival) {
  return sources.map(s => {
    if (!s.picks.length) return `<tr><td class="src">${esc(s.source)}</td><td><span class="pill grey">Pas de prono</span></td></tr>`;
    const k = s.picks.filter(n => arrival.includes(n)).length;
    const won = s.won_eur == null ? "" : ` · gagné ${euros(s.won_eur)}`;
    return `<tr><td class="src">${esc(s.source)}<br><span class="pill ${k || s.won_eur ? "" : "grey"}">${k} sur ${s.picks.length}${won}</span></td><td>${minis(s.picks, arrival)}</td></tr>`;
  }).join("");
}

function renderHistory() {
  el("history").innerHTML = (D.history || []).map(d => `<div class="day"><div class="eyebrow">${esc(d.label)}</div>
    ${d.cancelled ? `<div class="small muted">Course annulée : mises remboursées.</div>`
      : `<div class="small muted">Arrivée</div>${minis(d.arrival)}<table class="compare"><tbody>${ticketRows(d.tickets, d.arrival)}</tbody></table>`}</div>`).join("")
    || `<div class="small muted">Pas encore de résultat.</div>`;
}

function renderMatch() {
  const y = D.yesterday;
  el("y-label").textContent = y.label;
  el("arr-chips").innerHTML = y.arrival.map(n => cloth(n)).join("");
  el("yesterday").innerHTML = ticketRows(y.sources, y.arrival);
  renderHistory();
  el("season").innerHTML = `<span class="h">Joueur</span><span class="h" style="text-align:right">Misé</span><span class="h" style="text-align:right">Gagné</span>`
    + D.season.players.map(p => `<span><b>${esc(p.name)}</b></span><span class="v">${p.bet_eur} €</span><span class="v">${p.won_eur} €</span>`).join("");
  el("season-note").textContent = D.season.note || "";
}

function setMode(m) {
  el("m-prono").setAttribute("aria-pressed", m === "prono");
  el("m-value").setAttribute("aria-pressed", m === "value");
  el("mode-prono").hidden = m !== "prono"; el("mode-value").hidden = m !== "value";
  store.set("q-mode", m);
}

/* ---------- wiring ---------- */
function wire() {
  el("scrim").onclick = closeSheet;
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeSheet(); });
  document.addEventListener("click", e => {
    const r = e.target.closest("[data-refresh]");
    if (r) return refreshOdds(r);
    const b = e.target.closest("[data-h]");
    if (b && BY_NUM[b.dataset.h] && !BY_NUM[b.dataset.h].non_runner) openSheet(+b.dataset.h);
  });
  el("m-prono").onclick = () => setMode("prono");
  el("m-value").onclick = () => setMode("value");
  if (store.get("q-mode", "prono") === "value") setMode("value");
  document.querySelectorAll(".tabs button").forEach(t => t.onclick = () => {
    document.querySelectorAll(".tabs button").forEach(x => x.setAttribute("aria-selected", x === t));
    document.querySelectorAll("section.view").forEach(v => v.hidden = v.id !== t.dataset.v);
    window.scrollTo(0, 0);
  });
}

function showError(msg) {
  document.querySelector(".wrap").hidden = true;
  el("load-error").hidden = false;
  el("load-error").innerHTML = msg;
}

async function boot() {
  try {
    const res = await fetch(DATA_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    D = await res.json();
  } catch (e) {
    const fileHint = location.protocol === "file:"
      ? `<p>La page est ouverte comme fichier. Lancez un petit serveur dans le dossier de l'appli :</p><p><code>python -m http.server 8000</code></p><p>puis ouvrez <code>http://localhost:8000</code>.</p>`
      : `<p>Vérifiez que <code>${DATA_URL}</code> existe.</p>`;
    showError(`<b>Impossible de charger le Quinté du jour.</b>${fileHint}<p class="small">Détail : ${esc(e.message)}</p>`);
    return;
  }
  D.runners.forEach(r => BY_NUM[r.num] = r);
  renderRace(); renderInput(); renderCompare(); renderMatch(); wire();
  checkOdds();
  document.addEventListener("visibilitychange", () => { if (!document.hidden) checkOdds(); });  // app reopened later
}
boot();
