'use strict';
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const S = {session: null, run: null, files: [], rows: [], errors: [], sel: 0, mode: 'overlay', zoom: 1, ox: 0, oy: 0, imgs: {}, lang: 'en', map: {}};
const ZH = {
  s_import: '导入并分割', s_inspect: '检查掩膜', s_controls: '对照（可选）', s_export: '导出',
  h_add: '添加图片', p_formats: '支持：8 位灰度或 RGB（PNG、JPEG、TIFF），以及单通道 16 位 TIFF/PNG。每个文件一个视野。多页文件、带透明通道及其他格式会被拒绝并说明原因，不会被静默转换。',
  p_drop: '将图片或文件夹拖到这里', b_files: '选择文件', b_folder: '选择文件夹', t_file: '文件', t_check: '检查',
  h_settings: '设置', l_name: '运行名称', l_resize: '输入尺寸', o_native: '原始分辨率（已验证：CRC、DeepOrganoid）',
  o_factor: '按比例缩放，双线性（已验证：GemcitabineScreen，0.25）', o_square: '缩放到 S×S 正方形，box（已验证：NViR，256；仅限正方形图片）',
  n_scale: '大的单个球体在原始分辨率下可能分割失败。请通过检查少量图片的掩膜来选择尺寸，不要依据你要研究的功能结果来选。本工具不提供自动尺寸选择。',
  l_out: '结果文件夹', n_local: '图片只在本机处理。每次运行保存在新的文件夹中，不会覆盖已有结果，也不会修改原始图片。',
  b_start: '开始分割', b_cancel: '取消（保留已完成的图片）', h_images: '图片', v_overlay: '叠加', v_original: '原图', v_mask: '掩膜', v_side: '并排',
  l_opacity: '掩膜透明度', b_fit: '适应窗口', c_alive: 'Alive（模型类别）', c_dead: 'Dead（模型类别，尚未经外部验证）', c_bubble: '气泡',
  c_hint: '滚轮缩放，拖动平移。检查是否有漏分、边缘假阳性和相邻类器官粘连。', h_measure: '图像测量值', n_tier1: '第一层：仅由图像测得，单位为像素。不是活率。',
  h_controls: '对照与分组（可选）', p_controls: '相对对照的读出：将每张图片的数值除以同一组（例如同一块板、同一天）对照图片的中位数。请为需要归一化的每张图片指定分组，并标记对照图片。不同分组之间不会混用。',
  b_setgroup: '为所选图片设置分组', b_ctrl: '将所选标记为对照', b_sample: '将所选标记为样本', b_template: '下载表格模板', b_load: '导入表格（CSV）',
  t_group: '分组', t_role: '角色', t_area: '前景面积（像素）', b_preview: '检查分组', b_apply: '生成归一化表格',
  h_export: '结果', p_saved: '所有结果已保存在本次运行的文件夹中：', b_open: '打开文件夹', b_zip: '下载 ZIP',
  f_masks: '标签掩膜（0 背景、1 Alive、2 Dead、3 气泡），每张图片一个 PNG，文件名以输入序号开头。', f_overlays: '叠加预览图。',
  f_meas: '第一层图像测量值（像素），每张图片一行，含输入文件名。', f_norm: '第二层相对对照读出（仅在生成后存在），附所用分组表和分组汇总。',
  f_err: '被拒绝或失败的图片及原因。', f_man: '模型、权重 SHA256、设置、软件版本和输入文件校验值。', h_tiers: '如何理解输出',
  t1: '<b>图像测量值</b>（面积、区域数量、形态）只来自掩膜。“区域数量”是连通区域的数量；相互接触的类器官会合并，因此不等于类器官数量。',
  t2: '<b>相对对照的读出</b>只在同一分组内比较图片，不是活率百分比。', t3: '<b>经功能实验校准的预测</b>（例如 CellTiter-Glo）需要用你自己的实验数据拟合；本工具不提供。',
  t4: 'Alive 和 Dead 是模型类别。Dead 预测尚未经外部验证，不能报告为细胞死亡。'
};
const T = {
  en: {accepted: 'accepted', rejected: 'rejected', ready: n => `${n} image(s) ready`, nrej: n => `, ${n} rejected (not processed)`,
       running: p => `${p.done} / ${p.total} processed - ${p.ok} done, ${p.failed} failed - ${p.current || ''}`, loading: 'Loading the model...',
       finished: p => `Finished: ${p.ok} done, ${p.failed} failed or rejected (${p.elapsed_s} s).`, cancelled: p => `Cancelled: ${p.ok} done, kept.`,
       factor: 'Scale factor (0-1]', side: 'Square side S (px)', needSquare: 'Square-side mode needs square images; non-square images will be reported as failed.',
       wOk: w => `Model: VitaFormer M2 seed 42 - verified (SHA256 ${w.sha256.slice(0, 12)}..., ${w.size_mb} MB)`, wBad: w => `Model problem: ${w.message}`,
       errs: 'Rejected / failed', noGroup: 'Assign at least one group first.', sumOk: 'All groups have controls.', sumBad: 'Some groups cannot be normalised (see status). Their images will have no relative value.',
       created: (f, s) => `Created ${f}. Status: ${JSON.stringify(s)}`, unassigned: n => `${n} image(s) have no group and will be listed as "not in plate map".`,
       sumHead: ['group', 'images', 'controls', 'control median (px)', 'status', 'control images'],
       control: 'control', sample: 'sample', meas: {foreground_area_px: 'Foreground area (Alive+Dead), px', alive_area_px: 'Alive area, px',
       dead_area_px_unvalidated: 'Dead area, px (not validated)', bubble_area_px: 'Bubble area, px', foreground_area_fraction: 'Foreground fraction',
       n_regions: 'Region count (connected regions)', region_area_median_px: 'Median region area, px', region_eq_diameter_median_px: 'Median region diameter, px',
       region_circularity_mean: 'Mean region circularity', preprocessing: 'Preprocessing'}},
  zh: {accepted: '可处理', rejected: '已拒绝', ready: n => `${n} 张图片可处理`, nrej: n => `，${n} 张被拒绝（不处理）`,
       running: p => `已处理 ${p.done} / ${p.total} —— 完成 ${p.ok}，失败 ${p.failed} —— ${p.current || ''}`, loading: '正在加载模型……',
       finished: p => `完成：${p.ok} 张成功，${p.failed} 张失败或被拒绝（${p.elapsed_s} 秒）。`, cancelled: p => `已取消：保留已完成的 ${p.ok} 张。`,
       factor: '缩放比例 (0-1]', side: '正方形边长 S（像素）', needSquare: '正方形模式只适用于正方形图片；非正方形图片会被记为失败。',
       wOk: w => `模型：VitaFormer M2 种子 42 —— 已校验（SHA256 ${w.sha256.slice(0, 12)}…，${w.size_mb} MB）`, wBad: w => `模型文件有问题：${w.message}`,
       errs: '被拒绝 / 失败', noGroup: '请先为至少一张图片指定分组。', sumOk: '所有分组都有对照。', sumBad: '有分组无法归一化（见状态列），其中的图片不会有相对值。',
       created: (f, s) => `已生成 ${f}。状态：${JSON.stringify(s)}`, unassigned: n => `${n} 张图片没有分组，将标记为“不在分组表中”。`,
       sumHead: ['分组', '图片数', '对照数', '对照中位数（像素）', '状态', '对照图片'],
       control: '对照', sample: '样本', meas: {foreground_area_px: '前景面积（Alive+Dead），像素', alive_area_px: 'Alive 面积，像素',
       dead_area_px_unvalidated: 'Dead 面积，像素（未验证）', bubble_area_px: '气泡面积，像素', foreground_area_fraction: '前景占比',
       n_regions: '区域数量（连通区域）', region_area_median_px: '区域面积中位数，像素', region_eq_diameter_median_px: '区域直径中位数，像素',
       region_circularity_mean: '区域平均圆度', preprocessing: '预处理'}}
};
const L = () => T[S.lang];
const EN = {}; $$('[data-i]').forEach(e => EN[e.dataset.i] = e.innerHTML);
function applyLang() {
  $$('[data-i]').forEach(e => { const k = e.dataset.i; e.innerHTML = S.lang === 'zh' && ZH[k] ? ZH[k] : EN[k]; });
  $('#lang').textContent = S.lang === 'zh' ? 'English' : '中文'; document.documentElement.lang = S.lang === 'zh' ? 'zh' : 'en';
  updateResize(); renderFiles(); if (S.rows.length) { renderList(); renderMeas(); renderMap(); }
}
$('#lang').onclick = () => { S.lang = S.lang === 'en' ? 'zh' : 'en'; try { localStorage.setItem('vf_lang', S.lang); } catch (e) {} applyLang(); };
function toast(m, ms = 4000) { const t = $('#toast'); t.textContent = m; t.classList.remove('hidden'); clearTimeout(t._h); t._h = setTimeout(() => t.classList.add('hidden'), ms); }
async function api(path, opt = {}) {
  const r = await fetch(path, opt); const ct = r.headers.get('content-type') || '';
  const d = ct.includes('json') ? await r.json() : await r.blob(); if (!r.ok) throw new Error(d.error || r.statusText); return d;
}
const post = (p, b) => api(p, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b)});

// ---------------- steps
function go(step) { $$('.panel').forEach(p => p.classList.toggle('show', p.id === step)); $$('.step').forEach(b => b.classList.toggle('active', b.dataset.step === step)); if (step === 'inspect') { fit(); draw(); } }
$$('.step').forEach(b => b.onclick = () => !b.disabled && go(b.dataset.step));

// ---------------- config
async function init() {
  try { S.lang = localStorage.getItem('vf_lang') || 'en'; } catch (e) {}
  const ql = new URLSearchParams(location.search).get('lang'); if (ql === 'zh' || ql === 'en') S.lang = ql;
  const c = await api('/api/config'); $('#ver').textContent = 'v' + c.version; $('#outRoot').value = c.out_root;
  $('#device').textContent = c.device.cuda ? 'GPU detected: ' + c.device.name : 'CPU only';
  S.weights = c.weights; applyLang(); renderWeights();
}
function renderWeights() { const w = S.weights, b = $('#weightsBox'); b.innerHTML = w.ok ? L().wOk(w) : `<span class="bad">${L().wBad(w)}</span>`; updateStart(); }
function updateResize() {
  const m = $('#resizeMode').value; $('#resizeValueRow').classList.toggle('hidden', m === 'native');
  $('#resizeValueLabel').textContent = m === 'factor_bilinear' ? L().factor : L().side;
  if (m === 'factor_bilinear' && !$('#resizeValue').value) $('#resizeValue').value = 0.25;
  if (m === 'square_box' && (!$('#resizeValue').value || +$('#resizeValue').value < 2)) $('#resizeValue').value = 256;
  if (m === 'square_box' && S.files.some(f => f.ok && !f.square)) toast(L().needSquare);
}
$('#resizeMode').onchange = () => { $('#resizeValue').value = ''; updateResize(); };

// ---------------- import
async function ensureSession() {
  if (S.session) return;
  await post('/api/settings', {out_root: $('#outRoot').value});
  const s = await post('/api/session', {name: $('#runName').value}); S.session = s.session;
  $('#outRoot').disabled = true; $('#runName').disabled = true;
}
async function addFiles(list) {
  if (S.run) return toast('Start a new run by reloading the page.');
  try { await ensureSession(); } catch (e) { return toast(e.message, 8000); }
  for (const f of list) {
    const ent = {name: f.webkitRelativePath || f.name, ok: null, reason: '...'}; S.files.push(ent); renderFiles();
    try {
      const r = await api(`/api/upload?session=${encodeURIComponent(S.session)}&name=${encodeURIComponent(f.name)}`, {method: 'PUT', body: f});
      Object.assign(ent, r, {name: ent.name});
    } catch (e) { ent.ok = false; ent.reason = e.message; }
    renderFiles();
  }
}
function renderFiles() {
  const tb = $('#fileTable tbody'); tb.innerHTML = '';
  S.files.forEach((f, i) => { const tr = document.createElement('tr');
    tr.innerHTML = `<td>${i + 1}</td><td>${esc(f.name)}</td><td class="${f.ok ? 'ok' : f.ok === false ? 'bad' : ''}">${f.ok === null ? '...' : (f.ok ? L().accepted + ': ' : L().rejected + ': ') + esc(f.reason)}</td>`; tb.appendChild(tr); });
  const n = S.files.filter(f => f.ok).length, r = S.files.filter(f => f.ok === false).length;
  $('#fileSummary').textContent = S.files.length ? L().ready(n) + (r ? L().nrej(r) : '') : ''; updateStart();
}
function updateStart() { $('#start').disabled = !(S.weights && S.weights.ok && S.files.some(f => f.ok) && !S.run && !S.files.some(f => f.ok === null)); }
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
$('#pickFiles').onchange = e => addFiles([...e.target.files]);
$('#pickFolder').onchange = e => addFiles([...e.target.files].filter(f => !f.name.startsWith('.')));
const drop = $('#drop');
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); }; drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = async e => { e.preventDefault(); drop.classList.remove('over'); const out = [];
  const walk = async (en) => { if (en.isFile) out.push(await new Promise(r => en.file(r))); else if (en.isDirectory) { const rd = en.createReader();
    let batch; do { batch = await new Promise(r => rd.readEntries(r)); for (const x of batch) await walk(x); } while (batch.length); } };
  for (const it of e.dataTransfer.items) { const en = it.webkitGetAsEntry && it.webkitGetAsEntry(); if (en) await walk(en); }
  addFiles(out.filter(f => !f.name.startsWith('.'))); };

// ---------------- run
$('#start').onclick = async () => {
  const m = $('#resizeMode').value; const body = {session: S.session, resize_mode: m, resize_value: m === 'native' ? null : $('#resizeValue').value};
  try { const r = await post('/api/run', body); S.run = r.run; } catch (e) { return toast(e.message, 8000); }
  $('#start').disabled = true; $('#progressBox').classList.remove('hidden'); $('#resizeMode').disabled = $('#resizeValue').disabled = true; poll();
};
$('#cancel').onclick = () => post('/api/cancel', {run: S.run}).catch(e => toast(e.message));
async function poll() {
  const p = await api('/api/progress?run=' + encodeURIComponent(S.run));
  $('#bar').style.width = (p.total ? 100 * p.done / p.total : 0) + '%';
  $('#progressText').textContent = p.state === 'loading model' ? L().loading : p.state === 'running' ? L().running(p) :
    p.state === 'finished' ? L().finished(p) : p.state === 'cancelled' ? L().cancelled(p) : (p.message || p.state);
  if (p.message && p.state !== 'error') $('#progressText').textContent += '  [' + p.message + ']';
  if (p.device) $('#device').textContent = 'This run: ' + (p.device.startsWith('cuda') ? 'GPU' : 'CPU');
  if (p.ok && (!S.rows.length || p.done % 2 === 0 || ['finished', 'cancelled', 'error'].includes(p.state))) await loadResults();
  if (['finished', 'cancelled', 'error'].includes(p.state)) { $('#cancel').classList.add('hidden'); if (S.rows.length) go('inspect'); return; }
  setTimeout(poll, 1500);
}
async function loadResults() {
  const r = await api('/api/results?run=' + encodeURIComponent(S.run)); S.rows = r.rows; S.errors = r.errors;
  $('#runDir').textContent = r.run_dir; $('#zip').href = '/api/zip?run=' + encodeURIComponent(S.run); $('#template').href = '/api/plate_template?run=' + encodeURIComponent(S.run);
  $$('.step').forEach(b => b.disabled = b.dataset.step !== 'import' && !S.rows.length);
  renderList(); renderMap(); if (S.rows.length && !S.imgs[S.rows[S.sel].index]) select(S.sel);
}

// ---------------- inspect
function renderList() {
  const ul = $('#imgList'); ul.innerHTML = '';
  S.rows.forEach((r, i) => { const li = document.createElement('li'); li.textContent = `${r.index}. ${r.image}`; li.className = i === S.sel ? 'sel' : ''; li.onclick = () => select(i); ul.appendChild(li); });
  $('#errBox').innerHTML = S.errors.length ? `<h3 class="bad">${L().errs} (${S.errors.length})</h3>` + S.errors.map(e => `<p class="muted">${e.index}. ${esc(e.image)}: ${esc(e.reason)}</p>`).join('') : '';
}
async function loadImg(src) { return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; }); }
async function select(i) {
  S.sel = i; renderList(); renderMeas(); const r = S.rows[i]; if (!r) return;
  if (!S.imgs[r.index]) { const q = `/api/image?run=${encodeURIComponent(S.run)}&index=${r.index}&kind=`;
    S.imgs[r.index] = {orig: await loadImg(q + 'original'), mask: await loadImg(q + 'mask')}; }
  fit(); draw();
}
function renderMeas() {
  const r = S.rows[S.sel]; const t = $('#measTable'); t.innerHTML = ''; if (!r) return;
  const fmt = v => typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3)) : esc(v);
  Object.entries(L().meas).forEach(([k, lab]) => { if (k in r) t.insertAdjacentHTML('beforeend', `<tr><td>${lab}</td><td>${r[k] === null ? 'n/a' : fmt(r[k])}</td></tr>`); });
}
function fit() { const cur = S.rows[S.sel], im = cur ? S.imgs[cur.index] : null; if (!im) return; const cv = $('#cv'), W = cv.clientWidth, H = cv.clientHeight;
  S.zoom = Math.min(W / im.orig.width, H / im.orig.height); S.ox = (W - im.orig.width * S.zoom) / 2; S.oy = (H - im.orig.height * S.zoom) / 2; }
function paint(cv, layer) {
  const cur = S.rows[S.sel], im = cur ? S.imgs[cur.index] : null; const ctx = cv.getContext('2d'); const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = '#111'; ctx.fillRect(0, 0, cv.width, cv.height);
  if (!im) return; ctx.imageSmoothingEnabled = S.zoom < 2; ctx.setTransform(dpr * S.zoom, 0, 0, dpr * S.zoom, dpr * S.ox, dpr * S.oy);
  if (layer !== 'mask') ctx.drawImage(im.orig, 0, 0);
  if (layer === 'mask') { ctx.drawImage(im.mask, 0, 0); }
  else if (layer === 'overlay') { ctx.globalAlpha = $('#opacity').value / 100; ctx.drawImage(im.mask, 0, 0); ctx.globalAlpha = 1; }
}
function draw() {
  const side = S.mode === 'side'; $('#cv2').classList.toggle('hidden', !side);
  paint($('#cv'), side ? 'original' : S.mode); if (side) paint($('#cv2'), 'overlay');
}
$$('.vm').forEach(b => b.onclick = () => { S.mode = b.dataset.mode; $$('.vm').forEach(x => x.classList.toggle('active', x === b)); fit(); draw(); });
$('#opacity').oninput = draw; $('#fit').onclick = () => { fit(); draw(); };
$('#prev').onclick = () => S.sel > 0 && select(S.sel - 1); $('#next').onclick = () => S.sel < S.rows.length - 1 && select(S.sel + 1);
document.addEventListener('keydown', e => { if (!$('#inspect').classList.contains('show') || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') $('#prev').click(); if (e.key === 'ArrowRight') $('#next').click(); });
const vw = $('#viewer');
vw.addEventListener('wheel', e => { e.preventDefault(); const r = e.target.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const f = e.deltaY < 0 ? 1.2 : 1 / 1.2; S.ox = x - (x - S.ox) * f; S.oy = y - (y - S.oy) * f; S.zoom *= f; draw(); }, {passive: false});
let drag = null; vw.onmousedown = e => { drag = [e.clientX, e.clientY, S.ox, S.oy]; vw.style.cursor = 'grabbing'; };
window.onmouseup = () => { drag = null; vw.style.cursor = 'grab'; };
window.onmousemove = e => { if (!drag) return; S.ox = drag[2] + e.clientX - drag[0]; S.oy = drag[3] + e.clientY - drag[1]; draw(); };
window.onresize = () => { if ($('#inspect').classList.contains('show')) { fit(); draw(); } };

// ---------------- controls
function renderMap() {
  const tb = $('#mapTable tbody'); tb.innerHTML = '';
  S.rows.forEach(r => { const m = S.map[r.image] || (S.map[r.image] = {group: '', role: 'sample', sel: false}); const tr = document.createElement('tr');
    tr.innerHTML = `<td><input type="checkbox" ${m.sel ? 'checked' : ''}></td><td>${r.index}. ${esc(r.image)}</td><td><input value="${esc(m.group)}"></td>
      <td><select><option value="sample">${L().sample}</option><option value="control">${L().control}</option></select></td><td>${r.foreground_area_px.toLocaleString()}</td>`;
    const [cb, gi, rs] = tr.querySelectorAll('input,select'); rs.value = m.role;
    cb.onchange = () => m.sel = cb.checked; gi.oninput = () => { m.group = gi.value; $('#applyNorm').disabled = true; };
    rs.onchange = () => { m.role = rs.value; $('#applyNorm').disabled = true; }; tb.appendChild(tr); });
}
$('#selAll').onchange = e => { S.rows.forEach(r => S.map[r.image].sel = e.target.checked); renderMap(); };
const selRows = () => S.rows.filter(r => S.map[r.image].sel);
$('#applyGroup').onclick = () => { selRows().forEach(r => S.map[r.image].group = $('#bulkGroup').value.trim()); $('#applyNorm').disabled = true; renderMap(); };
$('#markCtrl').onclick = () => { selRows().forEach(r => S.map[r.image].role = 'control'); $('#applyNorm').disabled = true; renderMap(); };
$('#markSample').onclick = () => { selRows().forEach(r => S.map[r.image].role = 'sample'); $('#applyNorm').disabled = true; renderMap(); };
$('#loadMap').onchange = async e => { const txt = (await e.target.files[0].text()).replace(/^﻿/, ''); const lines = txt.split(/\r?\n/).filter(Boolean);
  const head = lines[0].split(',').map(s => s.trim().toLowerCase()); const ii = head.indexOf('image'), gi = head.indexOf('group'), ri = head.indexOf('role');
  if (ii < 0 || gi < 0 || ri < 0) return toast('CSV needs columns: image, group, role'); let n = 0, miss = 0;
  lines.slice(1).forEach(l => { const c = l.split(','); const m = S.map[c[ii]]; if (!m) { miss++; return; } m.group = (c[gi] || '').trim(); m.role = (c[ri] || '').trim().toLowerCase() === 'control' ? 'control' : 'sample'; n++; });
  renderMap(); toast(`${n} rows loaded` + (miss ? `, ${miss} names not found in this run` : '')); $('#applyNorm').disabled = true; };
const plateMap = () => S.rows.map(r => ({image: r.image, group: S.map[r.image].group, role: S.map[r.image].role}));
$('#preview').onclick = async () => {
  if (!plateMap().some(x => x.group.trim())) return toast(L().noGroup);
  try { const r = await post('/api/controls/preview', {run: S.run, plate_map: plateMap()});
    const bad = r.summary.some(s => s.status !== 'ok');
    $('#summary').innerHTML = `<p class="${bad ? 'warn' : 'ok'}">${bad ? L().sumBad : L().sumOk}</p>` + (r.unassigned ? `<p class="muted">${L().unassigned(r.unassigned)}</p>` : '') +
      '<table class="tbl"><tr>' + L().sumHead.map(h => `<th>${h}</th>`).join('') + '</tr>' +
      r.summary.map(s => `<tr><td>${esc(s.group)}</td><td>${s.n_images}</td><td>${s.n_controls}</td><td>${s.control_median_foreground_area_px == null ? 'n/a' : Math.round(s.control_median_foreground_area_px).toLocaleString()}</td><td class="${s.status === 'ok' ? 'ok' : 'bad'}">${esc(s.status)}</td><td class="muted">${esc(s.controls)}</td></tr>`).join('') + '</table>';
    $('#applyNorm').disabled = false; } catch (e) { toast(e.message, 8000); }
};
$('#applyNorm').onclick = async () => { try { const r = await post('/api/controls/apply', {run: S.run, plate_map: plateMap()}); toast(L().created(r.folder, r.status_counts), 9000); } catch (e) { toast(e.message, 8000); } };
$('#openFolder').onclick = () => post('/api/open_folder', {run: S.run}).catch(e => toast(e.message));
async function resume() {  // reopen a run of this session: /?run=<id>&step=inspect|controls|export
  const u = new URLSearchParams(location.search); if (!u.get('run')) return;
  S.run = u.get('run'); await loadResults();
  $('#runName').value = S.run.replace(/^\d{8}-\d{6}_/, '');
  S.files = S.rows.map(r => ({name: r.image, ok: true, reason: r.preprocessing})).concat(S.errors.map(e => ({name: e.image, ok: false, reason: e.reason.replace(/^rejected: /, '')})));
  if (S.rows.length) { $('#resizeMode').value = S.rows[0].resize_mode; updateResize(); if (S.rows[0].resize_value != null) $('#resizeValue').value = S.rows[0].resize_value;
    $('#resizeMode').disabled = $('#resizeValue').disabled = $('#outRoot').disabled = $('#runName').disabled = true; $('#start').disabled = true; }
  renderFiles(); $('#progressBox').classList.remove('hidden'); $('#cancel').classList.add('hidden'); $('#bar').style.width = '100%';
  const p = await api('/api/progress?run=' + encodeURIComponent(S.run)); $('#progressText').textContent = L().finished(p);
  if (p.device) $('#device').textContent = 'This run: ' + (p.device.startsWith('cuda') ? 'GPU' : 'CPU');
  const m = await api('/api/last_plate_map?run=' + encodeURIComponent(S.run));
  m.rows.forEach(x => { if (S.map[x.image]) { S.map[x.image].group = String(x.group); S.map[x.image].role = x.role; } }); renderMap();
  if (u.get('sel')) await select(+u.get('sel')); if (u.get('mode')) $(`.vm[data-mode="${u.get('mode')}"]`).click();
  go(u.get('step') || 'inspect'); if (u.get('step') === 'controls' && m.rows.length) $('#preview').click();
}
init().then(resume).catch(e => toast('Cannot reach the VitaFormer service: ' + e.message, 15000));
