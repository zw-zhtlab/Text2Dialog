(() => {
  const $ = (sel) => document.querySelector(sel);
  const api = (p, opt = {}) => fetch(p, opt).then(r => r.json()).catch(() => ({}));

  let jobId = null;
  let defaults = null;
  let pollTimer = null;

  // -------- 工具函数 --------
  function enableRun(enable) { $('#run').disabled = !enable; }
  function enableAfterExtract(enable) {
    $('#validate').disabled = !enable;
    $('#buildPairs').disabled = !enable;
    $('#buildChatML').disabled = true; // 只有 pairs 完成后才开启
    $('#downloadExtract').toggleAttribute('disabled', !enable);
  }
  function clearPollTimer() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }
  function nOrNull(v) {
    const t = String(v ?? '').trim();
    if (t === '') return null;
    const num = Number(t);
    return Number.isFinite(num) ? num : null;
  }
  function asNumber(x, fallback = 0) {
    const n = Number(x);
    return Number.isFinite(n) ? n : fallback;
  }
  function formatETA(sec) {
    const s = Math.max(0, Math.floor(sec));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    if (h > 0) return `${h}小时${m}分${ss}秒`;
    if (m > 0) return `${m}分${ss}秒`;
    return `${ss}秒`;
  }

  // -------- 新增：运行控制工具 --------
  const btnPause = $('#pause');
  const btnResume = $('#resume');
  const btnCancel = $('#cancel');

  async function control(action, reason = '') {
    if (!jobId) return {};
    return api(`/api/jobs/${jobId}/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, reason })
    });
  }

  function setControlButtonsByStatus(status) {
    // 缺省：全部禁用
    const disableAll = () => {
      btnPause.disabled = true;
      btnResume.disabled = true;
      btnCancel.disabled = true;
    };
    if (!btnPause || !btnResume || !btnCancel) return;
    switch (status) {
      case 'running':
        btnPause.disabled = false;
        btnResume.disabled = true;
        btnCancel.disabled = false;
        break;
      case 'paused':
        btnPause.disabled = true;
        btnResume.disabled = false;
        btnCancel.disabled = false;
        break;
      case 'cancelling':
        disableAll(); // 取消过程中不允许再操作
        break;
      case 'cancelled':
      case 'succeeded':
      case 'failed':
        disableAll();
        break;
      default:
        disableAll();
    }
  }

  // -------- 初始化：平台与默认值 --------
  async function init() {
    defaults = await api('/api/defaults');
    const platformSel = $('#platform');
    platformSel.innerHTML = '';
    if (defaults && defaults.platforms) {
      for (const [k, v] of Object.entries(defaults.platforms)) {
        const opt = document.createElement('option');
        opt.value = k; opt.textContent = `${k}（${v}）`;
        platformSel.appendChild(opt);
      }
    }
    if (defaults && defaults.config) {
      $('#saveChunkText').value = String(defaults.config.SAVE_CHUNK_TEXT);
      $('#sortOutput').value = String(defaults.config.DEFAULT_SORT_OUTPUT);
    }
    // 初始禁用控制按钮
    setControlButtonsByStatus('idle');
  }

  // -------- 上传 --------
  const dz = $('#dropzone');
  const file = $('#file');
  const pick = $('#pick');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => { dz.classList.remove('dragover'); });
  dz.addEventListener('drop', async e => {
    e.preventDefault(); dz.classList.remove('dragover');
    if (e.dataTransfer.files.length) await upload(e.dataTransfer.files[0]);
  });
  pick.addEventListener('click', () => file.click());
  file.addEventListener('change', async () => { if (file.files.length) await upload(file.files[0]); });

  async function upload(f) {
    const fd = new FormData();
    fd.append('file', f);
    const resp = await fetch('/api/jobs/create', { method: 'POST', body: fd }).then(r => r.json());
    jobId = resp.job_id;
    $('#uploadResult').textContent = `已上传：${f.name}，Job ID = ${jobId}`;
    enableRun(true);
    enableAfterExtract(false);
    clearPollTimer();
    resetProgressUI();
    setControlButtonsByStatus('idle');
  }

  // -------- 启动提取 --------
  $('#run').addEventListener('click', async () => {
    if (!jobId) return;
    clearPollTimer();
    enableRun(false);
    enableAfterExtract(false);
    setProgressIndeterminate('启动中…');
    setControlButtonsByStatus('running'); // 预设按钮状态

    const body = {
      platform: $('#platform').value || null,
      api_key: $('#apiKey').value || null,
      base_url: $('#baseUrl').value || null,
      model_name: $('#modelName').value || null,
      concurrent: $('#concurrent').value === 'true',
      save_chunk_text: $('#saveChunkText').value === 'true',
      sort_output: $('#sortOutput').value === 'true',
      threads: nOrNull($('#threads').value),
      TEMPERATURE: nOrNull($('#temperature').value),
      MAX_TOKEN_LEN: nOrNull($('#maxTokenLen').value),
      COVER_CONTENT: nOrNull($('#coverContent').value),
      REPLY_WINDOW: nOrNull($('#replyWindow').value),
      REPLY_CONFIDENCE_TH: nOrNull($('#replyConf').value),
    };

    const startResp = await api(`/api/jobs/${jobId}/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    // 显示 PID（若后端返回）
    if (startResp && typeof startResp.pid === 'number') {
      setProgressIndeterminate(`已启动作业（PID=${startResp.pid}）…`);
    } else {
      setProgressIndeterminate('已启动作业…');
    }

    poll(); // 开始轮询
  });

  if (btnPause) {
    btnPause.addEventListener('click', async () => {
      await control('pause');
      // 等待轮询刷新 UI
    });
  }
  if (btnResume) {
    btnResume.addEventListener('click', async () => {
      await control('resume');
    });
  }
  if (btnCancel) {
    btnCancel.addEventListener('click', async () => {
      const ok = confirm('确定取消当前作业吗？已完成的部分将保留。');
      if (ok) {
        await control('cancel');
      }
    });
  }

  // -------- 进度渲染 --------
  function resetProgressUI() {
    const bar = $('#bar');
    if (bar) {
      bar.style.display = 'block';
      bar.removeAttribute('value'); // 不确定态
      bar.max = 100;
    }
    const statusEl = $('#status');
    const etaEl = $('#eta');
    if (statusEl) statusEl.textContent = '';
    if (etaEl) etaEl.textContent = '';
  }

  function setProgressIndeterminate(text) {
    $('#bar').style.display = 'block';
    $('#bar').removeAttribute('value'); // 不确定态
    $('#status').textContent = text || '准备中…';
    const etaEl = $('#eta'); if (etaEl) etaEl.textContent = '';
  }

  function setProgressDeterminate(processed, total, stageText, message, pct, etaSec) {
    const bar = $('#bar');
    bar.style.display = 'block';
    bar.value = Math.max(0, Math.min(100, pct));
    const pctText = `${pct.toFixed(1)}%`;
    const parts = [];
    parts.push(`${processed} / ${total}（${pctText}）`);
    if (stageText) parts.push(stageText);
    if (message) parts.push(message);
    $('#status').textContent = parts.join(' · ');

    const etaEl = $('#eta');
    if (etaEl) {
      if (Number.isFinite(etaSec) && etaSec > 0) {
        etaEl.textContent = `预计剩余 ${formatETA(etaSec)}`;
      } else {
        etaEl.textContent = '';
      }
    }
  }

  const stageMap = {
    initializing: '初始化…',
    chunking: '正在分块…',
    processing: '处理中…',
    sorting: '正在排序…',
    done: '已完成',
    failed: '已失败',
    paused: '已暂停',
    cancelling: '正在取消…',
    cancelled: '已取消',
    running: '运行中…',
  };

  // -------- 轮询 --------
  async function poll() {
    if (!jobId) return;

    try {
      const info = await api(`/api/jobs/${jobId}/progress`);
      const prog = info.progress || {};

      // 兼容字段名/类型
      const processed = (prog.processed_chunks !== undefined)
        ? asNumber(prog.processed_chunks, 0)
        : asNumber(prog.processed, 0);
      const total = (prog.total_chunks !== undefined)
        ? asNumber(prog.total_chunks, 0)
        : asNumber(prog.total, 0);

      // 以 status 优先（后端 control_job 会更新 job.status）
      const status = (typeof info.status === 'string' && info.status) ? info.status : (prog.stage || '');
      const stage = (typeof prog.stage === 'string' && prog.stage) ? prog.stage : '';
      const message = (typeof prog.message === 'string') ? prog.message : (info.message || '');
      const stageText = stageMap[status] || stageMap[stage] || stage || status || '';

      // 根据状态更新控制按钮
      setControlButtonsByStatus(status || stage || '');

      // 状态特殊处理：暂停/取消中/已取消
      if (status === 'paused') {
        setProgressIndeterminate(stageText || '已暂停');
        // 继续轮询，等待恢复/取消
        pollTimer = setTimeout(poll, 1200);
        return;
      }
      if (status === 'cancelling') {
        setProgressIndeterminate(stageText || '正在取消…');
        pollTimer = setTimeout(poll, 1200);
        return;
      }
      if (status === 'cancelled') {
        $('#status').textContent = stageText || '已取消';
        enableRun(true);
        enableAfterExtract(false);
        clearPollTimer();
        return;
      }

      // 渲染逻辑：当 total 不可用/<=0 时，进入不确定态
      if (!(Number.isFinite(total) && total > 0)) {
        const hint = message || stageText || '准备中…';
        setProgressIndeterminate(hint);
      } else {
        const pct = (processed / total) * 100;
        const eta = (typeof prog.eta_sec === 'number') ? prog.eta_sec : null;
        setProgressDeterminate(processed, total, stageText, message, isFinite(pct) ? pct : 0, eta);
      }

      // 收敛条件
      if (status === 'succeeded' || stage === 'done' || (Number.isFinite(total) && total > 0 && processed >= total)) {
        // 成功收尾
        $('#status').textContent = '完成';
        enableAfterExtract(true);
        enableRun(true);
        setControlButtonsByStatus('succeeded');
        // 配置下载链接并展示预览
        const url = `/api/jobs/${jobId}/download?which=extraction`;
        $('#downloadExtract').href = url;
        await preview();
        await loadStats();
        clearPollTimer();
        return;
      }

      if (status === 'failed' || stage === 'failed') {
        $('#status').textContent = '失败：' + (message || '');
        enableAfterExtract(false);
        enableRun(true);
        setControlButtonsByStatus('failed');
        clearPollTimer();
        return;
      }

      // 继续轮询
      pollTimer = setTimeout(poll, 1200);
    } catch (err) {
      console.error('轮询进度失败：', err);
      // 轻量回退：继续尝试轮询
      pollTimer = setTimeout(poll, 1500);
    }
  }

  // -------- 预览 --------
  async function preview() {
    try {
      const resp = await fetch(`/api/jobs/${jobId}/download?which=extraction`);
      const txt = await resp.text();
      const lines = txt.trim().split('\n').slice(0, 8);
      const previewEl = $('#preview');
      if (!previewEl) return;
      previewEl.textContent = '';

      const frag = document.createDocumentFragment();
      const hint = document.createElement('div');
      hint.className = 'hint';
      hint.textContent = '前 8 条：';
      frag.appendChild(hint);

      lines.forEach((line, i) => {
        const pre = document.createElement('pre');
        let content = line;
        try {
          content = JSON.stringify(JSON.parse(line), null, 2);
        } catch {
          // Keep raw line when JSON parsing fails.
        }
        pre.textContent = `${i + 1}. ${content}`;
        frag.appendChild(pre);
      });
      previewEl.appendChild(frag);
    } catch (e) {
      console.warn('预览失败：', e);
    }
  }

  // -------- 统计（完成后显示） --------
  async function loadStats() {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      if (job && job.stats) renderStats(job.stats);
    } catch (e) {
      console.warn('加载统计失败：', e);
    }
  }

  const nf = new Intl.NumberFormat('zh-CN');

  function escapeHtml(s) {
    return String(s || '').replace(/[&<>"']/g, c => (
      {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}[c]
    ));
  }

  function roleRow([role, count], max) {
    return `<div class="role-item">
      <div class="role-name" title="${escapeHtml(role)}">${escapeHtml(role)}</div>
      <div class="role-bar"><progress value="${count}" max="${max || 1}"></progress></div>
      <div class="role-count">${nf.format(count)}</div>
    </div>`;
  }

  function renderStats(stats) {
    if (!stats) return;
    // 顶部 KPI
    const card = $('#statsCard');
    if (!card) return;
    card.style.display = 'block';
    $('#statTotal').textContent = nf.format(stats.total_dialogues || 0);
    $('#statRoles').textContent = nf.format(stats.unique_roles || 0);
    const avg = Number(stats.average_dialogue_length || 0);
    $('#statAvg').textContent = (Math.round(avg * 10) / 10) + ' 字';

    // 角色分布（Top 10 + 展开全部）
    const dist = Object.entries(stats.role_distribution || {}).sort((a,b) => b[1] - a[1]);
    const max = dist.length ? dist[0][1] : 0;
    const TOP = 10;
    $('#roleBars').innerHTML = dist.slice(0, TOP).map(d => roleRow(d, max)).join('');
    const btn = $('#toggleRoles');
    if (dist.length > TOP) {
      btn.style.display = 'inline';
      btn.onclick = () => {
        $('#roleBars').innerHTML = dist.map(d => roleRow(d, max)).join('');
        btn.style.display = 'none';
      };
    } else {
      btn.style.display = 'none';
    }
  }

  // -------- 校验 --------
  $('#validate').addEventListener('click', async () => {
    const res = await api('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId })
    });
    if (res.ok) $('#validateLog').textContent = '✅ 校验通过\n' + (res.log || '');
    else $('#validateLog').textContent = '⚠️ 校验未通过，请检查日志\n' + (res.log || '');
  });

  // -------- 生成 Pair 数据集 --------
  $('#buildPairs').addEventListener('click', async () => {
    const pairStr = $('#pairs').value.trim();
    const pairs = pairStr ? pairStr.split('|').map(s => s.trim()).filter(Boolean) : null;
    const req = {
      job_id: jobId,
      pairs,
      all_ordered_pairs: !pairs,
      min_confidence: Number($('#minConfidence').value || 0.8),
      strict: $('#strict').value === 'true',
      require_confidence: $('#requireConf').value === 'true',
    };
    const res = await api('/api/pairs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    });
    if (res.ok) {
      $('#downloadPairs').href = `/api/jobs/${jobId}/download?which=pairs_zip`;
      $('#downloadPairs').removeAttribute('disabled');
      $('#buildChatML').disabled = false; // 允许导出 ChatML
    } else {
      alert('生成失败：' + (res.log || ''));
    }
  });

  // -------- 导出 ChatML --------
  $('#buildChatML').addEventListener('click', async () => {
    const req = {
      job_id: jobId,
      mode: $('#mode').value,
      max_turns: Number($('#maxTurns').value || 4),
      min_confidence: $('#chatmlMinConf').value ? Number($('#chatmlMinConf').value) : null,
      dedupe: $('#dedupe').value === 'true',
      reverse: $('#reverse').value === 'true',
      system_text: $('#systemText').value || null,
    };
    const res = await api('/api/chatml', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req)
    });
    if (res.ok) {
      $('#downloadChatML').href = `/api/jobs/${jobId}/download?which=chatml`;
      $('#downloadChatML').removeAttribute('disabled');
    } else {
      alert('导出失败：' + (res.log || ''));
    }
  });

  // -------- 启动 --------
  init().catch(err => {
    console.error(err);
    alert('初始化失败：' + err);
  });
})();
