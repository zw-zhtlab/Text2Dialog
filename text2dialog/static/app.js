(() => {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const api = async (p, opt = {}) => {
    try {
      const r = await fetch(p, opt);
      const text = await r.text();
      const data = text ? JSON.parse(text) : {};
      if (!r.ok) {
        return { ...data, ok: false, http_status: r.status };
      }
      return data;
    } catch (err) {
      return { ok: false, detail: err && err.message ? err.message : '请求失败' };
    }
  };

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
    $('#downloadQuality').toggleAttribute('disabled', !enable);
  }
  function showToast(text, type = 'info') {
    const toast = $('#toast');
    if (!toast) return;
    toast.textContent = text || '';
    toast.className = `toast ${type}`;
    toast.hidden = !text;
    if (text) {
      clearTimeout(showToast._timer);
      showToast._timer = setTimeout(() => { toast.hidden = true; }, 4200);
    }
  }
  function setStatus(text, isError = false) {
    const el = $('#status');
    if (!el) return;
    el.textContent = text || '';
    el.classList.toggle('error', Boolean(isError));
  }
  function setWorkflowState(states = {}) {
    let activeStep = '';
    $$('.workflow-step').forEach(step => {
      const state = states[step.dataset.step] || '';
      if (state === 'active') activeStep = step.dataset.step;
      step.classList.toggle('active', state === 'active');
      step.classList.toggle('done', state === 'done');
      step.classList.toggle('error', state === 'error');
    });
    $$('[data-step-panel]').forEach(panel => {
      const state = states[panel.dataset.stepPanel] || '';
      panel.classList.toggle('active', state === 'active');
      panel.classList.toggle('done', state === 'done');
      panel.classList.toggle('error', state === 'error');
    });
    document.body.dataset.workflowStep = activeStep;
  }
  function resetWorkflow() {
    setWorkflowState({ upload: 'active' });
  }
  function resetResultUI() {
    ['#statsCard', '#qualityCard'].forEach(sel => {
      const el = $(sel);
      if (el) el.style.display = 'none';
    });
    ['#preview', '#validateLog', '#roleBars', '#qualityReasons'].forEach(sel => {
      const el = $(sel);
      if (el) el.textContent = '';
    });
    ['#downloadPairs', '#downloadChatML', '#downloadValidationReport', '#downloadPairDiagnostics'].forEach(sel => {
      const el = $(sel);
      if (el) {
        el.href = '#';
        el.toggleAttribute('disabled', true);
      }
    });
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
    document.body.dataset.jobStatus = status || 'idle';
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
    resetWorkflow();
  }

  // -------- 上传 --------
  const dz = $('#dropzone');
  const file = $('#file');
  const pick = $('#pick');
  dz.addEventListener('dragenter', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => { dz.classList.remove('dragover'); });
  dz.addEventListener('drop', async e => {
    e.preventDefault(); dz.classList.remove('dragover');
    if (e.dataTransfer.files.length) await upload(e.dataTransfer.files[0]);
  });
  pick.addEventListener('click', () => file.click());
  file.addEventListener('change', async () => { if (file.files.length) await upload(file.files[0]); });

  async function upload(f) {
    dz.classList.add('is-busy');
    dz.classList.remove('has-file');
    $('#uploadResult').textContent = `正在上传：${f.name}…`;
    try {
      const fd = new FormData();
      fd.append('file', f);
      const resp = await api('/api/jobs/create', { method: 'POST', body: fd });
      if (!resp.job_id) {
        $('#uploadResult').textContent = `上传失败：${resp.detail || '服务端未返回 Job ID'}`;
        showToast('上传失败', 'error');
        setWorkflowState({ upload: 'error' });
        enableRun(false);
        return;
      }
      jobId = resp.job_id;
      dz.classList.add('has-file');
      $('#uploadResult').textContent = `已上传：${f.name}，Job ID = ${jobId}`;
      $('#uploadState').textContent = f.name;
      showToast('文本已上传', 'success');
      enableRun(true);
      enableAfterExtract(false);
      clearPollTimer();
      resetResultUI();
      resetProgressUI();
      setControlButtonsByStatus('idle');
      setWorkflowState({ upload: 'done', extract: 'active' });
    } finally {
      dz.classList.remove('is-busy');
    }
  }

  // -------- 启动提取 --------
  $('#run').addEventListener('click', async () => {
    if (!jobId) return;
    clearPollTimer();
    enableRun(false);
    enableAfterExtract(false);
    setProgressIndeterminate('启动中…');
    setControlButtonsByStatus('running'); // 预设按钮状态
    setWorkflowState({ upload: 'done', extract: 'active' });

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

    if (!startResp.ok) {
      const detail = startResp.detail || startResp.error || `HTTP ${startResp.http_status || ''}`;
      setStatus(`启动失败：${detail}`, true);
      showToast('启动失败', 'error');
      setWorkflowState({ upload: 'done', extract: 'error' });
      enableRun(true);
      setControlButtonsByStatus('failed');
      return;
    }

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
    setStatus('');
    if (etaEl) etaEl.textContent = '';
  }

  function setProgressIndeterminate(text) {
    $('#bar').style.display = 'block';
    $('#bar').removeAttribute('value'); // 不确定态
    setStatus(text || '准备中…');
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
    setStatus(parts.join(' · '));

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
      const isPaused = (status === 'paused');
      const isCancelling = (status === 'cancelling');
      const isCancelled = (status === 'cancelled' || stage === 'cancelled');

      if (isPaused) {
        setProgressIndeterminate(stageText || '已暂停');
        // 继续轮询，等待恢复/取消
        pollTimer = setTimeout(poll, 1200);
        return;
      }
      if (isCancelling) {
        setProgressIndeterminate(stageText || '正在取消…');
        pollTimer = setTimeout(poll, 1200);
        return;
      }
      if (isCancelled) {
      setStatus(stageText || '已取消');
      showToast('作业已取消', 'error');
        setWorkflowState({ upload: 'done', extract: 'error' });
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

      // 收敛条件：失败优先于成功，避免 failed + processed>=total 被误判为成功
      const isFailed = (status === 'failed' || stage === 'failed');
      const isSucceeded = (status === 'succeeded' || stage === 'done') && !isFailed && !isCancelled;

      if (isFailed) {
        setStatus('失败：' + (message || ''), true);
        showToast('抽取失败', 'error');
        setWorkflowState({ upload: 'done', extract: 'error' });
        enableAfterExtract(false);
        enableRun(true);
        setControlButtonsByStatus('failed');
        clearPollTimer();
        return;
      }

      if (isSucceeded) {
        // 成功收尾
        setStatus('完成');
        showToast('抽取完成', 'success');
        setWorkflowState({ upload: 'done', extract: 'done', validate: 'active' });
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
      if (job) renderQuality(job);
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

  function topReasons(...summaries) {
    const merged = {};
    summaries.forEach(summary => {
      const reasons = summary && summary.by_reason ? summary.by_reason : {};
      Object.entries(reasons).forEach(([reason, count]) => {
        merged[reason] = (merged[reason] || 0) + Number(count || 0);
      });
    });
    return Object.entries(merged).sort((a, b) => b[1] - a[1]).slice(0, 6);
  }

  function setDownload(id, href, enabled) {
    const el = $(id);
    if (!el) return;
    if (href) el.href = href;
    el.toggleAttribute('disabled', !enabled);
  }

  function renderQuality(job) {
    const card = $('#qualityCard');
    if (!card) return;
    const extraction = job.quality_summary || {};
    const validation = job.validation_summary || {};
    const pair = job.pair_quality_summary || {};
    const hasQuality = Boolean(
      extraction.events || validation.error_count || pair.events ||
      (job.artifacts && (job.artifacts.quality_report || job.artifacts.validation_report || job.artifacts.pair_diagnostics))
    );
    if (!hasQuality) return;

    card.style.display = 'block';
    $('#qualityEvents').textContent = nf.format(extraction.events || 0);
    $('#validationIssues').textContent = nf.format(validation.error_count || 0);
    $('#pairQualityEvents').textContent = nf.format(pair.events || 0);

    const reasons = topReasons(extraction, pair);
    const reasonsEl = $('#qualityReasons');
    if (reasonsEl) {
      reasonsEl.innerHTML = reasons.map(([reason, count]) => `
        <div class="quality-reason">
          <div class="quality-reason-name" title="${escapeHtml(reason)}">${escapeHtml(reason)}</div>
          <div class="quality-reason-count">${nf.format(count)}</div>
        </div>
      `).join('');
    }

    const artifacts = job.artifacts || {};
    setDownload('#downloadQuality', `/api/jobs/${jobId}/download?which=quality_report`, Boolean(artifacts.quality_report));
    setDownload('#downloadValidationReport', `/api/jobs/${jobId}/download?which=validation_report`, Boolean(artifacts.validation_report));
    setDownload('#downloadPairDiagnostics', `/api/jobs/${jobId}/download?which=pair_diagnostics`, Boolean(artifacts.pair_diagnostics));
  }

  // -------- 校验 --------
  $('#validate').addEventListener('click', async () => {
    const res = await api('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: jobId })
    });
    if (res.ok) {
      $('#validateLog').textContent = '校验通过\n' + (res.log || '');
      showToast('校验通过', 'success');
      const job = await api(`/api/jobs/${jobId}`);
      renderQuality(job);
      setWorkflowState({ upload: 'done', extract: 'done', validate: 'done', pairs: 'active' });
    } else {
      $('#validateLog').textContent = '校验未通过，请检查日志\n' + (res.log || res.detail || '');
      showToast('校验未通过', 'error');
      const job = await api(`/api/jobs/${jobId}`);
      renderQuality(job);
      setWorkflowState({ upload: 'done', extract: 'done', validate: 'error' });
    }
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
      const job = await api(`/api/jobs/${jobId}`);
      renderQuality(job);
      setWorkflowState({ upload: 'done', extract: 'done', validate: 'done', pairs: 'done', chatml: 'active' });
    } else {
      setWorkflowState({ upload: 'done', extract: 'done', pairs: 'error' });
      showToast('Pair 数据集生成失败', 'error');
      alert('生成失败：' + (res.log || res.detail || ''));
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
      showToast('ChatML 已生成', 'success');
      setWorkflowState({ upload: 'done', extract: 'done', validate: 'done', pairs: 'done', chatml: 'done' });
    } else {
      setWorkflowState({ upload: 'done', extract: 'done', pairs: 'done', chatml: 'error' });
      showToast('ChatML 导出失败', 'error');
      alert('导出失败：' + (res.log || res.detail || ''));
    }
  });

  // -------- 启动 --------
  init().catch(err => {
    console.error(err);
    alert('初始化失败：' + err);
  });
})();
