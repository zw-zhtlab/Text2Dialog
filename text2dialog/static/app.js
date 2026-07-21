(() => {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const THEME_KEY = 't2d-theme';
  const ACTIVE_JOB_KEY = 't2d-active-job';
  const PREVIEW_LIMIT = 8;
  const POLL_MAX_FAILURES = 8;
  const POLL_MAX_MS = 6 * 60 * 60 * 1000;

  function storageGet(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  }
  function storageSet(key, value) {
    try { localStorage.setItem(key, value); } catch { /* Storage may be unavailable in privacy modes. */ }
  }
  function storageRemove(key) {
    try { localStorage.removeItem(key); } catch { /* Storage may be unavailable in privacy modes. */ }
  }
  function initTheme() {
    const themes = ['dark', 'light-doc'];
    const saved = storageGet(THEME_KEY);
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = themes.includes(saved) ? saved : (prefersDark ? 'dark' : 'light-doc');
    const apply = value => {
      const selected = themes.includes(value) ? value : 'light-doc';
      document.documentElement.dataset.theme = selected;
      document.documentElement.style.colorScheme = selected === 'dark' ? 'dark' : 'light';
    };
    const select = $('#themeSelect');
    apply(theme);
    storageSet(THEME_KEY, theme);
    if (select) {
      select.value = theme;
      select.addEventListener('change', () => {
        storageSet(THEME_KEY, select.value);
        apply(select.value);
      });
    }
  }

  let serverAccessToken = '';

  const api = async (p, opt = {}) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), opt.timeoutMs || 30000);
    try {
      const headers = new Headers(opt.headers || {});
      if (serverAccessToken && !headers.has('Authorization') && !headers.has('X-API-Key')) {
        headers.set('Authorization', `Bearer ${serverAccessToken}`);
      }
      const requestOptions = { ...opt, headers, signal: opt.signal || controller.signal };
      delete requestOptions.timeoutMs;
      const r = await fetch(p, requestOptions);
      const text = await r.text();
      let data = {};
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = { detail: text.slice(0, 1000) || `HTTP ${r.status}` };
        }
      }
      if (!r.ok) {
        return { ...(data && typeof data === 'object' ? data : {}), ok: false, http_status: r.status };
      }
      if (data && typeof data === 'object' && data.ok === false) return data;
      return data && typeof data === 'object' ? data : {};
    } catch (err) {
      const detail = err && err.name === 'AbortError'
        ? '请求超时'
        : (err && err.message ? err.message : '请求失败');
      return { ok: false, detail, network_error: true };
    } finally {
      clearTimeout(timeout);
    }
  };

  let jobId = null;
  let defaults = null;
  let capabilities = null;
  let pollTimer = null;
  let requestGeneration = 0;

  // -------- 工具函数 --------
  function enableRun(enable) { $('#run').disabled = !enable; }
  function enableAfterExtract(enable) {
    $('#validate').disabled = !enable;
    // Pair generation is gated by validation_summary.ok in applyJobArtifacts.
    $('#buildPairs').disabled = true;
    if (!enable) $('#buildChatML').disabled = true;
    if (!enable) {
      setDownload('#downloadExtract', '', false);
      setDownload('#downloadQuality', '', false);
    }
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
      if (state === 'active') step.setAttribute('aria-current', 'step');
      else step.removeAttribute('aria-current');
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
    ['#run', '#validate', '#buildPairs', '#buildChatML'].forEach(sel => {
      const button = $(sel);
      if (button) {
        button.removeAttribute('aria-busy');
        delete button.dataset.wasDisabled;
      }
    });
    ['#statsCard', '#qualityCard'].forEach(sel => {
      const el = $(sel);
      if (el) el.style.display = 'none';
    });
    ['#preview', '#validateLog', '#roleBars', '#qualityReasons'].forEach(sel => {
      const el = $(sel);
      if (el) el.textContent = '';
    });
    ['#downloadPairs', '#downloadChatML', '#downloadValidationReport', '#downloadPairDiagnostics'].forEach(sel => {
      setDownload(sel, '', false);
    });
  }
  function clearPollTimer() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }
  function currentRequest(expectedJobId, expectedGeneration) {
    return jobId === expectedJobId && requestGeneration === expectedGeneration;
  }
  function setBusy(button, busy) {
    if (!button) return;
    if (busy) {
      button.dataset.wasDisabled = String(button.disabled);
      button.disabled = true;
      button.setAttribute('aria-busy', 'true');
    } else {
      const wasDisabled = button.dataset.wasDisabled === 'true';
      button.disabled = wasDisabled;
      button.removeAttribute('aria-busy');
      delete button.dataset.wasDisabled;
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
  function rememberJob(id) {
    if (!id) return;
    storageSet(ACTIVE_JOB_KEY, id);
    const forgetButton = $('#forgetJob');
    if (forgetButton) forgetButton.hidden = false;
    const url = new URL(window.location.href);
    url.searchParams.set('job', id);
    window.history.replaceState(null, '', url);
  }
  function forgetJob() {
    storageRemove(ACTIVE_JOB_KEY);
    const forgetButton = $('#forgetJob');
    if (forgetButton) forgetButton.hidden = true;
    const url = new URL(window.location.href);
    url.searchParams.delete('job');
    window.history.replaceState(null, '', url);
    clearPollTimer();
    requestGeneration += 1;
    jobId = null;
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
    [defaults, capabilities] = await Promise.all([
      api('/api/defaults'),
      api('/api/capabilities'),
    ]);
    if (defaults && defaults.ok === false) {
      throw new Error(defaults.detail || '无法加载服务端默认配置');
    }
    if (capabilities && capabilities.ok === false) capabilities = null;
    const versionElement = $('#appVersion');
    if (versionElement && capabilities && capabilities.version) {
      versionElement.textContent = String(capabilities.version);
    }
    const platformSel = $('#platform');
    platformSel.innerHTML = '';
    if (defaults && defaults.platforms) {
      for (const [k, v] of Object.entries(defaults.platforms)) {
        const opt = document.createElement('option');
        opt.value = k; opt.textContent = `${k}（${v}）`;
        platformSel.appendChild(opt);
      }
    }
    if (defaults && defaults.current_platform && Array.from(platformSel.options).some(opt => opt.value === defaults.current_platform)) {
      platformSel.value = defaults.current_platform;
    }
    if (defaults && defaults.config) {
      const cfg = defaults.config;
      $('#concurrent').value = String(Boolean(cfg.DEFAULT_CONCURRENT));
      $('#threads').value = cfg.MAX_WORKERS ?? '';
      $('#temperature').value = cfg.TEMPERATURE ?? '';
      $('#maxTokenLen').value = cfg.MAX_TOKEN_LEN ?? '';
      $('#coverContent').value = cfg.COVER_CONTENT ?? '';
      $('#saveChunkText').value = String(Boolean(cfg.SAVE_CHUNK_TEXT));
      $('#sortOutput').value = String(Boolean(cfg.DEFAULT_SORT_OUTPUT));
      $('#replyWindow').value = cfg.REPLY_WINDOW ?? '';
      $('#replyConf').value = cfg.REPLY_CONFIDENCE_TH ?? '';
    }
    // 初始禁用控制按钮
    setControlButtonsByStatus('idle');
    resetWorkflow();
    await restoreJob();
  }

  async function restoreJob() {
    const urlJobId = new URLSearchParams(window.location.search).get('job');
    const savedJobId = urlJobId || storageGet(ACTIVE_JOB_KEY);
    if (!savedJobId || !/^[0-9a-f]{12}$/i.test(savedJobId)) {
      forgetJob();
      return;
    }
    const restoreGeneration = ++requestGeneration;
    const job = await api(`/api/jobs/${savedJobId}`);
    if (requestGeneration !== restoreGeneration) return;
    if (!job || job.ok === false || !job.id) {
      forgetJob();
      showToast('上次作业已不可用，请重新上传', 'info');
      return;
    }

    jobId = job.id;
    rememberJob(jobId);
    $('#uploadState').textContent = '已恢复';
    $('#uploadResult').textContent = `已恢复 Job ID = ${jobId}`;
    dz.classList.add('has-file');
    resetResultUI();
    applyJobArtifacts(job);

    const status = job.status || 'created';
    if (status === 'succeeded') {
      await showCompletedJob(job, false);
    } else if (['running', 'paused', 'cancelling'].includes(status)) {
      enableRun(false);
      setWorkflowState({ upload: 'done', extract: 'active' });
      setControlButtonsByStatus(status);
      setProgressIndeterminate(job.message || stageMap[status] || '恢复作业…');
      poll(jobId, restoreGeneration);
    } else if (status === 'failed' || status === 'cancelled') {
      enableRun(true);
      enableAfterExtract(false);
      setControlButtonsByStatus(status);
      setStatus(job.message || stageMap[status], status === 'failed');
      setWorkflowState({ upload: 'done', extract: 'error' });
    } else {
      enableRun(true);
      enableAfterExtract(false);
      setWorkflowState({ upload: 'done', extract: 'active' });
    }
  }

  function applyJobArtifacts(job) {
    const artifacts = job.artifacts || {};
    const validationOk = Boolean(job.validation_summary && job.validation_summary.ok === true && artifacts.validated);
    setDownload('#downloadExtract', `/api/jobs/${jobId}/download?which=extraction`, Boolean(artifacts.extraction));
    setDownload('#downloadQuality', `/api/jobs/${jobId}/download?which=quality_report`, Boolean(artifacts.quality_report));
    setDownload('#downloadValidationReport', `/api/jobs/${jobId}/download?which=validation_report`, Boolean(artifacts.validation_report));
    setDownload('#downloadPairDiagnostics', `/api/jobs/${jobId}/download?which=pair_diagnostics`, Boolean(artifacts.pair_diagnostics));
    setDownload('#downloadPairs', `/api/jobs/${jobId}/download?which=pairs_zip`, Boolean(artifacts.pairs_zip));
    setDownload('#downloadChatML', `/api/jobs/${jobId}/download?which=chatml`, Boolean(artifacts.chatml));
    $('#buildPairs').disabled = !validationOk;
    $('#buildChatML').disabled = !artifacts.pairs_zip;
  }

  async function showCompletedJob(job, announce = true) {
    const completedJobId = jobId;
    const completedGeneration = requestGeneration;
    enableAfterExtract(true);
    enableRun(true);
    setControlButtonsByStatus('succeeded');
    applyJobArtifacts(job);
    const artifacts = job.artifacts || {};
    const validationOk = Boolean(job.validation_summary && job.validation_summary.ok === true && artifacts.validated);
    const states = { upload: 'done', extract: 'done', validate: 'active' };
    if (validationOk) {
      states.validate = 'done';
      states.pairs = 'active';
    } else if (job.validation_status === 'failed') {
      states.validate = 'error';
    }
    if (artifacts.pairs_zip) {
      states.validate = 'done';
      states.pairs = 'done';
      states.chatml = 'active';
    }
    if (artifacts.chatml) states.chatml = 'done';
    setWorkflowState(states);
    setStatus('完成');
    await preview(completedJobId, completedGeneration);
    if (!currentRequest(completedJobId, completedGeneration)) return;
    if (job.stats) renderStats(job.stats);
    renderQuality(job);
    if (announce) showToast('抽取完成', 'success');
  }

  // -------- 上传 --------
  const dz = $('#dropzone');
  const file = $('#file');
  const pick = $('#pick');
  const forgetButton = $('#forgetJob');
  if (forgetButton) {
    forgetButton.addEventListener('click', () => {
      forgetJob();
      dz.classList.remove('has-file', 'is-busy');
      $('#uploadResult').textContent = '';
      $('#uploadState').textContent = '待上传';
      enableRun(false);
      enableAfterExtract(false);
      resetResultUI();
      resetProgressUI();
      resetWorkflow();
    });
  }
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
    const uploadGeneration = ++requestGeneration;
    clearPollTimer();
    dz.classList.add('is-busy');
    dz.classList.remove('has-file');
    $('#uploadResult').textContent = `正在上传：${f.name}…`;
    try {
      const fd = new FormData();
      fd.append('file', f);
      const resp = await api('/api/jobs/create', { method: 'POST', body: fd });
      if (requestGeneration !== uploadGeneration) return;
      if (!resp.job_id) {
        $('#uploadResult').textContent = `上传失败：${resp.detail || '服务端未返回 Job ID'}`;
        showToast('上传失败', 'error');
        setWorkflowState({ upload: 'error' });
        enableRun(false);
        return;
      }
      jobId = resp.job_id;
      rememberJob(jobId);
      dz.classList.add('has-file');
      $('#uploadResult').textContent = `已上传：${f.name}，Job ID = ${jobId}`;
      $('#uploadState').textContent = f.name;
      showToast('文本已上传', 'success');
      enableRun(true);
      enableAfterExtract(false);
      resetResultUI();
      resetProgressUI();
      setControlButtonsByStatus('idle');
      setWorkflowState({ upload: 'done', extract: 'active' });
    } finally {
      if (requestGeneration === uploadGeneration) dz.classList.remove('is-busy');
    }
  }

  // -------- 启动提取 --------
  $('#extractionForm').addEventListener('submit', async event => {
    event.preventDefault();
    if (!jobId) return;
    const runJobId = jobId;
    const runGeneration = ++requestGeneration;
    const runButton = $('#run');
    setBusy(runButton, true);
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

    const startResp = await api(`/api/jobs/${runJobId}/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!currentRequest(runJobId, runGeneration)) {
      runButton.removeAttribute('aria-busy');
      return;
    }

    if (!startResp.ok) {
      const detail = startResp.detail || startResp.error || `HTTP ${startResp.http_status || ''}`;
      setStatus(`启动失败：${detail}`, true);
      showToast('启动失败', 'error');
      setWorkflowState({ upload: 'done', extract: 'error' });
      enableRun(true);
      setBusy(runButton, false);
      setControlButtonsByStatus('failed');
      return;
    }

    // 显示 PID（若后端返回）
    if (startResp && typeof startResp.pid === 'number') {
      setProgressIndeterminate(`已启动作业（PID=${startResp.pid}）…`);
    } else {
      setProgressIndeterminate('已启动作业…');
    }

    setBusy(runButton, false);
    enableRun(false);
    poll(runJobId, runGeneration); // 开始轮询
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
  function schedulePoll(delay, expectedJobId, expectedGeneration, failures, startedAt) {
    clearPollTimer();
    pollTimer = setTimeout(
      () => poll(expectedJobId, expectedGeneration, failures, startedAt),
      delay,
    );
  }

  async function poll(
    expectedJobId = jobId,
    expectedGeneration = requestGeneration,
    failures = 0,
    startedAt = Date.now(),
  ) {
    if (!currentRequest(expectedJobId, expectedGeneration)) return;
    if (Date.now() - startedAt > POLL_MAX_MS) {
      setStatus('轮询已达时间上限，请刷新作业状态', true);
      clearPollTimer();
      enableRun(true);
      return;
    }

    try {
      const info = await api(`/api/jobs/${expectedJobId}/progress`);
      if (!currentRequest(expectedJobId, expectedGeneration)) return;
      if (!info || info.ok === false) {
        const nextFailures = failures + 1;
        if (nextFailures >= POLL_MAX_FAILURES) {
          setStatus(`轮询停止：${info && info.detail ? info.detail : '服务不可用'}`, true);
          showToast('无法获取作业状态', 'error');
          clearPollTimer();
          enableRun(true);
          return;
        }
        schedulePoll(Math.min(5000, 1000 + nextFailures * 500), expectedJobId, expectedGeneration, nextFailures, startedAt);
        return;
      }
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
        schedulePoll(1200, expectedJobId, expectedGeneration, 0, startedAt);
        return;
      }
      if (isCancelling) {
        setProgressIndeterminate(stageText || '正在取消…');
        schedulePoll(1200, expectedJobId, expectedGeneration, 0, startedAt);
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
      const isSucceeded = status === 'succeeded' && !isFailed && !isCancelled;

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
        const completedJob = await api(`/api/jobs/${expectedJobId}`);
        if (!currentRequest(expectedJobId, expectedGeneration) || completedJob.ok === false) return;
        await showCompletedJob(completedJob, true);
        clearPollTimer();
        return;
      }

      // 继续轮询
      schedulePoll(1200, expectedJobId, expectedGeneration, 0, startedAt);
    } catch (err) {
      console.error('轮询进度失败：', err);
      // 轻量回退：继续尝试轮询
      const nextFailures = failures + 1;
      if (nextFailures < POLL_MAX_FAILURES && currentRequest(expectedJobId, expectedGeneration)) {
        schedulePoll(1500, expectedJobId, expectedGeneration, nextFailures, startedAt);
      } else {
        setStatus('轮询失败次数过多，请刷新后重试', true);
        clearPollTimer();
        enableRun(true);
      }
    }
  }

  // -------- 预览 --------
  async function preview(
    expectedJobId = jobId,
    expectedGeneration = requestGeneration,
  ) {
    try {
      const query = new URLSearchParams({ limit: String(PREVIEW_LIMIT) });
      const resp = await api(`/api/jobs/${expectedJobId}/preview?${query.toString()}`);
      if (!currentRequest(expectedJobId, expectedGeneration)) return;
      if (!resp || resp.ok === false) throw new Error(resp && (resp.detail || resp.error) || '预览不可用');
      const lines = Array.isArray(resp.items) ? resp.items : [];
      const previewEl = $('#preview');
      if (!previewEl) return;
      previewEl.textContent = '';

      const frag = document.createDocumentFragment();
      const hint = document.createElement('div');
      hint.className = 'hint';
      const truncated = !Array.isArray(resp) && Boolean(resp.truncated);
      hint.textContent = `前 ${lines.length} 条${truncated ? '（预览已截断）' : ''}：`;
      frag.appendChild(hint);

      lines.forEach((line, i) => {
        const pre = document.createElement('pre');
        let content = typeof line === 'string' ? line : JSON.stringify(line, null, 2);
        if (typeof line === 'string') {
          try { content = JSON.stringify(JSON.parse(line), null, 2); } catch { /* Keep malformed records readable. */ }
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
    if (enabled && href) {
      el.href = href;
      el.setAttribute('aria-disabled', 'false');
      el.removeAttribute('tabindex');
    } else {
      el.removeAttribute('href');
      el.setAttribute('aria-disabled', 'true');
      el.setAttribute('tabindex', '-1');
    }
  }

  async function downloadWithAccessToken(event) {
    const link = event.currentTarget;
    const href = link && link.getAttribute('href');
    if (!serverAccessToken || !href) return;
    event.preventDefault();
    link.setAttribute('aria-busy', 'true');
    try {
      const response = await fetch(href, {
        headers: { Authorization: `Bearer ${serverAccessToken}` },
      });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          detail = body.detail || detail;
        } catch { /* Keep the bounded HTTP status fallback. */ }
        throw new Error(detail);
      }
      const blob = await response.blob();
      const disposition = response.headers.get('content-disposition') || '';
      const encodedName = disposition.match(/filename\*=UTF-8''([^;]+)/i);
      const plainName = disposition.match(/filename="?([^";]+)"?/i);
      let filename = plainName ? plainName[1] : 'text2dialog-artifact';
      if (encodedName) {
        try { filename = decodeURIComponent(encodedName[1]); } catch { /* Use fallback. */ }
      }
      const objectUrl = URL.createObjectURL(blob);
      const temporary = document.createElement('a');
      temporary.href = objectUrl;
      temporary.download = filename;
      document.body.appendChild(temporary);
      temporary.click();
      temporary.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      showToast(`下载失败：${err && err.message ? err.message : '请求失败'}`, 'error');
    } finally {
      link.removeAttribute('aria-busy');
    }
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
    if (!jobId) return;
    const expectedJobId = jobId;
    const expectedGeneration = requestGeneration;
    const validateButton = $('#validate');
    setBusy(validateButton, true);
    const res = await api('/api/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: expectedJobId })
    });
    if (!currentRequest(expectedJobId, expectedGeneration)) {
      setBusy(validateButton, false);
      return;
    }
    if (res.ok) {
      $('#validateLog').textContent = '校验通过\n' + (res.log || '');
      showToast('校验通过', 'success');
      const job = await api(`/api/jobs/${expectedJobId}`);
      if (!currentRequest(expectedJobId, expectedGeneration)) {
        setBusy(validateButton, false);
        return;
      }
      if (!job.validation_summary || job.validation_summary.ok !== true || !(job.artifacts || {}).validated) {
        $('#buildPairs').disabled = true;
        setWorkflowState({ upload: 'done', extract: 'done', validate: 'error' });
        showToast('服务端未确认验证成功', 'error');
        setBusy(validateButton, false);
        return;
      }
      renderQuality(job);
      applyJobArtifacts(job);
      setWorkflowState({ upload: 'done', extract: 'done', validate: 'done', pairs: 'active' });
    } else {
      $('#validateLog').textContent = '校验未通过，请检查日志\n' + (res.log || res.detail || '');
      showToast('校验未通过', 'error');
      const job = await api(`/api/jobs/${expectedJobId}`);
      if (!currentRequest(expectedJobId, expectedGeneration)) {
        setBusy(validateButton, false);
        return;
      }
      renderQuality(job);
      applyJobArtifacts(job);
      setWorkflowState({ upload: 'done', extract: 'done', validate: 'error' });
    }
    setBusy(validateButton, false);
  });

  // -------- 生成 Pair 数据集 --------
  $('#buildPairs').addEventListener('click', async () => {
    if (!jobId) return;
    const expectedJobId = jobId;
    const expectedGeneration = requestGeneration;
    const pairsButton = $('#buildPairs');
    setBusy(pairsButton, true);
    const authoritativeJob = await api(`/api/jobs/${expectedJobId}`);
    if (!currentRequest(expectedJobId, expectedGeneration)) {
      setBusy(pairsButton, false);
      return;
    }
    if (
      authoritativeJob.ok === false
      || !authoritativeJob.validation_summary
      || authoritativeJob.validation_summary.ok !== true
      || !(authoritativeJob.artifacts || {}).validated
    ) {
      showToast('必须先通过验证', 'error');
      applyJobArtifacts(authoritativeJob || {});
      setBusy(pairsButton, false);
      return;
    }
    const pairStr = $('#pairs').value.trim();
    const pairs = pairStr ? pairStr.split('|').map(s => s.trim()).filter(Boolean) : null;
    const expandsAllRoles = !pairs || pairs.some(item => ['ALL', '全部'].includes(item.toUpperCase()));
    if (expandsAllRoles) {
      const rolesResponse = await api(`/api/roles?job_id=${encodeURIComponent(expectedJobId)}`);
      if (!currentRequest(expectedJobId, expectedGeneration)) {
        setBusy(pairsButton, false);
        return;
      }
      const roleCount = Array.isArray(rolesResponse.roles) ? rolesResponse.roles.length : 0;
      const estimatedPairs = roleCount * Math.max(0, roleCount - 1);
      const maxPairs = capabilities && capabilities.limits
        ? Number(capabilities.limits.pair_combinations || 256)
        : 256;
      if (estimatedPairs > maxPairs) {
        alert(`角色展开将生成 ${estimatedPairs} 个有序对，超过服务端上限 ${maxPairs}。`);
        setBusy(pairsButton, false);
        return;
      }
      if (!confirm(`将按 ${roleCount} 个角色生成最多 ${estimatedPairs} 个有序对，是否继续？`)) {
        setBusy(pairsButton, false);
        return;
      }
    }
    const req = {
      job_id: expectedJobId,
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
    if (!currentRequest(expectedJobId, expectedGeneration)) {
      setBusy(pairsButton, false);
      return;
    }
    if (res.ok) {
      $('#buildChatML').disabled = false; // 允许导出 ChatML
      const job = await api(`/api/jobs/${expectedJobId}`);
      if (!currentRequest(expectedJobId, expectedGeneration)) {
        setBusy(pairsButton, false);
        return;
      }
      renderQuality(job);
      applyJobArtifacts(job);
      setWorkflowState({ upload: 'done', extract: 'done', validate: 'done', pairs: 'done', chatml: 'active' });
      setBusy(pairsButton, false);
    } else {
      setWorkflowState({ upload: 'done', extract: 'done', pairs: 'error' });
      setBusy(pairsButton, false);
      showToast('Pair 数据集生成失败', 'error');
      alert('生成失败：' + (res.log || res.detail || ''));
    }
  });

  // -------- 导出 ChatML --------
  $('#buildChatML').addEventListener('click', async () => {
    if (!jobId) return;
    const expectedJobId = jobId;
    const expectedGeneration = requestGeneration;
    const chatmlButton = $('#buildChatML');
    setBusy(chatmlButton, true);
    const req = {
      job_id: expectedJobId,
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
    if (!currentRequest(expectedJobId, expectedGeneration)) {
      setBusy(chatmlButton, false);
      return;
    }
    if (res.ok) {
      const job = await api(`/api/jobs/${expectedJobId}`);
      if (!currentRequest(expectedJobId, expectedGeneration)) {
        setBusy(chatmlButton, false);
        return;
      }
      applyJobArtifacts(job);
      showToast('ChatML 已生成', 'success');
      setWorkflowState({ upload: 'done', extract: 'done', validate: 'done', pairs: 'done', chatml: 'done' });
      setBusy(chatmlButton, false);
    } else {
      setWorkflowState({ upload: 'done', extract: 'done', pairs: 'done', chatml: 'error' });
      setBusy(chatmlButton, false);
      showToast('ChatML 导出失败', 'error');
      alert('导出失败：' + (res.log || res.detail || ''));
    }
  });

  // -------- 启动 --------
  function syncReverseAvailability() {
    const pairMode = $('#mode').value === 'pair';
    $('#reverse').disabled = pairMode;
    if (pairMode) $('#reverse').value = 'false';
  }
  $('#mode').addEventListener('change', syncReverseAvailability);
  syncReverseAvailability();

  async function connectServer() {
    const tokenInput = $('#serverAccessToken');
    const connectButton = $('#reconnectServer');
    const connectionState = $('#serverConnectionState');
    serverAccessToken = tokenInput ? tokenInput.value.trim() : '';
    setBusy(connectButton, true);
    if (connectionState) connectionState.textContent = '连接中…';
    try {
      await init();
      if (connectionState) connectionState.textContent = '已连接';
    } catch (err) {
      const detail = err && err.message ? err.message : '服务不可用';
      if (connectionState) connectionState.textContent = '需要有效的服务访问令牌';
      showToast(`连接失败：${detail}`, 'error');
    } finally {
      setBusy(connectButton, false);
    }
  }
  const serverTokenInput = $('#serverAccessToken');
  if (serverTokenInput) {
    serverTokenInput.addEventListener('input', () => {
      serverAccessToken = serverTokenInput.value.trim();
    });
  }
  $('#serverAccessForm').addEventListener('submit', event => {
    event.preventDefault();
    connectServer();
  });
  [
    '#downloadExtract', '#downloadQuality', '#downloadValidationReport',
    '#downloadPairDiagnostics', '#downloadPairs', '#downloadChatML',
  ].forEach(selector => {
    const link = $(selector);
    if (link) link.addEventListener('click', downloadWithAccessToken);
  });

  initTheme();
  connectServer();
})();
