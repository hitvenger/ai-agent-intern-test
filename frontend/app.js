/**
 * Aster & Row AI Customer Support — Demo Frontend Application
 * Lightweight client for FastAPI /chat endpoint
 */

(function () {
  'use strict';

  // Determine API base URL dynamically
  const API_BASE = (window.location.protocol.startsWith('http'))
    ? window.location.origin
    : 'http://127.0.0.1:8000';

  // Application State
  let sessionId = generateSessionId();
  let isDebugActive = false;
  let isSubmitting = false;

  // DOM References
  const chatMain = document.getElementById('chatMain');
  const welcomeState = document.getElementById('welcomeState');
  const messagesStream = document.getElementById('messagesStream');
  const typingIndicator = document.getElementById('typingIndicator');
  const chatForm = document.getElementById('chatForm');
  const messageInput = document.getElementById('messageInput');
  const sendBtn = document.getElementById('sendBtn');
  const debugToggleBtn = document.getElementById('debugToggleBtn');
  const debugBtnText = document.getElementById('debugBtnText');
  const newChatBtn = document.getElementById('newChatBtn');
  const sessionDisplay = document.getElementById('sessionDisplay');
  const errorBanner = document.getElementById('errorBanner');
  const errorMessage = document.getElementById('errorMessage');
  const errorCloseBtn = document.getElementById('errorCloseBtn');

  function init() {
    updateSessionUI();
    bindEvents();
    checkHealth();
  }

  function generateSessionId() {
    return 'demo_' + Date.now().toString(36) + '_' + Math.random().toString(36).substring(2, 6);
  }

  function updateSessionUI() {
    if (sessionDisplay) {
      sessionDisplay.textContent = `Session: ${sessionId}`;
    }
  }

  function bindEvents() {
    // Form submission
    chatForm.addEventListener('submit', function (e) {
      e.preventDefault();
      sendMessage();
    });

    // Enter to submit, Shift+Enter for new line
    messageInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    // Textarea auto-height
    messageInput.addEventListener('input', function () {
      messageInput.style.height = 'auto';
      messageInput.style.height = Math.min(messageInput.scrollHeight, 100) + 'px';
    });

    // Debug toggle
    debugToggleBtn.addEventListener('click', function () {
      isDebugActive = !isDebugActive;
      debugToggleBtn.classList.toggle('active', isDebugActive);
      debugBtnText.textContent = isDebugActive ? 'Debug: On' : 'Debug: Off';

      // Update all existing debug boxes
      document.querySelectorAll('.msg-debug-box').forEach(function (box) {
        box.style.display = isDebugActive ? 'block' : 'none';
      });
    });

    // New Chat button
    newChatBtn.addEventListener('click', function () {
      sessionId = generateSessionId();
      updateSessionUI();
      messagesStream.innerHTML = '';
      welcomeState.style.display = 'block';
      hideError();
      messageInput.value = '';
      messageInput.style.height = 'auto';
      messageInput.focus();
    });

    // Welcome cards & Quick chips prompt selection
    document.querySelectorAll('[data-prompt]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        const prompt = btn.getAttribute('data-prompt');
        if (prompt) {
          messageInput.value = prompt;
          sendMessage();
        }
      });
    });

    // Error banner dismiss
    if (errorCloseBtn) {
      errorCloseBtn.addEventListener('click', hideError);
    }
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, { method: 'GET' });
      if (!res.ok) {
        showError('FastAPI backend health check returned a non-200 response.');
      }
    } catch (err) {
      showError('Unable to reach the support agent. Make sure the FastAPI server is running (`uvicorn app.api.server:app --reload`).');
    }
  }

  function showError(msg) {
    if (errorMessage && errorBanner) {
      errorMessage.textContent = msg;
      errorBanner.style.display = 'flex';
    }
  }

  function hideError() {
    if (errorBanner) {
      errorBanner.style.display = 'none';
    }
  }

  async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isSubmitting) return;

    hideError();
    welcomeState.style.display = 'none';

    // 1. Render User Message
    appendUserBubble(text);

    // 2. Clear input & set loading state
    messageInput.value = '';
    messageInput.style.height = 'auto';
    setLoadingState(true);
    showTyping();
    scrollToBottom();

    // 3. Request API
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: text,
          session_id: sessionId
        })
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server status ${response.status}`);
      }

      const data = await response.json();
      hideTyping();
      appendAgentCard(data);
    } catch (err) {
      hideTyping();
      showError(`Request failed: ${err.message}`);
      appendAgentCard({
        answer: 'Sorry, I was unable to process your request. Please ensure the backend server is running and try again.',
        sources: [],
        handoff_recommended: false
      });
    } finally {
      setLoadingState(false);
      scrollToBottom();
      messageInput.focus();
    }
  }

  function setLoadingState(loading) {
    isSubmitting = loading;
    sendBtn.disabled = loading;
    messageInput.disabled = loading;
  }

  function showTyping() {
    typingIndicator.style.display = 'flex';
    scrollToBottom();
  }

  function hideTyping() {
    typingIndicator.style.display = 'none';
  }

  function scrollToBottom() {
    setTimeout(function () {
      chatMain.scrollTop = chatMain.scrollHeight;
    }, 50);
  }

  function appendUserBubble(text) {
    const group = document.createElement('div');
    group.className = 'msg-group user-group';
    group.innerHTML = `
      <div class="msg-sender-tag">Customer</div>
      <div class="msg-bubble-user">${escapeHtml(text)}</div>
    `;
    messagesStream.appendChild(group);
  }

  function appendAgentCard(data) {
    const group = document.createElement('div');
    group.className = 'msg-group agent-group';

    // Format Markdown
    const formattedAnswer = renderCleanMarkdown(data.answer || '');

    // Render Sources section if citations exist
    let sourcesHtml = '';
    if (data.sources && Array.isArray(data.sources) && data.sources.length > 0) {
      const badges = data.sources.map(src => {
        const parts = src.split('#');
        const file = parts[0] || src;
        const heading = parts[1] ? ` #${parts[1]}` : '';
        return `
          <div class="source-badge" title="${escapeHtml(src)}">
            <span class="source-file">${escapeHtml(file)}</span>
            <span class="source-heading-name">${escapeHtml(heading)}</span>
          </div>
        `;
      }).join('');

      sourcesHtml = `
        <div class="msg-sources-section">
          <div class="sources-heading">Sources</div>
          <div class="sources-chips-list">${badges}</div>
        </div>
      `;
    }

    // Render Handoff Warning Card if recommended
    let handoffHtml = '';
    if (data.handoff_recommended) {
      const reason = data.handoff_reason || 'Human support specialist assistance required.';
      handoffHtml = `
        <div class="msg-handoff-alert">
          <div class="handoff-alert-title">⚠ Human handoff recommended</div>
          <div class="handoff-alert-reason">${escapeHtml(reason)}</div>
        </div>
      `;
    }

    // Render Debug Trace Box
    let debugHtml = '';
    if (data.trace) {
      const t = data.trace;
      const displayStyle = isDebugActive ? 'block' : 'none';

      let toolStr = 'None (Policy Retrieval)';
      if (t.tool_called) {
        const args = t.tool_args ? JSON.stringify(t.tool_args) : '{}';
        toolStr = `${t.tool_called}(${args})`;
      }

      let toolResultRow = '';
      if (t.tool_result) {
        // Safe tool summary (order_id, status, carrier, ETA only)
        const safeSummary = {
          order_id: t.tool_result.order_id,
          status: t.tool_result.status,
          carrier: t.tool_result.carrier,
          estimated_delivery: t.tool_result.estimated_delivery,
          requires_handoff: t.tool_result.requires_handoff
        };
        toolResultRow = `
          <div class="debug-row">
            <span class="debug-label">Tool Result:</span>
            <span class="debug-value">${escapeHtml(JSON.stringify(safeSummary))}</span>
          </div>
        `;
      }

      const citationsStr = (t.retrieved_citations && t.retrieved_citations.length > 0)
        ? t.retrieved_citations.join(', ')
        : 'None';

      const conflictStr = t.conflict_detected
        ? `⚠️ Conflict Detected (${t.conflict_details || 'Care Guide vs Product Card'})`
        : 'False';

      const handoffStr = t.handoff_recommended
        ? `Yes (${t.handoff_reason || 'Escalation required'})`
        : 'No';

      debugHtml = `
        <div class="msg-debug-box" style="display: ${displayStyle};">
          <div class="debug-box-header">
            <span>Execution Trace (Debug)</span>
            <span>${t.latency_ms || 0}ms</span>
          </div>
          <div class="debug-box-body">
            <div class="debug-row">
              <span class="debug-label">Intent:</span>
              <span class="debug-value debug-pill">${escapeHtml(t.intent || 'policy_inquiry')}</span>
            </div>
            <div class="debug-row">
              <span class="debug-label">Tool Call:</span>
              <span class="debug-value">${escapeHtml(toolStr)}</span>
            </div>
            ${toolResultRow}
            <div class="debug-row">
              <span class="debug-label">Retrieved:</span>
              <span class="debug-value">${escapeHtml(citationsStr)}</span>
            </div>
            <div class="debug-row">
              <span class="debug-label">Conflict:</span>
              <span class="debug-value">${escapeHtml(conflictStr)}</span>
            </div>
            <div class="debug-row">
              <span class="debug-label">Handoff:</span>
              <span class="debug-value">${escapeHtml(handoffStr)}</span>
            </div>
          </div>
        </div>
      `;
    }

    group.innerHTML = `
      <div class="msg-sender-tag">Aster &amp; Row</div>
      <div class="msg-card-agent">
        <div class="msg-body-text">${formattedAnswer}</div>
        ${sourcesHtml}
        ${handoffHtml}
        ${debugHtml}
      </div>
    `;

    messagesStream.appendChild(group);
  }

  function renderCleanMarkdown(rawText) {
    if (!rawText) return '';
    let text = escapeHtml(rawText);

    // Strip or convert any markdown heading markers anywhere in text:
    // e.g. "## Breeze Tumbler" -> "<strong>Breeze Tumbler</strong>"
    text = text.replace(/(?:,\s*|:\s*)#+\s+([^\n\r<]+)/g, ': <strong>$1</strong>');
    text = text.replace(/^#+\s+([^\n\r<]+)/gm, '<strong>$1</strong>');
    text = text.replace(/(?:\s+)#+\s+([^\n\r<]+)/g, ' <strong>$1</strong>');
    text = text.replace(/#+\s+/g, '');

    // Bold formatting: **bold** -> <strong>bold</strong>
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Bullet points conversion
    const lines = text.split('\n');
    let inList = false;
    let result = '';

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) {
        if (inList) {
          result += '</ul>';
          inList = false;
        }
        continue;
      }

      if (line.startsWith('• ') || line.startsWith('- ') || line.startsWith('* ')) {
        if (!inList) {
          result += '<ul>';
          inList = true;
        }
        result += `<li>${line.substring(2)}</li>`;
      } else {
        if (inList) {
          result += '</ul>';
          inList = false;
        }
        result += `<p>${line}</p>`;
      }
    }

    if (inList) {
      result += '</ul>';
    }

    return result;
  }


  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Self start
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
