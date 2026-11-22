/* ============================================================
   coPPeR Workspace — app.js
   Handles: SSE streaming, context chips, model selector,
            numbered point-back citations (❶ ❷ ❸), split diff.
   ============================================================ */

(function () {
  "use strict";

  // ---- State ----
  let currentModel = "copper-1b-local";
  let sessionId = Date.now();
  let contextChipData = null;
  let isStreaming = false;

  // ---- DOM refs ----
  const chatStream     = document.getElementById("chat-stream");
  const promptInput    = document.getElementById("prompt-input");
  const btnSend        = document.getElementById("btn-send");
  const btnNewSession  = document.getElementById("btn-new-session");
  const btnApplyToApp  = document.getElementById("btn-apply-to-app");
  const btnFilePicker  = document.getElementById("btn-file-picker");
  const btnSelTool     = document.getElementById("btn-selection-tool");
  const contextArea    = document.getElementById("context-chip-area");
  const modelSelector  = document.getElementById("model-selector");
  const modelDropdown  = document.getElementById("model-dropdown");
  const docContent     = document.getElementById("doc-content");
  const splitDiffView  = document.getElementById("split-diff-view");
  const diffActions    = document.getElementById("diff-actions");
  const btnKeepEdit    = document.getElementById("btn-keep-edit");
  const btnUndo        = document.getElementById("btn-undo");

  // ============================================================
  //  Context Chip Management
  // ============================================================
  function showContextChip(label, data) {
    contextArea.innerHTML = "";
    contextArea.classList.remove("hidden");
    const chip = document.createElement("div");
    chip.className = "context-chip";
    chip.title = label;
    chip.innerHTML = `<span>${label}</span><span class="chip-remove" title="Remove">&times;</span>`;
    chip.querySelector(".chip-remove").addEventListener("click", function (e) {
      e.stopPropagation();
      contextArea.classList.add("hidden");
      contextArea.innerHTML = "";
      contextChipData = null;
    });
    contextArea.appendChild(chip);
    contextChipData = data || label;
  }

  // ============================================================
  //  Model Selector Dropdown
  // ============================================================
  modelSelector.addEventListener("click", function (e) {
    e.stopPropagation();
    modelDropdown.classList.toggle("hidden");
  });
  document.querySelectorAll(".dropdown-item").forEach(function (item) {
    item.addEventListener("click", function () {
      currentModel = this.dataset.model;
      document.getElementById("model-label").textContent = this.textContent.trim();
      document.querySelectorAll(".dropdown-item").forEach(function (i) { i.classList.remove("active"); });
      this.classList.add("active");
      modelDropdown.classList.add("hidden");
    });
  });
  document.addEventListener("click", function () { modelDropdown.classList.add("hidden"); });

  // ============================================================
  //  Citation Badge Interaction
  // ============================================================
  function highlightCitation(num) {
    // Clear previous highlights
    docContent.querySelectorAll(".citation-badge.doc-badge").forEach(function (b) { b.classList.remove("highlighted"); });
    document.querySelectorAll(".citation-badge").forEach(function (b) { b.classList.remove("active"); });

    const block = docContent.querySelector('[data-citation="' + num + '"]');
    if (block) {
      block.scrollIntoView({ behavior: "smooth", block: "center" });
      var badge = block.querySelector(".citation-badge.doc-badge");
      if (badge) badge.classList.add("highlighted");
    }

    // Highlight matching AI citation badges in chat
    document.querySelectorAll('.citation-badge[data-num="' + num + '"]').forEach(function (b) { b.classList.add("active"); });
  }

  // ============================================================
  //  Add User Message Bubble
  // ============================================================
  function addUserBubble(text) {
    var bubble = document.createElement("div");
    bubble.className = "user-bubble";
    bubble.textContent = text;
    chatStream.appendChild(bubble);
    chatStream.scrollTop = chatStream.scrollHeight;
  }

  // ============================================================
  //  Add AI Response Card (with citation badges)
  // ============================================================
  function addAiCard(htmlContent, citations) {
    var card = document.createElement("div");
    card.className = "ai-card";

    // Build content with citation badges
    var innerHtml = htmlContent;
    if (citations && citations.length > 0) {
      innerHtml += '<div class="citation-row" style="margin-top:8px;padding-top:6px;border-top:1px solid #EBF1F8;">';
      citations.forEach(function (num, idx) {
        var badge = document.createElement("span");
        badge.className = "citation-badge";
        badge.setAttribute("data-num", num);
        badge.textContent = String.fromCharCode(0x276A + idx); // ❶ ❷ ❸
        badge.title = "Citation #" + num;
        badge.addEventListener("click", function () { highlightCitation(num); });
        innerHtml += '<span class="citation-badge" data-num="' + num + '">&#x' + (0x276A + idx).toString(16) + ';</span>';
      });
      innerHtml += '</div>';
    }

    card.innerHTML = innerHtml;
    chatStream.appendChild(card);
    chatStream.scrollTop = chatStream.scrollHeight;
    return card;
  }

  // ============================================================
  //  SSE Streaming Client
  // ============================================================
  async function streamChat(messages, maxNewTokens) {
    if (isStreaming) return;
    isStreaming = true;

    var aiCard = addAiCard("", []);
    var textContent = "";
    var citations = [];

    try {
      var resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: messages,
          max_new_tokens: maxNewTokens || 256,
          temperature: 0.7,
          stream: true
        })
      });

      if (!resp.ok) throw new Error("Server error " + resp.status);

      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";

      while (true) {
        var { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        var lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (var i = 0; i < lines.length; i++) {
          var line = lines[i].trim();
          if (!line.startsWith("data: ")) continue;
          var payload = line.slice(6);
          if (payload === "[DONE]") break;

          try {
            var chunk = JSON.parse(payload);
            var delta = chunk.choices && chunk.choices[0] && chunk.choices[0].delta;
            if (delta && delta.content) {
              textContent += delta.content;
              aiCard.innerHTML = textContent + '<div class="citation-row" style="margin-top:8px;padding-top:6px;border-top:1px solid #EBF1F8;">' + citations.map(function(c,idx){ return '<span class="citation-badge" data-num="'+c+'">&#x'+(0x276A+idx).toString(16)+';</span>'; }).join('') + '</div>';
              chatStream.scrollTop = chatStream.scrollHeight;
            }
          } catch (e) { /* skip malformed */ }
        }
      }

      // Update diff view if in split-diff mode
      updateDiffView(textContent);

    } catch (err) {
      aiCard.innerHTML += '<br><span style="color:#DC2626;font-size:12px;">Error: ' + err.message + '</span>';
    } finally {
      isStreaming = false;
    }
  }

  // ============================================================
  //  Send Handler (Enter key + button click)
  // ============================================================
  function sendMessage() {
    var text = promptInput.value.trim();
    if (!text || isStreaming) return;

    addUserBubble(text);
    promptInput.value = "";

    var messages = [];
    if (contextChipData && typeof contextChipData === "string") {
      messages.push({ role: "system", content: "Context document:\n" + contextChipData });
    }
    messages.push({ role: "user", content: text });

    streamChat(messages, 256);
  }

  btnSend.addEventListener("click", sendMessage);
  promptInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  // ============================================================
  //  New Session
  // ============================================================
  btnNewSession.addEventListener("click", function () {
    sessionId = Date.now();
    chatStream.innerHTML = "";
    contextArea.classList.add("hidden");
    contextArea.innerHTML = "";
    contextChipData = null;
    promptInput.value = "";
    document.getElementById("doc-title").textContent = "Untitled";
  });

  // ============================================================
  //  Apply to Active App (calls bridge or API)
  // ============================================================
  btnApplyToApp.addEventListener("click", function () {
    var lastCard = chatStream.querySelector(".ai-card");
    if (!lastCard) return;
    var text = lastCard.textContent.replace(/[❶❷❸]/g, "").trim();

    if (window.WorkspaceBridge && typeof window.WorkspaceBridge.applyToActiveWindow === "function") {
      window.WorkspaceBridge.applyToActiveWindow(text);
    } else {
      fetch("/api/workspace/apply", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      }).catch(function () {});
    }
  });

  // ============================================================
  //  File Picker (calls bridge or API)
  // ============================================================
  btnFilePicker.addEventListener("click", function () {
    if (window.WorkspaceBridge && typeof window.WorkspaceBridge.loadFile === "function") {
      window.WorkspaceBridge.loadFile();
    } else {
      fetch("/api/workspace/capture", { method: "POST" })
        .then(function (r) { return r.json(); })
        .then(function (data) { showContextChip("📄 " + data.filename || "Document", data.content); });
    }
  });

  // ============================================================
  //  Dashed-Square Selection Tool
  // ============================================================
  btnSelTool.addEventListener("click", function () {
    if (window.WorkspaceBridge && typeof window.WorkspaceBridge.captureSelection === "function") {
      window.WorkspaceBridge.captureSelection();
    } else {
      fetch("/api/workspace/capture", { method: "POST" })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          var label = data.selection ? data.selection.substring(0, 40) + (data.selection.length > 40 ? "…" : "") : "Active Selection";
          showContextChip("📄 Active Selection · " + (data.charCount || "?") + " chars", data.selection);
        })
        .catch(function () {});
    }
  });

  // ============================================================
  //  Split Diff View
  // ============================================================
  var pendingEdit = "";

  function updateDiffView(editText) {
    pendingEdit = editText;
    document.getElementById("diff-original").textContent = "Original content would appear here…";
    document.getElementById("diff-proposed").textContent = editText || "(no edit yet)";
  }

  // Mode tabs toggle Document / Split Diff views
  document.querySelectorAll(".tab-btn").forEach(function (tab) {
    tab.addEventListener("click", function () {
      document.querySelectorAll(".tab-btn").forEach(function (t) { t.classList.remove("active"); });
      this.classList.add("active");
      var mode = this.dataset.mode;

      docContent.style.display = (mode === "document") ? "" : "none";
      splitDiffView.classList.toggle("hidden", mode !== "split-diff");
      diffActions.classList.toggle("hidden", mode !== "split-diff");
    });
  });

  // Keep Edit — applies the proposed edit to doc content
  btnKeepEdit.addEventListener("click", function () {
    if (pendingEdit) {
      var block1 = document.getElementById("block-1");
      if (block1) block1.textContent = pendingEdit;
    }
  });

  // Undo — reverts the last edit
  btnUndo.addEventListener("click", function () {
    var block1 = document.getElementById("block-1");
    if (block1 && !pendingEdit) block1.textContent = "Loading document…";
  });

})();
