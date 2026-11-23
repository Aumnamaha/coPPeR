/* ============================================================
   coPPeR Workspace — app.js
   Handles: SSE streaming, context chips, model selector,
            numbered point-back citations (❶ ❷ ❸), split diff,
            Notes Drawer, interactive checklists/tables, export.
   ============================================================ */

(function () {
  "use strict";

  // ---- State ----
  let currentModel = "copper-1b-local";
  let sessionId = Date.now();
  let contextChipData = null;
  let isStreaming = false;
  let currentArtifactSlug = "";
  let currentArtifactType = "note";
  let currentArtifactContent = "";

  // ---- DOM refs ----
  const chatStream       = document.getElementById("chat-stream");
  const promptInput      = document.getElementById("prompt-input");
  const btnSend          = document.getElementById("btn-send");
  const btnNewSession    = document.getElementById("btn-new-session");
  const btnApplyToApp    = document.getElementById("btn-apply-to-app");
  const btnFilePicker    = document.getElementById("btn-file-picker");
  const btnSelTool       = document.getElementById("btn-selection-tool");
  const contextArea      = document.getElementById("context-chip-area");
  const modelSelector    = document.getElementById("model-selector");
  const modelDropdown    = document.getElementById("model-dropdown");
  const docContent       = document.getElementById("doc-content");
  const splitDiffView    = document.getElementById("split-diff-view");
  const diffActions      = document.getElementById("diff-actions");
  const exportActions    = document.getElementById("export-actions");
  const btnKeepEdit      = document.getElementById("btn-keep-edit");
  const btnUndo          = document.getElementById("btn-undo");
  // Notes Drawer
  const btnNotesDrawer   = document.getElementById("btn-notes-drawer");
  const notesDrawer      = document.getElementById("notes-drawer");
  const drawerBackdrop   = document.getElementById("drawer-backdrop");
  const btnCloseDrawer   = document.getElementById("btn-close-drawer");
  const btnNewNote       = document.getElementById("btn-new-note");
  const btnNewChecklist  = document.getElementById("btn-new-checklist");
  const btnNewTable      = document.getElementById("btn-new-table");
  const notesList        = document.getElementById("notes-list");
  // Artifact views
  const checklistView    = document.getElementById("checklist-view");
  const checklistItems   = document.getElementById("checklist-items");
  const tableView        = document.getElementById("table-view");
  const tableGridContainer = document.getElementById("table-grid-container");
  const tableGrid        = document.getElementById("table-grid");
  const btnAddRow        = document.getElementById("btn-add-row");
  const btnAddCol        = document.getElementById("btn-add-col");
  // Export buttons
  const btnExportMd      = document.getElementById("btn-export-md");
  const btnExportDocx    = document.getElementById("btn-export-docx");

  // ============================================================
  //  Context Chip Management
  // ============================================================
  function showContextChip(label, data) {
    contextArea.innerHTML = "";
    contextArea.classList.remove("hidden");
    var chip = document.createElement("div");
    chip.className = "context-chip";
    chip.title = label;
    chip.innerHTML = '<span>' + label + '</span><span class="chip-remove" title="Remove">&times;</span>';
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
    docContent.querySelectorAll(".citation-badge.doc-badge").forEach(function (b) { b.classList.remove("highlighted"); });
    document.querySelectorAll(".citation-badge").forEach(function (b) { b.classList.remove("active"); });

    var block = docContent.querySelector('[data-citation="' + num + '"]');
    if (block) {
      block.scrollIntoView({ behavior: "smooth", block: "center" });
      var badge = block.querySelector(".citation-badge.doc-badge");
      if (badge) badge.classList.add("highlighted");
    }

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

    var innerHtml = htmlContent;
    if (citations && citations.length > 0) {
      innerHtml += '<div class="citation-row" style="margin-top:8px;padding-top:6px;border-top:1px solid #EBF1F8;">';
      citations.forEach(function (num, idx) {
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
        var result = await reader.read();
        if (result.done) break;

        buffer += decoder.decode(result.value, { stream: true });
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
  var currentMode = "document";

  function setViewMode(mode) {
    currentMode = mode;
    docContent.style.display = (mode === "document") ? "" : "none";
    checklistView.classList.toggle("hidden", mode !== "table");
    tableView.classList.toggle("hidden", mode !== "table");
    splitDiffView.classList.toggle("hidden", mode !== "split-diff");

    if (mode === "split-diff") {
      diffActions.classList.remove("hidden");
      exportActions.classList.add("hidden");
    } else if (mode === "table") {
      diffActions.classList.add("hidden");
      exportActions.classList.remove("hidden");
    } else {
      diffActions.classList.add("hidden");
      exportActions.classList.remove("hidden");
    }

    document.querySelectorAll(".tab-btn").forEach(function (t) { t.classList.remove("active"); });
    document.querySelector('.tab-btn[data-mode="' + mode + '"]').classList.add("active");
  }

  document.querySelectorAll(".tab-btn").forEach(function (tab) {
    tab.addEventListener("click", function () {
      setViewMode(this.dataset.mode);
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

  // ============================================================
  //  Notes Drawer (Phase 10)
  // ============================================================
  function openDrawer() {
    notesDrawer.classList.add("open");
    drawerBackdrop.classList.remove("hidden");
    loadArtifactsList();
  }

  function closeDrawer() {
    notesDrawer.classList.remove("open");
    drawerBackdrop.classList.add("hidden");
  }

  btnNotesDrawer.addEventListener("click", openDrawer);
  btnCloseDrawer.addEventListener("click", closeDrawer);
  drawerBackdrop.addEventListener("click", closeDrawer);

  async function loadArtifactsList() {
    try {
      var resp = await fetch("/api/artifacts");
      if (!resp.ok) return;
      var data = await resp.json();
      notesList.innerHTML = "";
      var items = data.artifacts || [];
      if (items.length === 0) {
        notesList.innerHTML = '<p class="empty-hint">No artifacts yet. Create one above!</p>';
        return;
      }
      items.forEach(function (item) {
        var el = document.createElement("div");
        el.className = "note-item";
        var badgeClass = item.type || "note";
        if (!["note", "checklist", "table"].includes(badgeClass)) badgeClass = "note";
        el.innerHTML = '<div class="note-info"><div class="note-title-text">' + escapeHtml(item.title) + '</div><div class="note-meta">' + (item.size_bytes || 0) + ' bytes</div></div>' +
          '<span class="note-type-badge ' + badgeClass + '">' + item.type + '</span>' +
          '<button class="note-delete-btn" title="Delete">&times;</button>';

        el.addEventListener("click", function (e) {
          if (e.target.classList.contains("note-delete-btn")) return;
          loadArtifact(item.slug);
          closeDrawer();
        });

        var delBtn = el.querySelector(".note-delete-btn");
        delBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          deleteArtifact(item.slug, el);
        });

        notesList.appendChild(el);
      });
    } catch (err) { /* silent */ }
  }

  async function loadArtifact(slug) {
    try {
      var resp = await fetch("/api/artifacts/" + slug);
      if (!resp.ok) return;
      var data = await resp.json();
      currentArtifactSlug = data.slug || slug;
      currentArtifactType = data.type || "note";
      currentArtifactContent = data.content || "";

      document.getElementById("doc-title").textContent = data.title || slug;

      // Show content in doc-content
      docContent.style.display = "";
      checklistView.classList.add("hidden");
      tableView.classList.add("hidden");

      if (currentArtifactType === "checklist") {
        renderChecklist(currentArtifactContent);
      } else if (currentArtifactType === "table") {
        renderTableGrid(currentArtifactContent);
      } else {
        docContent.innerHTML = "";
        var lines = currentArtifactContent.split("\n");
        lines.forEach(function(line, idx) {
          if (!line.trim()) return;
          var block = document.createElement("div");
          block.className = "doc-block numbered";
          block.setAttribute("data-citation", (idx + 1).toString());
          var badge = document.createElement("span");
          badge.className = "citation-badge doc-badge";
          badge.textContent = String.fromCharCode(0x276A + ((idx) % 3));
          block.appendChild(badge);
          var p = document.createElement("p");
          p.id = "block-" + (idx + 1);
          // Render headings and paragraphs from markdown
          if (line.startsWith("# ")) {
            p.innerHTML = '<strong>' + escapeHtml(line.slice(2)) + '</strong>';
          } else if (line.startsWith("## ")) {
            p.innerHTML = '<em>' + escapeHtml(line.slice(3)) + '</em>';
          } else {
            p.textContent = line;
          }
          block.appendChild(p);
          docContent.appendChild(block);
        });
      }

      // Switch to document mode view for notes, table mode for checklists/tables
      if (currentArtifactType === "checklist" || currentArtifactType === "table") {
        setViewMode("table");
      } else {
        setViewMode("document");
      }
    } catch (err) { /* silent */ }
  }

  async function deleteArtifact(slug, el) {
    try {
      var resp = await fetch("/api/artifacts/" + slug, { method: "DELETE" });
      if (resp.ok) {
        el.remove();
        loadArtifactsList();
      }
    } catch (err) { /* silent */ }
  }

  async function createArtifact(type) {
    var title = prompt("Enter artifact title:", type === "note" ? "Untitled Note" : type === "checklist" ? "My Checklist" : "Data Table");
    if (!title) return;

    var slug = title.toLowerCase().replace(/[^a-z0-9\s-]/g, "").replace(/\s+/g, "-").replace(/-+/, "-") || ("new-" + Date.now());

    var content = "";
    if (type === "checklist") {
      content = "# My Checklist\n\n- [ ] Item one\n- [ ] Item two\n- [x] Completed item\n";
    } else if (type === "table") {
      content = "| Name | Role | Status |\n| --- | --- | --- |\n| Alice | Engineer | Active |\n| Bob | Designer | Review |\n";
    }

    try {
      var resp = await fetch("/api/artifacts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title, content: content, type: type, slug: slug })
      });
      if (resp.ok) {
        loadArtifactsList();
        loadArtifact(slug);
      }
    } catch (err) { /* silent */ }
  }

  btnNewNote.addEventListener("click", function () { createArtifact("note"); });
  btnNewChecklist.addEventListener("click", function () { createArtifact("checklist"); });
  btnNewTable.addEventListener("click", function () { createArtifact("table"); });

  // ============================================================
  //  Interactive Checklist Rendering (Phase 10)
  // ============================================================
  function renderChecklist(mdContent) {
    checklistView.classList.remove("hidden");
    tableView.classList.add("hidden");
    docContent.style.display = "none";
    checklistItems.innerHTML = "";

    var lines = mdContent.split("\n");
    var checkIndex = 0;
    lines.forEach(function (line) {
      var m = line.match(/^(- \[([ xX])\])(.*)$/);
      if (!m) return;

      var checked = (m[2] === "x" || m[2] === "X");
      var text = m[3].trim();

      var itemEl = document.createElement("div");
      itemEl.className = "checklist-item";
      itemEl.innerHTML = '<span class="checkbox' + (checked ? ' checked' : '') + '">' + (checked ? '&#x2714;' : '') + '</span><span class="checkbox-text">' + escapeHtml(text) + '</span>';

      itemEl.addEventListener("click", function () {
        var cb = this.querySelector(".checkbox");
        var isChecked = cb.classList.toggle("checked");
        if (isChecked) {
          cb.innerHTML = "&#x2714;";
        } else {
          cb.innerHTML = "";
        }
        // Update underlying content and auto-save
        currentArtifactContent = ArtifactStore.toggleChecklist(currentArtifactContent, checkIndex);
        autoSaveArtifact();
      });

      checklistItems.appendChild(itemEl);
      checkIndex++;
    });
  }

  // ============================================================
  //  Interactive Table Grid Rendering (Phase 10)
  // ============================================================
  var _tableHeaders = [];
  var _tableRows = [];

  function renderTableGrid(mdContent) {
    tableView.classList.remove("hidden");
    checklistView.classList.add("hidden");
    docContent.style.display = "none";

    // Parse table using a simple inline parser
    var parsed = ArtifactStore.parseMarkdownTable(mdContent);
    if (!parsed || !parsed.headers.length) {
      // Fallback: create empty table
      _tableHeaders = ["Column 1", "Column 2"];
      _tableRows = [["", ""], ["", ""]];
    } else {
      _tableHeaders = parsed.headers.slice();
      _tableRows = parsed.rows.map(function(r) { return r.slice(); });
    }

    renderTableGridInternal();
  }

  function renderTableGridInternal() {
    if (!tableGrid) return;
    tableGrid.innerHTML = "";

    // Header row
    var thead = document.createElement("thead");
    var headerRow = document.createElement("tr");
    _tableHeaders.forEach(function (h, ci) {
      var th = document.createElement("th");
      th.textContent = h;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    tableGrid.appendChild(thead);

    // Body rows
    var tbody = document.createElement("tbody");
    _tableRows.forEach(function (row, ri) {
      var tr = document.createElement("tr");
      row.forEach(function (val, ci) {
        var td = document.createElement("td");
        var input = document.createElement("input");
        input.type = "text";
        input.value = val || "";
        input.addEventListener("change", function () {
          _tableRows[ri][ci] = this.value;
          autoSaveArtifact();
        });
        td.appendChild(input);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    tableGrid.appendChild(tbody);
  }

  btnAddRow.addEventListener("click", function () {
    var newRow = [];
    for (var i = 0; i < _tableHeaders.length; i++) newRow.push("");
    _tableRows.push(newRow);
    renderTableGridInternal();
    autoSaveArtifact();
  });

  btnAddCol.addEventListener("click", function () {
    var colName = prompt("Column name:", "New Column");
    if (!colName) return;
    _tableHeaders.push(colName);
    _tableRows.forEach(function (row) { row.push(""); });
    renderTableGridInternal();
    autoSaveArtifact();
  });

  // ============================================================
  //  Auto-save artifact to disk
  // ============================================================
  var _saveTimeout = null;
  function autoSaveArtifact() {
    clearTimeout(_saveTimeout);
    _saveTimeout = setTimeout(function () {
      saveCurrentArtifact();
    }, 800);
  }

  async function saveCurrentArtifact() {
    if (!currentArtifactSlug) return;
    try {
      await fetch("/api/artifacts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: document.getElementById("doc-title").textContent, content: currentArtifactContent, type: currentArtifactType, slug: currentArtifactSlug })
      });
    } catch (err) { /* silent */ }
  }

  // ============================================================
  //  Export Functions (Phase 10)
  // ============================================================
  btnExportMd.addEventListener("click", function () {
    if (!currentArtifactContent) return;
    downloadFile(currentArtifactContent, (document.getElementById("doc-title").textContent || "untitled") + ".md", "text/markdown");
  });

  btnExportDocx.addEventListener("click", async function () {
    var title = document.getElementById("doc-title").textContent || "Untitled";
    if (!currentArtifactContent) return;
    try {
      var resp = await fetch("/api/artifacts/export-docx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title, content: currentArtifactContent })
      });
      if (!resp.ok) throw new Error("Export failed");
      var data = await resp.json();
      // Trigger download of the .docx file
      var docUrl = data.path || "";
      if (docUrl) {
        var a = document.createElement("a");
        a.href = "/api/artifacts/download?path=" + encodeURIComponent(docUrl);
        a.download = title + ".docx";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    } catch (err) {
      alert("DOCX export error: " + err.message);
    }
  });

  // ============================================================
  //  Utility helpers
  // ============================================================
  function escapeHtml(text) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
  }

  function downloadFile(content, filename, mimeType) {
    var blob = new Blob([content], { type: mimeType });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // ============================================================
  //  Inline artifact parser helpers (mirrors artifact_store.py)
  // ============================================================
  var ArtifactStore = {
    toggleChecklist: function (mdText, index) {
      var lines = mdText.split("\n");
      var checkIdx = 0;
      for (var i = 0; i < lines.length; i++) {
        var m2 = lines[i].match(/^(- \[)( |x|X)(\])(.*)$/);
        if (m2) {
          if (checkIdx === index) {
            var curState = m2[2].toLowerCase();
            var newState = (curState === "x") ? " " : "x";
            lines[i] = m2[1] + newState + m2[3] + m2[4];
            break;
          }
          checkIdx++;
        }
      }
      return lines.join("\n");
    },

    parseMarkdownTable: function (mdText) {
      var lines = mdText.split("\n").map(function(l){ return l.trim(); }).filter(function(l){ return l.length > 0; });
      var sepIdx = null;
      for (var i = 0; i < lines.length; i++) {
        if (/^\|?\s*:?-{1,}:\s*(\|\s*:?-{1,}:\s*)+\|?$/.test(lines[i])) {
          sepIdx = i; break;
        }
      }
      if (sepIdx === null || sepIdx === 0) return { headers: [], rows: [] };

      var headerCells = lines[sepIdx - 1].split("|").map(function(c){ return c.trim(); }).filter(function(c){ return c.length > 0; });
      var rows = [];
      for (var j = sepIdx + 1; j < lines.length; j++) {
        var cells = lines[j].split("|").map(function(c){ return c.trim(); }).filter(function(c){ return c.length > 0; });
        if (cells.length === headerCells.length) rows.push(cells);
      }
      return { headers: headerCells, rows: rows };
    }
  };

  // ========================================================================
  // Work Agents — Multi-Agent Document Runner UI (Phase 11)
  // ========================================================================

  var _currentJobId = null;
  var _pollInterval = null;

  function openWorkAgentsPanel() {
    document.getElementById('work-agents-panel').classList.remove('hidden');
  }

  function closeWorkAgentsPanel() {
    document.getElementById('work-agents-panel').classList.add('hidden');
    if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }
    _currentJobId = null;
  }

  function updateAgentCards(agentsData) {
    var cards = document.querySelectorAll('.agent-card');
    cards.forEach(function(card) {
      var agentName = card.getAttribute('data-agent');
      var info = agentsData[agentName];
      if (!info) return;

      // Update status badge
      var badge = card.querySelector('.agent-status-badge');
      badge.textContent = info.status.charAt(0).toUpperCase() + info.status.slice(1);

      // Clear previous state classes
      card.classList.remove('running', 'done', 'failed');

      if (info.status === 'running') {
        card.classList.add('running');
      } else if (info.status === 'done') {
        card.classList.add('done');
      } else if (info.status === 'failed') {
        card.classList.add('failed');
      }
    });
  }

  function showProposedPreview(markdown) {
    var preview = document.getElementById('proposed-preview');
    var content = document.getElementById('preview-content');
    if (markdown && markdown.length > 0) {
      content.textContent = markdown;
      preview.classList.remove('hidden');
    } else {
      preview.classList.add('hidden');
    }
  }

  function showApprovalGate(show) {
    var gate = document.getElementById('approval-gate');
    if (show) {
      gate.classList.remove('hidden');
    } else {
      gate.classList.add('hidden');
    }
  }

  function updateJobStatus(status) {
    var badge = document.getElementById('job-status-badge');
    badge.textContent = status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ');
    badge.className = 'status-badge';
    badge.classList.add(status.replace(/\s+/g, '_').toLowerCase());
  }

  function startJobPolling(jobId) {
    if (_pollInterval) { clearInterval(_pollInterval); }
    _currentJobId = jobId;

    document.getElementById('job-id-display').textContent = 'Job: ' + jobId;
    document.getElementById('job-info-bar').classList.remove('hidden');

    // Initial fetch
    fetch('/api/work/jobs/' + jobId)
      .then(function(r) { return r.json(); })
      .then(function(data) {
        updateAgentCards(data.agents || {});
        updateJobStatus(data.status);
        showProposedPreview(data.proposed_markdown);
        if (data.status === 'awaiting_approval') {
          showApprovalGate(true);
        } else {
          showApprovalGate(false);
        }
      })
      .catch(function() {});

    // Poll every 1.5 seconds while awaiting approval
    _pollInterval = setInterval(function() {
      fetch('/api/work/jobs/' + jobId)
        .then(function(r) { return r.json(); })
        .then(function(data) {
          updateAgentCards(data.agents || {});
          updateJobStatus(data.status);
          showProposedPreview(data.proposed_markdown);
          if (data.status === 'awaiting_approval') {
            showApprovalGate(true);
          } else {
            showApprovalGate(false);
          }
        })
        .catch(function() {});
    }, 1500);
  }

  function sendToWork(content, instruction) {
    if (!content || !instruction) return;
    fetch('/api/work/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'Multi-Agent Work', content: content, instruction: instruction })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      openWorkAgentsPanel();
      startJobPolling(data.job_id);
    })
    .catch(function(err) {
      console.error('Failed to send to work:', err);
    });
  }

  // Wire up event listeners for Work Agents UI
  (function initWorkAgentsUI() {
    var btnSendToWork = document.getElementById('btn-send-to-work');
    if (btnSendToWork) {
      btnSendToWork.addEventListener('click', function() {
        var promptInput = document.getElementById('prompt-input');
        var content = window._currentDocumentContent || '';
        sendToWork(content, promptInput ? promptInput.value : 'Process this document');
      });
    }

    var btnWorkAgents = document.getElementById('btn-work-agents');
    if (btnWorkAgents) {
      btnWorkAgents.addEventListener('click', function() {
        openWorkAgentsPanel();
      });
    }

    var btnCloseAgents = document.getElementById('btn-close-agents');
    if (btnCloseAgents) {
      btnCloseAgents.addEventListener('click', closeWorkAgentsPanel);
    }

    var btnApprove = document.getElementById('btn-approve-artifact');
    if (btnApprove && _currentJobId) {
      btnApprove.addEventListener('click', function() {
        fetch('/api/work/jobs/' + _currentJobId + '/approve', { method: 'POST' })
          .then(function(r) { return r.json(); })
          .then(function(data) {
            if (data.success) {
              updateJobStatus('approved');
              showApprovalGate(false);
            }
          });
      });
    }

    var btnReject = document.getElementById('btn-reject-job');
    if (btnReject && _currentJobId) {
      btnReject.addEventListener('click', function() {
        fetch('/api/work/jobs/' + _currentJobId + '/reject', { method: 'POST' })
          .then(function(r) { return r.json(); })
          .then(function(data) {
            if (data.success) {
              updateJobStatus('rejected');
              showApprovalGate(false);
              setTimeout(closeWorkAgentsPanel, 1500);
            }
          });
      });
    }
  })();

})();
