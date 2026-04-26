async function getDemoData() {
  const res = await fetch("./data/demo-data.json");
  if (!res.ok) throw new Error("Failed to load demo data");
  return res.json();
}

function escapeHtml(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function parseInlineMd(text) {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  out = out.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width:100%; border-radius:10px; border:1px solid var(--border);" />');
  out = out.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  return out;
}

function mdToHtmlLight(md) {
  const lines = md.split("\n");
  let html = "";
  let inList = false;
  let inCode = false;
  let inTable = false;
  let tableHeaderDone = false;

  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };
  const closeTable = () => {
    if (inTable) {
      html += "</tbody></table>";
      inTable = false;
      tableHeaderDone = false;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (line.startsWith("```")) {
      closeList();
      closeTable();
      if (!inCode) {
        inCode = true;
        html += "<pre><code>";
      } else {
        inCode = false;
        html += "</code></pre>";
      }
      continue;
    }
    if (inCode) {
      html += `${escapeHtml(raw)}\n`;
      continue;
    }

    if (!line) {
      closeList();
      closeTable();
      continue;
    }

    if (/^\|.*\|$/.test(line)) {
      closeList();
      const cells = line
        .split("|")
        .slice(1, -1)
        .map((c) => c.trim());
      const isDivider = cells.every((c) => /^:?-{3,}:?$/.test(c));
      if (!inTable) {
        inTable = true;
        html += "<table><thead>";
      }
      if (isDivider) {
        if (!tableHeaderDone) {
          html += "</tr></thead><tbody>";
          tableHeaderDone = true;
        }
      } else {
        const tag = tableHeaderDone ? "td" : "th";
        html += "<tr>";
        cells.forEach((c) => {
        html += `<${tag}>${parseInlineMd(c)}</${tag}>`;
        });
        html += "</tr>";
      }
      continue;
    } else {
      closeTable();
    }

    if (line.startsWith("### ")) {
      closeList();
      html += `<h3>${parseInlineMd(line.slice(4))}</h3>`;
      continue;
    }
    if (line.startsWith("## ")) {
      closeList();
      html += `<h2>${parseInlineMd(line.slice(3))}</h2>`;
      continue;
    }
    if (line.startsWith("# ")) {
      closeList();
      html += `<h1>${parseInlineMd(line.slice(2))}</h1>`;
      continue;
    }
    if (line.startsWith("- ")) {
      if (!inList) {
        html += "<ul>";
        inList = true;
      }
      html += `<li>${parseInlineMd(line.slice(2))}</li>`;
      continue;
    }

    closeList();
    html += `<p>${parseInlineMd(line)}</p>`;
  }

  closeList();
  closeTable();
  if (inCode) html += "</code></pre>";
  return html;
}

function setActiveNav() {
  const file = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".nav-links a").forEach((a) => {
    if (a.getAttribute("href") === file) a.classList.add("active");
  });
}

function fillLinks(project) {
  const spaceLinks = document.querySelectorAll("[data-space-link]");
  const codeLinks = document.querySelectorAll("[data-code-link]");
  const sftV5cLinks = document.querySelectorAll("[data-sft-v5c-link]");
  const sftV5dLinks = document.querySelectorAll("[data-sft-v5d-link]");
  const readmeLinks = document.querySelectorAll("[data-readme-link]");
  const blogLinks = document.querySelectorAll("[data-blog-link]");

  spaceLinks.forEach((el) => (el.href = project.space_url));
  codeLinks.forEach((el) => (el.href = project.code_url));
  sftV5cLinks.forEach((el) => (el.href = project.sft_v5c_url));
  sftV5dLinks.forEach((el) => (el.href = project.sft_v5d_best_url));
  readmeLinks.forEach((el) => (el.href = project.readme_url));
  blogLinks.forEach((el) => (el.href = project.blog_url));
}

function renderMetrics(metrics) {
  const map = {
    baselineScore: metrics.local_baseline_score.toFixed(3),
    bestScore: metrics.sft_v5c_best_score.toFixed(3),
    successRate: `${Math.round(metrics.sft_peak_success_rate * 100)}%`,
    multiplier: `${metrics.improvement_multiplier.toFixed(1)}x`,
  };
  Object.entries(map).forEach(([id, v]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  });
}

function renderDemo(demoCases) {
  let idx = 0;
  const caseTitle = document.getElementById("caseTitle");
  const caseSummary = document.getElementById("caseSummary");
  const baselineList = document.getElementById("baselineList");
  const trainedList = document.getElementById("trainedList");
  const caseHook = document.getElementById("caseHook");
  const policyRule = document.getElementById("policyRule");
  const docGallery = document.getElementById("docGallery");

  function writeActions(listEl, actions) {
    listEl.innerHTML = "";
    actions.forEach((a) => {
      const li = document.createElement("li");
      li.textContent = a;
      if (a === "submit_final_resolution") li.classList.add("submit");
      listEl.appendChild(li);
    });
  }

  function classifyEvidenceTag(tag) {
    if (tag.startsWith("submit_")) return "submit";
    if (tag.startsWith("compare_")) return "compare";
    return "inspect";
  }

  function buildEvidenceCaption(figcaptionEl, label, rawTags) {
    figcaptionEl.innerHTML = "";
    const labelSpan = document.createElement("span");
    labelSpan.className = "doc-label";
    labelSpan.textContent = label;
    figcaptionEl.appendChild(labelSpan);
    if (!rawTags) return;

    const tags = rawTags
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    if (!tags.length) return;

    const tagWrap = document.createElement("span");
    tagWrap.className = "evidence-tag-wrap";
    tags.forEach((tag) => {
      const pill = document.createElement("span");
      pill.className = `evidence-pill ${classifyEvidenceTag(tag)}`;
      pill.textContent = tag;
      tagWrap.appendChild(pill);
    });
    figcaptionEl.appendChild(tagWrap);
  }

  function paint() {
    const c = demoCases[idx];
    caseTitle.textContent = `${c.title} (${c.task_id})`;
    caseSummary.textContent = c.summary;
    if (caseHook) caseHook.textContent = c.story_hook || "";
    if (policyRule) policyRule.textContent = c.policy_rule || "";
    writeActions(baselineList, c.baseline_behavior);
    writeActions(trainedList, c.trained_behavior);
    if (docGallery) {
      docGallery.innerHTML = "";
      (c.document_images || []).forEach((src, i) => {
        const wrap = document.createElement("figure");
        wrap.className = "doc-card";
        const img = document.createElement("img");
        img.src = src;
        img.alt = `${c.task_id} document ${i + 1}`;
        const cap = document.createElement("figcaption");
        const labels = ["Invoice", "Purchase Order", "GRN", "Policy"];
        const tag =
          (c.document_evidence_tags && c.document_evidence_tags[i]) || "";
        buildEvidenceCaption(cap, labels[i] || `Document ${i + 1}`, tag);
        wrap.appendChild(img);
        wrap.appendChild(cap);
        docGallery.appendChild(wrap);
      });
    }
  }

  document.getElementById("prevCase")?.addEventListener("click", () => {
    idx = (idx - 1 + demoCases.length) % demoCases.length;
    paint();
  });
  document.getElementById("nextCase")?.addEventListener("click", () => {
    idx = (idx + 1) % demoCases.length;
    paint();
  });

  paint();
}

function renderSimulator(simCases) {
  if (!Array.isArray(simCases) || !simCases.length) return;
  let caseIdx = 0;
  let stepIdx = 0;
  let autoplayTimer = null;
  let autoplayOn = false;
  const AUTOPLAY_MS = 1800;

  const titleEl = document.getElementById("simCaseTitle");
  const goalEl = document.getElementById("simGoal");
  const stepHeadEl = document.getElementById("simStepHead");
  const readsEl = document.getElementById("simReads");
  const toolEl = document.getElementById("simTool");
  const resultEl = document.getElementById("simResult");
  const traceEl = document.getElementById("simTrace");
  const autoplayBtn = document.getElementById("toggleSimAutoplay");
  const progressEl = document.getElementById("simStepProgress");

  function currentSteps() {
    return simCases[caseIdx].steps || [];
  }

  function updateAutoplayButton() {
    if (!autoplayBtn) return;
    autoplayBtn.textContent = autoplayOn ? "Pause auto-play" : "Auto-play";
    autoplayBtn.classList.toggle("autoplay-on", autoplayOn);
  }

  function stopAutoplay() {
    autoplayOn = false;
    if (autoplayTimer) {
      clearInterval(autoplayTimer);
      autoplayTimer = null;
    }
    updateAutoplayButton();
  }

  function nextStep() {
    const steps = currentSteps();
    if (!steps.length) return;
    stepIdx = (stepIdx + 1) % steps.length;
    paint();
  }

  function startAutoplay() {
    stopAutoplay();
    autoplayOn = true;
    autoplayTimer = setInterval(() => {
      nextStep();
    }, AUTOPLAY_MS);
    updateAutoplayButton();
  }

  function paint() {
    const sc = simCases[caseIdx];
    const steps = currentSteps();
    if (!steps.length) return;
    const current = steps[stepIdx];
    titleEl.textContent = `${sc.difficulty.toUpperCase()} - ${sc.title} (${sc.task_id})`;
    if (progressEl) progressEl.textContent = `Step ${stepIdx + 1}/${steps.length}`;
    goalEl.textContent = sc.goal || "";
    stepHeadEl.textContent = `Step ${current.step_no}: ${current.agent_action}`;
    readsEl.textContent = (current.reads || []).join(", ");
    toolEl.textContent = current.tool_used || current.agent_action || "";
    resultEl.textContent = current.result || "";

    traceEl.innerHTML = "";
    steps.forEach((s, i) => {
      const li = document.createElement("li");
      li.textContent = `${s.step_no}. ${s.agent_action}`;
      if (s.agent_action === "submit_final_resolution") li.classList.add("submit");
      if (i === stepIdx) li.classList.add("active-step");
      traceEl.appendChild(li);
    });
  }

  document.getElementById("prevSimCase")?.addEventListener("click", () => {
    caseIdx = (caseIdx - 1 + simCases.length) % simCases.length;
    stepIdx = 0;
    paint();
  });
  document.getElementById("nextSimCase")?.addEventListener("click", () => {
    caseIdx = (caseIdx + 1) % simCases.length;
    stepIdx = 0;
    paint();
  });
  document.getElementById("nextSimStep")?.addEventListener("click", nextStep);
  autoplayBtn?.addEventListener("click", () => {
    if (autoplayOn) stopAutoplay();
    else startAutoplay();
  });

  updateAutoplayButton();
  paint();
}

async function renderBlogPage() {
  const container = document.getElementById("blogContainer");
  if (!container) return;
  try {
    const md = await fetch("./blog.md").then((r) => {
      if (!r.ok) throw new Error("blog.md fetch failed");
      return r.text();
    });
    if (window.marked && typeof window.marked.parse === "function") {
      container.innerHTML = window.marked.parse(md);
    } else {
      container.innerHTML = mdToHtmlLight(md);
    }
  } catch (e) {
    container.innerHTML =
      "<p>Blog preview could not be loaded in this runtime. Open <a href='./blog.md' target='_blank' rel='noreferrer'>blog.md</a> directly.</p>";
  }
}

async function init() {
  setActiveNav();
  try {
    const data = await getDemoData();
    fillLinks(data.project);
    renderMetrics(data.metrics);
    renderDemo(data.demo_cases);
    renderSimulator(data.simulator_cases);
  } catch (e) {
    const err = document.getElementById("appError");
    if (err) err.textContent = `demo-data load issue: ${e.message}`;
  }
  await renderBlogPage();
}

init().catch((e) => {
  const err = document.getElementById("appError");
  if (err) err.textContent = e.message;
});
