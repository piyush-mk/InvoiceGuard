async function getDemoData() {
  const res = await fetch("./data/demo-data.json");
  if (!res.ok) throw new Error("Failed to load demo data");
  return res.json();
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

  spaceLinks.forEach((el) => (el.href = project.space_url));
  codeLinks.forEach((el) => (el.href = project.code_url));
  sftV5cLinks.forEach((el) => (el.href = project.sft_v5c_url));
  sftV5dLinks.forEach((el) => (el.href = project.sft_v5d_best_url));
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

  function writeActions(listEl, actions) {
    listEl.innerHTML = "";
    actions.forEach((a) => {
      const li = document.createElement("li");
      li.textContent = a;
      if (a === "submit_final_resolution") li.classList.add("submit");
      listEl.appendChild(li);
    });
  }

  function paint() {
    const c = demoCases[idx];
    caseTitle.textContent = `${c.title} (${c.task_id})`;
    caseSummary.textContent = c.summary;
    writeActions(baselineList, c.baseline_behavior);
    writeActions(trainedList, c.trained_behavior);
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

async function renderBlogPage() {
  const container = document.getElementById("blogContainer");
  if (!container) return;
  const md = await fetch("./blog.md").then((r) => r.text());
  container.innerHTML = window.marked.parse(md);
}

async function init() {
  setActiveNav();
  const data = await getDemoData();
  fillLinks(data.project);
  renderMetrics(data.metrics);
  renderDemo(data.demo_cases);
  await renderBlogPage();
}

init().catch((e) => {
  const err = document.getElementById("appError");
  if (err) err.textContent = e.message;
});
