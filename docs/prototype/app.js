const NODES = [
  { id: "start", label: "START", kind: "component", x: 70, y: 95, w: 90, h: 52 },
  { id: "context_gate", label: "Context Gate", kind: "gate", x: 205, y: 95, w: 120, h: 72 },
  { id: "contextualize", label: "Contextualize", kind: "agent", x: 385, y: 95, w: 142, h: 64 },
  { id: "plan", label: "Plan", kind: "agent", x: 555, y: 95, w: 116, h: 64 },
  { id: "retrieve", label: "Retrieve", kind: "component", x: 745, y: 105, w: 126, h: 60 },
  { id: "assess_evidence", label: "Assess Evidence", kind: "agent", x: 940, y: 105, w: 145, h: 64 },
  { id: "refine_query", label: "Refine Query", kind: "agent", x: 1115, y: 205, w: 132, h: 62 },
  { id: "generate", label: "Generate", kind: "agent", x: 735, y: 355, w: 126, h: 62 },
  { id: "validate_citations", label: "Validate Citations", kind: "component", x: 920, y: 355, w: 154, h: 62 },
  { id: "verify", label: "Verify", kind: "agent", x: 1095, y: 355, w: 116, h: 62 },
  { id: "repair_answer", label: "Repair Answer", kind: "agent", x: 1245, y: 445, w: 132, h: 62 },
  { id: "final_authorization", label: "Final Authorization", kind: "gate", x: 885, y: 545, w: 154, h: 82 },
  { id: "commit", label: "Commit Turn", kind: "component", x: 1085, y: 545, w: 136, h: 60 },
  { id: "terminal_answered", label: "Answered", kind: "terminal", x: 1275, y: 545, w: 118, h: 58 },
  { id: "terminal_clarification_requested", label: "Clarification", kind: "terminal", x: 390, y: 630, w: 132, h: 56 },
  { id: "terminal_incomplete", label: "Incomplete", kind: "terminal bad", x: 1060, y: 650, w: 122, h: 56 },
  { id: "terminal_refused", label: "Refused", kind: "terminal bad", x: 200, y: 630, w: 108, h: 56 },
  { id: "terminal_failed", label: "Failed", kind: "terminal bad", x: 700, y: 650, w: 108, h: 56 },
];

const EDGES = [
  ["start", "context_gate"], ["context_gate", "contextualize"],
  ["context_gate", "terminal_refused", -120],
  ["contextualize", "plan"], ["contextualize", "terminal_clarification_requested", 140],
  ["plan", "retrieve"], ["plan", "generate", 80],
  ["retrieve", "assess_evidence"], ["retrieve", "terminal_failed", 105],
  ["assess_evidence", "generate", 70], ["assess_evidence", "refine_query"],
  ["assess_evidence", "terminal_incomplete", 140], ["refine_query", "retrieve", 110],
  ["generate", "validate_citations"], ["validate_citations", "verify"],
  ["validate_citations", "terminal_refused", 135],
  ["verify", "final_authorization"], ["verify", "repair_answer"],
  ["verify", "terminal_refused", 150], ["repair_answer", "generate", 115],
  ["final_authorization", "commit"], ["final_authorization", "plan", -150],
  ["commit", "terminal_answered"],
];

const NODE_SPECS = {
  context_gate: {
    title: "Context Gate", kind: "gate", fn: "authorize_turn_context()",
    purpose: "Bind the run to trusted Runtime Context and revalidate historical dependencies.",
    contract: `def authorize_turn_context(\n    state: ThreadState,\n    *, runtime: RuntimeContext,\n) -> ContextGateResult:\n    assert runtime.principal_id == state.thread_principal_id\n    return revalidate_history(state, runtime)`,
    invariants: ["User text cannot select a Principal", "Historical citations never grant present access", "Principal mismatch fails before retrieval"],
  },
  contextualize: {
    title: "Contextual Rewriter", kind: "agent", fn: "rewrite_standalone_question()",
    purpose: "Resolve references using only currently authorized Thread Memory.",
    contract: `model.with_structured_output(StandaloneQuestionDecision)\n\nAllowed: rewrite references, request clarification\nForbidden: answer, invent facts, expand Access Scope`,
    invariants: ["Original constraints are preserved", "Ambiguity routes to clarification", "No chain-of-thought is stored"],
  },
  plan: {
    title: "Retrieval Planner", kind: "agent", fn: "plan_retrieval()",
    purpose: "Decide whether and where to retrieve under a code-owned budget.",
    contract: `model.with_structured_output(RetrievalPlan)\n\nInputs: StandaloneQuestion, allowed sources\nOutputs: direct_response | RetrievalRequest[]\nCannot set: principal_id, grants, SQL, unlimited top_k`,
    invariants: ["Source choices come from a fixed enum", "Authority fields are absent from the schema", "Greeting may skip retrieval"],
  },
  retrieve: {
    title: "Authorized Retriever", kind: "component", fn: "retrieve_authorized_evidence()",
    purpose: "Execute hybrid retrieval with the same authorization predicate in every branch.",
    contract: `def retrieve_authorized_evidence(\n    request: RetrievalRequest,\n    *, runtime: RuntimeContext,\n) -> RetrievalResult:\n    scope = resolve_access_scope(runtime.principal_id)\n    return repository.hybrid_search(request, scope)`,
    invariants: ["Dense and lexical SQL bind the same predicate", "Unknown tool fields are rejected", "Failures never silently fall back"],
  },
  assess_evidence: {
    title: "Evidence Assessor", kind: "agent", fn: "assess_evidence_sufficiency()",
    purpose: "Judge coverage without treating document instructions as control authority.",
    contract: `model.with_structured_output(EvidenceAssessment)\n\nOutputs: sufficient | refine | incomplete\nBudget is read-only and enforced by code`,
    invariants: ["Evidence is untrusted content", "Refinement is bounded", "No evidence is distinct from retrieval failure"],
  },
  refine_query: {
    title: "Query Refiner", kind: "agent", fn: "refine_retrieval_query()",
    purpose: "Perform one bounded refinement using retrieval diagnostics, not new authority.",
    contract: `if budget.research_iterations < limits.research_iterations:\n    return refine(standalone_question, retrieval_result)\nreturn Incomplete(reason="research_budget_exhausted")`,
    invariants: ["One retry in this prototype", "Counters never reset", "The Principal remains runtime-owned"],
  },
  generate: {
    title: "Answer Generator", kind: "agent", fn: "generate_cited_answer()",
    purpose: "Produce a buffered draft using only the current authorized Evidence Set.",
    contract: `model.with_structured_output(CitedAnswerDraft)\n\nInputs: question, Evidence Set, answer-local keys\nOutput remains buffered until all gates pass`,
    invariants: ["Only supplied citation keys may be used", "Draft tokens are developer-only", "Historical answers are not direct evidence"],
  },
  validate_citations: {
    title: "Citation Validator", kind: "component", fn: "validate_citation_map()",
    purpose: "Deterministically reject missing, unknown, or conflicting citation keys.",
    contract: `def validate_citation_map(\n    draft: CitedAnswerDraft,\n    evidence: EvidenceSet,\n) -> ValidatedCitationMap:\n    assert set(draft.keys) <= evidence.answer_local_keys`,
    invariants: ["The model cannot invent source paths", "Factual answers require citations", "Validation does not claim factual truth"],
  },
  verify: {
    title: "Answer Verifier", kind: "agent", fn: "verify_answer_support()",
    purpose: "Check whether each factual claim is semantically supported by cited Evidence.",
    contract: `model.with_structured_output(VerificationResult)\n\nOutputs: passed | repairable | unsupported\nVerifier cannot add Evidence or broaden the search`,
    invariants: ["Repair is answer-only", "Citation proves traceability, not source truth", "Unsupported output never reaches Chat"],
  },
  repair_answer: {
    title: "Answer Repair", kind: "agent", fn: "repair_answer_from_same_evidence()",
    purpose: "Repair one unsupported draft without reopening retrieval.",
    contract: `if budget.answer_repairs < limits.answer_repairs:\n    return regenerate_from_same_evidence(verification.feedback)\nreturn Refusal(reason="unsupported_answer")`,
    invariants: ["Same Evidence Set", "One repair in this prototype", "No new tools or authority"],
  },
  final_authorization: {
    title: "Final Authorization", kind: "gate", fn: "revalidate_answer_authorization()",
    purpose: "Prevent a response from mixing Evidence gathered under different grant versions.",
    contract: `if runtime.authorization_version != state.authorization_snapshot:\n    discard_current_turn_work()\n    return "plan" if restart_budget_available() else "refused"\nreturn "commit"`,
    invariants: ["Discard Evidence and draft together", "Restart at Plan", "At most one authorization restart"],
  },
  commit: {
    title: "Turn Record Commit", kind: "component", fn: "commit_terminal_turn_record()",
    purpose: "Atomically retain semantic Thread Memory and clear Current Turn Work.",
    contract: `def commit_terminal_turn_record(state: TurnState) -> ThreadState:\n    thread.turn_records.append(to_terminal_record(state))\n    return clear_current_turn_work(thread)`,
    invariants: ["Only terminal records become Thread Memory", "Raw Evidence is cleared", "Citation identities remain revalidatable"],
  },
};

const DEFAULT_SPEC = {
  title: "Terminal Outcome", kind: "terminal", fn: "finish_turn()",
  purpose: "Expose one explicit terminal outcome without leaking internal failure detail.",
  contract: `return TerminalOutcome(\n    status="answered | clarification | incomplete | refused | failed",\n    safe_message=validated_output,\n)`,
  invariants: ["Terminal state is explicit", "No unsupported guess", "User-visible errors are sanitized"],
};

const EVENT_COPY = {
  "context.authorized_history_loaded": "Principal matched and historical dependencies remain authorized.",
  "question.contextualized": "The message is now a standalone retrieval question.",
  "plan.retrieval_requested": "The planner selected bounded retrieval from fixed sources.",
  "plan.direct_response": "This greeting needs no knowledge retrieval.",
  "retrieval.completed": "Authorized hybrid retrieval completed with a typed result.",
  "evidence.sufficient": "Current Evidence covers the question, so generation may begin.",
  "evidence.refinement_needed": "No eligible Evidence was found and one refinement remains.",
  "query.refined": "The query changed; authority and consumed budgets did not reset.",
  "answer.draft_buffered": "A draft exists only in the trusted developer trace.",
  "citations.validated": "Every answer-local key maps to supplied Evidence.",
  "answer.verified": "Cited Evidence semantically supports the draft claims.",
  "answer.repair_needed": "The draft contains an unsupported claim; one answer repair remains.",
  "answer.repair_started": "The draft was discarded; repair reuses the same Evidence.",
  "authorization.changed_work_discarded": "The grant version changed, so all Current Turn Work was discarded.",
  "authorization.final_check_passed": "Every final citation is still authorized under the same snapshot.",
  "turn.answered": "The terminal Turn Record was committed and transient work was cleared.",
  "turn.clarification_requested": "Authorized context cannot resolve the reference safely.",
  "turn.incomplete": "The graph exhausted its bounded research path without enough Evidence.",
  "turn.refused": "A security or validation boundary refused the Turn.",
  "turn.failed": "A required retrieval stage failed and no silent fallback was used.",
};

const STATUS_COPY = {
  context_gate: "Checking trusted runtime context…",
  contextualize: "Rewriting the question…",
  plan: "Planning bounded retrieval…",
  retrieve: "Searching authorized sources…",
  assess_evidence: "Assessing evidence coverage…",
  refine_query: "Refining the retrieval query…",
  generate: "Generating a buffered draft…",
  validate_citations: "Validating citation keys…",
  verify: "Verifying factual support…",
  repair_answer: "Repairing the answer…",
  final_authorization: "Revalidating access…",
  commit: "Committing the terminal Turn Record…",
};

const state = {
  data: null,
  scenarioId: "happy",
  turnIndex: 0,
  submitted: false,
  stepIndex: -1,
  playing: false,
  speed: 800,
  graphZoom: 1,
  inspectorTab: "detail",
  revealedTurns: new Set(),
  typingTurnKey: null,
  timer: null,
  typingTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[char]));
const pretty = (value) => JSON.stringify(value, null, 2);

function scenario() { return state.data.scenarios.find((item) => item.id === state.scenarioId); }
function turn() { return scenario().turns[state.turnIndex]; }
function isTerminal() { return state.submitted && state.stepIndex === turn().steps.length - 1; }
function currentStep() { return state.stepIndex >= 0 ? turn().steps[state.stepIndex] : null; }
function snapshot() { return currentStep()?.after ?? turn().initial_state; }
function graphNodeForSnapshot(value) {
  return value.stage === "terminal" ? `terminal_${value.terminal.outcome}` : value.stage;
}

function specFor(nodeId) {
  if (NODE_SPECS[nodeId]) return NODE_SPECS[nodeId];
  if (nodeId === "start") return { ...DEFAULT_SPEC, title: "Start Turn", kind: "component", fn: "START" };
  if (nodeId.startsWith("terminal_")) return { ...DEFAULT_SPEC, title: nodeId.replace("terminal_", "").replaceAll("_", " ") };
  return DEFAULT_SPEC;
}

async function load() {
  const response = await fetch("scenarios.json");
  if (!response.ok) throw new Error(`Unable to load scenarios.json: ${response.status}`);
  state.data = await response.json();
  bindEvents();
  renderScenarioLibrary();
  renderAll();
}

function bindEvents() {
  $("#scenarioButton").addEventListener("click", () => { $("#scenarioModal").hidden = false; });
  $("#closeScenarioModal").addEventListener("click", closeScenarioModal);
  $("#scenarioModal").addEventListener("click", (event) => { if (event.target.id === "scenarioModal") closeScenarioModal(); });
  $("#restartButton").addEventListener("click", restartScenario);
  $("#submitButton").addEventListener("click", submitTurn);
  $("#playButton").addEventListener("click", togglePlayback);
  $("#stepForwardButton").addEventListener("click", () => stepBy(1));
  $("#stepBackButton").addEventListener("click", () => stepBy(-1));
  $("#timeline").addEventListener("input", (event) => {
    pause();
    state.stepIndex = Number(event.target.value) - 1;
    renderAll();
  });
  $("#speedSelect").addEventListener("change", (event) => { state.speed = Number(event.target.value); schedule(); });
  $("#zoomOutButton").addEventListener("click", () => setGraphZoom(state.graphZoom - .25));
  $("#zoomInButton").addEventListener("click", () => setGraphZoom(state.graphZoom + .25));
  $("#fitButton").addEventListener("click", () => resetGraphView(true));
  $$(".inspector-tab").forEach((button) => button.addEventListener("click", () => {
    state.inspectorTab = button.dataset.tab;
    renderInspector();
  }));
  $("#closeEvidenceDrawer").addEventListener("click", closeEvidenceDrawer);
  document.addEventListener("keydown", (event) => {
    const tag = document.activeElement?.tagName;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
    if (event.key === " ") { event.preventDefault(); togglePlayback(); }
  });
}

function closeScenarioModal() { $("#scenarioModal").hidden = true; }

function renderScenarioLibrary() {
  const groups = {};
  state.data.scenarios.forEach((item) => { (groups[item.group] ??= []).push(item); });
  $("#scenarioGroups").innerHTML = Object.entries(groups).map(([group, items]) => `
    <section class="scenario-group">
      <h3>${escapeHtml(group)}</h3>
      <div class="scenario-grid">
        ${items.map((item) => `
          <button class="scenario-card ${item.id === state.scenarioId ? "selected" : ""}" data-scenario="${item.id}" type="button">
            <strong>${escapeHtml(item.title)}</strong>
            <p>${escapeHtml(item.description)}</p>
            <footer>
              <span>${item.turns.length} Turn${item.turns.length > 1 ? "s" : ""} · ${escapeHtml(item.expected)}</span>
              ${item.known_limitation ? '<span class="limitation">Known limitation</span>' : ""}
            </footer>
          </button>`).join("")}
      </div>
    </section>`).join("");
  $$(".scenario-card").forEach((button) => button.addEventListener("click", () => {
    selectScenario(button.dataset.scenario);
    closeScenarioModal();
  }));
}

function selectScenario(id) {
  pause();
  clearInterval(state.typingTimer);
  state.scenarioId = id;
  state.turnIndex = 0;
  state.submitted = false;
  state.stepIndex = -1;
  state.revealedTurns.clear();
  state.typingTurnKey = null;
  state.graphZoom = 1;
  closeEvidenceDrawer();
  renderScenarioLibrary();
  renderAll();
  resetGraphScroll(false);
}

function restartScenario() { selectScenario(state.scenarioId); }

function submitTurn() {
  if (!state.submitted) {
    state.submitted = true;
    state.stepIndex = -1;
  } else if (isTerminal() && state.turnIndex + 1 < scenario().turns.length) {
    state.turnIndex += 1;
    state.stepIndex = -1;
  } else {
    return;
  }
  state.playing = true;
  renderAll();
  schedule();
}

function togglePlayback() {
  if (!state.submitted) { submitTurn(); return; }
  if (isTerminal()) return;
  state.playing = !state.playing;
  renderPlayback();
  schedule();
}

function pause() { state.playing = false; clearTimeout(state.timer); state.timer = null; }

function schedule() {
  clearTimeout(state.timer);
  if (!state.playing || !state.submitted || isTerminal()) return;
  state.timer = setTimeout(() => advancePlayback(true), state.speed);
}

function advancePlayback(animate = false) {
  if (state.stepIndex >= turn().steps.length - 1) { pause(); return; }
  const previous = state.stepIndex;
  state.stepIndex += 1;
  if (isTerminal()) state.playing = false;
  renderAll({ animatePacket: animate && previous !== state.stepIndex, animateAssistant: isTerminal() });
  schedule();
}

function stepBy(delta) {
  if (!state.submitted) return;
  pause();
  const next = Math.max(-1, Math.min(turn().steps.length - 1, state.stepIndex + delta));
  const forward = next > state.stepIndex;
  state.stepIndex = next;
  renderAll({ animatePacket: forward, animateAssistant: forward && isTerminal() });
}

function renderAll(options = {}) {
  renderHeader();
  renderChat(options.animateAssistant);
  renderGraph(options.animatePacket);
  renderPlayback();
  renderInspector();
}

function renderHeader() {
  $("#scenarioButtonLabel").textContent = scenario().title;
  $("#turnBadge").textContent = `Turn ${state.turnIndex + 1} / ${scenario().turns.length}`;
  $("#scenarioRiskBadge").hidden = !scenario().known_limitation;
  $("#scenarioButton").disabled = state.submitted && !isTerminal();
}

function answerHtml(text, turnData) {
  let result = escapeHtml(text);
  turnData.final_documents.forEach((document) => {
    const key = `[${document.citation_key}]`;
    result = result.replaceAll(key, `<button class="citation" data-evidence="${document.id}" type="button">${key}</button>`);
  });
  return result;
}

function messageMarkup(role, label, body, classes = "") {
  return `<article class="message ${role} ${classes}">
    <div class="message-avatar">${role === "user" ? "U" : "AI"}</div>
    <div class="message-content">
      <p class="message-meta">${escapeHtml(label)}</p>
      <div class="message-bubble">${body}</div>
    </div>
  </article>`;
}

function renderChat(animateAssistant = false) {
  const activeTurnKey = `${state.scenarioId}:${state.turnIndex}`;
  if (!isTerminal() && state.typingTurnKey === activeTurnKey) {
    clearInterval(state.typingTimer);
    state.typingTurnKey = null;
  }
  const parts = [];
  for (let index = 0; index < scenario().turns.length; index += 1) {
    const item = scenario().turns[index];
    const completed = index < state.turnIndex || (index === state.turnIndex && isTerminal());
    const activeSubmitted = index === state.turnIndex && state.submitted;
    if (index > state.turnIndex || (!completed && !activeSubmitted)) break;
    parts.push(messageMarkup("user", `User · ${item.principal_id}`, escapeHtml(item.message)));
    if (completed) {
      const turnKey = `${state.scenarioId}:${index}`;
      const shouldType = animateAssistant && index === state.turnIndex && !state.revealedTurns.has(turnKey);
      const body = shouldType ? `<span class="typewriter" data-turn-key="${turnKey}"></span>` : answerHtml(item.assistant, item);
      parts.push(messageMarkup("assistant", `Assistant · ${item.terminal.outcome}`, body));
      if (shouldType) {
        state.revealedTurns.add(turnKey);
        state.typingTurnKey = turnKey;
        setTimeout(() => typeVerifiedAnswer(turnKey, item), 0);
      }
    } else if (activeSubmitted) {
      const stage = graphNodeForSnapshot(snapshot());
      const status = STATUS_COPY[stage] ?? "Running a bounded graph transition…";
      parts.push(messageMarkup("assistant status", "Assistant · verified buffer pending", `<span class="status-pulse"><i></i><i></i><i></i></span>${escapeHtml(status)}`));
    }
  }
  if (!parts.length) {
    parts.push(`<div class="empty-chat"><p>Select a fixed Scenario, then submit its read-only message.</p></div>`);
  }
  $("#chatMessages").innerHTML = parts.join("");
  $("#chatMessages").scrollTop = $("#chatMessages").scrollHeight;
  bindEvidenceLinks();
  renderDocuments();
  renderComposer();
}

function typeVerifiedAnswer(turnKey, turnData) {
  clearInterval(state.typingTimer);
  const element = document.querySelector(`[data-turn-key="${CSS.escape(turnKey)}"]`);
  if (!element) return;
  let index = 0;
  const text = turnData.assistant;
  state.typingTimer = setInterval(() => {
    index += 1;
    element.textContent = text.slice(0, index);
    if (index >= text.length) {
      clearInterval(state.typingTimer);
      element.parentElement.innerHTML = answerHtml(text, turnData);
      state.typingTurnKey = null;
      bindEvidenceLinks();
      renderDocuments();
    }
  }, 14);
}

function latestCompletedTurn() {
  if (isTerminal()) return turn();
  if (state.turnIndex > 0) return scenario().turns[state.turnIndex - 1];
  return null;
}

function renderDocuments() {
  const item = latestCompletedTurn();
  const documents = state.typingTurnKey ? [] : (item?.final_documents ?? []);
  $("#retrievedSection").hidden = documents.length === 0;
  $("#documentCount").textContent = `${documents.length} authorized`;
  $("#documentList").innerHTML = documents.map((document) => `
    <button class="document-card" data-evidence="${document.id}" type="button">
      <strong>${escapeHtml(document.title)}</strong>
      <span>${escapeHtml(document.knowledge_source)} · ${escapeHtml(document.path)}</span>
      <span class="doc-key">[${escapeHtml(document.citation_key)}]</span>
    </button>`).join("");
  bindEvidenceLinks();
}

function renderComposer() {
  const hasNext = isTerminal() && state.turnIndex + 1 < scenario().turns.length;
  const composerTurn = hasNext ? scenario().turns[state.turnIndex + 1] : turn();
  $("#composer").value = composerTurn.message;
  const button = $("#submitButton");
  if (!state.submitted) {
    button.disabled = false;
    button.textContent = "Submit fixed message";
  } else if (hasNext) {
    button.disabled = false;
    button.textContent = `Submit Turn ${state.turnIndex + 2}`;
  } else if (isTerminal()) {
    button.disabled = true;
    button.textContent = "Scenario complete";
  } else {
    button.disabled = true;
    button.textContent = "Graph running…";
  }
}

function bindEvidenceLinks() {
  $$('[data-evidence]').forEach((element) => element.addEventListener("click", () => {
    const id = element.dataset.evidence;
    const documents = scenario().turns.flatMap((item) => [...item.final_documents, ...item.trace_evidence]);
    const document = documents.find((candidate) => candidate.id === id);
    if (document) openEvidence(document);
    $$(`[data-evidence="${CSS.escape(id)}"]`).forEach((match) => match.classList.add("highlighted"));
  }));
}

function openEvidence(document) {
  $("#drawerTitle").textContent = document.title;
  const instructionIndex = document.text.indexOf("INSTRUCTION:");
  const content = instructionIndex >= 0
    ? `${escapeHtml(document.text.slice(0, instructionIndex))}<mark>${escapeHtml(document.text.slice(instructionIndex))}</mark>`
    : escapeHtml(document.text);
  $("#drawerContent").innerHTML = `
    ${document.known_limitation ? `<div class="limitation-banner"><strong>Known limitation</strong><br>${escapeHtml(document.known_limitation)}</div>` : ""}
    ${instructionIndex >= 0 ? '<div class="limitation-banner"><strong>Untrusted content — not instruction authority</strong><br>The factual sentence may be cited; the embedded command has no control authority.</div>' : ""}
    <div class="drawer-meta">
      <div><strong>Citation mapping</strong><span>[${escapeHtml(document.citation_key)}] → ${escapeHtml(document.id)}</span></div>
      <div><strong>Knowledge Source</strong><span>${escapeHtml(document.knowledge_source)}</span></div>
      <div><strong>Source path</strong><span>${escapeHtml(document.path)}</span></div>
      <div><strong>Source Locator</strong><span>${escapeHtml(document.locator)}</span></div>
      <div><strong>Source Revision</strong><span>${escapeHtml(document.source_revision)}</span></div>
      <div><strong>Evidence identity</strong><span>${escapeHtml(document.id)}</span></div>
    </div>
    <section class="inspector-section"><h3>Mock chunk</h3><div class="evidence-content">${content}</div></section>
    <section class="inspector-section"><h3>Retrieval provenance</h3><pre class="json-block">${escapeHtml(pretty(document.provenance))}</pre></section>`;
  $("#evidenceDrawer").classList.add("open");
  $("#evidenceDrawer").setAttribute("aria-hidden", "false");
}

function closeEvidenceDrawer() {
  $("#evidenceDrawer").classList.remove("open");
  $("#evidenceDrawer").setAttribute("aria-hidden", "true");
}

function edgeKey(from, to) { return `${from}->${to}`; }
function nodeById(id) { return NODES.find((node) => node.id === id); }

function boundaryPoint(node, target) {
  const dx = target.x - node.x;
  const dy = target.y - node.y;
  if (!dx && !dy) return { x: node.x, y: node.y };
  const halfWidth = node.w / 2;
  const halfHeight = node.h / 2;
  let scale;
  if (node.kind === "gate") {
    scale = 1 / (Math.abs(dx) / halfWidth + Math.abs(dy) / halfHeight);
  } else if (node.kind.includes("terminal")) {
    scale = 1 / Math.sqrt((dx * dx) / (halfWidth * halfWidth) + (dy * dy) / (halfHeight * halfHeight));
  } else {
    scale = Math.min(
      dx ? halfWidth / Math.abs(dx) : Number.POSITIVE_INFINITY,
      dy ? halfHeight / Math.abs(dy) : Number.POSITIVE_INFINITY,
    );
  }
  return { x: node.x + dx * scale, y: node.y + dy * scale };
}

function edgeGeometry(from, to, curve = 0) {
  const a = nodeById(from); const b = nodeById(to);
  if (!a || !b) return null;
  const control = curve ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 + curve } : null;
  const start = boundaryPoint(a, control ?? b);
  const end = boundaryPoint(b, control ?? a);
  return { start, end, control };
}

function edgePath(from, to, curve = 0) {
  const geometry = edgeGeometry(from, to, curve);
  if (!geometry) return "";
  const { start, end, control } = geometry;
  if (!control) return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
  return `M ${start.x} ${start.y} Q ${control.x} ${control.y} ${end.x} ${end.y}`;
}

function renderGraph(animate = false) {
  const active = graphNodeForSnapshot(snapshot());
  const visibleSteps = state.submitted ? turn().steps.slice(0, state.stepIndex + 1) : [];
  const completedNodes = new Set(["start", ...visibleSteps.map((step) => step.node)]);
  const executedEdges = new Set(visibleSteps.map((step) => edgeKey(step.node, step.next_node)));
  const currentEdge = currentStep() ? edgeKey(currentStep().node, currentStep().next_node) : null;
  const svg = $("#graphSvg");
  svg.innerHTML = `
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#4b6688"></path>
      </marker>
    </defs>
    <g class="graph-group"><rect x="655" y="28" width="565" height="235"></rect><text x="675" y="52">RESEARCH SUBGRAPH</text></g>
    <g class="graph-group"><rect x="655" y="285" width="675" height="210"></rect><text x="675" y="309">ANSWER SUBGRAPH</text></g>
    <g class="edges">
      ${EDGES.map(([from, to, curve = 0]) => {
        const key = edgeKey(from, to);
        const classes = ["graph-edge", executedEdges.has(key) ? "executed" : "", key === currentEdge ? "current" : ""].join(" ");
        return `<path class="${classes}" data-edge="${key}" d="${edgePath(from, to, curve)}"></path>`;
      }).join("")}
    </g>
    <g class="nodes">
      ${NODES.map((node) => nodeMarkup(node, active, completedNodes)).join("")}
    </g>`;
  $$(".graph-node").forEach((node) => node.addEventListener("click", () => jumpToNode(node.dataset.node)));
  applyGraphZoom();
  if (animate && currentStep()) animatePacket(currentStep());
}

function setGraphZoom(nextZoom) {
  const viewport = $(".graph-viewport");
  const centerX = viewport.scrollWidth ? (viewport.scrollLeft + viewport.clientWidth / 2) / viewport.scrollWidth : .5;
  const centerY = viewport.scrollHeight ? (viewport.scrollTop + viewport.clientHeight / 2) / viewport.scrollHeight : .5;
  state.graphZoom = Math.max(.75, Math.min(2.5, Math.round(nextZoom * 4) / 4));
  applyGraphZoom();
  requestAnimationFrame(() => {
    viewport.scrollLeft = Math.max(0, centerX * viewport.scrollWidth - viewport.clientWidth / 2);
    viewport.scrollTop = Math.max(0, centerY * viewport.scrollHeight - viewport.clientHeight / 2);
  });
}

function applyGraphZoom() {
  const svg = $("#graphSvg");
  const viewport = $(".graph-viewport");
  svg.style.width = `${Math.max(1, viewport.clientWidth) * state.graphZoom}px`;
  svg.style.height = `${Math.max(1, viewport.clientHeight) * state.graphZoom}px`;
  $("#zoomLabel").textContent = `${Math.round(state.graphZoom * 100)}%`;
  $("#zoomOutButton").disabled = state.graphZoom <= .75;
  $("#zoomInButton").disabled = state.graphZoom >= 2.5;
}

function resetGraphScroll(smooth) {
  requestAnimationFrame(() => $(".graph-viewport").scrollTo({
    top: 0,
    left: 0,
    behavior: smooth ? "smooth" : "auto",
  }));
}

function resetGraphView(smooth = false) {
  state.graphZoom = 1;
  applyGraphZoom();
  resetGraphScroll(smooth);
}

function nodeMarkup(node, active, completedNodes) {
  const status = node.id === active ? "active" : completedNodes.has(node.id) ? "completed" : "inactive";
  const kindLabel = node.kind.includes("terminal") ? "TERMINAL" : node.kind.toUpperCase();
  let shape;
  if (node.kind === "gate") {
    const points = `${node.x},${node.y - node.h / 2} ${node.x + node.w / 2},${node.y} ${node.x},${node.y + node.h / 2} ${node.x - node.w / 2},${node.y}`;
    shape = `<polygon class="node-shape" points="${points}"></polygon>`;
  } else {
    shape = `<rect class="node-shape" x="${node.x - node.w / 2}" y="${node.y - node.h / 2}" width="${node.w}" height="${node.h}" rx="${node.kind.includes("terminal") ? 28 : 12}"></rect>`;
  }
  return `<g class="graph-node ${node.kind} ${status}" data-node="${node.id}" role="button" tabindex="0">
    ${shape}
    <text class="node-kind" x="${node.x}" y="${node.y - 6}">${kindLabel}</text>
    <text x="${node.x}" y="${node.y + 12}">${escapeHtml(node.label)}</text>
    ${completedNodes.has(node.id) && node.id !== active ? `<text class="node-check" x="${node.x + node.w / 2 - 8}" y="${node.y - node.h / 2 + 14}">✓</text>` : ""}
  </g>`;
}

function animatePacket(step) {
  const edge = EDGES.find(([from, to]) => from === step.node && to === step.next_node);
  const geometry = edgeGeometry(step.node, step.next_node, edge?.[2] ?? 0);
  if (!geometry) return;
  const { start: from, end: to, control } = geometry;
  const svg = $("#graphSvg");
  const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
  group.setAttribute("class", "graph-packet");
  const width = Math.max(86, step.packet.length * 6.3);
  group.innerHTML = `<rect x="${-width / 2}" y="-14" width="${width}" height="28"></rect><text y="4">${escapeHtml(step.packet)}</text>`;
  svg.appendChild(group);
  const started = performance.now();
  const duration = Math.min(760, Math.max(260, state.speed * .72));
  const tick = (now) => {
    const progress = Math.min(1, (now - started) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    const inverse = 1 - eased;
    const x = control
      ? inverse * inverse * from.x + 2 * inverse * eased * control.x + eased * eased * to.x
      : from.x + (to.x - from.x) * eased;
    const y = control
      ? inverse * inverse * from.y + 2 * inverse * eased * control.y + eased * eased * to.y
      : from.y + (to.y - from.y) * eased;
    group.setAttribute("transform", `translate(${x} ${y})`);
    if (progress < 1) requestAnimationFrame(tick); else group.remove();
  };
  requestAnimationFrame(tick);
}

function jumpToNode(nodeId) {
  if (!state.submitted) return;
  pause();
  const candidates = turn().steps.map((step, index) => ({ step, index })).filter(({ step }) => step.node === nodeId || step.next_node === nodeId);
  if (!candidates.length) return;
  const reachable = candidates.filter(({ index }) => index <= state.stepIndex);
  state.stepIndex = (reachable.at(-1) ?? candidates[0]).index;
  renderAll();
}

function renderPlayback() {
  const steps = turn().steps;
  $("#timeline").max = steps.length;
  $("#timeline").value = state.stepIndex + 1;
  $("#timeline").disabled = !state.submitted;
  $("#stepBackButton").disabled = !state.submitted || state.stepIndex < 0;
  $("#stepForwardButton").disabled = !state.submitted || state.stepIndex >= steps.length - 1;
  $("#playButton").disabled = isTerminal();
  $("#playButton").textContent = state.playing ? "Ⅱ" : "▶";
  $("#stepLabel").textContent = state.stepIndex < 0 ? "Ready" : `Step ${state.stepIndex + 1} / ${steps.length}`;
}

function renderInspector() {
  const nodeId = currentStep()?.node ?? graphNodeForSnapshot(snapshot());
  const spec = specFor(nodeId);
  $("#inspectorTitle").textContent = spec.title;
  const badge = $("#nodeTypeBadge");
  badge.className = `node-type-badge ${spec.kind.split(" ")[0]}`;
  badge.textContent = spec.kind.split(" ")[0].replace(/^./, (char) => char.toUpperCase());
  $$(".inspector-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === state.inspectorTab));
  const renderers = { detail: renderInspectorDetail, contract: renderContract, diff: renderDiff, json: renderRawJson };
  $("#inspectorContent").innerHTML = renderers[state.inspectorTab](spec);
}

function actualInputs(step) {
  const source = step?.before ?? snapshot();
  return {
    runtime_context: {
      principal_id: source.principal_id,
      authorization_snapshot: source.authorization_snapshot,
      limits: source.limits,
    },
    graph_state: {
      stage: source.stage,
      user_message: source.user_message,
      standalone_question: source.standalone_question,
      evidence_ids: source.evidence.map((item) => item.id),
      counters: source.counters,
      thread_records: source.thread_memory.length,
    },
  };
}

function actualOutputs(step) {
  if (!step) return { status: "awaiting_submit", next_node: "context_gate" };
  return {
    event: step.event,
    packet: step.packet,
    next_node: step.next_node,
    evidence_ids: step.after.evidence.map((item) => item.id),
    terminal: step.after.terminal,
  };
}

function renderInspectorDetail(spec) {
  const step = currentStep();
  const route = step ? (EVENT_COPY[step.event] ?? step.event) : "Submit the fixed message to start this Turn.";
  const securityEvents = step?.after.security_events ?? [];
  const finalIds = new Set(turn().final_documents.map((item) => item.id));
  const visibleEvidenceById = new Map();
  if (state.submitted) {
    turn().steps.slice(0, state.stepIndex + 1).forEach((visibleStep) => {
      visibleStep.after.evidence.forEach((item) => visibleEvidenceById.set(item.id, item));
    });
  }
  const visibleTraceEvidence = [...visibleEvidenceById.values()];
  return `
    <section class="inspector-section"><h3>Purpose</h3><p>${escapeHtml(spec.purpose)}</p></section>
    <section class="inspector-section"><h3>Function</h3><div class="data-row"><code>${escapeHtml(spec.fn)}</code></div></section>
    <section class="inspector-section"><h3>Inputs</h3><pre class="json-block">${escapeHtml(pretty(actualInputs(step)))}</pre></section>
    <section class="inspector-section"><h3>Outputs</h3><pre class="json-block">${escapeHtml(pretty(actualOutputs(step)))}</pre></section>
    <section class="inspector-section"><h3>Why this route</h3><div class="event-callout"><p>${escapeHtml(route)}</p></div></section>
    ${securityEvents.length ? `<section class="inspector-section"><h3>Security events</h3><div class="event-callout security-callout"><code>${securityEvents.map(escapeHtml).join("<br>")}</code></div></section>` : ""}
    <section class="inspector-section"><h3>Trace Evidence</h3><div class="trace-evidence">
      ${visibleTraceEvidence.length ? visibleTraceEvidence.map((item) => `<div class="trace-card ${finalIds.has(item.id) ? "" : "discarded"}"><strong>${escapeHtml(item.id)}</strong><span>${escapeHtml(item.title)}</span><span>${finalIds.has(item.id) ? "Final authorized Evidence" : "Developer trace only · not in final Chat"}</span></div>`).join("") : "<span>No Evidence has reached this step.</span>"}
    </div></section>`;
}

function renderContract(spec) {
  return `
    <section class="inspector-section"><h3>Python contract sketch</h3><pre class="code-block">${escapeHtml(spec.contract)}</pre></section>
    <section class="inspector-section"><h3>Enforced invariants</h3><div class="data-grid">${spec.invariants.map((item) => `<div class="data-row"><code>${escapeHtml(item)}</code></div>`).join("")}</div></section>
    <section class="inspector-section"><h3>Boundary</h3><p>Design-level contract, not production implementation. No chain-of-thought is shown.</p></section>`;
}

function renderDiff() {
  const diff = currentStep()?.diff ?? [];
  if (!diff.length) return `<section class="inspector-section"><h3>State diff</h3><p>Submit or advance one transition to see changed fields.</p></section>`;
  return `<section class="inspector-section"><h3>${diff.length} changed fields</h3><div class="diff-list">${diff.map((item) => `
    <div class="diff-row"><div class="diff-path">${escapeHtml(item.path)}</div><div class="diff-before">− ${escapeHtml(JSON.stringify(item.before))}</div><div class="diff-after">+ ${escapeHtml(JSON.stringify(item.after))}</div></div>`).join("")}</div></section>`;
}

function renderRawJson() {
  return `<section class="inspector-section"><h3>Current snapshot</h3><pre class="json-block">${escapeHtml(pretty(snapshot()))}</pre></section>`;
}

load().catch((error) => {
  document.body.innerHTML = `<main style="padding:32px;color:#ffabb4"><h1>Prototype failed to load</h1><pre>${escapeHtml(error.stack)}</pre></main>`;
  console.error(error);
});
