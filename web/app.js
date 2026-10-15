const form = document.querySelector('#prediction-form');
const fieldsRoot = document.querySelector('#feature-fields');
const button = document.querySelector('#submit-button');
const loadSample = document.querySelector('#load-sample');
const connection = document.querySelector('#connection-status');
const connectionDot = document.querySelector('#connection-dot');
const version = document.querySelector('#model-version');
const formError = document.querySelector('#form-error');
let modelInfo;

const readable = (name) => ({
  TDS: 'Total dissolved solids (TDS)',
  Organic_carbon: 'Organic carbon',
  Trihalomethanes: 'Trihalomethanes',
}[name] || name.replaceAll('_', ' '));

const prettyNumber = (value, digits = 3) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });

function buildFields(info) {
  fieldsRoot.replaceChildren();
  info.user_input_features.forEach((feature) => {
    const range = info.feature_ranges[feature];
    const spec = info.user_input_specs[feature];
    const item = document.createElement('div');
    item.className = 'field';
    const inputId = `field-${feature}`;
    const bisHint = `BIS guidance: acceptable ≤ ${prettyNumber(spec.acceptable_limit)} ${spec.unit}; permissible ≤ ${prettyNumber(spec.permissible_limit)} ${spec.unit} when no alternate source is available.`;
    item.innerHTML = `
      <label for="${inputId}">${spec.label || readable(feature)} <span>${spec.unit}</span><span class="tooltip" tabindex="0" title="${spec.tooltip}" aria-label="${spec.tooltip}">ⓘ</span></label>
      <input id="${inputId}" name="${feature}" type="number" min="${spec.min}" max="${spec.max}" step="any" inputmode="decimal" required aria-describedby="hint-${feature}" />
      <small id="hint-${feature}">${bisHint} Form range: ${prettyNumber(spec.min)}–${prettyNumber(spec.max)} ${spec.unit}.</small>`;
    fieldsRoot.append(item);
  });
}

function populateDefaults() {
  const demoSample = { Turbidity: 1, Hardness: 180, TDS: 450 };
  Object.entries(demoSample).forEach(([feature, value]) => {
    const input = form.elements.namedItem(feature);
    if (input) input.value = Number(value).toFixed(4);
  });
  clearErrors();
}

function clearErrors() {
  formError.hidden = true;
  formError.textContent = '';
  form.querySelectorAll('input').forEach((input) => {
    input.classList.remove('invalid');
    const spec = modelInfo?.user_input_specs[input.name];
    const helper = document.querySelector(`#hint-${CSS.escape(input.name)}`);
    if (spec && helper) helper.textContent = `BIS guidance: acceptable ≤ ${prettyNumber(spec.acceptable_limit)} ${spec.unit}; permissible ≤ ${prettyNumber(spec.permissible_limit)} ${spec.unit} when no alternate source is available. Form range: ${prettyNumber(spec.min)}–${prettyNumber(spec.max)} ${spec.unit}.`;
  });
}

function displayFieldErrors(errors) {
  Object.entries(errors).forEach(([field, message]) => {
    const input = form.elements.namedItem(field);
    if (input) {
      input.classList.add('invalid');
      const helper = document.querySelector(`#hint-${CSS.escape(field)}`);
      if (helper) helper.textContent = message;
    }
  });
}

function showWarnings(items) {
  const box = document.querySelector('#warnings');
  if (!items || !items.length) { box.hidden = true; box.replaceChildren(); return; }
  const list = document.createElement('ul');
  items.forEach((item) => { const node = document.createElement('li'); node.textContent = item; list.append(node); });
  box.replaceChildren(list); box.hidden = false;
}

function displayGuidance(items) {
  const grid = document.querySelector('#guidance-grid');
  grid.replaceChildren();
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = `guidance-item ${item.status}`;
    const metric = document.createElement('strong');
    metric.textContent = `${item.metric}: ${prettyNumber(item.value)} ${item.unit}`;
    const guidance = document.createElement('span');
    guidance.textContent = item.guidance;
    row.append(metric, guidance);
    grid.append(row);
  });
}

function displayResult(result) {
  document.querySelector('#empty-state').hidden = true;
  document.querySelector('#result-content').hidden = false;
  const card = document.querySelector('#safety-card');
  const safe = result.safety_code === 1;
  card.classList.toggle('unsafe', !safe);
  document.querySelector('#safety-label').textContent = result.safety_label;
  document.querySelector('#safe-probability').textContent = `${(result.safe_probability * 100).toFixed(1)}%`;
  document.querySelector('#safety-note').textContent = result.safety_prediction_caution
    ? 'held-out false-safe errors — not a safety guarantee'
    : 'model probability of safe (1)';
  document.querySelector('#ph-value').textContent = result.predicted_ph.toFixed(2);
  const safeProbability = `${(result.ph_classifier_safe_probability * 100).toFixed(1)}% Safe`;
  document.querySelector('#ph-note').textContent = result.ph_prediction_caution
    ? `weak validation signal · uses classifier probability (${safeProbability})`
    : result.ph_within_physical_range
      ? `experimental estimate · uses classifier probability (${safeProbability})`
      : 'outside physical pH range — verify reading';
  document.querySelector('#disclaimer').textContent = result.disclaimer;
  displayGuidance(result.input_guidance);
  showWarnings(result.warnings);
}

async function initialize() {
  try {
    const response = await fetch('/api/model-info');
    if (!response.ok) throw new Error('Model service unavailable');
    modelInfo = await response.json();
    buildFields(modelInfo);
    connection.textContent = 'Model service ready';
    connectionDot.classList.add('ready');
    version.textContent = `Model v${modelInfo.model_version}`;
    const cm = modelInfo.classification.metrics;
    const rm = modelInfo.regression.metrics;
    const classifierName = modelInfo.classification.display_name || 'Random Forest';
    const regressionName = modelInfo.regression.display_name || modelInfo.regression.model;
    document.querySelector('#class-metrics').textContent = `${classifierName} · ROC-AUC ${cm.roc_auc.toFixed(3)} · safe-class F1 ${cm.f1_safe.toFixed(3)}`;
    document.querySelector('#reg-metrics').textContent = `${regressionName} · MAE ${rm.mae.toFixed(3)} · RMSE ${rm.rmse.toFixed(3)} · R² ${rm.r2.toFixed(3)} (${modelInfo.regression.signal_status} signal)`;
  } catch (error) {
    connection.textContent = 'Unable to load model service. Train models and start the server.';
    connectionDot.classList.add('error');
    formError.textContent = 'The dashboard needs the local API. Run `python run.py train`, then `python run.py serve`.';
    formError.hidden = false;
    button.disabled = true;
  }
}

loadSample.addEventListener('click', () => { if (modelInfo) populateDefaults(); });
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearErrors();
  const payload = Object.fromEntries(new FormData(form).entries());
  for (const [feature, value] of Object.entries(payload)) {
    const spec = modelInfo.user_input_specs[feature];
    if (value.trim() === '' || !Number.isFinite(Number(value)) || Number(value) < spec.min || Number(value) > spec.max) {
      displayFieldErrors({ [feature]: `Enter a realistic value from ${spec.min} to ${spec.max} ${spec.unit}.` });
    }
  }
  if (form.querySelector('.invalid')) {
    formError.textContent = 'Correct the highlighted fields before predicting.';
    formError.hidden = false;
    return;
  }
  button.disabled = true;
  button.firstChild.textContent = 'Predicting… ';
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const result = await response.json();
    if (!response.ok) {
      displayFieldErrors(result.fields || {});
      formError.textContent = result.error || 'Prediction request could not be completed.';
      formError.hidden = false;
      return;
    }
    displayResult(result);
  } catch (error) {
    formError.textContent = 'Network error: the prediction service is unavailable.';
    formError.hidden = false;
  } finally {
    button.disabled = false;
    button.firstChild.textContent = 'Run prediction ';
  }
});

initialize();
