const canvas = document.getElementById("game-canvas");
const ctx = canvas.getContext("2d");
const stateView = document.getElementById("state-view");

const startBtn = document.getElementById("start-btn");
const pauseBtn = document.getElementById("pause-btn");
const resetBtn = document.getElementById("reset-btn");
const saveModelBtn = document.getElementById("save-model-btn");
const loadModelBtn = document.getElementById("load-model-btn");
const clearModelBtn = document.getElementById("clear-model-btn");
const exportResultsBtn = document.getElementById("export-results-btn");
const trainToggle = document.getElementById("train-toggle");
const autoPlayerToggle = document.getElementById("auto-player-toggle");
const epsilonInput = document.getElementById("epsilon-input");
const alphaInput = document.getElementById("alpha-input");
const gammaInput = document.getElementById("gamma-input");
const applyParamsBtn = document.getElementById("apply-params-btn");
const trainingChart = document.getElementById("training-chart");
const recentTrainingChart = document.getElementById("training-chart-recent");
const trainingMetrics = document.getElementById("training-metrics");
const modelStatus = document.getElementById("model-status");
const chartCtx = trainingChart.getContext("2d");
const recentChartCtx = recentTrainingChart ? recentTrainingChart.getContext("2d") : null;

const WORLD = {
  width: canvas.width,
  height: canvas.height,
};

const EPSILON_DECAY = 0.997;
const EPSILON_MIN = 0.05;
const DECISION_INTERVAL = 0.17;
const AUTO_DECISION_INTERVAL = 0.12;
const EPISODE_TIME_LIMIT = 60;
const TRAINING_STOP_EPISODES = 1000;
const CONVERGENCE_RETURN_DELTA = 1.0;
const CONVERGENCE_WINRATE_DELTA = 5;
const MODEL_STORAGE_KEY = "cds524_tank_q_model_v1";
const MOVING_WINDOW = 20;
const CHART_RENDER_MAX_POINTS = 600;

const ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY", "SHOOT"];

const keys = new Set();

const game = {
  mode: "menu",
  time: 0,
  episodeTime: 0,
  episode: 1,
  playerWins: 0,
  enemyWins: 0,
  trainingHistory: {
    episodeReturns: [],
    outcomes: [],
    episodes: [],
  },
  walls: [
    { x: 280, y: 160, w: 64, h: 240 },
    { x: 620, y: 90, w: 55, h: 200 },
    { x: 460, y: 380, w: 120, h: 56 },
  ],
  bullets: [],
  player: null,
  enemy: null,
  ai: {
    qTable: new Map(),
    alpha: 0.18,
    gamma: 0.92,
    epsilon: 0.5,
    training: true,
    lastState: null,
    lastAction: null,
    currentAction: 4,
    rewardBuffer: 0,
    episodeReturn: 0,
    lastReward: 0,
    lastRewardReason: "N/A",
    lastActionName: "STAY",
    explored: 0,
    exploited: 0,
    lastDecisionAt: 0,
    previousDistance: 0,
    stateKey: "",
    stoppedAtEpisode: null,
  },
  autoPlayer: {
    lastX: 0,
    lastY: 0,
    stuckFrames: 0,
    lastDecisionAt: 0,
    moveX: 0,
    moveY: 0,
    shootHold: 0,
    mode: "chase",
    orbitDir: 1,
    recoverFrames: 0,
  },
  modelMeta: {
    savedAt: null,
    loadedAt: null,
  },
};

function createTank(config) {
  return {
    x: config.x,
    y: config.y,
    radius: 16,
    speed: config.speed,
    hp: 3,
    maxHp: 3,
    dirX: config.dirX,
    dirY: config.dirY,
    shootCooldown: 0.42,
    cooldownLeft: 0,
    color: config.color,
    label: config.label,
  };
}

function resetEpisode() {
  game.bullets = [];
  game.player = createTank({
    x: 120,
    y: 300,
    speed: 168,
    dirX: 1,
    dirY: 0,
    color: "#126d7f",
    label: "Player",
  });
  game.enemy = createTank({
    x: 840,
    y: 300,
    speed: 146,
    dirX: -1,
    dirY: 0,
    color: "#b03f37",
    label: "Enemy",
  });
  game.episodeTime = 0;
  game.autoPlayer.lastX = game.player.x;
  game.autoPlayer.lastY = game.player.y;
  game.autoPlayer.stuckFrames = 0;
  game.autoPlayer.lastDecisionAt = 0;
  game.autoPlayer.moveX = 0;
  game.autoPlayer.moveY = 0;
  game.autoPlayer.shootHold = 0;
  game.autoPlayer.mode = "chase";
  game.autoPlayer.orbitDir = Math.random() < 0.5 ? -1 : 1;
  game.autoPlayer.recoverFrames = 0;
  game.ai.lastState = null;
  game.ai.lastAction = null;
  game.ai.currentAction = 4;
  game.ai.rewardBuffer = 0;
  game.ai.episodeReturn = 0;
  game.ai.lastReward = 0;
  game.ai.lastRewardReason = "episode_reset";
  game.ai.lastDecisionAt = 0;
  game.ai.previousDistance = distance(game.enemy.x, game.enemy.y, game.player.x, game.player.y);
  game.ai.stoppedAtEpisode = null;
}

function averageRange(values, startIdx, endIdx) {
  if (endIdx <= startIdx) return 0;
  let sum = 0;
  for (let i = startIdx; i < endIdx; i += 1) sum += values[i];
  return sum / (endIdx - startIdx);
}

function pushHistory(arr, value) {
  arr.push(value);
}

function toFiniteNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function setModelStatus(text) {
  modelStatus.textContent = `模型状态：${text}`;
}

function makeFileSafeTimestamp() {
  return new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
}

function downloadTextFile(filename, content, mimeType = "application/json") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadDataUrlFile(filename, dataUrl) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function serializeQTable(map) {
  return Array.from(map.entries()).map(([key, row]) => [key, row.slice(0, ACTIONS.length)]);
}

function hydrateQTable(serialized) {
  const map = new Map();
  if (!Array.isArray(serialized)) return map;
  for (const pair of serialized) {
    if (!Array.isArray(pair) || pair.length !== 2) continue;
    const key = String(pair[0]);
    const rawRow = Array.isArray(pair[1]) ? pair[1] : [];
    const row = new Array(ACTIONS.length).fill(0).map((_, idx) => toFiniteNumber(rawRow[idx], 0));
    map.set(key, row);
  }
  return map;
}

function getStorageSnapshot() {
  return {
    version: 1,
    savedAt: new Date().toISOString(),
    ai: {
      epsilon: game.ai.epsilon,
      alpha: game.ai.alpha,
      gamma: game.ai.gamma,
      explored: game.ai.explored,
      exploited: game.ai.exploited,
      qTable: serializeQTable(game.ai.qTable),
    },
    stats: {
      episode: game.episode,
      playerWins: game.playerWins,
      enemyWins: game.enemyWins,
      episodeReturns: game.trainingHistory.episodeReturns.slice(),
      outcomes: game.trainingHistory.outcomes.slice(),
      episodes: game.trainingHistory.episodes.slice(),
    },
  };
}

function saveModelToLocalStorage() {
  try {
    const snapshot = getStorageSnapshot();
    localStorage.setItem(MODEL_STORAGE_KEY, JSON.stringify(snapshot));
    game.modelMeta.savedAt = snapshot.savedAt;
    setModelStatus(`已保存 (${new Date(snapshot.savedAt).toLocaleTimeString()})`);
  } catch (error) {
    setModelStatus("保存失败");
    console.error(error);
  }
}

function loadModelFromLocalStorage() {
  try {
    const raw = localStorage.getItem(MODEL_STORAGE_KEY);
    if (!raw) {
      setModelStatus("未找到已保存模型");
      return;
    }
    const parsed = JSON.parse(raw);
    const qTable = hydrateQTable(parsed?.ai?.qTable);
    game.ai.qTable = qTable;
    game.ai.epsilon = clamp(toFiniteNumber(parsed?.ai?.epsilon, game.ai.epsilon), 0, 1);
    game.ai.alpha = clamp(toFiniteNumber(parsed?.ai?.alpha, game.ai.alpha), 0, 1);
    game.ai.gamma = clamp(toFiniteNumber(parsed?.ai?.gamma, game.ai.gamma), 0, 1);
    game.ai.explored = Math.max(0, Math.floor(toFiniteNumber(parsed?.ai?.explored, game.ai.explored)));
    game.ai.exploited = Math.max(0, Math.floor(toFiniteNumber(parsed?.ai?.exploited, game.ai.exploited)));

    game.episode = Math.max(1, Math.floor(toFiniteNumber(parsed?.stats?.episode, game.episode)));
    game.playerWins = Math.max(0, Math.floor(toFiniteNumber(parsed?.stats?.playerWins, game.playerWins)));
    game.enemyWins = Math.max(0, Math.floor(toFiniteNumber(parsed?.stats?.enemyWins, game.enemyWins)));
    game.trainingHistory.episodeReturns = Array.isArray(parsed?.stats?.episodeReturns)
      ? parsed.stats.episodeReturns.map((v) => toFiniteNumber(v, 0))
      : [];
    game.trainingHistory.outcomes = Array.isArray(parsed?.stats?.outcomes)
      ? parsed.stats.outcomes.map((v) => (toFiniteNumber(v, 0) > 0 ? 1 : 0))
      : [];
    game.trainingHistory.episodes = Array.isArray(parsed?.stats?.episodes)
      ? parsed.stats.episodes.map((item) => ({
        episode: Math.max(1, Math.floor(toFiniteNumber(item?.episode, 1))),
        reason: String(item?.reason ?? "unknown"),
        episodeReturn: toFiniteNumber(item?.episodeReturn, 0),
        epsilon: toFiniteNumber(item?.epsilon, game.ai.epsilon),
        alpha: toFiniteNumber(item?.alpha, game.ai.alpha),
        gamma: toFiniteNumber(item?.gamma, game.ai.gamma),
        qStates: Math.max(0, Math.floor(toFiniteNumber(item?.qStates, 0))),
        enemyWins: Math.max(0, Math.floor(toFiniteNumber(item?.enemyWins, 0))),
        playerWins: Math.max(0, Math.floor(toFiniteNumber(item?.playerWins, 0))),
        explored: Math.max(0, Math.floor(toFiniteNumber(item?.explored, 0))),
        exploited: Math.max(0, Math.floor(toFiniteNumber(item?.exploited, 0))),
        endedAt: typeof item?.endedAt === "string" ? item.endedAt : null,
      }))
      : [];

    epsilonInput.value = game.ai.epsilon.toFixed(3);
    alphaInput.value = game.ai.alpha.toFixed(3);
    gammaInput.value = game.ai.gamma.toFixed(3);
    game.modelMeta.loadedAt = new Date().toISOString();
    setModelStatus(`已加载 (Q状态 ${game.ai.qTable.size})`);
  } catch (error) {
    setModelStatus("加载失败");
    console.error(error);
  }
}

function clearStoredModel() {
  localStorage.removeItem(MODEL_STORAGE_KEY);
  game.ai.qTable = new Map();
  game.ai.explored = 0;
  game.ai.exploited = 0;
  game.trainingHistory.episodeReturns = [];
  game.trainingHistory.outcomes = [];
  game.trainingHistory.episodes = [];
  game.episode = 1;
  game.playerWins = 0;
  game.enemyWins = 0;
  setModelStatus("已清空并重置");
  resetEpisode();
}

function initializeModelStatus() {
  const raw = localStorage.getItem(MODEL_STORAGE_KEY);
  if (!raw) {
    setModelStatus("未保存");
    return;
  }
  try {
    const parsed = JSON.parse(raw);
    game.modelMeta.savedAt = typeof parsed.savedAt === "string" ? parsed.savedAt : null;
    const qCount = Array.isArray(parsed?.ai?.qTable) ? parsed.ai.qTable.length : 0;
    if (game.modelMeta.savedAt) {
      setModelStatus(`已存在保存模型 (${new Date(game.modelMeta.savedAt).toLocaleString()})`);
    } else {
      setModelStatus(`已存在保存模型 (Q状态 ${qCount})`);
    }
  } catch {
    setModelStatus("发现损坏模型，建议清空");
  }
}

function getMovingAverage(values, index, windowSize) {
  const start = Math.max(0, index - windowSize + 1);
  let sum = 0;
  for (let i = start; i <= index; i += 1) sum += values[i];
  return sum / (index - start + 1);
}

function buildTrainingExportPayload(imageFile, chartMeta) {
  const snapshot = getStorageSnapshot();
  return {
    exportedAt: new Date().toISOString(),
    version: 1,
    snapshot,
    chart: {
      movingWindow: MOVING_WINDOW,
      exportMovingWindow: chartMeta?.movingWindow ?? MOVING_WINDOW,
      exportMaxPoints: chartMeta?.maxPoints ?? CHART_RENDER_MAX_POINTS,
      imageFile,
    },
    trainingStop: {
      type: "fixed-episodes",
      threshold: TRAINING_STOP_EPISODES,
      stoppedAtEpisode: game.ai.stoppedAtEpisode,
    },
    summary: {
      episodesRecorded: game.trainingHistory.episodeReturns.length,
      qStates: game.ai.qTable.size,
      playerWins: game.playerWins,
      enemyWins: game.enemyWins,
      epsilon: game.ai.epsilon,
      alpha: game.ai.alpha,
      gamma: game.ai.gamma,
    },
  };
}

function exportTrainingResults() {
  try {
    const stamp = makeFileSafeTimestamp();
    const imageFile = `tank-training-chart-${stamp}.png`;
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = trainingChart.width;
    exportCanvas.height = trainingChart.height;
    const exportCtx = exportCanvas.getContext("2d");
    const exportWindow = Math.max(1, game.trainingHistory.episodeReturns.length);
    const exportMaxPoints = Math.max(1, game.trainingHistory.episodeReturns.length);
    renderTrainingChart(exportCtx, exportCanvas, {
      movingWindow: exportWindow,
      maxPoints: exportMaxPoints,
    });
    const payload = buildTrainingExportPayload(imageFile, {
      movingWindow: exportWindow,
      maxPoints: exportMaxPoints,
    });
    downloadTextFile(`tank-training-results-${stamp}.json`, JSON.stringify(payload, null, 2));
    downloadDataUrlFile(imageFile, exportCanvas.toDataURL("image/png"));
    setModelStatus(`训练结果JSON与曲线PNG已导出 (${new Date().toLocaleTimeString()})`);
  } catch (error) {
    setModelStatus("导出失败");
    console.error(error);
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance(ax, ay, bx, by) {
  return Math.hypot(ax - bx, ay - by);
}

function circleHitsRect(cx, cy, radius, rect) {
  const nearestX = clamp(cx, rect.x, rect.x + rect.w);
  const nearestY = clamp(cy, rect.y, rect.y + rect.h);
  const dx = cx - nearestX;
  const dy = cy - nearestY;
  return dx * dx + dy * dy < radius * radius;
}

function collidesWalls(x, y, radius) {
  for (const wall of game.walls) {
    if (circleHitsRect(x, y, radius, wall)) return true;
  }
  return false;
}

function tryMoveTank(tank, dx, dy, dt) {
  let nx = dx;
  let ny = dy;
  const len = Math.hypot(nx, ny);
  if (len > 1e-6) {
    nx /= len;
    ny /= len;
    tank.dirX = nx;
    tank.dirY = ny;
  } else {
    nx = 0;
    ny = 0;
  }

  const moveX = nx * tank.speed * dt;
  const moveY = ny * tank.speed * dt;
  let moved = false;
  let collided = false;

  const xTry = tank.x + moveX;
  const xWithin = clamp(xTry, tank.radius, WORLD.width - tank.radius);
  if (!collidesWalls(xWithin, tank.y, tank.radius)) {
    if (Math.abs(xWithin - tank.x) > 1e-3) moved = true;
    tank.x = xWithin;
  } else if (Math.abs(moveX) > 1e-3) {
    collided = true;
  }

  const yTry = tank.y + moveY;
  const yWithin = clamp(yTry, tank.radius, WORLD.height - tank.radius);
  if (!collidesWalls(tank.x, yWithin, tank.radius)) {
    if (Math.abs(yWithin - tank.y) > 1e-3) moved = true;
    tank.y = yWithin;
  } else if (Math.abs(moveY) > 1e-3) {
    collided = true;
  }

  return { moved, collided };
}

function spawnBullet(tank, owner) {
  if (tank.cooldownLeft > 0) return false;
  tank.cooldownLeft = tank.shootCooldown;

  let dx = tank.dirX;
  let dy = tank.dirY;
  if (owner === "enemy") {
    const tx = game.player.x - tank.x;
    const ty = game.player.y - tank.y;
    const d = Math.hypot(tx, ty) || 1;
    dx = tx / d;
    dy = ty / d;
  }

  const bulletSpeed = 330;
  game.bullets.push({
    x: tank.x + dx * (tank.radius + 6),
    y: tank.y + dy * (tank.radius + 6),
    vx: dx * bulletSpeed,
    vy: dy * bulletSpeed,
    radius: 4,
    owner,
    bouncesLeft: 1,
  });
  return true;
}

function getStateVector() {
  const dx = game.player.x - game.enemy.x;
  const dy = game.player.y - game.enemy.y;

  const bucket = (v) => {
    if (v < -170) return -2;
    if (v < -55) return -1;
    if (v <= 55) return 0;
    if (v <= 170) return 1;
    return 2;
  };

  const enemyHp = game.enemy.hp >= 2 ? 1 : 0;
  const playerHp = game.player.hp >= 2 ? 1 : 0;

  let threat = 0;
  for (const b of game.bullets) {
    if (b.owner !== "player") continue;
    const bx = game.enemy.x - b.x;
    const by = game.enemy.y - b.y;
    const dist = Math.hypot(bx, by);
    if (dist > 170) continue;
    const bSpeed = Math.hypot(b.vx, b.vy) || 1;
    const dot = (bx * b.vx + by * b.vy) / (dist * bSpeed);
    if (dot > 0.7) {
      threat = 1;
      break;
    }
  }

  return [bucket(dx), bucket(dy), enemyHp, playerHp, threat];
}

function stateKeyFromVector(vec) {
  return vec.join("|");
}

function ensureQRow(stateKey) {
  if (!game.ai.qTable.has(stateKey)) {
    game.ai.qTable.set(stateKey, new Array(ACTIONS.length).fill(0));
  }
  return game.ai.qTable.get(stateKey);
}

function chooseAction(stateKey) {
  const qRow = ensureQRow(stateKey);
  let action = 0;
  let exploratory = false;
  if (game.ai.training && Math.random() < game.ai.epsilon) {
    action = Math.floor(Math.random() * ACTIONS.length);
    exploratory = true;
    game.ai.explored += 1;
  } else {
    let maxVal = qRow[0];
    let best = [0];
    for (let i = 1; i < qRow.length; i += 1) {
      if (qRow[i] > maxVal + 1e-9) {
        maxVal = qRow[i];
        best = [i];
      } else if (Math.abs(qRow[i] - maxVal) <= 1e-9) {
        best.push(i);
      }
    }
    action = best[Math.floor(Math.random() * best.length)];
    game.ai.exploited += 1;
  }
  return { action, exploratory };
}

function updateQ(prevStateKey, action, reward, newStateKey, done) {
  const prevQ = ensureQRow(prevStateKey);
  const current = prevQ[action];
  const nextQ = ensureQRow(newStateKey);
  const maxNext = nextQ.reduce((m, v) => Math.max(m, v), nextQ[0] ?? 0);
  const target = reward + (done ? 0 : game.ai.gamma * maxNext);
  prevQ[action] = current + game.ai.alpha * (target - current);
}

function addReward(value, reason) {
  game.ai.rewardBuffer += value;
  game.ai.episodeReturn += value;
  game.ai.lastReward = value;
  game.ai.lastRewardReason = reason;
}

function enemyDecisionTick() {
  const currentState = stateKeyFromVector(getStateVector());
  game.ai.stateKey = currentState;

  if (game.ai.lastState !== null && game.ai.lastAction !== null && game.ai.training) {
    updateQ(game.ai.lastState, game.ai.lastAction, game.ai.rewardBuffer, currentState, false);
  }

  game.ai.rewardBuffer = 0;
  const decision = chooseAction(currentState);
  game.ai.currentAction = decision.action;
  game.ai.lastAction = decision.action;
  game.ai.lastActionName = ACTIONS[decision.action];
  game.ai.lastState = currentState;
  game.ai.lastDecisionAt = game.episodeTime;

  if (decision.action === 5 && spawnBullet(game.enemy, "enemy")) {
    addReward(0.03, "enemy_shot");
  }
}

function applyEnemyAction(dt) {
  let vx = 0;
  let vy = 0;
  const act = game.ai.currentAction;
  if (act === 0) vy = -1;
  if (act === 1) vy = 1;
  if (act === 2) vx = -1;
  if (act === 3) vx = 1;

  const enemyMove = tryMoveTank(game.enemy, vx, vy, dt);
  if (enemyMove.collided) addReward(-0.08, "wall_collision");

  const prev = game.ai.previousDistance;
  const now = distance(game.enemy.x, game.enemy.y, game.player.x, game.player.y);
  if (now < prev - 1) {
    addReward(0.025, "approach_player");
  } else if (now > prev + 1) {
    addReward(-0.015, "lost_pressure");
  }
  game.ai.previousDistance = now;
}

function updateBullets(dt) {
  const remaining = [];
  for (const b of game.bullets) {
    let nextX = b.x + b.vx * dt;
    let nextY = b.y + b.vy * dt;
    let bounced = false;

    if (nextX <= b.radius || nextX >= WORLD.width - b.radius) {
      b.vx *= -1;
      nextX = clamp(nextX, b.radius, WORLD.width - b.radius);
      bounced = true;
    }
    if (nextY <= b.radius || nextY >= WORLD.height - b.radius) {
      b.vy *= -1;
      nextY = clamp(nextY, b.radius, WORLD.height - b.radius);
      bounced = true;
    }

    let hitWall = false;
    for (const wall of game.walls) {
      if (circleHitsRect(nextX, nextY, b.radius, wall)) {
        hitWall = true;
        const tryX = !circleHitsRect(nextX, b.y, b.radius, wall);
        const tryY = !circleHitsRect(b.x, nextY, b.radius, wall);
        if (tryX && !tryY) b.vy *= -1;
        else if (!tryX && tryY) b.vx *= -1;
        else {
          b.vx *= -1;
          b.vy *= -1;
        }
        nextX = b.x + b.vx * dt;
        nextY = b.y + b.vy * dt;
        bounced = true;
        break;
      }
    }

    if (bounced || hitWall) {
      b.bouncesLeft -= 1;
      if (b.bouncesLeft < 0) {
        continue;
      }
    }

    b.x = nextX;
    b.y = nextY;

    const target = b.owner === "player" ? game.enemy : game.player;
    if (distance(b.x, b.y, target.x, target.y) <= b.radius + target.radius) {
      target.hp -= 1;
      if (b.owner === "player") addReward(-2.6, "enemy_got_hit");
      if (b.owner === "enemy") addReward(2.8, "player_got_hit");
      continue;
    }

    remaining.push(b);
  }
  game.bullets = remaining;
}

function handlePlayerInput(dt) {
  if (autoPlayerToggle && autoPlayerToggle.checked) {
    handleAutoPlayerInput(dt);
    return;
  }
  const right = keys.has("KeyD") || keys.has("ArrowRight");
  const left = keys.has("KeyA") || keys.has("ArrowLeft");
  const down = keys.has("KeyS") || keys.has("ArrowDown");
  const up = keys.has("KeyW") || keys.has("ArrowUp");
  const vx = (right ? 1 : 0) - (left ? 1 : 0);
  const vy = (down ? 1 : 0) - (up ? 1 : 0);
  tryMoveTank(game.player, vx, vy, dt);
  
  if (keys.has("Space")) {
    spawnBullet(game.player, "player");
  }
}

function handleAutoPlayerInput(dt) {
  const player = game.player;
  const enemy = game.enemy;
  const auto = game.autoPlayer;

  const tryDirection = (tx, ty) => {
    const len = Math.hypot(tx, ty);
    if (len < 1e-6) return false;
    const nx = tx / len;
    const ny = ty / len;
    // 【修复1】降低预测距离，原为 36，避免坦克过于“胆小”引发全向拒绝移动
    const lookAhead = 20;
    const testX = player.x + nx * lookAhead;
    const testY = player.y + ny * lookAhead;
    // 【修复2】稍微放宽地图边缘判定，原为 player.radius + 8
    const margin = player.radius + 4;
    if (testX < margin || testX > WORLD.width - margin || testY < margin || testY > WORLD.height - margin) return false;
    if (collidesWalls(testX, testY, player.radius)) return false;
    auto.moveX = nx;
    auto.moveY = ny;
    return true;
  };

  const shouldDecide = game.episodeTime - auto.lastDecisionAt >= AUTO_DECISION_INTERVAL;
  if (shouldDecide) {
    auto.lastDecisionAt = game.episodeTime;
    let shouldShoot = false;

    const dx = enemy.x - player.x;
    const dy = enemy.y - player.y;
    const dist = Math.hypot(dx, dy) || 1;
    const aimX = dx / dist;
    const aimY = dy / dist;

    if (auto.recoverFrames > 0) {
      auto.recoverFrames -= 1;
    } else {
      let danger = false;
      for (const b of game.bullets) {
        if (b.owner !== "enemy") continue;
        const bx = player.x - b.x;
        const by = player.y - b.y;
        const bDist = Math.hypot(bx, by);
        if (bDist < 200) {
          const bSpeed = Math.hypot(b.vx, b.vy) || 1;
          const dot = (bx * b.vx + by * b.vy) / (bDist * bSpeed);
          if (dot > 0.75) {
            danger = true;
            if (!tryDirection(-b.vy, b.vx)) {
              tryDirection(b.vy, -b.vx);
            }
            break;
          }
        }
      }

      if (!danger) {
        if (auto.mode === "chase" && dist < 150) auto.mode = "orbit";
        else if (auto.mode === "orbit" && dist > 230) auto.mode = "chase";
        else if (auto.mode === "orbit" && dist < 110) auto.mode = "retreat";
        else if (auto.mode === "retreat" && dist > 170) auto.mode = "orbit";

        if (auto.mode === "chase") {
          tryDirection(aimX, aimY);
        } else if (auto.mode === "retreat") {
          tryDirection(-aimX, -aimY);
        } else {
          const ox = -aimY * auto.orbitDir;
          const oy = aimX * auto.orbitDir;
          if (!tryDirection(ox, oy)) {
            auto.orbitDir *= -1;
            // 【修复3】打破 Orbit 死循环：如果换向后仍被挡，强制切为后退模式
            if (!tryDirection(-ox, -oy)) {
              auto.mode = "retreat";
              tryDirection(-aimX, -aimY);
            }
          }
        }

        const dotAim = player.dirX * aimX + player.dirY * aimY;
        if (dotAim > 0.92 && dist < 520) {
          shouldShoot = true;
        }
      }
    }
    auto.shootHold = shouldShoot ? 2 : Math.max(0, auto.shootHold - 1);
  }

  // 确保卡死恢复在尝试移动前运算
  if (auto.stuckFrames > 12) {
    const rx = Math.random() - 0.5;
    const ry = Math.random() - 0.5;
    if (!tryDirection(rx, ry)) {
      tryDirection(-rx, -ry);
    }
    auto.recoverFrames = 15; // 延长恢复时间
    auto.stuckFrames = 0;
  }

  if (Math.hypot(auto.moveX, auto.moveY) < 1e-6) {
    tryDirection(Math.random() - 0.5, Math.random() - 0.5);
  }

  // 【修复4】获取实际移动的物理碰撞结果
  const moveResult = tryMoveTank(player, auto.moveX, auto.moveY, dt);

  // 【修复5】卡死检测升级：只要位移过小【或者】发生碰墙，立刻开始累计卡死帧数
  const movedDist = Math.hypot(player.x - auto.lastX, player.y - auto.lastY);
  if (movedDist < 0.5 || moveResult.collided) {
    auto.stuckFrames += 1;
  } else {
    auto.stuckFrames = 0;
  }
  
  auto.lastX = player.x;
  auto.lastY = player.y;

  if (auto.shootHold > 0) {
    spawnBullet(player, "player");
    auto.shootHold -= 1;
  }
}

function applyCooldowns(dt) {
  game.player.cooldownLeft = Math.max(0, game.player.cooldownLeft - dt);
  game.enemy.cooldownLeft = Math.max(0, game.enemy.cooldownLeft - dt);
}

function recordEpisodeMetrics(reason) {
  const episodeReturn = Number(game.ai.episodeReturn.toFixed(3));
  pushHistory(game.trainingHistory.episodeReturns, episodeReturn);
  pushHistory(game.trainingHistory.outcomes, reason === "enemy_win" ? 1 : 0);
  pushHistory(game.trainingHistory.episodes, {
    episode: game.episode,
    reason,
    episodeReturn,
    epsilon: Number(game.ai.epsilon.toFixed(6)),
    alpha: game.ai.alpha,
    gamma: game.ai.gamma,
    qStates: game.ai.qTable.size,
    enemyWins: game.enemyWins,
    playerWins: game.playerWins,
    explored: game.ai.explored,
    exploited: game.ai.exploited,
    endedAt: new Date().toISOString(),
  });
}

function finishEpisode(reason) {
  let terminalReward = 0;
  if (reason === "enemy_win") {
    game.enemyWins += 1;
    terminalReward = 8;
  } else if (reason === "player_win") {
    game.playerWins += 1;
    terminalReward = -8;
  } else {
    terminalReward = -1.4;
  }
  addReward(terminalReward, reason);

  const terminalState = stateKeyFromVector(getStateVector());
  if (game.ai.lastState !== null && game.ai.lastAction !== null && game.ai.training) {
    updateQ(game.ai.lastState, game.ai.lastAction, game.ai.rewardBuffer, terminalState, true);
  }

  if (game.ai.training) {
    game.ai.epsilon = Math.max(EPSILON_MIN, game.ai.epsilon * EPSILON_DECAY);
    epsilonInput.value = game.ai.epsilon.toFixed(3);
  }

  recordEpisodeMetrics(reason);
  game.episode += 1;
  if (game.ai.training && game.trainingHistory.episodeReturns.length >= TRAINING_STOP_EPISODES) {
    game.ai.training = false;
    game.ai.stoppedAtEpisode = game.episode;
    if (trainToggle) trainToggle.checked = false;
    game.mode = "paused";
    setModelStatus(`训练已达到 ${TRAINING_STOP_EPISODES} 回合，已停止训练 (${new Date().toLocaleTimeString()})`);
  }
  resetEpisode();
}

function update(dt) {
  if (game.mode !== "running") return;

  game.time += dt;
  game.episodeTime += dt;
  addReward(0.01, "survival");

  handlePlayerInput(dt);
  applyCooldowns(dt);

  if (game.episodeTime - game.ai.lastDecisionAt >= DECISION_INTERVAL) {
    enemyDecisionTick();
  }
  applyEnemyAction(dt);
  updateBullets(dt);

  if (game.player.hp <= 0) finishEpisode("enemy_win");
  else if (game.enemy.hp <= 0) finishEpisode("player_win");
  else if (game.episodeTime >= EPISODE_TIME_LIMIT) finishEpisode("timeout");
}

function drawBackground() {
  const grd = ctx.createLinearGradient(0, 0, 0, WORLD.height);
  grd.addColorStop(0, "#c5d4ab");
  grd.addColorStop(1, "#9db786");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, WORLD.width, WORLD.height);

  ctx.strokeStyle = "rgba(33, 60, 35, 0.13)";
  ctx.lineWidth = 1;
  for (let x = 0; x < WORLD.width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, WORLD.height);
    ctx.stroke();
  }
  for (let y = 0; y < WORLD.height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(WORLD.width, y);
    ctx.stroke();
  }
}

function drawWalls() {
  for (const wall of game.walls) {
    ctx.fillStyle = "#60715f";
    ctx.fillRect(wall.x, wall.y, wall.w, wall.h);
    ctx.strokeStyle = "#354937";
    ctx.lineWidth = 2;
    ctx.strokeRect(wall.x + 0.5, wall.y + 0.5, wall.w - 1, wall.h - 1);
  }
}

function drawTank(tank) {
  const angle = Math.atan2(tank.dirY, tank.dirX);
  ctx.save();
  ctx.translate(tank.x, tank.y);
  ctx.rotate(angle);

  ctx.fillStyle = tank.color;
  ctx.fillRect(-16, -12, 32, 24);
  ctx.fillStyle = "#efefe8";
  ctx.fillRect(-5, -8, 10, 16);
  ctx.fillStyle = "#1f1f1a";
  ctx.fillRect(0, -3, 22, 6);

  ctx.restore();

  const hpWidth = 42;
  const ratio = tank.hp / tank.maxHp;
  ctx.fillStyle = "rgba(30, 40, 35, 0.7)";
  ctx.fillRect(tank.x - hpWidth / 2, tank.y - 28, hpWidth, 5);
  ctx.fillStyle = ratio > 0.4 ? "#5fbc65" : "#de5d51";
  ctx.fillRect(tank.x - hpWidth / 2, tank.y - 28, hpWidth * ratio, 5);
}

function drawBullets() {
  for (const b of game.bullets) {
    ctx.fillStyle = b.owner === "player" ? "#f4f4f4" : "#ffd38a";
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawHud() {
  ctx.fillStyle = "rgba(10, 15, 14, 0.75)";
  ctx.fillRect(12, 10, 350, 84);
  ctx.fillStyle = "#f8f8f2";
  ctx.font = "14px monospace";
  ctx.fillText(`Episode: ${game.episode} | Mode: ${game.mode}`, 20, 33);
  ctx.fillText(`Player HP: ${game.player.hp} | Enemy HP: ${game.enemy.hp}`, 20, 53);
  ctx.fillText(`AI Action: ${game.ai.lastActionName} | Reward: ${game.ai.lastReward.toFixed(2)}`, 20, 73);
  ctx.fillText(`Episode Return: ${game.ai.episodeReturn.toFixed(2)} | Epsilon: ${game.ai.epsilon.toFixed(3)}`, 20, 93);
}

function renderTrainingChart(targetCtx, targetCanvas, options) {
  if (!targetCtx) return;
  const w = targetCanvas.width;
  const h = targetCanvas.height;
  targetCtx.clearRect(0, 0, w, h);
  targetCtx.fillStyle = "#fbf9f0";
  targetCtx.fillRect(0, 0, w, h);

  const history = game.trainingHistory.episodeReturns;
  const outcomes = game.trainingHistory.outcomes;
  if (history.length === 0) {
    targetCtx.fillStyle = "#45544d";
    targetCtx.font = "13px monospace";
    targetCtx.fillText("暂无训练数据，完成几回合后显示曲线", 20, h / 2);
    return;
  }

  const left = 42;
  const right = 14;
  const top = 16;
  const bottom = 28;
  const innerW = w - left - right;
  const innerH = h - top - bottom;
  const totalPoints = history.length;
  const startIndex = Math.max(0, options?.startIndex ?? 0);
  const endIndex = Math.min(totalPoints, options?.endIndex ?? totalPoints);
  const viewHistory = history.slice(startIndex, endIndex);
  const viewOutcomes = outcomes.slice(startIndex, endIndex);
  const maxPoints = options?.maxPoints ?? CHART_RENDER_MAX_POINTS;
  const viewTotal = viewHistory.length;
  const step = Math.max(1, Math.ceil(viewTotal / maxPoints));
  const sampledIndices = [];
  for (let i = 0; i < viewTotal; i += step) sampledIndices.push(i);
  if (sampledIndices[sampledIndices.length - 1] !== viewTotal - 1) {
    sampledIndices.push(viewTotal - 1);
  }
  const n = sampledIndices.length;

  let minReturn = viewHistory[sampledIndices[0]];
  let maxReturn = viewHistory[sampledIndices[0]];
  for (const idx of sampledIndices) {
    const v = viewHistory[idx];
    minReturn = Math.min(minReturn, v);
    maxReturn = Math.max(maxReturn, v);
  }
  if (Math.abs(maxReturn - minReturn) < 1e-4) {
    maxReturn += 1;
    minReturn -= 1;
  }
  const pad = (maxReturn - minReturn) * 0.1;
  const yMin = minReturn - pad;
  const yMax = maxReturn + pad;
  const scaleReturn = (value) => top + ((yMax - value) / (yMax - yMin)) * innerH;
  const scaleWinRate = (value) => top + ((100 - value) / 100) * innerH;
  const scaleX = (i) => left + (n === 1 ? 0 : (i / (n - 1)) * innerW);

  targetCtx.strokeStyle = "#91a196";
  targetCtx.lineWidth = 1;
  targetCtx.beginPath();
  targetCtx.moveTo(left, top);
  targetCtx.lineTo(left, top + innerH);
  targetCtx.lineTo(left + innerW, top + innerH);
  targetCtx.stroke();

  targetCtx.fillStyle = "#516257";
  targetCtx.font = "11px monospace";
  targetCtx.fillText(`${yMax.toFixed(1)}`, 4, top + 4);
  targetCtx.fillText(`${yMin.toFixed(1)}`, 4, top + innerH);
  targetCtx.fillText("回合", left + innerW - 26, h - 6);
  targetCtx.fillText("回报", 4, 12);
  targetCtx.fillText("胜率%", w - 38, 12);

  targetCtx.strokeStyle = "#e06a2c";
  targetCtx.lineWidth = 2;
  targetCtx.beginPath();
  for (let i = 0; i < n; i += 1) {
    const idx = sampledIndices[i];
    const x = scaleX(i);
  const y = scaleReturn(viewHistory[idx]);
    if (i === 0) targetCtx.moveTo(x, y);
    else targetCtx.lineTo(x, y);
  }
  targetCtx.stroke();

  const movingWindow = Math.max(1, Math.min(viewHistory.length, options?.movingWindow ?? MOVING_WINDOW));
  targetCtx.strokeStyle = "#1f9a8a";
  targetCtx.lineWidth = 2;
  targetCtx.beginPath();
  for (let i = 0; i < n; i += 1) {
    const idx = sampledIndices[i];
    const moving = getMovingAverage(viewHistory, idx, movingWindow);
    const x = scaleX(i);
    const y = scaleReturn(moving);
    if (i === 0) targetCtx.moveTo(x, y);
    else targetCtx.lineTo(x, y);
  }
  targetCtx.stroke();

  targetCtx.strokeStyle = "#5b4bdb";
  targetCtx.lineWidth = 2;
  targetCtx.setLineDash([6, 4]);
  targetCtx.beginPath();
  for (let i = 0; i < n; i += 1) {
    const idx = sampledIndices[i];
    const winRate = getMovingAverage(viewOutcomes, idx, movingWindow) * 100;
    const x = scaleX(i);
    const y = scaleWinRate(winRate);
    if (i === 0) targetCtx.moveTo(x, y);
    else targetCtx.lineTo(x, y);
  }
  targetCtx.stroke();
  targetCtx.setLineDash([]);

  targetCtx.fillStyle = "#e06a2c";
  targetCtx.fillRect(left, h - 18, 10, 2);
  targetCtx.fillStyle = "#1f9a8a";
  targetCtx.fillRect(left + 110, h - 18, 10, 2);
  targetCtx.fillStyle = "#5b4bdb";
  targetCtx.fillRect(left + 240, h - 18, 10, 2);
  targetCtx.fillStyle = "#45544d";
  targetCtx.fillText("回报", left + 14, h - 14);
  targetCtx.fillText(`回报MA${movingWindow}`, left + 124, h - 14);
  targetCtx.fillText(`胜率MA${movingWindow}`, left + 254, h - 14);

  const latestIdx = viewHistory.length - 1;
  const latestWinRate = getMovingAverage(viewOutcomes, latestIdx, movingWindow) * 100;
  targetCtx.fillStyle = "#5b4bdb";
  targetCtx.fillText(`最新胜率：${latestWinRate.toFixed(1)}%`, left + innerW - 120, top + 12);
}

function drawTrainingChart() {
  renderTrainingChart(chartCtx, trainingChart, {
    movingWindow: MOVING_WINDOW,
    maxPoints: CHART_RENDER_MAX_POINTS,
  });
}

function drawRecentTrainingChart() {
  if (!recentTrainingChart || !recentChartCtx) return;
  const recentWindow = Math.min(20, game.trainingHistory.episodeReturns.length);
  const startIndex = Math.max(0, game.trainingHistory.episodeReturns.length - recentWindow);
  renderTrainingChart(recentChartCtx, recentTrainingChart, {
    movingWindow: Math.max(1, Math.min(10, recentWindow)),
    maxPoints: recentWindow,
    startIndex,
    endIndex: game.trainingHistory.episodeReturns.length,
  });
}

function drawOverlay() {
  if (game.mode === "running") return;
  ctx.fillStyle = "rgba(20, 23, 20, 0.55)";
  ctx.fillRect(0, 0, WORLD.width, WORLD.height);
  ctx.fillStyle = "#fefefe";
  ctx.textAlign = "center";
  ctx.font = "bold 42px Trebuchet MS";
  const title = game.mode === "menu" ? "Tank Q-Learning Arena" : "Paused";
  ctx.fillText(title, WORLD.width / 2, WORLD.height / 2 - 22);
  ctx.font = "20px Trebuchet MS";
  ctx.fillText("按开始按钮进入对战，观察AI策略逐回合优化", WORLD.width / 2, WORLD.height / 2 + 12);
  ctx.textAlign = "left";
}

function render() {
  drawBackground();
  drawWalls();
  drawTank(game.player);
  drawTank(game.enemy);
  drawBullets();
  drawHud();
  drawOverlay();
  drawTrainingChart();
  drawRecentTrainingChart();
  updateStatePanel();
}

function updateStatePanel() {
  const hist = game.trainingHistory.episodeReturns;
  const episodeRecords = game.trainingHistory.episodes;
  const lastRecord = episodeRecords[episodeRecords.length - 1] || null;
  const recentIdx = hist.length - 1;
  const recentAvgReturn = recentIdx >= 0 ? getMovingAverage(hist, recentIdx, MOVING_WINDOW) : 0;
  const recentWinRate = recentIdx >= 0
    ? getMovingAverage(game.trainingHistory.outcomes, recentIdx, MOVING_WINDOW) * 100
    : 0;

  const payload = {
    mode: game.mode,
    episode: game.episode,
    timeSec: Number(game.episodeTime.toFixed(2)),
    score: { player: game.playerWins, enemy: game.enemyWins },
    ai: {
      state: game.ai.stateKey,
      action: game.ai.lastActionName,
      reward: Number(game.ai.lastReward.toFixed(3)),
      rewardReason: game.ai.lastRewardReason,
      episodeReturn: Number(game.ai.episodeReturn.toFixed(3)),
      epsilon: Number(game.ai.epsilon.toFixed(3)),
      alpha: game.ai.alpha,
      gamma: game.ai.gamma,
      training: game.ai.training,
  stoppedAtEpisode: game.ai.stoppedAtEpisode,
      qStates: game.ai.qTable.size,
      explored: game.ai.explored,
      exploited: game.ai.exploited,
    },
    training: {
      episodesRecorded: game.trainingHistory.episodeReturns.length,
      detailedRecords: episodeRecords.length,
      movingAverageReturn: Number(recentAvgReturn.toFixed(3)),
      movingEnemyWinRatePct: Number(recentWinRate.toFixed(1)),
      movingWindow: MOVING_WINDOW,
      lastEpisode: lastRecord,
    },
    model: {
      hasSavedModel: Boolean(localStorage.getItem(MODEL_STORAGE_KEY)),
      savedAt: game.modelMeta.savedAt,
      loadedAt: game.modelMeta.loadedAt,
    },
    player: {
      x: Number(game.player.x.toFixed(1)),
      y: Number(game.player.y.toFixed(1)),
      hp: game.player.hp,
    },
    enemy: {
      x: Number(game.enemy.x.toFixed(1)),
      y: Number(game.enemy.y.toFixed(1)),
      hp: game.enemy.hp,
    },
    bullets: game.bullets.length,
  };
  stateView.textContent = JSON.stringify(payload, null, 2);

  if (trainingMetrics) {
    const totalEpisodes = game.trainingHistory.episodeReturns.length;
    const latestReturn = totalEpisodes > 0 ? game.trainingHistory.episodeReturns[totalEpisodes - 1] : 0;
    const movingReturn = Number(recentAvgReturn.toFixed(2));
    const movingWinRate = Number(recentWinRate.toFixed(1));
    const scoreText = `${game.playerWins} / ${game.enemyWins}`;
    const eps = Number(game.ai.epsilon.toFixed(3));
    const qStates = game.ai.qTable.size;
    const stoppedText = game.ai.stoppedAtEpisode
      ? `已停止（第 ${game.ai.stoppedAtEpisode} 回合）`
      : "未停止";

    trainingMetrics.innerHTML = `
      <p>回合数：${totalEpisodes}</p>
      <p>最新回报：${latestReturn.toFixed(2)}</p>
      <p>回报移动平均：${movingReturn}</p>
      <p>胜率移动平均：${movingWinRate}%</p>
      <p>当前比分（玩家/AI）：${scoreText}</p>
      <p>Epsilon：${eps}</p>
      <p>Q状态数：${qStates}</p>
      <p>训练停止：${stoppedText}</p>
      <p>停止规则：训练达到 ${TRAINING_STOP_EPISODES} 回合后自动停止</p>
    `;
  }
}

function togglePause() {
  if (game.mode === "running") game.mode = "paused";
  else if (game.mode === "paused" || game.mode === "menu") game.mode = "running";
}

function toggleFullscreen() {
  const host = document.querySelector(".game-panel");
  if (!document.fullscreenElement) {
    host.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen().catch(() => {});
  }
}

startBtn.addEventListener("click", () => {
  game.mode = "running";
});

pauseBtn.addEventListener("click", () => {
  togglePause();
});

resetBtn.addEventListener("click", () => {
  resetEpisode();
  game.mode = "running";
});

saveModelBtn.addEventListener("click", () => {
  saveModelToLocalStorage();
});

loadModelBtn.addEventListener("click", () => {
  loadModelFromLocalStorage();
});

clearModelBtn.addEventListener("click", () => {
  clearStoredModel();
});

exportResultsBtn.addEventListener("click", () => {
  exportTrainingResults();
});

trainToggle.addEventListener("change", () => {
  game.ai.training = trainToggle.checked;
});

applyParamsBtn.addEventListener("click", () => {
  game.ai.epsilon = clamp(Number(epsilonInput.value), 0, 1);
  game.ai.alpha = clamp(Number(alphaInput.value), 0, 1);
  game.ai.gamma = clamp(Number(gammaInput.value), 0, 1);
});

window.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    if (game.mode === "running") spawnBullet(game.player, "player");
  }
  if (event.code === "KeyP") {
    togglePause();
  }
  if (event.code === "KeyF") {
    toggleFullscreen();
  }
  keys.add(event.code);
});

window.addEventListener("keyup", (event) => {
  keys.delete(event.code);
});

window.render_game_to_text = () => {
  const hist = game.trainingHistory.episodeReturns;
  const episodeRecords = game.trainingHistory.episodes;
  const lastRecord = episodeRecords[episodeRecords.length - 1] || null;
  const recentIdx = hist.length - 1;
  const recentAvgReturn = recentIdx >= 0 ? getMovingAverage(hist, recentIdx, MOVING_WINDOW) : 0;
  const recentWinRate = recentIdx >= 0
    ? getMovingAverage(game.trainingHistory.outcomes, recentIdx, MOVING_WINDOW) * 100
    : 0;
  return JSON.stringify({
    coordinate_system: "origin=(0,0) at top-left, +x right, +y down",
    mode: game.mode,
    episode: game.episode,
    player: {
      x: Number(game.player.x.toFixed(1)),
      y: Number(game.player.y.toFixed(1)),
      hp: game.player.hp,
      dir: [Number(game.player.dirX.toFixed(2)), Number(game.player.dirY.toFixed(2))],
      cooldown: Number(game.player.cooldownLeft.toFixed(2)),
    },
    enemy: {
      x: Number(game.enemy.x.toFixed(1)),
      y: Number(game.enemy.y.toFixed(1)),
      hp: game.enemy.hp,
      action: game.ai.lastActionName,
      epsilon: Number(game.ai.epsilon.toFixed(3)),
      reward: Number(game.ai.lastReward.toFixed(2)),
    },
    bullets: game.bullets.map((b) => ({
      x: Number(b.x.toFixed(1)),
      y: Number(b.y.toFixed(1)),
      owner: b.owner,
      vx: Number(b.vx.toFixed(1)),
      vy: Number(b.vy.toFixed(1)),
    })),
    training: {
      episodesRecorded: hist.length,
      detailedRecords: episodeRecords.length,
      movingAverageReturn: Number(recentAvgReturn.toFixed(2)),
      movingEnemyWinRatePct: Number(recentWinRate.toFixed(1)),
      qStates: game.ai.qTable.size,
      lastEpisode: lastRecord,
    },
    model: {
      saved: Boolean(localStorage.getItem(MODEL_STORAGE_KEY)),
      savedAt: game.modelMeta.savedAt,
    },
    score: { player: game.playerWins, enemy: game.enemyWins },
  });
};

window.advanceTime = (ms) => {
  const frameMs = 1000 / 60;
  const steps = Math.max(1, Math.round(ms / frameMs));
  for (let i = 0; i < steps; i += 1) {
    update(1 / 60);
  }
  render();
};

let lastTs = performance.now();
function frame(ts) {
  const dt = Math.min(0.05, (ts - lastTs) / 1000);
  lastTs = ts;
  update(dt);
  render();
  window.requestAnimationFrame(frame);
}

initializeModelStatus();
resetEpisode();
render();
window.requestAnimationFrame(frame);
