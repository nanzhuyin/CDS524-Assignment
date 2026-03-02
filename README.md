# CDS524 Assignment 1 - Tank Design Game (Q-Learning)


## 1. 游戏设计 (Game Design)

### 1.1 Objective and Rules
- 玩家坦克（蓝色）与 AI 坦克（红色）在地图中对战。
- 每辆坦克初始 3 点生命值。
- 子弹可与边界/墙体发生有限反弹。
- 玩家生命值归零则 AI 获胜；AI 生命值归零则玩家获胜；超时按超时结束。

### 1.2 State Space
敌方 AI 的状态向量离散化为：
- `dx_bucket`: 玩家与敌方 x 相对位置分桶（-2/-1/0/1/2）
- `dy_bucket`: 玩家与敌方 y 相对位置分桶（-2/-1/0/1/2）
- `enemy_hp_bucket`: 敌方血量桶（高/低）
- `player_hp_bucket`: 玩家血量桶（高/低）
- `threat`: 是否存在接近敌方的玩家子弹（0/1）

状态 key 形如：`dx|dy|enemy_hp|player_hp|threat`。

### 1.3 Action Space
离散动作共 6 个：
- `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `SHOOT`

### 1.4 Reward Function
在 `game.js` 中实现了同时包含正负奖励的 reward 机制：
- 生存奖励：`+0.01 / frame`
- 向玩家施压（距离变近）：`+0.025`
- 远离玩家（压力下降）：`-0.015`
- 敌方发射子弹：`+0.03`
- 敌方撞墙：`-0.08`
- 敌方命中玩家：`+2.8`
- 敌方被玩家命中：`-2.6`
- 终局奖励：敌胜 `+8`，敌败 `-8`，超时 `-1.4`

## 2. Q-Learning Implementation

### 2.1 算法
采用 Q-table 的离散 Q-learning：

`Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))`

### 2.2 探索与利用
- 策略：epsilon-greedy
- 参数：
  - `epsilon` 默认 0.5（每回合衰减，最小 0.05）
  - `alpha` 默认 0.18
  - `gamma` 默认 0.92
- 可在 UI 中实时修改 `epsilon/alpha/gamma`。

### 2.3 学习流程
- 每个决策间隔（`DECISION_INTERVAL = 0.17s`）更新一次 Q 值。
- 终局时进行一次 terminal update。
- 回合结束后自动重置，并继续累积经验。

## 3. Game Interaction (UI)

项目提供完整交互界面并满足“显示状态、动作、奖励”要求：
- Canvas 实时渲染战场、坦克、子弹、生命条。
- 右侧面板实时显示：
  - 当前状态 key
  - AI 当前动作
  - 最新奖励与奖励原因
  - 回合累计回报
  - epsilon/alpha/gamma 与 Q 表大小
  - 训练曲线（回报曲线、移动平均回报、移动平均敌方胜率）
- 控制按钮：
  - `开始/继续`、`暂停`、`重置回合`、`AI训练模式` 开关、`玩家自动控制` 开关
  - `保存模型`（保存全量训练数据到 localStorage）
  - `加载模型`（恢复已保存训练结果）
  - `清空模型`（删除本地模型并重置）
  - `导出训练结果`（导出单个 JSON，内含训练曲线图数据）
- 键盘（当关闭“玩家自动控制”时）：
  - `WASD` 移动
  - `Space` 射击
  - `P` 暂停
  - `F` 全屏切换

## 4. 运行方法与自动训练

在项目根目录执行：

```bash
python3 -m http.server 8000
```

然后浏览器打开：`http://localhost:8000`

**如何进行自动训练：**
1. 页面加载后，默认勾选了“玩家自动控制”和“AI训练模式”。
2. 点击“开始 / 继续”，玩家坦克将由内置的规则策略自动控制（自动躲避子弹、瞄准射击、保持距离）。
3. 游戏将自动进行多回合对战，AI 坦克会持续更新 Q-table。
4. 训练一段时间后，点击“导出训练结果”即可获得包含训练曲线和 Q-table 的 JSON 文件，用于作业提交。
5. 如果需要手动演示，只需取消勾选“玩家自动控制”，即可使用键盘操作玩家坦克。


## 5. 训练结果查看与模型复用

- 训练曲线位置：右侧 `训练曲线` 面板
  - 橙线：每回合回报
  - 青线：移动平均回报（窗口 20 回合）
  - 紫虚线：移动平均敌方胜率（窗口 20 回合）
  - 新增：`近20回合曲线` 用于观察最近训练是否收敛
  - 训练达到 1000 回合后自动停止，并在状态栏标注停止回合
  - 新增 `训练指标` 文本面板：实时显示回合数、回报、胜率、epsilon 与停止状态
- 保存模型：点击 `保存模型`（保存到浏览器 localStorage）
- 加载模型：点击 `加载模型`（恢复 Q-table、参数、统计）
- 加载本地模型：点击 `加载本地模型` 选择本地 JSON（支持导出的训练结果 JSON 或单独模型快照 JSON）
- 清空模型：点击 `清空模型`（删除本地模型并重置训练）
- 导出训练结果：点击 `导出训练结果`
  - `tank-training-results-*.json`：包含 Q-table、全量每回合记录、汇总指标
  - `tank-training-chart-*.png`：训练曲线图像（PNG 文件，独立导出）
  - 导出的 PNG 使用“全量回合”绘制（不是仅 20/50 回合窗口）

## 6. 代码结构

- `/Users/zhishixuebao/Documents/New project/index.html`
- `/Users/zhishixuebao/Documents/New project/style.css`
- `/Users/zhishixuebao/Documents/New project/game.js`
- `/Users/zhishixuebao/Documents/New project/README.md`
- `/Users/zhishixuebao/Documents/New project/progress.md`
