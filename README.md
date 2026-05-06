# AI 安全與價值對齊 — 實作課程

從零實作一套「AI 憲法 → Reward 函數 → GRPO 對齊訓練」的完整安全對齊流程。

---

## 課程概覽

| Lab | 主題 | 預估時間 | 程式檔案 | 詳細引導 |
|-----|------|---------|---------|---------|
| Lab 1 | AI 憲法定義 — 制定模型行為規範 | 1 小時 | `lab1/constitution.py` | [`lab1/README.md`](lab1/README.md) |
| Lab 2 | Reward 函數 — Regex 硬限制 + LLM-as-a-Judge 軟限制 | 2 小時 | `lab2/reward.py` | [`lab2/README.md`](lab2/README.md) |
| Lab 2 驗證 | 安全測試 — 化武配方 / 偏見引導 / 角色扮演攻擊 | 0.5 小時 | `lab2/test_cases.py` | [`lab2/README.md`](lab2/README.md) |
| Lab 3 | GRPO 安全對齊訓練 — `trl.GRPOTrainer` + LoRA | 2.5 小時 | `lab3/grpo_training.py` | [`lab3/README.md`](lab3/README.md) |

---

## 推薦學習路徑（兩階段）

> **重要：請務必照下面的順序走。第一階段先用 ROUTE_A 跑完整套流程，第二階段再切到 ROUTE_B 比較差異。** 這樣才能體會「同一個模型 + 不同憲法 = 不同行為」的對齊概念。

### 階段一：使用 `CONSTITUTION_ROUTE_A`（極致防禦）跑完整套流程

| 順序 | 動作 | 對應 TODO |
|---|---|---|
| 1 | 閱讀 Lab 1 的 ROUTE_A 規則設計（**先不要動 ROUTE_A**） | — |
| 2 | 完成 Lab 2 的 reward 函數（硬限制 + 軟限制 + 綜合計算） | Lab 2-1c / 2-2a / 2-3 |
| 3 | 執行 `python main.py`，驗證 Lab 2 的 reward 區分能力 | — |
| 4 | 完成 Lab 3 的 reward 整合並啟動 GRPO 訓練（`ACTIVE_ROUTE="A"`） | Lab 3-3 / 3-6 |
| 5 | 觀察 Reward / KL / Loss 三條曲線是否健康 | — |

### 階段二：自己設計 `CONSTITUTION_ROUTE_B`（學術開放）並比較差異

| 順序 | 動作 | 對應 TODO |
|---|---|---|
| 1 | 完成 Lab 1-1，定義 ROUTE_B 的 3 條規則 | Lab 1-1 |
| 2 | 把 `main.py` 第 33 行的 `ACTIVE_ROUTE = "A"` 改成 `"B"` | — |
| 3 | 重跑 `python main.py`，觀察「相同回應在不同憲法下的軟限制分數差異」 | — |
| 4 | 把 `lab3/grpo_training.py` 第 47 行的 `ACTIVE_ROUTE = "A"` 改成 `"B"` | — |
| 5 | 重跑 GRPO 訓練（建議改 `output_dir` 避免覆蓋階段一結果），比較兩次曲線 | — |

> 階段二的核心問題：**換了憲法之後，「邊界案例」（例如純粹的學術討論）的 reward 是否因此提升？模型訓練後的拒答行為是否也跟著變寬鬆？**

---

## 關鍵連動：Lab 1 的憲法如何影響 Lab 2 的軟限制？

這是整套課程的**核心設計思想**，也是 Constitutional AI 概念的具體化。

```
┌──────────────────────────────────────────────────────────────────┐
│  [Lab 1]  CONSTITUTION_ROUTE_A / ROUTE_B (dict)                  │
│           rule_1, rule_2, rule_3 ...                             │
└──────────────────────────────────────────────────────────────────┘
                            │
                            │  get_active_constitution(ACTIVE_ROUTE)
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  [Lab 2]  reward_soft_constraint(response, constitution)         │
│           └─► build_judge_prompt(response, constitution)         │
│                  └─► 把 constitution 的每條 rule.description     │
│                      寫進 LLM-as-a-Judge 的 system message       │
└──────────────────────────────────────────────────────────────────┘
                            │
                            │  soft_score
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  [Lab 2]  compute_total_reward                                   │
│           total = 0.6 * hard_score + 0.4 * soft_score            │
└──────────────────────────────────────────────────────────────────┘
                            │
                            │  total_reward
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│  [Lab 3]  safety_reward_function (GRPOTrainer.reward_funcs)      │
│           → 變成模型訓練的梯度方向                                │
└──────────────────────────────────────────────────────────────────┘
```

**換句話說：**

> **改寫 Lab 1 的憲法 ＝ 重新定義 Lab 2 軟限制 Judge 的評分標準 ＝ 重新定義 Lab 3 模型「想追求的目標」。**

這就是為什麼階段二只要改 Lab 1 的 ROUTE_B 字典，**整條 reward 鏈、整個訓練目標都會跟著變**——你不需要改 Lab 2/3 任何一行程式碼，模型的行為偏好就會被重塑。

> **硬限制（regex）不受憲法影響**——它是固定的紅線；只有**軟限制**會吸收憲法的「價值觀」。這也是雙層防禦的設計理由：硬限制守底線、軟限制做語意判斷。

---

## 專案結構

```
2026_ai_rookie_course9/
├── main.py                  ← 主程式入口（Lab 1 + Lab 2 + 驗證區）
├── pyproject.toml           ← 套件相依性（uv sync）
├── README.md                ← ★ 本檔（課程總覽 + 學習路徑）
│
├── lab1/                    ← Lab 1：AI 憲法定義區
│   ├── README.md            ← ★ Lab 1 詳細引導
│   ├── __init__.py
│   └── constitution.py      ← 定義路線 A/B 的評分規則
│
├── lab2/                    ← Lab 2：Reward 腳本區 + 驗證測試
│   ├── README.md            ← ★ Lab 2 詳細引導
│   ├── __init__.py
│   ├── reward.py            ← 硬限制 / 軟限制 / 綜合獎勵
│   └── test_cases.py        ← 三個攻擊測試案例 + 評估函數
│
└── lab3/                    ← Lab 3：GRPO 訓練與監控
    ├── README.md            ← ★ Lab 3 詳細引導
    ├── __init__.py
    ├── grpo_training.py     ← GRPOTrainer + LoRA + Reward 整合 + 監控曲線
    └── system_prompt.txt    ← 訓練時注入的 system prompt
```

---

## 環境設置

```bash
uv sync
```

---

## 使用方式

**Lab 1 + Lab 2 + 驗證區**（不需 GPU）：

```bash
python main.py
```

**Lab 3 GRPO 訓練**（需 GPU，建議 ≥ 12 GB 顯存）：

```bash
python -m lab3.grpo_training
```

> 沒有 GPU 也可以閱讀 Lab 3 的程式碼了解流程，或使用 Google Colab / Kaggle 提供的免費 GPU。

---

## 補充：KL Divergence

GRPO 訓練中用來監控策略模型 (π) 與參考模型 (π\_ref) 的偏移程度，防止 reward hacking：

$$KL(P \| Q) = \sum P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$

> 訓練時若 KL 在前幾個 step 就快速衝高（例如 > 5），代表 learning rate 太大或 `kl_coef` 太小，請參考 [`lab3/README.md`](lab3/README.md) 的調參指引。

---

## TODO 總檢查清單

> 各 TODO 的詳細說明請見對應 lab 的 README。

### 階段一（必做）

#### Lab 2 — `lab2/reward.py`
- [ ] **Lab 2-1c**：完成 `reward_hard_constraint` 中 high / medium 的扣分邏輯
- [ ] **Lab 2-2a**：設計 LLM-as-a-Judge 評分 Prompt（**這裡會用到 Lab 1 的 constitution**）
- [ ] **Lab 2-3**：完成 `compute_total_reward` 的 `total_reward` 計算

#### Lab 3 — `lab3/grpo_training.py`
- [ ] **Lab 3-3**：完成 `safety_reward_function` 的獎勵整合邏輯
- [ ] **Lab 3-6**：調整 GRPO 超參數（`kl_coef` / `temperature` / `num_generations` / `learning_rate`）

### 階段二（必做）

#### Lab 1 — `lab1/constitution.py`
- [ ] **Lab 1-1**：定義 `CONSTITUTION_ROUTE_B`（學術開放）的 3 條評分規則

### 選做擴充
- [ ] **Lab 2-1a/b**：補充更多 critical / high / medium 等級的敏感詞 regex
- [ ] **Lab 2-2b**：實作 LLM-as-a-Judge 的真實 API 呼叫
- [ ] **Lab 3-2**：新增更多訓練 prompts
- [ ] **Lab 3-4**：調整 LoRA 參數（`r` / `lora_alpha`）觀察效果差異
