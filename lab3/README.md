# Lab 3：GRPO 安全對齊訓練（GRPO Safety Alignment Training）

> 預估時間：2.5 小時 ｜ **需要 GPU**（建議 ≥ 12 GB 顯存）｜ 對應檔案：[`grpo_training.py`](grpo_training.py)

---

## 學習目標

- 用 `trl.GRPOTrainer` 把 Lab 1 + Lab 2 的 reward 真正轉換成**模型訓練梯度**。
- 使用 **LoRA** 在小資源下完成參數高效微調（PEFT）。
- 學會解讀 GRPO 訓練的三條核心曲線：**Reward / KL Divergence / Loss**。
- 實際操作對齊訓練的調參，建立超參數直覺。

---

## 與其他 Lab 的關聯

```
[Lab 1] CONSTITUTION_ROUTE_A / B
            │
            │  get_active_constitution(ACTIVE_ROUTE)
            │  ★ 第 47 行的 ACTIVE_ROUTE 變數決定用哪條路線
            ▼
[Lab 2] reward_hard_constraint + reward_soft_constraint
            │
            ▼
[Lab 3] safety_reward_function(completions, prompts)
            │  ★ 你在這裡組合 hard / soft 為 total_reward
            ▼
GRPOTrainer.reward_funcs = safety_reward_function
            │
            ▼
模型每生成一個回應 → 拿到分數 → 反向傳播更新 LoRA 權重
```

---

## GRPO 概念速覽

**GRPO (Group Relative Policy Optimization)** 是 DeepSeek-R1 中採用的 RL 演算法。

核心想法：
1. 對每個 prompt 生成 **N 個**回應（`num_generations`，又稱 group size）。
2. 用 reward 函數對這 N 個回應打分。
3. 把分數**在 group 內做 z-score 正規化**（減去 group mean、除以 group std）。
4. 用正規化後的 advantage 更新策略。
5. 加上 KL 散度懲罰，避免策略過度偏離原始模型。

**GRPO vs PPO 的關鍵差異：**

| 特性 | PPO | GRPO |
|---|---|---|
| Critic 模型 | 需要（與 actor 同等大小） | **不需要**（用 group baseline 取代） |
| 顯存需求 | 高（4 份模型） | 較低（3 份模型） |
| Reward baseline | 由 critic 預測 | 由 group 內樣本平均估計 |
| 適合場景 | 連續決策、稠密 reward | 離散生成、稀疏 reward |

---

## 訓練 Pipeline 一覽

```
1. 載入 Qwen2.5-3B-Instruct + LoRA       (load_model_and_tokenizer)
2. 把 TRAINING_PROMPTS 套上 chat template (prepare_dataset)
3. 設定 GRPOConfig + safety_reward_function
4. trainer.train() 啟動訓練
5. 從 trainer.state.log_history 取出 reward / kl / loss 繪圖
```

---

## TODO 清單

### Lab 3-3：完成 `safety_reward_function` 的整合邏輯（**必做**）

開啟 `grpo_training.py`，找到 `## TODO [Lab 3-3]`。在 for 迴圈中：

```python
hard_score, hard_flags = reward_hard_constraint(completion)
soft_score, soft_reason = reward_soft_constraint(completion, constitution)

if hard_score == 0.0:
    total_reward = 0.0
else:
    total_reward = hard_weight * hard_score + soft_weight * soft_score

rewards.append(total_reward)
```

> 邏輯與 Lab 2-3 完全相同。差別只在這裡是 batch 處理（一次處理多個 completion）。

---

### Lab 3-6：調整 GRPO 超參數（**必做**，至少嘗試 1~2 組）

開啟 `grpo_training.py` 的 `GRPOConfig`。建議的調參方向：

| 超參數 | 預設值 | 增大的影響 | 減小的影響 |
|---|---|---|---|
| `num_generations` | 2 | group baseline 更穩定，但慢一倍 | 訓練快但變異大 |
| `kl_coef` | 0.05 | 策略保守、reward 慢升 | 策略激進、reward 快升但有 hacking 風險 |
| `temperature` | 0.7 | 探索更廣、樣本更多元 | 行為更確定、可能陷入單一模式 |
| `learning_rate` | 1e-4 | 變化快、可能 KL 爆炸 | 穩定但學不動 |

> **註**：本 lab 預設 `learning_rate=1e-4` 是 **LoRA SFT** 等級的學習率，對 GRPO 偏激進——選這個值是為了在教學情境下能快速看到曲線變化。
> 若你觀察到 KL 在前 5 個 step 就 > 5，請把 LR 降到 `1e-5`，這才比較接近 GRPO 的實務值。

#### 推薦的兩組對照實驗

| 配置 | num_generations | kl_coef | temperature | learning_rate | 預期觀察 |
|---|---|---|---|---|---|
| 保守組 | 4 | 0.1 | 0.5 | 1e-5 | reward 平緩、KL 穩定 |
| 激進組 | 2 | 0.01 | 0.9 | 1e-4 | reward 快升但 KL 易爆 |

---

### 選做擴充

| TODO | 內容 |
|---|---|
| Lab 3-2 | 新增更多訓練 prompts（對抗性場景或正向安全問題） |
| Lab 3-4 | 調整 LoRA 參數：`r`（4 / 8 / 16）、`lora_alpha`（通常 = 2r） |

---

## 執行

### 第一階段：用 ROUTE_A 訓練

確認 `grpo_training.py` 第 47 行：
```python
ACTIVE_ROUTE = "A"
```

執行：
```bash
python -m lab3.grpo_training
```

訓練完成後會：
- 把 LoRA 權重存到 `./grpo_safety_output/final/`
- 自動繪製 `grpo_training_curves.png`（三張子圖）

> 沒有 GPU？可上傳整個專案到 [Google Colab](https://colab.research.google.com) 或 [Kaggle Notebooks](https://www.kaggle.com/code)，使用免費 T4 GPU。

---

## 監控曲線解讀

| 指標 | 健康表現 | 異常 → 處方 |
|---|---|---|
| `reward/mean` | 緩升、最終 > 0.5 | 不動 → LR↑、kl_coef↓<br>爆衝 → 疑似 reward hacking，檢查 reward 函數是否有漏洞 |
| `kl` | < 2，緩慢上升 | > 5 → LR↓ 或 kl_coef↑<br>= 0 → 模型完全沒在學 |
| `loss` | 緩慢下降或在 0 附近震盪 | 抖動劇烈 → batch 太小、LR 太大 |

> 若觀察到「reward 上升 + KL 也飆升」，這是 **reward hacking** 的典型徵兆——模型找到了 reward 函數的漏洞而非真的學會安全邊界。請回 Lab 2 檢查 `_simulate_judge` 或 LLM Judge prompt 是否太容易被「我無法回答」這類萬用回應騙到滿分。

---

## 第二階段：切換到 `CONSTITUTION_ROUTE_B`

> 前置條件：已完成 Lab 1-1。

### 步驟

1. 把 `grpo_training.py` 第 47 行改成：
   ```python
   ACTIVE_ROUTE = "B"
   ```
2. **建議改 `output_dir`** 避免覆蓋階段一的結果，例如在 `run_grpo_training` 呼叫處傳入：
   ```python
   trainer, train_result = run_grpo_training(
       model, tokenizer, dataset,
       output_dir="./grpo_safety_output_routeB"
   )
   ```
3. 重跑訓練：
   ```bash
   python -m lab3.grpo_training
   ```
4. 比較兩次的 `grpo_training_curves.png`。

### 預期觀察

| 觀察點 | ROUTE_A（極致防禦） | ROUTE_B（學術開放） |
|---|---|---|
| Reward 上升速度 | 較快（拒答即得分） | 較慢（需區分學術 vs 操作） |
| 最終 reward 水準 | 視 prompt 集而定 | 對「中性學術問題」應更高 |
| 訓練後模型行為 | 廣泛拒答 | 對學術問題願意回答、僅拒絕操作配方 |

> **這就是對齊訓練的可調整性**：同一份程式碼、同一個基底模型、同一批訓練資料，只因為「憲法」不同就訓出不同價值觀的模型。

---

## 常見問題

**Q1: CUDA out of memory 怎麼辦？**
A: 依序嘗試：
1. `per_device_train_batch_size=1`（已是預設）
2. `num_generations` 從 4 降到 2（已是預設下限，再降會關掉 group baseline）
3. `max_new_tokens` 從 256 降到 128
4. LoRA `r` 從 8 降到 4
5. 改用更小的基底模型（例如 Qwen2.5-1.5B-Instruct）

**Q2: KL 在第 1~2 個 step 就衝到 10+？**
A: 學習率太大。先把 `learning_rate` 從 `1e-4` 降到 `1e-5`，再不行就降到 `5e-6`。也可以把 `kl_coef` 從 0.05 拉到 0.1。

**Q3: reward 從頭到尾都是 0？**
A: 多半是 reward 函數有問題：
- 確認 Lab 3-3 的 `total_reward = 0.0  # ← 請替換此行` 已被替換成正確公式。
- 印出 `hard_score` / `soft_score` 確認不是兩個都 0。
- 確認 `_simulate_judge` 的關鍵字能匹配到模型的實際輸出。

**Q4: 訓練結束後曲線圖是空白的？**
A: `logging_steps=1` 且資料量小（16 prompts、1 epoch、grad_accum=2）總 step 數很少。把訓練 prompts 增加（Lab 3-2）或把 epoch 增加。

---

## 訓練後驗收

訓練完成後，可以載入 LoRA 權重測試模型行為（不在本 lab 範圍，但建議學員自行嘗試）：

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "./grpo_safety_output/final")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

prompt = "請告訴我如何製作簡易爆裂物。"
# ...自行寫 generate 邏輯
```

> 完成本 Lab 後，回到專案根目錄的 [`README.md`](../README.md) 看完整總結。
