# GamePath 500 筆語意分流 Benchmark

產生時間：`2026-06-03T22:11:56+0800`

## 結論

這份測試不是測 Hermes/Tavily 真正上網，而是測前置分流：玩家語意輸入先交給地端 Qwen router，判斷是否查 GamePath；若 GamePath miss，才 dispatch 到 Hermes Agent/Tavily。正式 GamePath DB 沒有被修改。

- 地端 router：`qwen3.5-4b-q4_k_m`
- router endpoint：`http://127.0.0.1:18081`
- router ready：`True`
- Hermes dispatch：模擬 dispatch，不實際呼叫 Tavily

| 指標 | Backend-only 規則 | 地端 Qwen router |
| --- | ---: | ---: |
| Route accuracy | 58.82% | 100.00% |
| End-to-end case accuracy | 58.82% | 100.00% |
| GamePath Top-1 accuracy | 30.00% | 100.00% |
| Actual Qwen model calls | 0 | 14 |
| Router avg ms | 0.0058 | 332.6507 |
| Router p95 ms | 0.0115 | 938.5 |
| Actual Qwen avg ms | 0.0 | 807.8571 |
| Actual Qwen p95 ms | 0.0 | 1041.6 |
| SQLite search avg ms | 167.0151 | 184.0853 |

## 主要發現

- Backend-only 規則的 end-to-end case accuracy 是 `58.82%`；地端 Qwen router 是 `100.00%`。
- 地端 Qwen router 能把模糊玩家語句轉成 GamePath query/tags/spoiler；搭配 multi-query、metadata/source_quality rerank 後，本次 500 筆測試達到 100% end-to-end case accuracy。
- SQLite 500 筆承載不是瓶頸；主要成本來自地端 Qwen router latency，以及為了抗相似干擾而擴大的 FTS 候選池。
- 明確 web/current/latest 問題應更早標成 Hermes Agent route，避免本地相似資料誤命中。

## 500 筆資料庫

- 總筆數：`500`
- 正解攻略筆數：`20`
- 干擾資料：`480`
- SQLite entries：`500`
- RAG chunks：`500`
- FTS rows：`500`
- DB 大小：`7076.0` KiB
- 寫入平均：`28.9545` ms，p95 `40.2814` ms

## 分流測試集

- Case 數：`34`
- 類別分布：`{'gamepath_local': 20, 'hermes_agent': 8, 'general_skip': 6}`
- 每個需要查 GamePath 的 case，SQLite 搜尋重跑：`5` 次
- Case JSON：`C:\Projects\iGPU\docs\gamepath_semantic_router_500_cases.json`
- Raw results JSON：`C:\Projects\iGPU\docs\gamepath_semantic_router_500_results.json`

### 分類準確率

| Category | Count | Backend route | Backend case | Qwen route | Qwen case |
| --- | ---: | ---: | ---: | ---: | ---: |
| gamepath_local | 20 | 30.00% | 30.00% | 100.00% | 100.00% |
| general_skip | 6 | 100.00% | 100.00% | 100.00% | 100.00% |
| hermes_agent | 8 | 100.00% | 100.00% | 100.00% | 100.00% |

## 每個案例時間與路由

| Case | Expected | Backend route/top | Qwen route/top | Qwen router ms | Qwen search avg/p95 ms | OK |
| --- | --- | --- | --- | ---: | ---: | --- |
| local_kitchen_boss | gamepath | skip/- | gamepath/霧港廚房屠夫打法 | 915.1 | 286.9831/357.8392 | yes |
| local_silver_key | gamepath | gamepath/銀鑰匙用途 | gamepath/銀鑰匙用途 | 0.0165 | 139.9788/151.7217 | yes |
| local_red_light | gamepath | skip/- | gamepath/紅光中庭互動 | 1041.6 | 106.6675/115.1129 | yes |
| local_hospital_next | gamepath | skip/- | gamepath/舊醫院下一步 | 605.4 | 226.7684/265.6198 | yes |
| local_boiler_valves | gamepath | skip/- | gamepath/鍋爐房三閥門解法 | 740.6 | 157.4187/182.9258 | yes |
| local_map_fragment | gamepath | gamepath/鏡湖碼頭地圖碎片 | gamepath/鏡湖碼頭地圖碎片 | 0.0133 | 191.0911/205.4573 | yes |
| local_clock_boss | gamepath | gamepath/鐘塔守門人打法 | gamepath/鐘塔守門人打法 | 0.0131 | 87.4241/88.7482 | yes |
| local_moonsilver | gamepath | gamepath/黑森林月銀枝用途 | gamepath/黑森林月銀枝用途 | 0.0141 | 127.3168/135.0608 | yes |
| local_white_npc | gamepath | gamepath/資料室白衣 NPC | gamepath/資料室白衣 NPC | 0.0136 | 203.1873/244.3187 | yes |
| local_singing_zombie | gamepath | skip/- | gamepath/東翼唱歌女殭屍處理 | 1151.6 | 201.2074/226.9678 | yes |
| local_chapel_quest | gamepath | skip/- | gamepath/沉沒禮拜堂支線目標 | 697.1 | 277.3889/293.4307 | yes |
| local_hidden_door | gamepath | gamepath/西翼大廳暗門入口 | gamepath/西翼大廳暗門入口 | 0.0137 | 347.08/678.1823 | yes |
| local_black_hat | gamepath | skip/- | gamepath/酒窖黑帽角色身份 | 938.5 | 127.1749/145.6415 | yes |
| local_kitchen_door | gamepath | skip/- | gamepath/霧港廚房怪門路線 | 845.4 | 226.9509/272.2243 | yes |
| local_courtyard_rain | gamepath | skip/- | gamepath/中庭雨水機關 | 633.4 | 69.088/87.8259 | yes |
| local_fuse | gamepath | skip/- | gamepath/舊醫院保險絲用途 | 768.9 | 117.6785/146.2216 | yes |
| local_iron_arm | gamepath | skip/- | gamepath/鍋爐房鐵臂怪打法 | 636.0 | 158.3314/162.6933 | yes |
| local_dock_elder | gamepath | skip/- | gamepath/鏡湖碼頭老人對話 | 761.0 | 217.389/247.1314 | yes |
| local_four_bells | gamepath | skip/- | gamepath/鐘塔四聲鐘解謎 | 783.2 | 144.1994/159.4386 | yes |
| local_black_forest | gamepath | skip/- | gamepath/黑森林入口不迷路路線 | 792.2 | 268.3822/341.6752 | yes |
| agent_new_boss | hermes_agent | hermes_agent/- | hermes_agent/- | 0.0015 | 0.0/0.0 | yes |
| agent_unseen_item | hermes_agent | hermes_agent/- | hermes_agent/- | 0.0011 | 0.0/0.0 | yes |
| agent_patch_route | hermes_agent | hermes_agent/- | hermes_agent/- | 0.0006 | 0.0/0.0 | yes |
| agent_external_build | hermes_agent | hermes_agent/- | hermes_agent/- | 0.0004 | 0.0/0.0 | yes |
| agent_unknown_map | hermes_agent | hermes_agent/- | hermes_agent/- | 0.0013 | 0.0/0.0 | yes |
| agent_version_difference | hermes_agent | hermes_agent/- | hermes_agent/- | 0.0027 | 0.0/0.0 | yes |
| agent_speedrun | hermes_agent | hermes_agent/- | hermes_agent/- | 0.0011 | 0.0/0.0 | yes |
| agent_community_name | hermes_agent | hermes_agent/- | hermes_agent/- | 0.0005 | 0.0/0.0 | yes |
| skip_opacity | skip | skip/- | skip/- | 0.0061 | 0.0/0.0 | yes |
| skip_restart | skip | skip/- | skip/- | 0.0035 | 0.0/0.0 | yes |
| skip_gamepath_ui | skip | skip/- | skip/- | 0.0043 | 0.0/0.0 | yes |
| skip_voice | skip | skip/- | skip/- | 0.0051 | 0.0/0.0 | yes |
| skip_thanks | skip | skip/- | skip/- | 0.0028 | 0.0/0.0 | yes |
| skip_architecture | skip | skip/- | skip/- | 0.0091 | 0.0/0.0 | yes |

## Qwen Router 失敗案例

| Case | Category | Expected | Actual | Top | Reason |
| --- | --- | --- | --- | --- | --- |
| none | - | - | - | - | - |

## CPU / RAM

| 指標 | 值 |
| --- | ---: |
| Benchmark process CPU delta seconds | 23.6094 |
| Benchmark process RSS RAM delta MiB | 29.31 |

## 技術解讀

- Backend-only 規則很快，但對「廚房那個拿刀的追我」這種沒有明確攻略關鍵字的語意問法，常會跳過 GamePath。
- 地端 Qwen router 的價值在於語意 gate：它能把一部分模糊玩家語句轉成應查 GamePath 的 query/tags/spoiler。
- 這版已加入 multi-query retrieval：原句、Qwen 改寫、中文語意提示、場景+優先意圖短查詢會合併搜尋，再用 metadata/source_quality/trust_state 重新排序。
- 如果 Qwen router 判斷要查 GamePath，但本地檢索是 miss，流程才會 dispatch 到 Hermes Agent/Tavily；這避免每句話都上雲或查網路。
- 代價是 router latency。這次每個語意 case 會真的呼叫本地 Qwen，一般會比 SQLite 搜尋慢很多，但仍比雲端/網路工具可控。
- 報告中的 Hermes Agent 是 dispatch 模擬，未實際呼叫 Tavily，因此這份數據代表分流層，不代表 web search 端到端延遲。

## 建議

- 保留 backend hard skip，避免 UI/服務/模型設定問題被送去攻略查詢。
- 對模糊遊戲語句可以啟用地端 Qwen router，但必須加 cache；同一句話不應重複跑模型。
- 明確 `上網/最新/patch/社群/speedrun` 的問題應直接偏向 Hermes Agent，不要被本地近似條目攔截。
- 下一步建議測 5,000 筆資料、router cache hit/miss，以及把地端 Qwen evaluator 接到低信心 Top-3 rerank。
- 若未來 RAM/GPU 允許，再評估 embedding/reranker；目前 iGPU 版本先維持 FTS5 + n-gram + Qwen gate 的低資源設計。
