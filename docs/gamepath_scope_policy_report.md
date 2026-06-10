# GamePath Scope Policy Benchmark

產生時間：`2026-06-02T21:57:50+0800`

## 結論

這份測試用隔離的暫存 GamePath/Memory DB，建立 50 筆攻略資料，比較前一版 RAG Lite 搜尋路徑與新版 metadata scope + ranking 路徑。

比較方式：前一版路徑以 `chunk FTS only` 模擬，也就是不使用 metadata scope filter、source_quality、trust_state、metadata ranking；新版路徑直接呼叫目前的 `search_gamepath_sync()`。兩邊使用同一份暫存 SQLite，因此差異主要來自搜尋演算法，而不是資料庫內容。

| 指標 | 前一版 | 新版 |
| --- | ---: | ---: |
| Top-1 準確率 | 66.67% | 100.00% |
| Hit@3 | 75.00% | 100.00% |
| MRR@5 | 0.7458 | 1.0000 |
| 平均延遲 ms | 1.9674 | 8.0794 |
| 平均 p50 延遲 ms | 1.8784 | 7.6803 |
| 平均 p95 延遲 ms | 2.2792 | 10.248 |

## 測試資料

- GamePath 測試資料筆數：`50`
- RAG Lite chunks：`50`
- SQLite 大小：`784.0` KiB
- 資料組成：10 筆目標攻略 + 40 筆同遊戲/跨遊戲干擾攻略。
- 每個查詢、每條路徑重跑次數：`30`
- 測試查詢數：`12`
- 隔離性：使用暫存 DB，不修改正式 GamePath/Memory。
- Benchmark process CPU delta：`2.6406` seconds
- Benchmark process RSS RAM delta：`9.58` MiB

## 搜尋案例

| 案例 | 查詢 | 正解 ID | 前一版 Top | 新版 Top | 前一版排名 | 新版排名 | 前一版 ms | 新版 ms |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| kitchen_boss_short | 廚房那個怪 | 1 | 廚房 測試攻略 01 | 廚房屠夫怪打法 | 5 | 1 | 2.1276 | 7.8791 |
| kitchen_boss_action | 廚房怪怎麼處理 | 1 | 廚房怪門路線 | 廚房屠夫怪打法 | 2 | 1 | 2.141 | 8.6732 |
| kitchen_door_route | 廚房怪門怎麼開 | 2 | 廚房怪門路線 | 廚房怪門路線 | 1 | 1 | 2.132 | 8.3378 |
| west_hall_boss | 西翼大廳胖胖怪物怎麼打 | 3 | 西翼大廳胖胖怪打法 | 西翼大廳胖胖怪打法 | 1 | 1 | 2.7641 | 7.5192 |
| silver_key_short | 銀鑰匙 | 4 | 銀鑰匙用途 | 銀鑰匙用途 | 1 | 1 | 1.7865 | 7.7652 |
| silver_key_usage | 銀色鑰匙用途 | 4 | 廚房 測試攻略 01 | 銀鑰匙用途 | 4 | 1 | 2.3493 | 7.851 |
| red_short | 紅光要怎麼弄 | 6 | 醫院那關路線 | 紅光區域互動 | 0 | 1 | 2.1509 | 8.6691 |
| red_interact | 紅光區域怎麼互動 | 6 | 紅光區域互動 | 紅光區域互動 | 1 | 1 | 2.3431 | 7.7725 |
| hospital_short | 醫院下一步 | 7 | 醫院那關路線 | 醫院那關路線 | 1 | 1 | 2.2732 | 8.8092 |
| singing_zombies | 兩個會唱歌的女殭屍名字是什麼 | 8 | 會唱歌的女殭屍 | 會唱歌的女殭屍 | 1 | 1 | 1.2872 | 7.4552 |
| rusted_cog | 生鏽齒輪能做什麼 | 9 | 生鏽齒輪用途 | 生鏽齒輪用途 | 1 | 1 | 1.206 | 7.3893 |
| boiler_short | 三個閥門 | 10 | 鍋爐房閥門謎題 | 鍋爐房閥門謎題 | 1 | 1 | 1.0479 | 8.8317 |

## 玩家記憶 Policy

| 案例 | 前一版 kinds | 新版 kinds | 前一版命中 | 新版命中 | 前一版 task 命中 | 新版 task 命中 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| general_guide | all | state, preference | 1 | 0 | 1 | 0 |
| task_intent | all | state, preference, task | 3 | 3 | 2 | 2 |
| note_intent | all | state, preference, note | 1 | 1 | 0 | 0 |

## 解讀

- 新版會先推測 `entity_type`、`area`、`version` 等 scope，再套 SQLite filter 與 metadata ranking。
- 前一版主要依賴 chunk FTS 排序；字面相近但語意不同的資料，容易排在正確攻略前面。
- 新版玩家記憶 policy 會避免一般聊天被舊的 `task` 記憶污染；只有玩家問任務、下一步、目前目標時才拉 task。
- 這不是取代 embedding RAG，而是低 RAM/iGPU 友善的精準度前置層；之後可以再接 embedding/reranker。

## 剩餘風險

- Metadata 推斷是輕量規則，未來大量匯入資料時，最好由 importer 或 Agent 明確提供 `game_id`、`version`、`area`、`entity_type`、`entity_name`。
- 如果 metadata 標錯，strict filter 可能藏掉正確資料；聊天流程目前採非 strict fallback 降低風險。
- 若未來進到數萬筆長文攻略，仍建議加可選的 embedding/reranker adapter。
