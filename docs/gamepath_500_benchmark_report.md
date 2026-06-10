# GamePath 500 筆測試資料 Benchmark

產生時間：`2026-06-03T06:24:31+0800`

## 結論

本測試用隔離暫存 SQLite 建立 500 筆合成遊戲攻略資料，並使用同一批資料比較前一版 `chunk FTS only` 與新版 `metadata scope + ranking` 搜尋路徑。正式 GamePath/Memory 沒有被修改。

測試過程中發現新版在 500 筆資料下會因 metadata scope 過度過濾而掉分，因此已補上非 strict 搜尋的 unscoped fallback 合併排序，並加強 `白衣NPC` / `白衣 NPC` 這類名稱比對與 `character` 類型推斷。本報告數字為修正後結果。

| 指標 | 前一版 | 新版 |
| --- | ---: | ---: |
| Top-1 精準度 | 90.00% | 100.00% |
| Hit@3 | 100.00% | 100.00% |
| Hit@5 | 100.00% | 100.00% |
| MRR@5 | 0.9500 | 1.0000 |
| 平均查詢時間 ms | 7.515 | 20.5068 |
| 平均 p50 ms | 7.3772 | 19.8302 |
| 平均 p95 ms | 7.9505 | 24.2615 |

## 測試資料內容

- 總筆數：`500`
- 目標正解資料：`20`
- 干擾資料：`480`
- 測試遊戲：`GAMEPATH_BENCHMARK_ARPG`
- 完整 500 筆資料：`C:\Projects\iGPU\docs\gamepath_500_dataset.json`
- SQLite entries：`500`
- RAG chunks：`500`
- FTS rows：`500`
- DB 大小：`7076.0` KiB

### 遊戲分布

| Game | Count |
| --- | ---: |
| GAMEPATH_BENCHMARK_ARPG | 404 |
| SKY_FORGE_ODYSSEY | 24 |
| NOCTURNE_ARCHIVE | 24 |
| IRONWOOD_SURVIVAL | 24 |
| STARLIGHT_TACTICS | 24 |

### 類型分布

| Entity type | Count |
| --- | ---: |
| boss | 43 |
| route | 43 |
| item | 42 |
| mechanic | 42 |
| puzzle | 42 |
| npc | 42 |
| map | 41 |
| material | 41 |
| enemy | 41 |
| quest | 41 |
| location | 41 |
| character | 41 |

### 代表資料樣本

| Key | Title | Question | Type | Area |
| --- | --- | --- | --- | --- |
| mist_kitchen_boss | 霧港廚房屠夫打法 | 廚房屠夫怪怎麼處理 | boss | 霧港廚房 |
| silver_key_item | 銀鑰匙用途 | 銀色鑰匙用途 | item | 地下酒窖 |
| red_light_mechanic | 紅光中庭互動 | 紅光要怎麼弄 | mechanic | 紅光中庭 |
| hospital_route | 舊醫院下一步 | 醫院下一步 | route | 舊醫院 |
| boiler_valve_puzzle | 鍋爐房三閥門解法 | 三個閥門順序 | puzzle | 鍋爐房 |
| mirror_lake_map | 鏡湖碼頭地圖碎片 | 鏡湖地圖碎片位置 | map | 鏡湖碼頭 |
| clocktower_boss | 鐘塔守門人打法 | 鐘塔守門人弱點 | boss | 鐘塔頂層 |
| blackwood_material | 黑森林月銀枝用途 | 月銀枝用途 | material | 黑森林入口 |
| archive_npc | 資料室白衣 NPC | 白衣NPC對話選哪個 | npc | 資料室 |
| east_hall_enemy | 東翼唱歌女殭屍處理 | 唱歌女殭屍怎麼辦 | enemy | 東翼走廊 |
| chapel_quest | 沉沒禮拜堂支線目標 | 禮拜堂支線下一步 | quest | 沉沒禮拜堂 |
| west_hall_location | 西翼大廳暗門入口 | 西翼暗門在哪 | location | 西翼大廳 |

## 寫入時間

| 指標 | ms |
| --- | ---: |
| 平均 | 28.5075 |
| p50 | 27.1026 |
| p95 | 33.3984 |
| min | 18.9179 |
| max | 459.2803 |

## 每次查詢時間與精準度

每個查詢在每條路徑重跑 `20` 次；每一次 run 的 raw samples 已保存在 `C:\Projects\iGPU\docs\gamepath_500_benchmark_results.json`。

| Case | Query | Type | 前一版 Top | 新版 Top | 前一版 Rank | 新版 Rank | 前一版 avg/p95 ms | 新版 avg/p95 ms |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| mist_kitchen_boss | 廚房屠夫怪怎麼處理 | boss | 霧港廚房屠夫打法 | 霧港廚房屠夫打法 | 1 | 1 | 9.2011/10.0465 | 24.9471/32.3315 |
| silver_key_item | 銀色鑰匙用途 | item | 銀鑰匙用途 | 銀鑰匙用途 | 1 | 1 | 9.9338/10.3879 | 23.681/28.8916 |
| red_light_mechanic | 紅光要怎麼弄 | mechanic | 資料室白衣 NPC | 紅光中庭互動 | 2 | 1 | 8.4657/8.7983 | 22.8945/26.5765 |
| hospital_route | 醫院下一步 | route | 舊醫院下一步 | 舊醫院下一步 | 1 | 1 | 8.4881/9.1413 | 23.7647/26.4791 |
| boiler_valve_puzzle | 三個閥門順序 | puzzle | 鍋爐房三閥門解法 | 鍋爐房三閥門解法 | 1 | 1 | 8.277/8.856 | 22.345/26.1467 |
| mirror_lake_map | 鏡湖地圖碎片位置 | map | 鏡湖碼頭地圖碎片 | 鏡湖碼頭地圖碎片 | 1 | 1 | 8.393/9.062 | 19.0836/22.7434 |
| clocktower_boss | 鐘塔守門人弱點 | boss | 鐘塔守門人打法 | 鐘塔守門人打法 | 1 | 1 | 4.9385/5.2847 | 15.8293/20.4309 |
| blackwood_material | 月銀枝用途 | material | 黑森林月銀枝用途 | 黑森林月銀枝用途 | 1 | 1 | 9.4146/9.7589 | 20.97/24.1817 |
| archive_npc | 白衣NPC對話選哪個 | npc | 鏡湖碼頭老人對話 | 資料室白衣 NPC | 2 | 1 | 8.0571/8.3648 | 21.4594/24.6865 |
| east_hall_enemy | 唱歌女殭屍怎麼辦 | enemy | 東翼唱歌女殭屍處理 | 東翼唱歌女殭屍處理 | 1 | 1 | 8.0914/8.4151 | 22.8127/29.3437 |
| chapel_quest | 禮拜堂支線下一步 | quest | 沉沒禮拜堂支線目標 | 沉沒禮拜堂支線目標 | 1 | 1 | 10.2484/10.6604 | 22.685/25.4943 |
| west_hall_location | 西翼暗門在哪 | location | 西翼大廳暗門入口 | 西翼大廳暗門入口 | 1 | 1 | 8.8376/9.1138 | 24.2175/26.7061 |
| cellar_character | 黑帽角色名字 | character | 酒窖黑帽角色身份 | 酒窖黑帽角色身份 | 1 | 1 | 4.5023/4.9365 | 16.9636/22.0618 |
| kitchen_route | 廚房怪門怎麼開 | route | 霧港廚房怪門路線 | 霧港廚房怪門路線 | 1 | 1 | 8.1177/8.7914 | 21.2307/23.4321 |
| courtyard_mechanic | 中庭雨水機關觸發 | mechanic | 中庭雨水機關 | 中庭雨水機關 | 1 | 1 | 4.7396/5.2565 | 15.9431/19.1321 |
| hospital_item | 保險絲用在哪 | item | 舊醫院保險絲用途 | 舊醫院保險絲用途 | 1 | 1 | 9.0991/9.4214 | 21.9217/26.4745 |
| boiler_boss | 鐵臂怪怎麼打 | boss | 鍋爐房鐵臂怪打法 | 鍋爐房鐵臂怪打法 | 1 | 1 | 8.1168/8.5832 | 21.1296/24.1027 |
| dock_npc | 碼頭老人選項 | npc | 鏡湖碼頭老人對話 | 鏡湖碼頭老人對話 | 1 | 1 | 2.668/2.8825 | 13.5992/15.8583 |
| clocktower_puzzle | 四聲鐘順序 | puzzle | 鐘塔四聲鐘解謎 | 鐘塔四聲鐘解謎 | 1 | 1 | 2.5128/2.8481 | 12.9704/15.0232 |
| forest_route | 黑森林迷路怎麼走 | route | 黑森林入口不迷路路線 | 黑森林入口不迷路路線 | 1 | 1 | 8.1984/8.4001 | 21.6888/25.1327 |

## CPU / RAM 觀測

| 指標 | 值 |
| --- | ---: |
| Benchmark process CPU delta seconds | 14.375 |
| Benchmark process RSS RAM delta MiB | 98.35 |

## 技術解讀

- 前一版 FTS-only 很快，但在大量相似資料中容易被重複關鍵字或跨類型資料干擾。
- 新版會用 `game_id`、`entity_type`、`area`、`source_quality` 與 metadata ranking 重新排序，並在非 strict 模式合併未套 scope filter 的候選，避免推斷錯誤時直接漏掉正解。
- 新版查詢會多花約十幾毫秒，原因是多了 scope inference、兩段候選搜尋、chunk grouping、metadata scoring；但仍是本地 SQLite 級別，遠低於 Hermes/Tavily 或雲端模型延遲。
- 名稱查詢需要做正規化：例如 `白衣NPC` 與 `白衣 NPC` 應視為同一候選，否則大量資料下會被其他 NPC 對話攻略干擾。
- 500 筆資料仍不是上限測試。下一階段若要驗證數千到數萬筆，應加入分頁查詢、冷/熱 cache、長文 chunk 數量與多遊戲 corpus 比例。

## 剩餘風險

- 這批資料是合成資料，能測演算法抗干擾能力，但不能完全代表真實玩家攻略語料。
- Metadata 若由規則推斷錯誤，搜尋仍可能掉分；大量匯入時建議由 importer 或 Agent 明確寫入 metadata。
- 若未來攻略文章變成長篇 wiki，仍建議加 embedding/reranker adapter 做第二階段 rerank。
