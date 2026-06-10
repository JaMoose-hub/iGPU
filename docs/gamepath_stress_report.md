# GamePath 壓力測試技術報告

產生時間：`2026-06-06T18:02:08+0800`

## 結論
這次用隔離 SQLite 壓到 `5000` 筆本地攻略資料。核心 GamePath 檢索在最大資料量下 Top-1 準確度為 `100.00%`，Hit@5 為 `100.00%`，搜尋 p95 為 `484.17` ms。

判斷：目前演算法對「已被判定為攻略查詢」的數千筆本地攻略包 + 小筆記是夠用的；真正拖慢聊天體驗的通常不是 SQLite/RAG Lite，而是地端 Qwen 意圖判斷與提示整理。不過 hard negative 顯示，不能把每句玩家輸入都無腦丟進 SQLite，正式流程仍需要 Qwen intent gate 先分流一般聊天、UI 指令與最新版/網路意圖。如果資料量上萬或大量同義詞不共享字面線索，就需要下一階段快取或 optional semantic reranker。

## 測試設計

- 使用暫存 `IGPU_GAMEPATH_DIR`，不修改正式 `gamepath/gamepath.sqlite`。
- 每個資料量都重新建立 SQLite、FTS5、chunk index 與 Markdown notes。
- Golden 正解資料：12 筆，涵蓋 enemy、boss、item、route、puzzle、mechanic、npc、map、material、quest、location。
- 干擾資料：同遊戲與不同遊戲混合，包含泛用攻略包、同區域、同 tag、不同 entity。
- 查詢型態：exact、paraphrase、noisy_tag；另測 hard negative 與重複寫入。

## 壓力測試總覽

| 資料量 | DB KiB | 寫入 avg/p95 ms | 搜尋 Top-1 | Hit@5 | MRR@5 | 搜尋 avg/p95 ms | 負例誤命中 | 去重成功 | RSS delta MiB | CPU sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 876.00 | 24.478/28.939 | 100.00% | 100.00% | 1.0000 | 57.278/101.5 | 33.33% | 100.00% | 17.16 | 5.25 |
| 500 | 3904.00 | 21.376/25.208 | 100.00% | 100.00% | 1.0000 | 90.803/161.453 | 33.33% | 100.00% | 70.95 | 12.9062 |
| 2000 | 15088.00 | 28.619/33.811 | 100.00% | 100.00% | 1.0000 | 143.723/266.418 | 33.33% | 100.00% | 137.56 | 41.875 |
| 5000 | 37388.00 | 35.955/47.611 | 100.00% | 100.00% | 1.0000 | 248.932/484.17 | 33.33% | 100.00% | 91.35 | 148.9688 |

## 分類準確度

| 資料量 | exact | paraphrase | noisy_tag |
| ---: | ---: | ---: | ---: |
| 100 | 100.00% | 100.00% | 100.00% |
| 500 | 100.00% | 100.00% | 100.00% |
| 2000 | 100.00% | 100.00% | 100.00% |
| 5000 | 100.00% | 100.00% | 100.00% |

## 代表查詢樣本

| Query | Kind | Expected | Top | Rank | Score | Coverage | ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 唱歌女殭屍在哪出沒 | exact | 1 | 1 | 1 | 1.0 | 1.0 | 218.788 |
| 唱歌女殭屍在哪出沒 | exact | 1 | 1 | 1 | 1.0 | 1.0 | 219.642 |
| 唱歌女殭屍在哪出沒 | exact | 1 | 1 | 1 | 1.0 | 1.0 | 231.729 |
| 二樓聽到歌聲是哪隻怪 | paraphrase | 1 | 1 | 1 | 1.0 | 0.625 | 66.339 |
| 二樓聽到歌聲是哪隻怪 | paraphrase | 1 | 1 | 1 | 1.0 | 0.625 | 65.816 |
| 二樓聽到歌聲是哪隻怪 | paraphrase | 1 | 1 | 1 | 1.0 | 0.625 | 64.45 |
| 二樓聽到歌聲是哪隻怪 | noisy_tag | 1 | 1 | 1 | 0.923 | 0.625 | 65.129 |
| 二樓聽到歌聲是哪隻怪 | noisy_tag | 1 | 1 | 1 | 0.923 | 0.625 | 64.485 |
| 二樓聽到歌聲是哪隻怪 | noisy_tag | 1 | 1 | 1 | 0.923 | 0.625 | 66.268 |
| 廚房拿刀屠夫怪怎麼打 | exact | 2 | 2 | 1 | 1.0 | 1.0 | 255.914 |
| 廚房拿刀屠夫怪怎麼打 | exact | 2 | 2 | 1 | 1.0 | 1.0 | 219.124 |
| 廚房拿刀屠夫怪怎麼打 | exact | 2 | 2 | 1 | 1.0 | 1.0 | 219.445 |

## Top-1 失敗案例

最大資料量下沒有 Top-1 失敗案例。

## 重複寫入測試

最大資料量下，對 6 筆既有攻略用 Hermes 風格相似問法再次寫入：成功辨識重複 `100.00%`，entry count `5000 -> 5000`。

## Hard Negative 與正式路由

Hard negative 是刻意把非攻略、UI 指令、最新版/網路意圖丟進純 GamePath 檢索。這不是正式 `/chat` 的完整路徑，因為正式流程前面還有 Qwen intent router。

| 資料量 | 負例誤命中 | 負例 avg/p95 ms |
| ---: | ---: | ---: |
| 100 | 33.33% | 28.336/36.464 |
| 500 | 33.33% | 76.163/169.565 |
| 2000 | 33.33% | 346.906/876.44 |
| 5000 | 33.33% | 1941.069/5470.611 |

這代表不能「每句話無腦查 SQLite」。當 query 明顯不是本地攻略，純 FTS 仍可能因為共通詞抓到可整理候選；無命中掃描在 5000 筆時也會變慢。因此正式流程保留 Qwen intent gate 是必要的。

本機實測 `/intent/route`：

| Prompt | Route | search_gamepath | 補充 |
| --- | --- | ---: | --- |
| 這不是攻略問題只是心情不好 | general_chat | false | 判定為情緒/一般聊天 |
| 幫我打開 Game Search 視窗 | ui_command | false | 判定為 UI 指令 |
| 星霜羅盤最新版本用途 | hermes_web | false | 判定為最新版/網路意圖 |

## 解讀

1. SQLite FTS5 + chunk RAG Lite 對數千筆攻略的核心檢索成本仍在遊戲互動可接受範圍，通常低於一次地端 Qwen 呼叫。
2. 這版修正後，metadata/tag 不再壓過文字覆蓋率；noisy tag 壓力測試可用來防止大攻略包壓掉精準小筆記。
3. 重複寫入已能擋住語意相近的 Hermes 回答，避免 GamePath 被同一題不同問法洗版。
4. 目前弱點是純字面/中文 n-gram 還不是完整語意向量；如果玩家完全換一套說法、沒有共通詞，仍可能需要 Qwen retrieval evaluator 或未來 optional embedding。
5. 生產聊天流程的總延遲會比本報告高，因為 `/chat` 還包含 Qwen intent router、Qwen hint summary 或 Hermes；本報告主要測 GamePath 演算法本體。

## 建議門檻

- 目前可支撐：數千筆本地攻略、小型攻略包、玩家常見中文相似問法。
- 建議警戒：超過 10k 筆或大量長篇 wiki 匯入後，應加查詢快取、熱門 entry cache、或 optional reranker。
- 下一步：把生產 `/chat` 的 Qwen intent/hint 結果做短 TTL cache，這會比繼續壓 SQLite 更能改善玩家體感。

## 產物

- JSON raw results：`C:\Projects\iGPU\docs\gamepath_stress_results.json`
- Markdown report：`C:\Projects\iGPU\docs\gamepath_stress_report.md`
