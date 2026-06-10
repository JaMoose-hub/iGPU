# GamePath 流程圖與架構圖

最後更新：2026-06-08

這份文件只整理 GamePath 的圖與角色分工，方便簡報或快速 review。完整細節仍以 `docs/gamepath_architecture.md` 為主。

## 1. 總體架構圖

```mermaid
flowchart LR
    subgraph UI["遊戲陪伴前端"]
        Overlay["Overlay 主視窗"]
        Task["Task 視窗"]
        Search["Game Search 視窗"]
        Path["GamePath 視窗"]
        HUD["HUD / 截圖 / 虛擬游標"]
    end

    subgraph Backend["Backend API"]
        Chat["/chat"]
        GamePathAPI["GamePath API"]
        Health["/health"]
        RouterGate["路由與安全 allowlist"]
    end

    subgraph LocalAI["地端模型層 iGPU"]
        Qwen["llama.cpp Vulkan<br/>Qwen3.5-4B Q4_K_M"]
        Intent["User Intent Gate"]
        Eval["Retrieval Evaluator"]
        Hint["本地 Hint 整理"]
    end

    subgraph Storage["穩定資料層"]
        SQLite["gamepath.sqlite<br/>entry / chunk / FTS5"]
        Notes["Markdown Notes<br/>gamepath/notes/{game_id}/{entry_id}.md"]
        Memory["玩家記憶 / task memory"]
    end

    subgraph Cloud["雲端 Agent 層"]
        Hermes["Hermes Agent GPT5.5"]
        Tavily["Tavily Web Search"]
        Vision["雲端 Vision / 截圖理解"]
    end

    Overlay --> Chat
    Task --> Chat
    Search --> Chat
    Path --> GamePathAPI
    HUD --> Chat

    Chat --> RouterGate
    RouterGate --> Intent
    Intent --> Qwen
    Intent --> GamePathAPI
    GamePathAPI --> SQLite
    SQLite --> Notes
    GamePathAPI --> Eval
    Eval --> Qwen
    Eval --> Hint
    Chat --> Memory

    Chat --> Hermes
    Hermes --> Tavily
    Hermes --> Vision
    Hermes -->|"濃縮後可重用提示"| GamePathAPI
```

## 2. 文字聊天主流程

```mermaid
flowchart TD
    A["玩家輸入文字 / 語音轉文字"] --> B["前端送到 /chat"]
    B --> C["地端 Qwen 判斷 user intent"]
    C --> D{"route"}

    D -->|"ui_command / skip"| UI["跳過 GamePath<br/>Backend allowlist 決定 UI 動作"]
    D -->|"general_chat / task_memory"| General["一般聊天或玩家記憶流程"]
    D -->|"hermes_web"| Web["直接交給 Hermes / Tavily"]
    D -->|"gamepath_query"| Q["建立 GamePath 查詢"]

    Q --> MQ["Multi-query retrieval<br/>原句 + query variants + tags"]
    MQ --> FTS["SQLite FTS5 / RAG Lite chunks"]
    FTS --> Rank["metadata + coverage + trust rerank"]
    Rank --> E{"direct / summarize / miss"}

    E -->|"direct"| Local["本地 GamePath 命中"]
    Local --> Hint["地端 Qwen 整理 Hint 1/2/3"]
    Hint --> Player["回覆玩家"]

    E -->|"summarize"| Summ["找到相關資料但需濃縮"]
    Summ --> HermesLocal["Hermes 根據 GamePath 片段重組提示"]
    HermesLocal --> Player

    E -->|"miss"| Miss["本地不足或信心不夠"]
    Miss --> HermesWeb["Hermes Agent 判斷是否 Tavily web search"]
    HermesWeb --> Condense["濃縮成低劇透玩家提示"]
    Condense --> Store{"可重用？"}
    Store -->|"是"| Write["寫入 SQLite + Markdown"]
    Store -->|"否"| NoWrite["只回覆，不寫入"]
    Write --> Player
    NoWrite --> Player
```

## 3. 截圖 / HUD / 攻略混合流程

```mermaid
flowchart TD
    A["玩家截圖 + 問題"] --> B["Vision intent router"]
    B --> C{"圖片意圖"}

    C -->|"screenshot_visual"| V["一般畫面理解<br/>回答看到了什麼"]
    C -->|"screenshot_hud"| HUD["HUD 標記 / 圈選 / 指方向"]
    C -->|"screenshot_gamepath_query"| GPQ["Vision 產生 GamePath query"]

    GPQ --> Search["GamePath RAG Lite 搜尋"]
    Search --> Decision{"direct / summarize / miss"}

    Decision -->|"direct"| Local["本地資料回答<br/>整理成 Hint"]
    Decision -->|"summarize"| S["Hermes 濃縮本地片段"]
    Decision -->|"miss"| W["Hermes / Tavily 視需要查網路"]

    V --> Player["回覆玩家"]
    HUD --> Player
    Local --> Player
    S --> Player
    W --> Condense["濃縮為玩家提示"]
    Condense --> Store{"可重用攻略？"}
    Store -->|"是"| DB["寫入 GamePath"]
    Store -->|"否"| Player
    DB --> Player
```

## 4. RAG Lite 內部搜尋流程

```mermaid
flowchart TD
    A["GamePath query"] --> B["game_id 過濾"]
    B --> C["建立 query variants"]
    C --> D["查 gamepath_chunk_fts"]
    D --> E{"有 chunk 命中？"}

    E -->|"有"| Chunk["取最相關 chunks"]
    E -->|"沒有"| Entry["fallback 查 gamepath_fts"]

    Chunk --> Merge["合併同一 entry"]
    Entry --> Merge
    Merge --> Score["計算分數"]

    Score --> Coverage["match coverage"]
    Score --> Core["core overlap"]
    Score --> Meta["area / entity / tags"]
    Score --> Trust["trust_state / source_quality"]

    Coverage --> Rank["rerank"]
    Core --> Rank
    Meta --> Rank
    Trust --> Rank

    Rank --> Eval{"Evaluator 判斷"}
    Eval -->|"direct"| D1["高信心本地回答"]
    Eval -->|"summarize"| D2["交給模型濃縮"]
    Eval -->|"miss"| D3["交給 Hermes / Tavily"]
```

## 5. GamePath 寫回規則流程

```mermaid
flowchart TD
    A["Hermes / Vision 回答"] --> B["condense_agent_answer"]
    B --> C{"有 Agent 介入？"}
    C -->|"否"| NoAgent["不寫入：agent_not_used"]
    C -->|"是"| Len{"回答長度足夠？"}

    Len -->|"否"| Short["不寫入：answer_too_short"]
    Len -->|"是"| Intent{"攻略型且可重用？"}

    Intent -->|"否"| NotGuide["不寫入：not_guide_intent / UI / save-only"]
    Intent -->|"是"| Certain{"答案可靠且有可執行提示？"}

    Certain -->|"否"| Uncertain["不寫入：uncertain_answer / needs_specific_item"]
    Certain -->|"是"| Dup{"已有相似條目？"}

    Dup -->|"是，可更新"| Update["更新既有條目"]
    Dup -->|"是，不更新"| Existing["不新增：duplicate_existing"]
    Dup -->|"否"| Insert["新增 SQLite row + Markdown"]

    Update --> Index["更新 FTS5 + chunk index"]
    Insert --> Index
    Index --> Notify["發出 gamepath:changed<br/>GamePath 視窗刷新"]
```

## 6. direct / summarize / miss 意義

| 狀態 | 意義 | 下一步 |
| --- | --- | --- |
| `direct` | GamePath 本地資料高信心命中，答案範圍明確 | 地端 Qwen 整理成 Hint 1/2/3，直接回玩家 |
| `summarize` | 找到相關本地資料，但內容較長或需要重組 | Hermes 或模型根據本地片段濃縮，不查整個網路 |
| `miss` | 本地不足、候選太弱、或像硬湊 | 交給 Hermes GPT5.5，必要時 Tavily web search |

## 7. 狀態泡泡對照

| 泡泡狀態 | 代表意思 |
| --- | --- |
| `gamepath_skipped` | Qwen / Backend 判斷不用查 GamePath |
| `gamepath_hit` | GamePath 高信心命中，走本地快取 |
| `gamepath_summarizing` | GamePath 找到資料，但正在濃縮成玩家提示 |
| `gamepath_miss` | GamePath 未命中或信心不足，準備交給 Hermes / Tavily |
| `agent_may_search_web` | Hermes 可能會使用 Tavily 查網路 |
| `gamepath_stored` | 回答已濃縮並寫入 SQLite + Markdown |
| `gamepath_not_stored` | 回答不符合可重用規則，沒有寫入 |
| `gamepath_disputed` | 玩家回報提示不符，條目會降權或改走驗證流程 |

## 8. 一句話版

```text
地端 Qwen 判斷要不要查攻略；
GamePath 用 SQLite + Markdown 找本地可重用提示；
本地足夠就用 Hint 回答；
本地不足才交給雲端 Hermes GPT5.5 / Tavily；
雲端查到的內容會濃縮成可重用提示，再寫回 GamePath。
```
