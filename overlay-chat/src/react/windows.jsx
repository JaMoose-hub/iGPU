import React from "react";

const Icon = ({ name }) => <span className="icon" data-icon={name}></span>;

export const MainWindow = () => (
  <>
    <div className="drag-bar" data-cursor-action="window-move" data-cursor-label="Move window">
      <div className="title" data-tauri-drag-region="">
        <span id="cosmosLottie" className="app-lottie-icon" aria-hidden="true" data-tauri-drag-region=""></span>
        <span data-tauri-drag-region="">Game Companion Beta</span>
      </div>
      <div className="drag-bar-actions">
        <button id="toolsBtn" className="window-btn tools-window-btn" title="Open tools panel" aria-label="Open tools panel">
          <Icon name="list" />
        </button>
        <button id="minBtn" className="window-btn min-btn" title="Minimize" aria-label="Minimize">
          <Icon name="minus" />
        </button>
        <button id="maxBtn" className="window-btn max-btn" title="Maximize" aria-label="Maximize">
          <Icon name="square" />
        </button>
        <button id="closeBtn" className="window-btn close-btn" title="Close" aria-label="Close">
          <Icon name="x" />
        </button>
      </div>
    </div>

    <div id="chatWindow" className="chat-window" data-cursor-action="scroll-drag" data-cursor-scroll="" data-cursor-label="Scroll chat">
      <div className="message bot-message">
        Ready. I can watch the screen, answer questions, and help during play.
        <br />
        <small style={{ opacity: 0.72 }}>F4 protect, F5 search, F6 tasks, F7 task capture, F8 voice, F9 screenshot, F11 cursor.</small>
      </div>
    </div>

    <div id="imagePreviewArea" className="image-preview-area" style={{ display: "none" }}>
      <div className="preview-container">
        <img id="previewImg" src="" alt="Screenshot preview" />
        <button id="removeImgBtn" className="remove-img-btn" title="Remove image" aria-label="Remove image">
          <Icon name="x" />
        </button>
      </div>
    </div>

    <div className="input-area">
      <button id="voiceBtn" className="voice-btn" title="Voice mode" data-cursor-action="voice-toggle">
        <Icon name="mic" />
        <span className="button-label">Mic</span>
      </button>
      <select id="captureDisplaySelect" className="screen-select" title="Screenshot source" aria-label="Screenshot source">
        <option value="auto">Auto</option>
      </select>
      <button id="screenshotBtn" className="screenshot-btn" title="Screenshot (F9)">
        <Icon name="camera" />
        <span className="button-label">Shot</span>
      </button>
      <button id="taskCaptureBtn" className="task-btn" title="Create task from screen and input (F7)">
        <Icon name="flag" />
        <span className="button-label">Task</span>
      </button>
      <input type="text" id="messageInput" placeholder="Ask for help, or press F9 for screenshot analysis..." />
      <button id="stopBtn" className="stop-btn" style={{ display: "none" }}>
        <Icon name="square" />
        <span className="button-label">Stop</span>
      </button>
      <button id="sendBtn">
        <Icon name="send" />
        <span className="button-label">Send</span>
      </button>
    </div>

    <button id="resizeGrip" className="resize-grip" title="Resize" aria-label="Resize">
      <Icon name="grip" />
    </button>
  </>
);

export const ToolsWindow = () => (
  <>
    <div className="tools-drag" data-tauri-drag-region="" data-cursor-action="window-move" data-cursor-label="Move tools">
      <div className="tools-title" data-tauri-drag-region="">
        <Icon name="list" />
        <span>Tools</span>
      </div>
      <button id="closeBtn" className="window-btn" title="Close tools" aria-label="Close tools">
        <Icon name="x" />
      </button>
    </div>

    <main className="tools-body">
      <section className="tool-section">
        <div className="section-title">Game</div>
        <select id="gameSelect" className="game-select" title="Game"></select>
        <button id="gameAutoBtn" className="tool-btn wide active" title="Auto detect game">
          <Icon name="radio" />
          <span className="button-label">Auto</span>
        </button>
      </section>

      <section className="tool-section">
        <div className="section-title">State</div>
        <div className="tool-grid two">
          <button id="liveStateBtn" className="tool-btn" title="Live State observer">
            <Icon name="radio" />
            <span className="button-label">Live</span>
          </button>
          <button id="liveStateAnalyzeBtn" className="tool-btn" title="Analyze once">
            <Icon name="target" />
            <span className="button-label">Scan</span>
          </button>
        </div>
      </section>

      <section className="tool-section">
        <div className="section-title">Windows</div>
        <div className="tool-grid three">
          <button id="searchBtn" className="tool-btn" title="Game Search">
            <Icon name="search" />
            <span className="button-label">Search</span>
          </button>
          <button id="tasksBtn" className="tool-btn" title="Task log">
            <Icon name="list" />
            <span className="button-label">Tasks</span>
          </button>
          <button id="gamepathBtn" className="tool-btn" title="GamePath">
            <Icon name="database" />
            <span className="button-label">Path</span>
          </button>
        </div>
      </section>

      <section className="tool-section">
        <div className="section-title">Chat</div>
        <button id="clearChatBtn" className="tool-btn wide" title="Clear chat window">
          <Icon name="trash" />
          <span className="button-label">Clear chat</span>
        </button>
      </section>

      <section className="tool-section">
        <div className="section-title">Overlay</div>
        <div className="tool-grid two">
          <button id="hudBtn" className="tool-btn active" title="Clear HUD">
            <Icon name="crosshair" />
            <span className="button-label">HUD</span>
          </button>
          <button id="hudTestBtn" className="tool-btn" title="HUD test">
            <Icon name="target" />
            <span className="button-label">Test</span>
          </button>
        </div>
      </section>

      <section className="tool-section">
        <div className="section-title">Controls</div>
        <div className="tool-grid two">
          <button id="protectBtn" className="tool-btn" title="Content protection">
            <Icon name="shield" />
            <span className="button-label">Protect</span>
          </button>
          <button id="virtualCursorBtn" className="tool-btn" title="Virtual cursor">
            <Icon name="cursor" />
            <span className="button-label">Cursor</span>
          </button>
          <button id="perfBtn" className="tool-btn" title="Performance mode">
            <Icon name="gauge" />
            <span className="button-label">Perf</span>
          </button>
          <button id="gamePilotBtn" className="tool-btn" title="AI game pilot">
            <Icon name="gamepad" />
            <span className="button-label">Pilot</span>
          </button>
        </div>
      </section>

      <section className="tool-section opacity-section">
        <div className="section-title">Opacity</div>
        <div className="opacity-row">
          <Icon name="blend" />
          <input id="opacitySlider" type="range" min="10" max="100" step="5" defaultValue="100" aria-label="Window opacity" />
          <span id="opacityValue" className="opacity-value">100</span>
        </div>
      </section>
    </main>
  </>
);

export const TasksWindow = () => (
  <>
    <div className="task-drag" data-tauri-drag-region="" data-cursor-action="window-move" data-cursor-label="Move window">
      <div className="task-title" data-tauri-drag-region="">
        <Icon name="list" />
        <span>任務紀錄</span>
      </div>
      <div className="task-window-actions">
        <button id="refreshBtn" className="window-btn" title="Refresh" aria-label="Refresh">
          <Icon name="refresh" />
        </button>
        <button id="closeBtn" className="window-btn close-btn" title="Close" aria-label="Close">
          <Icon name="x" />
        </button>
      </div>
    </div>

    <div className="task-toolbar">
      <div className="filter-tabs" role="tablist" aria-label="Task filter">
        <button className="filter-btn active" data-filter="active">進行中</button>
        <button className="filter-btn" data-filter="done">完成</button>
        <button className="filter-btn" data-filter="all">全部</button>
      </div>
      <button id="clearDoneBtn" className="ghost-btn" title="Clear completed tasks">
        <Icon name="trash" />
      </button>
    </div>

    <div className="search-row">
      <Icon name="search" />
      <input id="searchInput" type="search" placeholder="搜尋任務..." />
    </div>

    <main id="taskList" className="task-list" data-cursor-action="scroll-drag" data-cursor-scroll="" data-cursor-label="Scroll tasks"></main>

    <template id="emptyTemplate">
      <section className="empty-state">
        <Icon name="flag" />
        <p>尚無任務</p>
        <small>等待下一個目標。</small>
      </section>
    </template>
  </>
);

export const SearchWindow = () => (
  <>
    <div className="search-drag" data-tauri-drag-region="" data-cursor-action="window-move" data-cursor-label="Move window">
      <div className="search-title" data-tauri-drag-region="">
        <Icon name="search" />
        <span>Game Search</span>
      </div>
      <div className="search-window-actions">
        <button id="closeBtn" className="window-btn close-btn" title="Close" aria-label="Close">
          <Icon name="x" />
        </button>
      </div>
    </div>

    <main className="search-panel">
      <label className="field-label" htmlFor="gameInput">Current game</label>
      <div className="field-row">
        <Icon name="gamepad" />
        <input id="gameInput" type="text" placeholder="Game name" autoComplete="off" />
      </div>

      <label className="field-label" htmlFor="keywordInput">Keyword</label>
      <form id="searchForm" className="search-form">
        <div className="field-row keyword-row">
          <Icon name="search" />
          <input id="keywordInput" type="search" placeholder="item, quest, build, map..." autoComplete="off" />
        </div>
        <button id="searchBtn" className="primary-btn" type="submit">
          <Icon name="panel" />
          <span>Search</span>
        </button>
        <button id="externalBtn" className="secondary-btn" type="button">
          <Icon name="external" />
          <span>External</span>
        </button>
      </form>

      <div className="engine-row" role="group" aria-label="Search engine">
        <button className="engine-btn active" data-engine="google" type="button">Google</button>
        <button className="engine-btn" data-engine="youtube" type="button">YouTube</button>
        <button className="engine-btn" data-engine="wiki" type="button">Wiki</button>
      </div>

      <div id="quickChips" className="quick-chips" aria-label="Quick keywords"></div>

      <div id="browserNav" className="browser-nav" aria-label="Browser controls">
        <button id="browserBackBtn" className="nav-btn" type="button" title="Back" aria-label="Back" disabled>
          <Icon name="back" />
        </button>
        <button id="browserForwardBtn" className="nav-btn" type="button" title="Forward" aria-label="Forward" disabled>
          <Icon name="forward" />
        </button>
        <button id="browserReloadBtn" className="nav-btn" type="button" title="Reload" aria-label="Reload" disabled>
          <Icon name="reload" />
        </button>
        <button id="browserHomeBtn" className="nav-btn" type="button" title="Current search" aria-label="Current search" disabled>
          <Icon name="home" />
        </button>
        <div id="browserAddress" className="browser-address">No page loaded</div>
      </div>

      <section id="browserHost" className="browser-host">
        <div id="browserEmpty" className="browser-empty">
          <Icon name="panel" />
          <span>Search results will open here.</span>
        </div>
      </section>
    </main>
  </>
);

export const GamePathWindow = () => (
  <>
    <div className="gamepath-drag" data-tauri-drag-region="" data-cursor-action="window-move" data-cursor-label="Move window">
      <div className="gamepath-title" data-tauri-drag-region="">
        <Icon name="database" />
        <span>GamePath</span>
      </div>
      <div className="gamepath-window-actions">
        <button id="refreshBtn" className="window-btn" title="Refresh" aria-label="Refresh">
          <Icon name="refresh" />
        </button>
        <button id="closeBtn" className="window-btn close-btn" title="Close" aria-label="Close">
          <Icon name="x" />
        </button>
      </div>
    </div>

    <section className="stats-row">
      <div className="stat">
        <span>SQLite</span>
        <strong id="entryCount">-</strong>
      </div>
      <div className="stat">
        <span>Game</span>
        <strong id="currentGame">All</strong>
      </div>
    </section>

    <div className="search-row">
      <Icon name="search" />
      <input id="searchInput" type="search" placeholder="搜尋 GamePath 攻略..." />
    </div>

    <main id="entryList" className="entry-list" data-cursor-action="scroll-drag" data-cursor-scroll="" data-cursor-label="Scroll GamePath"></main>

    <template id="emptyTemplate">
      <section className="empty-state">
        <Icon name="database" />
        <p>目前沒有可顯示的 GamePath 資料</p>
        <small>問攻略後，Hermes 濃縮出的提示會保存到這裡。</small>
      </section>
    </template>
  </>
);

export const HudWindow = () => <canvas id="hudCanvas"></canvas>;

export const StandbyWindow = () => (
  <main id="standbyRoot" className="standby-root collapsed" data-tone="pink">
    <span className="standby-glow" aria-hidden="true"></span>
    <span className="standby-handle" aria-hidden="true"></span>

    <button id="standbyCollapsedButton" className="standby-hit-zone" title="Open mini chat" aria-label="Open mini chat">
      <span className="sr-only">Open mini chat</span>
    </button>

    <span className="standby-typein-glow" aria-hidden="true"></span>

    <div id="standbyThinkingPanel" className="standby-thinking-panel" aria-hidden="true">
      <span id="standbyThinkingLight" className="standby-thinking-light" aria-hidden="true"></span>
      <div className="standby-thinking-body">
        <span id="standbyThinkingLoading" className="standby-thinking-loading" aria-hidden="true"></span>
        <span id="standbyThinkingText" className="standby-thinking-text">Analyzing request...</span>
      </div>
    </div>

    <section id="standbyResponsePanel" className="standby-response-panel" aria-hidden="true">
      <span className="standby-response-glow" aria-hidden="true">
        <span className="standby-response-glow-layer standby-response-glow-layer-1"></span>
        <span className="standby-response-glow-layer standby-response-glow-layer-2"></span>
        <span className="standby-response-glow-layer standby-response-glow-layer-3"></span>
        <span className="standby-response-glow-layer standby-response-glow-layer-4"></span>
      </span>
      <div className="standby-response-body">
        <p id="standbyResponseText" className="standby-response-text"></p>
        <div className="standby-response-detail">
          <div className="standby-response-hotkeys" aria-hidden="true">
            <span className="standby-response-hotkey">Shift</span>
            <span className="standby-response-hotkey">D</span>
          </div>
          <span className="standby-response-hint">for more detailed response</span>
        </div>
        <button
          id="standbyResponseEnterBtn"
          className="standby-response-action"
          type="button"
          title="Back to input"
          aria-label="Back to input"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M16.85 7.25V11.65C16.85 13.58 15.28 15.15 13.35 15.15H6.95" stroke="#0F0F0F" strokeWidth="1.5" strokeLinecap="round" />
            <path d="M10.15 11.95L6.85 15.25L10.15 18.55" stroke="#0F0F0F" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </section>

    <span className="standby-detail-glow standby-detail-glow-top" aria-hidden="true">
      <span className="standby-detail-glow-image"></span>
    </span>
    <span className="standby-detail-glow standby-detail-glow-bottom" aria-hidden="true">
      <span className="standby-detail-glow-image"></span>
    </span>
    <span className="standby-detail-side-highlight" aria-hidden="true"></span>

    <section id="standbyDetailPanel" className="standby-detail-panel" aria-hidden="true">
      <header className="standby-detail-header">
        <div className="standby-detail-title-row">
          <span className="standby-detail-feature-icon" aria-hidden="true">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M5 6.5C5 5.67157 5.67157 5 6.5 5H17.5C18.3284 5 19 5.67157 19 6.5V17.5C19 18.3284 18.3284 19 17.5 19H6.5C5.67157 19 5 18.3284 5 17.5V6.5Z" stroke="#F2F2F2" strokeWidth="1.4" />
              <path d="M8 9H16M8 12H16M8 15H13" stroke="#F2F2F2" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </span>
          <h1 id="standbyDetailTitle" className="standby-detail-title">Game Companion</h1>
        </div>
        <button id="standbyDetailCloseBtn" className="standby-detail-hotkey" type="button" title="Back to short response">
          Esc
        </button>
      </header>

      <div className="standby-detail-content">
        <div className="standby-detail-question-row">
          <p id="standbyDetailQuestion" className="standby-detail-question">Which weapon should I choose for this build</p>
        </div>
        <section id="standbyDetailContext" className="standby-detail-context" hidden>
          <div className="standby-detail-context-title">Conversation</div>
          <div
            id="standbyDetailContextList"
            className="standby-detail-context-list"
            data-cursor-action="scroll-drag"
            data-cursor-scroll=""
            data-cursor-label="Scroll detailed conversation"
          ></div>
        </section>
        <div id="standbyDetailAnswer" className="standby-detail-answer">
          I have a response ready.
        </div>
        <div className="standby-detail-divider" aria-hidden="true"></div>
        <form id="standbyDetailForm" className="standby-detail-form">
          <div className="standby-detail-type-left">
            <span className="standby-detail-type-line" aria-hidden="true"></span>
            <input
              id="standbyDetailInput"
              className="standby-detail-input"
              type="text"
              aria-label="Reply in detailed response"
              autoComplete="off"
              placeholder="Type or Shift M to speak"
            />
          </div>
          <button id="standbyDetailSendBtn" className="standby-detail-action default" type="button" title="Voice mode" aria-label="Voice mode">
            <svg className="standby-detail-mic-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path opacity="0.1" d="M10.85 15L10 14.125V4H14V14.125L13.15 15H10.85Z" fill="#F2F2F2" />
              <path fillRule="evenodd" clipRule="evenodd" d="M15 3V14.525L13.575 16H10.425L9 14.525V3H15ZM16 2H8V14.93L10 17H14L16 14.93V2ZM12.5 21V19H14.83L18 15.745V12H17V15.335L14.405 18H12.5H11.5H9.595L7 15.335V12H6V15.745L9.17 19H11.5V21H8L7 22H11.5H12.5H17L16 21H12.5Z" fill="#F2F2F2" />
            </svg>
            <svg className="standby-detail-send-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path d="M13.2673 4.20889C12.9674 3.9232 12.4926 3.93475 12.2069 4.23467C11.9212 4.5346 11.9328 5.00933 12.2327 5.29502L18.4841 11.2496H3.75C3.33579 11.2496 3 11.5854 3 11.9996C3 12.4138 3.33579 12.7496 3.75 12.7496H18.4842L12.2327 18.7043C11.9328 18.99 11.9212 19.4648 12.2069 19.7647C12.4926 20.0646 12.9674 20.0762 13.2673 19.7905L20.6862 12.7238C20.8551 12.5629 20.9551 12.3576 20.9861 12.1443C20.9952 12.0975 21 12.0491 21 11.9996C21 11.9501 20.9952 11.9016 20.986 11.8547C20.955 11.6415 20.855 11.4364 20.6862 11.2756L13.2673 4.20889Z" fill="#F2F2F2" />
            </svg>
          </button>
        </form>
      </div>
    </section>

    <form id="standbyForm" className="standby-mini-form" aria-hidden="true">
      <div className="standby-input-hint" aria-hidden="true">
        <span className="standby-hint-text">Type or</span>
        <span className="standby-hotkeys">
          <span className="standby-hotkey-key">Shift</span>
          <span className="standby-hotkey-key">M</span>
        </span>
        <span className="standby-hint-text">to speak</span>
      </div>
      <input
        id="standbyInput"
        className="standby-input"
        type="text"
        aria-label="Mini chat input"
        autoComplete="off"
        placeholder=""
      />
      <button id="standbySendBtn" className="standby-send" type="button" title="Voice mode" aria-label="Voice mode">
        <svg className="standby-mic-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path opacity="0.1" d="M10.85 15L10 14.125V4H14V14.125L13.15 15H10.85Z" fill="#0F0F0F" />
          <path fillRule="evenodd" clipRule="evenodd" d="M15 3V14.525L13.575 16H10.425L9 14.525V3H15ZM16 2H8V14.93L10 17H14L16 14.93V2ZM12.5 21V19H14.83L18 15.745V12H17V15.335L14.405 18H12.5H11.5H9.595L7 15.335V12H6V15.745L9.17 19H11.5V21H8L7 22H11.5H12.5H17L16 21H12.5Z" fill="#0F0F0F" />
        </svg>
        <svg className="standby-send-icon" width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M13.2673 4.20889C12.9674 3.9232 12.4926 3.93475 12.2069 4.23467C11.9212 4.5346 11.9328 5.00933 12.2327 5.29502L18.4841 11.2496H3.75C3.33579 11.2496 3 11.5854 3 11.9996C3 12.4138 3.33579 12.7496 3.75 12.7496H18.4842L12.2327 18.7043C11.9328 18.99 11.9212 19.4648 12.2069 19.7647C12.4926 20.0646 12.9674 20.0762 13.2673 19.7905L20.6862 12.7238C20.8551 12.5629 20.9551 12.3576 20.9861 12.1443C20.9952 12.0975 21 12.0491 21 11.9996C21 11.9501 20.9952 11.9016 20.986 11.8547C20.955 11.6415 20.855 11.4364 20.6862 11.2756L13.2673 4.20889Z" fill="black" />
        </svg>
      </button>
    </form>
  </main>
);
