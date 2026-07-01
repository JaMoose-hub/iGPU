use serde::Serialize;
use std::{
    fs::{self, OpenOptions},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};
use tauri::{
    webview::Color, Emitter, LogicalSize, Manager, PhysicalPosition, PhysicalSize, WindowEvent,
};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

#[cfg(windows)]
use windows_sys::Win32::Foundation::POINT;

#[cfg(windows)]
use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;

#[cfg(windows)]
use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;

#[cfg(windows)]
use windows_sys::Win32::UI::Input::XboxController::{
    XInputGetState, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_DPAD_DOWN,
    XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT, XINPUT_GAMEPAD_DPAD_UP,
    XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE,
    XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE,
};

const GAME_SEARCH_BROWSER_LABEL: &str = "game-search-results";
static CAPTURE_PROTECTION_ENABLED: AtomicBool = AtomicBool::new(false);
static VIRTUAL_CURSOR_GLOBAL_CONTROLS_ENABLED: AtomicBool = AtomicBool::new(false);
static VIRTUAL_CURSOR_TEXT_ENTRY_ACTIVE: AtomicBool = AtomicBool::new(false);
static VIRTUAL_CURSOR_STATE: OnceLock<Mutex<VirtualCursorState>> = OnceLock::new();
static CAPTURE_PROTECTION_BOOT_RESET_UNTIL: OnceLock<Instant> = OnceLock::new();
static STANDBY_CTRL_G_LAST_AT: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();
static NITROGEN_PILOT_PROCESS: OnceLock<Mutex<Option<NitrogenPilotProcess>>> = OnceLock::new();
const STANDBY_CTRL_G_DEBOUNCE_MS: u64 = 650;
const NITROGEN_REPO_DIR: &str = r"C:\Projects\ai_auto";
const NITROGEN_PYTHON: &str = r"C:\Users\Administrator\Miniconda3\python.exe";
const NITROGEN_PLAY_SCRIPT: &str = r"C:\Projects\ai_auto\scripts\play.py";
const NITROGEN_STOP_FILE: &str = r"C:\Projects\ai_auto\STOP_AGENT";
const NITROGEN_LOG_DIR: &str = r"C:\Projects\ai_auto\logs";
const VIRTUAL_CURSOR_STEP: f64 = 16.0;
const VIRTUAL_CURSOR_GAMEPAD_STEP: f64 = 20.0;
const VIRTUAL_CURSOR_SCREEN_MARGIN: f64 = 12.0;

#[derive(Clone, Serialize)]
struct VirtualCursorWindowFrame {
    label: &'static str,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    scale_factor: f64,
    visible: bool,
}

struct NitrogenPilotProcess {
    child: Child,
    host: String,
    port: u16,
    process_name: String,
    started_at: Instant,
}

#[derive(Clone, Serialize)]
struct NitrogenPilotStatus {
    running: bool,
    pid: Option<u32>,
    host: String,
    port: u16,
    process_name: String,
    elapsed_ms: Option<u64>,
    message: String,
}

struct VirtualCursorState {
    enabled: bool,
    initialized: bool,
    screen_x: f64,
    screen_y: f64,
    local_x: f64,
    local_y: f64,
    active_window: String,
    moving_window: Option<String>,
}

impl Default for VirtualCursorState {
    fn default() -> Self {
        Self {
            enabled: false,
            initialized: false,
            screen_x: 0.0,
            screen_y: 0.0,
            local_x: 0.0,
            local_y: 0.0,
            active_window: "main".to_string(),
            moving_window: None,
        }
    }
}

fn virtual_cursor_state() -> &'static Mutex<VirtualCursorState> {
    VIRTUAL_CURSOR_STATE.get_or_init(|| Mutex::new(VirtualCursorState::default()))
}

#[cfg(windows)]
const WDA_NONE: u32 = 0x00000000;

#[cfg(windows)]
const WDA_EXCLUDEFROMCAPTURE: u32 = 0x00000011;

#[cfg(windows)]
fn set_window_capture_affinity(window: &tauri::WebviewWindow, affinity: u32) -> Result<(), String> {
    let hwnd = window.hwnd().map_err(|err| err.to_string())?;
    unsafe {
        let ok = windows_sys::Win32::UI::WindowsAndMessaging::SetWindowDisplayAffinity(
            hwnd.0 as _,
            affinity,
        );
        if ok == 0 {
            return Err("SetWindowDisplayAffinity failed".to_string());
        }
    }
    Ok(())
}

#[cfg(windows)]
fn clear_window_display_affinity(window: &tauri::WebviewWindow) {
    let _ = set_window_capture_affinity(window, WDA_NONE);
}

#[cfg(not(windows))]
fn clear_window_display_affinity(_window: &tauri::WebviewWindow) {}

fn clear_window_capture_protection(window: &tauri::WebviewWindow) {
    let _ = window.set_content_protected(false);
    clear_window_display_affinity(window);
}

#[cfg(windows)]
fn set_window_display_excluded(window: &tauri::WebviewWindow, excluded: bool) {
    if excluded {
        let _ = set_window_capture_affinity(window, WDA_EXCLUDEFROMCAPTURE);
    } else {
        clear_window_capture_protection(window);
    }
}

#[cfg(not(windows))]
fn set_window_display_excluded(window: &tauri::WebviewWindow, excluded: bool) {
    let _ = window.set_content_protected(excluded);
}

fn apply_current_capture_protection(window: &tauri::WebviewWindow) {
    set_window_display_excluded(window, CAPTURE_PROTECTION_ENABLED.load(Ordering::Relaxed));
}

fn apply_capture_protection_to_all_windows(app: &tauri::AppHandle, excluded: bool) {
    for label in [
        "main", "hud", "tasks", "search", "gamepath", "tools", "standby",
    ] {
        if let Some(window) = app.get_webview_window(label) {
            set_window_display_excluded(&window, excluded);
        }
    }
}

fn clear_capture_protection_on_all_windows(app: &tauri::AppHandle) {
    CAPTURE_PROTECTION_ENABLED.store(false, Ordering::Relaxed);
    apply_capture_protection_to_all_windows(app, false);
}

fn capture_protection_boot_reset_active() -> bool {
    CAPTURE_PROTECTION_BOOT_RESET_UNTIL
        .get()
        .is_some_and(|deadline| Instant::now() < *deadline)
}

fn set_capture_protection_state(app: &tauri::AppHandle, excluded: bool) -> Result<(), String> {
    if app.get_webview_window("main").is_none() {
        return Err("main window not found".to_string());
    }

    CAPTURE_PROTECTION_ENABLED.store(excluded, Ordering::Relaxed);
    dismiss_input_experience_windows();
    apply_capture_protection_to_all_windows(app, excluded);
    park_hidden_companion_windows(app);
    dismiss_input_experience_windows();
    let _ = app.emit(
        "protect-state-changed",
        serde_json::json!({ "enabled": excluded }),
    );
    Ok(())
}

fn set_app_capture_exclusion_state(app: &tauri::AppHandle, excluded: bool) -> Result<(), String> {
    if app.get_webview_window("main").is_none() {
        return Err("main window not found".to_string());
    }

    dismiss_input_experience_windows();
    apply_capture_protection_to_all_windows(app, excluded);
    park_hidden_companion_windows(app);
    dismiss_input_experience_windows();
    Ok(())
}

fn toggle_capture_protection_state(app: tauri::AppHandle) {
    let enabled = !CAPTURE_PROTECTION_ENABLED.load(Ordering::Relaxed);
    let _ = set_capture_protection_state(&app, enabled);
}

#[cfg(not(windows))]
fn virtual_cursor_control_codes() -> Vec<Code> {
    vec![
        Code::KeyW,
        Code::KeyA,
        Code::KeyS,
        Code::KeyD,
        Code::ArrowUp,
        Code::ArrowLeft,
        Code::ArrowDown,
        Code::ArrowRight,
        Code::Numpad8,
        Code::Numpad4,
        Code::Numpad2,
        Code::Numpad6,
        Code::Numpad5,
        Code::Tab,
        Code::Enter,
        Code::NumpadEnter,
        Code::Space,
        Code::Escape,
    ]
}

fn virtual_cursor_control_payload(code: Code) -> Option<serde_json::Value> {
    match code {
        Code::KeyW | Code::ArrowUp | Code::Numpad8 => {
            Some(serde_json::json!({ "type": "move", "dx": 0, "dy": -1 }))
        }
        Code::KeyA | Code::ArrowLeft | Code::Numpad4 => {
            Some(serde_json::json!({ "type": "move", "dx": -1, "dy": 0 }))
        }
        Code::KeyS | Code::ArrowDown | Code::Numpad2 => {
            Some(serde_json::json!({ "type": "move", "dx": 0, "dy": 1 }))
        }
        Code::KeyD | Code::ArrowRight | Code::Numpad6 => {
            Some(serde_json::json!({ "type": "move", "dx": 1, "dy": 0 }))
        }
        Code::Tab => Some(serde_json::json!({ "type": "target_next" })),
        Code::Enter | Code::NumpadEnter | Code::Space | Code::Numpad5 => {
            Some(serde_json::json!({ "type": "activate" }))
        }
        Code::Escape => Some(serde_json::json!({ "type": "disable" })),
        _ => None,
    }
}

fn frame_contains_screen_point(frame: &VirtualCursorWindowFrame, x: f64, y: f64) -> bool {
    let left = frame.x as f64;
    let top = frame.y as f64;
    x >= left && x <= left + frame.width as f64 && y >= top && y <= top + frame.height as f64
}

fn frame_distance_to_point(frame: &VirtualCursorWindowFrame, x: f64, y: f64) -> f64 {
    let left = frame.x as f64;
    let top = frame.y as f64;
    let right = left + frame.width as f64;
    let bottom = top + frame.height as f64;
    let dx = if x < left {
        left - x
    } else if x > right {
        x - right
    } else {
        0.0
    };
    let dy = if y < top {
        top - y
    } else if y > bottom {
        y - bottom
    } else {
        0.0
    };
    (dx * dx + dy * dy).sqrt()
}

fn frame_center(frame: &VirtualCursorWindowFrame) -> (f64, f64) {
    (
        frame.x as f64 + frame.width as f64 / 2.0,
        frame.y as f64 + frame.height as f64 / 2.0,
    )
}

fn frame_local_size(frame: &VirtualCursorWindowFrame) -> (f64, f64) {
    let scale = if frame.scale_factor > 0.0 {
        frame.scale_factor
    } else {
        1.0
    };
    (
        (frame.width as f64 / scale).max(24.0),
        (frame.height as f64 / scale).max(24.0),
    )
}

fn screen_to_frame_local(frame: &VirtualCursorWindowFrame, x: f64, y: f64) -> (f64, f64) {
    let scale = if frame.scale_factor > 0.0 {
        frame.scale_factor
    } else {
        1.0
    };
    let (width, height) = frame_local_size(frame);
    (
        ((x - frame.x as f64) / scale).clamp(12.0, width - 12.0),
        ((y - frame.y as f64) / scale).clamp(12.0, height - 12.0),
    )
}

fn select_virtual_cursor_frame<'a>(
    frames: &'a [VirtualCursorWindowFrame],
    active_window: &str,
    screen_x: f64,
    screen_y: f64,
) -> Option<&'a VirtualCursorWindowFrame> {
    if let Some(frame) = frames.iter().find(|frame| {
        frame.label == active_window && frame_contains_screen_point(frame, screen_x, screen_y)
    }) {
        return Some(frame);
    }

    if let Some(frame) = frames
        .iter()
        .rev()
        .find(|frame| frame_contains_screen_point(frame, screen_x, screen_y))
    {
        return Some(frame);
    }

    frames
        .iter()
        .map(|frame| {
            let active_bias = if frame.label == active_window {
                -1.0
            } else {
                0.0
            };
            (
                frame_distance_to_point(frame, screen_x, screen_y) + active_bias,
                frame,
            )
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, frame)| frame)
}

fn virtual_cursor_exit_edge(
    frame: &VirtualCursorWindowFrame,
    next_x: f64,
    next_y: f64,
    delta_x: f64,
    delta_y: f64,
) -> Option<&'static str> {
    let left = frame.x as f64;
    let top = frame.y as f64;
    let right = left + frame.width as f64;
    let bottom = top + frame.height as f64;

    let horizontal = if delta_x > 0.0 && next_x > right {
        Some("right")
    } else if delta_x < 0.0 && next_x < left {
        Some("left")
    } else {
        None
    };
    let vertical = if delta_y > 0.0 && next_y > bottom {
        Some("down")
    } else if delta_y < 0.0 && next_y < top {
        Some("up")
    } else {
        None
    };

    match (horizontal, vertical) {
        (Some(x_edge), Some(y_edge)) => {
            if delta_x.abs() >= delta_y.abs() {
                Some(x_edge)
            } else {
                Some(y_edge)
            }
        }
        (Some(edge), None) | (None, Some(edge)) => Some(edge),
        (None, None) => None,
    }
}

fn transfer_virtual_cursor_at_edge(
    state: &mut VirtualCursorState,
    frames: &[VirtualCursorWindowFrame],
    source: &VirtualCursorWindowFrame,
    edge: &str,
    next_x: f64,
    next_y: f64,
) -> bool {
    let source_left = source.x as f64;
    let source_top = source.y as f64;
    let source_right = source_left + source.width as f64;
    let source_bottom = source_top + source.height as f64;
    let cursor_screen_x = match edge {
        "left" => source_left - 1.0,
        "right" => source_right + 1.0,
        _ => next_x.clamp(source_left, source_right),
    };
    let cursor_screen_y = match edge {
        "up" => source_top - 1.0,
        "down" => source_bottom + 1.0,
        _ => next_y.clamp(source_top, source_bottom),
    };

    let Some(target) =
        pick_virtual_cursor_transfer_target(frames, source, edge, cursor_screen_x, cursor_screen_y)
    else {
        return false;
    };

    let target_scale = if target.scale_factor > 0.0 {
        target.scale_factor
    } else {
        1.0
    };
    let (target_width, target_height) = frame_local_size(&target);
    let local_x =
        ((cursor_screen_x - target.x as f64) / target_scale).clamp(12.0, target_width - 12.0);
    let local_y =
        ((cursor_screen_y - target.y as f64) / target_scale).clamp(12.0, target_height - 12.0);

    state.active_window = target.label.to_string();
    state.moving_window = None;
    state.initialized = true;
    state.local_x = local_x;
    state.local_y = local_y;
    state.screen_x = target.x as f64 + local_x * target_scale;
    state.screen_y = target.y as f64 + local_y * target_scale;
    true
}

fn initialize_virtual_cursor_position(
    state: &mut VirtualCursorState,
    frames: &[VirtualCursorWindowFrame],
) {
    if state.initialized {
        return;
    }
    let frame = frames
        .iter()
        .find(|frame| frame.label == state.active_window)
        .or_else(|| frames.iter().find(|frame| frame.label == "main"))
        .or_else(|| frames.first());
    if let Some(frame) = frame {
        let (x, y) = frame_center(frame);
        state.screen_x = x;
        state.screen_y = y;
        state.active_window = frame.label.to_string();
        state.initialized = true;
    }
}

fn clamp_virtual_cursor_to_world(
    state: &mut VirtualCursorState,
    frames: &[VirtualCursorWindowFrame],
) {
    initialize_virtual_cursor_position(state, frames);
    if frames.is_empty() {
        return;
    }

    let left = frames
        .iter()
        .map(|frame| frame.x as f64)
        .fold(f64::INFINITY, f64::min);
    let top = frames
        .iter()
        .map(|frame| frame.y as f64)
        .fold(f64::INFINITY, f64::min);
    let right = frames
        .iter()
        .map(|frame| frame.x as f64 + frame.width as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let bottom = frames
        .iter()
        .map(|frame| frame.y as f64 + frame.height as f64)
        .fold(f64::NEG_INFINITY, f64::max);

    let min_x = left + VIRTUAL_CURSOR_SCREEN_MARGIN;
    let max_x = right - VIRTUAL_CURSOR_SCREEN_MARGIN;
    let min_y = top + VIRTUAL_CURSOR_SCREEN_MARGIN;
    let max_y = bottom - VIRTUAL_CURSOR_SCREEN_MARGIN;
    if min_x <= max_x {
        state.screen_x = state.screen_x.clamp(min_x, max_x);
    } else {
        state.screen_x = (left + right) / 2.0;
    }
    if min_y <= max_y {
        state.screen_y = state.screen_y.clamp(min_y, max_y);
    } else {
        state.screen_y = (top + bottom) / 2.0;
    }
}

fn anchor_virtual_cursor_to_active_frame(
    state: &mut VirtualCursorState,
    frames: &[VirtualCursorWindowFrame],
) {
    if !state.enabled || !state.initialized {
        clamp_virtual_cursor_to_world(state, frames);
        return;
    }

    let Some(frame) = frames
        .iter()
        .find(|frame| frame.label == state.active_window.as_str())
    else {
        state.moving_window = None;
        clamp_virtual_cursor_to_world(state, frames);
        return;
    };

    let scale = if frame.scale_factor > 0.0 {
        frame.scale_factor
    } else {
        1.0
    };
    let (width, height) = frame_local_size(frame);
    let local_x = state.local_x.clamp(
        VIRTUAL_CURSOR_SCREEN_MARGIN,
        width - VIRTUAL_CURSOR_SCREEN_MARGIN,
    );
    let local_y = state.local_y.clamp(
        VIRTUAL_CURSOR_SCREEN_MARGIN,
        height - VIRTUAL_CURSOR_SCREEN_MARGIN,
    );
    state.screen_x = frame.x as f64 + local_x * scale;
    state.screen_y = frame.y as f64 + local_y * scale;
    clamp_virtual_cursor_to_world(state, frames);
}

fn emit_virtual_cursor_render(app: &tauri::AppHandle) -> Result<(), String> {
    let frames = collect_virtual_cursor_window_frames(app)?;
    let mut state = virtual_cursor_state()
        .lock()
        .map_err(|_| "virtual cursor state poisoned".to_string())?;

    if !state.enabled || frames.is_empty() {
        app.emit(
            "virtual-cursor-render",
            serde_json::json!({
                "enabled": false,
                "activeWindow": state.active_window,
                "movingWindow": serde_json::Value::Null,
            }),
        )
        .map_err(|err| err.to_string())?;
        return Ok(());
    }

    clamp_virtual_cursor_to_world(&mut state, &frames);
    let Some(frame) = select_virtual_cursor_frame(
        &frames,
        &state.active_window,
        state.screen_x,
        state.screen_y,
    ) else {
        return Ok(());
    };
    state.active_window = frame.label.to_string();
    let (local_x, local_y) = screen_to_frame_local(frame, state.screen_x, state.screen_y);
    state.local_x = local_x;
    state.local_y = local_y;
    let active_window = state.active_window.clone();
    let moving_window = state.moving_window.clone();
    let screen_x = state.screen_x;
    let screen_y = state.screen_y;
    drop(state);

    let payload = serde_json::json!({
        "enabled": true,
        "activeWindow": active_window,
        "x": local_x,
        "y": local_y,
        "screenX": screen_x,
        "screenY": screen_y,
        "movingWindow": moving_window,
    });
    let _ = app.emit_to(
        active_window.as_str(),
        "virtual-cursor-render",
        payload.clone(),
    );
    app.emit("virtual-cursor-render", payload)
        .map_err(|err| err.to_string())
}

fn set_virtual_cursor_enabled(app: &tauri::AppHandle, enabled: bool) -> Result<(), String> {
    VIRTUAL_CURSOR_GLOBAL_CONTROLS_ENABLED.store(enabled, Ordering::Relaxed);
    VIRTUAL_CURSOR_TEXT_ENTRY_ACTIVE.store(false, Ordering::Relaxed);
    {
        let frames = collect_virtual_cursor_window_frames(app)?;
        let mut state = virtual_cursor_state()
            .lock()
            .map_err(|_| "virtual cursor state poisoned".to_string())?;
        state.enabled = enabled;
        if !enabled {
            state.moving_window = None;
        }
        if enabled {
            clamp_virtual_cursor_to_world(&mut state, &frames);
        }
    }
    emit_virtual_cursor_render(app)
}

fn move_virtual_cursor_window(app: &tauri::AppHandle, label: &str, delta_x: f64, delta_y: f64) {
    let Some(window) = app.get_webview_window(label) else {
        if let Ok(mut state) = virtual_cursor_state().lock() {
            state.moving_window = None;
        }
        let _ = emit_virtual_cursor_render(app);
        return;
    };

    let Ok(position) = window.outer_position() else {
        return;
    };

    let next_x = position.x.saturating_add(delta_x.round() as i32);
    let next_y = position.y.saturating_add(delta_y.round() as i32);
    let _ = window.set_position(PhysicalPosition::new(next_x, next_y));

    if let Ok(frames) = collect_virtual_cursor_window_frames(app) {
        if let Ok(mut state) = virtual_cursor_state().lock() {
            anchor_virtual_cursor_to_active_frame(&mut state, &frames);
        }
    }
    let _ = emit_virtual_cursor_render(app);
}

fn move_virtual_cursor_by(app: &tauri::AppHandle, delta_x: f64, delta_y: f64) {
    if delta_x.abs() < 0.05 && delta_y.abs() < 0.05 {
        return;
    }
    let moving_window = virtual_cursor_state().lock().ok().and_then(|state| {
        if state.enabled {
            state.moving_window.clone()
        } else {
            None
        }
    });
    if let Some(label) = moving_window {
        move_virtual_cursor_window(app, &label, delta_x, delta_y);
        return;
    }

    let mut moved = false;
    if let Ok(frames) = collect_virtual_cursor_window_frames(app) {
        if let Ok(mut state) = virtual_cursor_state().lock() {
            if !state.enabled {
                return;
            }
            clamp_virtual_cursor_to_world(&mut state, &frames);
            let next_x = state.screen_x + delta_x;
            let next_y = state.screen_y + delta_y;
            let active_window = state.active_window.clone();
            let transferred = frames
                .iter()
                .find(|frame| frame.label == active_window.as_str())
                .and_then(|source| {
                    if frame_contains_screen_point(source, next_x, next_y) {
                        return None;
                    }
                    virtual_cursor_exit_edge(source, next_x, next_y, delta_x, delta_y)
                        .map(|edge| (source, edge))
                })
                .map(|(source, edge)| {
                    transfer_virtual_cursor_at_edge(
                        &mut state, &frames, source, edge, next_x, next_y,
                    )
                })
                .unwrap_or(false);

            if !transferred {
                state.screen_x = next_x;
                state.screen_y = next_y;
                clamp_virtual_cursor_to_world(&mut state, &frames);
            }
            moved = true;
        }
    }
    if moved {
        let _ = emit_virtual_cursor_render(app);
    }
}

fn move_virtual_cursor(app: &tauri::AppHandle, dx: i32, dy: i32) {
    move_virtual_cursor_by(
        app,
        dx as f64 * VIRTUAL_CURSOR_STEP,
        dy as f64 * VIRTUAL_CURSOR_STEP,
    );
}

fn emit_virtual_cursor_action(app: &tauri::AppHandle, action: &str) {
    let active_window = virtual_cursor_state()
        .lock()
        .ok()
        .map(|state| state.active_window.clone())
        .unwrap_or_else(|| "main".to_string());
    let payload = serde_json::json!({ "type": action, "activeWindow": active_window });
    let _ = app.emit("virtual-cursor-action", payload.clone());
    let _ = app.emit_to(active_window.as_str(), "virtual-cursor-action", payload);
}

fn set_virtual_cursor_global_controls_state(
    app: &tauri::AppHandle,
    enabled: bool,
) -> Result<(), String> {
    #[cfg(windows)]
    {
        return set_virtual_cursor_enabled(app, enabled);
    }

    #[cfg(not(windows))]
    {
        let was_enabled = VIRTUAL_CURSOR_GLOBAL_CONTROLS_ENABLED.swap(enabled, Ordering::Relaxed);
        if was_enabled == enabled {
            return Ok(());
        }

        for code in virtual_cursor_control_codes() {
            let shortcut = Shortcut::new(None, code);
            let result = if enabled {
                app.global_shortcut().register(shortcut)
            } else {
                app.global_shortcut().unregister(shortcut)
            };
            if let Err(err) = result {
                eprintln!("virtual cursor shortcut {:?} failed: {}", code, err);
            }
        }
        set_virtual_cursor_enabled(app, enabled)
    }
}

#[cfg(windows)]
fn virtual_key_down(vkey: i32) -> bool {
    unsafe { (GetAsyncKeyState(vkey) as u16 & 0x8000) != 0 }
}

fn emit_standby_hotkey(app: &tauri::AppHandle, source: &str) {
    if app
        .emit_to(
            "main",
            "standby-hotkey",
            serde_json::json!({ "source": source }),
        )
        .is_err()
    {
        let _ = open_standby_typein_window(app);
    }
}

#[cfg(windows)]
fn start_virtual_cursor_keyboard_poll(app: tauri::AppHandle) {
    thread::spawn(move || {
        let mut was_standby_hotkey = false;
        let mut was_tab = false;
        let mut was_activate = false;
        let mut pending_activate = false;
        let mut controls_clear_since: Option<Instant> = None;
        let mut was_escape = false;
        let mut was_f11 = false;

        loop {
            let ctrl = virtual_key_down(0x11) || virtual_key_down(0xA2) || virtual_key_down(0xA3);
            let alt = virtual_key_down(0x12) || virtual_key_down(0xA4) || virtual_key_down(0xA5);
            let g = virtual_key_down(0x47);
            let standby_hotkey = ctrl && g;
            if standby_hotkey && !was_standby_hotkey && should_accept_standby_ctrl_g() {
                emit_standby_hotkey(
                    &app,
                    if alt {
                        "poll_ctrl_alt_g"
                    } else {
                        "poll_ctrl_g"
                    },
                );
            }
            was_standby_hotkey = standby_hotkey;

            let f11 = virtual_key_down(0x7A);
            if f11 && !was_f11 {
                let enabled = !VIRTUAL_CURSOR_GLOBAL_CONTROLS_ENABLED.load(Ordering::Relaxed);
                let _ = set_virtual_cursor_enabled(&app, enabled);
            }
            was_f11 = f11;

            if !VIRTUAL_CURSOR_GLOBAL_CONTROLS_ENABLED.load(Ordering::Relaxed) {
                was_tab = false;
                was_activate = false;
                pending_activate = false;
                controls_clear_since = None;
                was_escape = false;
                thread::sleep(Duration::from_millis(45));
                continue;
            }

            if VIRTUAL_CURSOR_TEXT_ENTRY_ACTIVE.load(Ordering::Relaxed) {
                was_tab = false;
                was_activate = false;
                pending_activate = false;
                controls_clear_since = None;
                was_escape = false;
                thread::sleep(Duration::from_millis(55));
                continue;
            }

            let up = virtual_key_down(0x57) || virtual_key_down(0x26) || virtual_key_down(0x68);
            let left = virtual_key_down(0x41) || virtual_key_down(0x25) || virtual_key_down(0x64);
            let down = virtual_key_down(0x53) || virtual_key_down(0x28) || virtual_key_down(0x62);
            let right = virtual_key_down(0x44) || virtual_key_down(0x27) || virtual_key_down(0x66);

            let dx = i32::from(right) - i32::from(left);
            let dy = i32::from(down) - i32::from(up);
            if dx != 0 || dy != 0 {
                move_virtual_cursor(&app, dx, dy);
            }

            let tab = virtual_key_down(0x09);
            if tab && !was_tab {
                emit_virtual_cursor_action(&app, "target_next");
            }
            was_tab = tab;

            let activate =
                virtual_key_down(0x0D) || virtual_key_down(0x20) || virtual_key_down(0x65);
            if activate && !was_activate {
                pending_activate = true;
                controls_clear_since = None;
            }
            was_activate = activate;

            let controls_down = up || left || down || right || tab || activate;
            if pending_activate {
                if controls_down {
                    controls_clear_since = None;
                } else {
                    let now = Instant::now();
                    let clear_since = controls_clear_since.get_or_insert(now);
                    if now.duration_since(*clear_since) >= Duration::from_millis(90) {
                        emit_virtual_cursor_action(&app, "activate");
                        pending_activate = false;
                        controls_clear_since = None;
                    }
                }
            }

            let escape = virtual_key_down(0x1B);
            if escape && !was_escape {
                if !clear_virtual_cursor_window_move(&app) {
                    let _ = set_virtual_cursor_enabled(&app, false);
                }
                pending_activate = false;
                controls_clear_since = None;
            }
            was_escape = escape;

            thread::sleep(Duration::from_millis(55));
        }
    });
}

#[cfg(not(windows))]
fn start_virtual_cursor_keyboard_poll(_app: tauri::AppHandle) {}

#[cfg(windows)]
fn xinput_button_down(buttons: u16, button: u16) -> bool {
    buttons & button != 0
}

#[cfg(windows)]
fn normalized_gamepad_axis(value: i16) -> f64 {
    let deadzone = XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE as f64;
    let raw = value as i32;
    let magnitude = raw.abs() as f64;
    if magnitude <= deadzone {
        return 0.0;
    }
    let sign = if raw < 0 { -1.0 } else { 1.0 };
    let normalized = ((magnitude - deadzone) / (32767.0 - deadzone)).clamp(0.0, 1.0);
    sign * normalized.powf(1.25)
}

#[cfg(windows)]
fn first_connected_xinput_state() -> Option<XINPUT_STATE> {
    for index in 0..4 {
        let mut state: XINPUT_STATE = unsafe { std::mem::zeroed() };
        let result = unsafe { XInputGetState(index, &mut state) };
        if result == 0 {
            return Some(state);
        }
    }
    None
}

fn exit_virtual_cursor_text_entry(app: &tauri::AppHandle, source: &str) {
    VIRTUAL_CURSOR_TEXT_ENTRY_ACTIVE.store(false, Ordering::Relaxed);
    let _ = app.emit(
        "virtual-cursor-exit-text-entry",
        serde_json::json!({ "source": source }),
    );
}

fn clear_virtual_cursor_window_move(app: &tauri::AppHandle) -> bool {
    let mut was_moving = false;
    if let Ok(mut state) = virtual_cursor_state().lock() {
        was_moving = state.moving_window.is_some();
        state.moving_window = None;
    }
    if was_moving {
        let _ = emit_virtual_cursor_render(app);
    }
    was_moving
}

#[cfg(windows)]
fn start_virtual_cursor_gamepad_poll(app: tauri::AppHandle) {
    thread::spawn(move || {
        let mut was_modifier = false;
        let mut was_a = false;
        let mut was_b = false;
        let mut was_x = false;
        let mut was_y = false;

        loop {
            let Some(state) = first_connected_xinput_state() else {
                was_modifier = false;
                was_a = false;
                was_b = false;
                was_x = false;
                was_y = false;
                thread::sleep(Duration::from_millis(250));
                continue;
            };

            let gamepad = state.Gamepad;
            let buttons = gamepad.wButtons;
            let modifier = xinput_button_down(buttons, XINPUT_GAMEPAD_LEFT_SHOULDER)
                && xinput_button_down(buttons, XINPUT_GAMEPAD_RIGHT_SHOULDER);

            if !modifier {
                was_modifier = false;
                was_a = false;
                was_b = false;
                was_x = false;
                was_y = false;
                thread::sleep(Duration::from_millis(35));
                continue;
            }

            if !was_modifier {
                let _ = set_virtual_cursor_enabled(&app, true);
            }
            was_modifier = true;

            let b = xinput_button_down(buttons, XINPUT_GAMEPAD_B);
            if b && !was_b {
                if VIRTUAL_CURSOR_TEXT_ENTRY_ACTIVE.load(Ordering::Relaxed) {
                    exit_virtual_cursor_text_entry(&app, "gamepad-b");
                } else if !clear_virtual_cursor_window_move(&app) {
                    let _ = set_virtual_cursor_enabled(&app, false);
                }
            }
            was_b = b;

            if VIRTUAL_CURSOR_TEXT_ENTRY_ACTIVE.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(35));
                continue;
            }

            let a = xinput_button_down(buttons, XINPUT_GAMEPAD_A);
            if a && !was_a {
                emit_virtual_cursor_action(&app, "activate");
            }
            was_a = a;

            let x = xinput_button_down(buttons, XINPUT_GAMEPAD_X);
            if x && !was_x {
                let _ = show_tasks_window(app.clone());
                let _ = set_virtual_cursor_active_window(app.clone(), "tasks".to_string());
            }
            was_x = x;

            let y = xinput_button_down(buttons, XINPUT_GAMEPAD_Y);
            if y && !was_y {
                let _ = show_search_window(app.clone());
                let _ = set_virtual_cursor_active_window(app.clone(), "search".to_string());
            }
            was_y = y;

            let mut dx = normalized_gamepad_axis(gamepad.sThumbLX) * VIRTUAL_CURSOR_GAMEPAD_STEP;
            let mut dy = -normalized_gamepad_axis(gamepad.sThumbLY) * VIRTUAL_CURSOR_GAMEPAD_STEP;

            if xinput_button_down(buttons, XINPUT_GAMEPAD_DPAD_LEFT) {
                dx -= VIRTUAL_CURSOR_STEP;
            }
            if xinput_button_down(buttons, XINPUT_GAMEPAD_DPAD_RIGHT) {
                dx += VIRTUAL_CURSOR_STEP;
            }
            if xinput_button_down(buttons, XINPUT_GAMEPAD_DPAD_UP) {
                dy -= VIRTUAL_CURSOR_STEP;
            }
            if xinput_button_down(buttons, XINPUT_GAMEPAD_DPAD_DOWN) {
                dy += VIRTUAL_CURSOR_STEP;
            }

            move_virtual_cursor_by(&app, dx, dy);
            thread::sleep(Duration::from_millis(24));
        }
    });
}

#[cfg(not(windows))]
fn start_virtual_cursor_gamepad_poll(_app: tauri::AppHandle) {}

fn park_window_offscreen(window: &tauri::WebviewWindow) {
    let _ = window.set_position(PhysicalPosition::new(-32000, -32000));
}

fn park_window_if_hidden(window: &tauri::WebviewWindow) {
    if !window.is_visible().unwrap_or(false) {
        park_window_offscreen(window);
    }
}

fn park_hidden_companion_windows(app: &tauri::AppHandle) {
    for label in ["hud", "tasks", "search", "gamepath", "tools", "standby"] {
        if let Some(window) = app.get_webview_window(label) {
            park_window_if_hidden(&window);
        }
    }
}

fn hide_companion_window(window: &tauri::WebviewWindow) -> Result<(), String> {
    window.hide().map_err(|err| err.to_string())?;
    park_window_offscreen(window);
    dismiss_input_experience_windows();
    Ok(())
}

#[cfg(windows)]
unsafe extern "system" fn enum_input_experience_windows(
    hwnd: windows_sys::Win32::Foundation::HWND,
    _lparam: windows_sys::Win32::Foundation::LPARAM,
) -> windows_sys::Win32::Foundation::BOOL {
    let mut class_buf = [0u16; 256];
    let class_len = windows_sys::Win32::UI::WindowsAndMessaging::GetClassNameW(
        hwnd,
        class_buf.as_mut_ptr(),
        class_buf.len() as i32,
    );
    if class_len <= 0 {
        return 1;
    }

    let class_name = String::from_utf16_lossy(&class_buf[..class_len as usize]);
    if class_name != "Windows.UI.Core.CoreWindow" {
        return 1;
    }

    let mut title_buf = [0u16; 256];
    let title_len = windows_sys::Win32::UI::WindowsAndMessaging::GetWindowTextW(
        hwnd,
        title_buf.as_mut_ptr(),
        title_buf.len() as i32,
    );
    let title = if title_len > 0 {
        String::from_utf16_lossy(&title_buf[..title_len as usize])
    } else {
        String::new()
    };
    let title_lower = title.to_lowercase();
    if title.contains("Windows 輸入") || title_lower.contains("input experience") {
        windows_sys::Win32::UI::WindowsAndMessaging::ShowWindow(
            hwnd,
            windows_sys::Win32::UI::WindowsAndMessaging::SW_HIDE,
        );
    }

    1
}

#[cfg(windows)]
fn dismiss_input_experience_windows() {
    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::EnumWindows(
            Some(enum_input_experience_windows),
            0,
        );
    }
}

#[cfg(not(windows))]
fn dismiss_input_experience_windows() {}

fn force_companion_window_repaint(window: &tauri::WebviewWindow, x: i32, y: i32) {
    dismiss_input_experience_windows();

    #[cfg(windows)]
    {
        if let Ok(hwnd) = window.hwnd() {
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::SetWindowPos(
                    hwnd.0 as _,
                    windows_sys::Win32::UI::WindowsAndMessaging::HWND_TOPMOST,
                    x,
                    y,
                    0,
                    0,
                    windows_sys::Win32::UI::WindowsAndMessaging::SWP_NOSIZE
                        | windows_sys::Win32::UI::WindowsAndMessaging::SWP_NOACTIVATE
                        | windows_sys::Win32::UI::WindowsAndMessaging::SWP_SHOWWINDOW
                        | windows_sys::Win32::UI::WindowsAndMessaging::SWP_FRAMECHANGED,
                );
            }
        }
    }

    let _ = window.eval(
        r#"
        window.dispatchEvent(new Event("resize"));
        requestAnimationFrame(() => {
          document.documentElement.style.transform = "translateZ(0)";
          requestAnimationFrame(() => {
            document.documentElement.style.transform = "";
            window.dispatchEvent(new Event("resize"));
          });
        });
        "#,
    );
    let _ = window.emit("companion-window-shown", ());
    dismiss_input_experience_windows();
}

#[cfg(windows)]
fn force_companion_window_focus(window: &tauri::WebviewWindow) {
    if let Ok(hwnd) = window.hwnd() {
        unsafe {
            let hwnd = hwnd.0 as _;
            let current_thread = windows_sys::Win32::System::Threading::GetCurrentThreadId();
            let target_thread =
                windows_sys::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId(
                    hwnd,
                    std::ptr::null_mut(),
                );
            let foreground = windows_sys::Win32::UI::WindowsAndMessaging::GetForegroundWindow();
            let foreground_thread = if foreground.is_null() {
                0
            } else {
                windows_sys::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId(
                    foreground,
                    std::ptr::null_mut(),
                )
            };

            if foreground_thread != 0 && foreground_thread != current_thread {
                windows_sys::Win32::System::Threading::AttachThreadInput(
                    current_thread,
                    foreground_thread,
                    1,
                );
            }
            if target_thread != 0 && target_thread != current_thread {
                windows_sys::Win32::System::Threading::AttachThreadInput(
                    current_thread,
                    target_thread,
                    1,
                );
            }

            windows_sys::Win32::UI::WindowsAndMessaging::ShowWindow(
                hwnd,
                windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOW,
            );
            windows_sys::Win32::UI::WindowsAndMessaging::SetWindowPos(
                hwnd,
                windows_sys::Win32::UI::WindowsAndMessaging::HWND_TOPMOST,
                0,
                0,
                0,
                0,
                windows_sys::Win32::UI::WindowsAndMessaging::SWP_NOMOVE
                    | windows_sys::Win32::UI::WindowsAndMessaging::SWP_NOSIZE
                    | windows_sys::Win32::UI::WindowsAndMessaging::SWP_SHOWWINDOW,
            );
            windows_sys::Win32::UI::WindowsAndMessaging::BringWindowToTop(hwnd);
            windows_sys::Win32::UI::WindowsAndMessaging::SetForegroundWindow(hwnd);
            windows_sys::Win32::UI::Input::KeyboardAndMouse::SetActiveWindow(hwnd);
            windows_sys::Win32::UI::Input::KeyboardAndMouse::SetFocus(hwnd);

            if target_thread != 0 && target_thread != current_thread {
                windows_sys::Win32::System::Threading::AttachThreadInput(
                    current_thread,
                    target_thread,
                    0,
                );
            }
            if foreground_thread != 0 && foreground_thread != current_thread {
                windows_sys::Win32::System::Threading::AttachThreadInput(
                    current_thread,
                    foreground_thread,
                    0,
                );
            }
        }
    }
}

#[cfg(not(windows))]
fn force_companion_window_focus(_window: &tauri::WebviewWindow) {}

#[cfg(windows)]
fn click_screen_point(screen_x: i32, screen_y: i32) -> Result<(), String> {
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        SendInput, INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP,
        MOUSEINPUT,
    };

    let mut previous = POINT { x: 0, y: 0 };
    let has_previous =
        unsafe { windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos(&mut previous) != 0 };

    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::SetCursorPos(screen_x, screen_y);
    }
    thread::sleep(Duration::from_millis(18));

    let inputs = [
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: 0,
                    dwFlags: MOUSEEVENTF_LEFTDOWN,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        },
        INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: 0,
                    dwFlags: MOUSEEVENTF_LEFTUP,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        },
    ];
    let sent = unsafe {
        SendInput(
            inputs.len() as u32,
            inputs.as_ptr(),
            std::mem::size_of::<INPUT>() as i32,
        )
    };
    thread::sleep(Duration::from_millis(18));

    if has_previous {
        unsafe {
            windows_sys::Win32::UI::WindowsAndMessaging::SetCursorPos(previous.x, previous.y);
        }
    }

    if sent != inputs.len() as u32 {
        return Err("SendInput did not deliver the virtual cursor click".to_string());
    }
    Ok(())
}

#[cfg(not(windows))]
fn click_screen_point(_screen_x: i32, _screen_y: i32) -> Result<(), String> {
    Ok(())
}

#[cfg(windows)]
fn find_hud_hwnd() -> windows_sys::Win32::Foundation::HWND {
    let title: Vec<u16> = "game-guidance-hud\0".encode_utf16().collect();
    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::FindWindowW(std::ptr::null(), title.as_ptr())
    }
}

#[tauri::command]
fn configure_hud_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(hud) = app.get_webview_window("hud") else {
        return Err("hud window not found".to_string());
    };

    let _ = hud.set_ignore_cursor_events(true);
    let _ = hud.set_skip_taskbar(true);
    let _ = hud.set_focusable(false);
    let _ = hud.set_always_on_top(true);

    Ok(())
}

#[tauri::command]
fn show_hud_window(
    app: tauri::AppHandle,
    width: f64,
    height: f64,
    x: Option<f64>,
    y: Option<f64>,
) -> Result<(), String> {
    let Some(hud) = app.get_webview_window("hud") else {
        return Err("hud window not found".to_string());
    };

    apply_current_capture_protection(&hud);
    let _ = hud.set_skip_taskbar(true);
    let _ = hud.set_ignore_cursor_events(true);
    let _ = hud.set_focusable(false);
    let _ = hud.set_always_on_top(true);

    let requested_width = width.max(1.0) as u32;
    let requested_height = height.max(1.0) as u32;
    let mut final_x = x.unwrap_or(0.0).round() as i32;
    let mut final_y = y.unwrap_or(0.0).round() as i32;
    let mut final_width = requested_width;
    let mut final_height = requested_height;

    if x.is_some() || y.is_some() {
        let _ = hud.set_position(PhysicalPosition::new(final_x, final_y));
        let _ = hud.set_size(PhysicalSize::new(final_width, final_height));
    } else if let Ok(Some(monitor)) = app.primary_monitor() {
        let position = monitor.position();
        let size = monitor.size();
        final_x = position.x;
        final_y = position.y;
        final_width = size.width;
        final_height = size.height;
        let _ = hud.set_position(PhysicalPosition::new(position.x, position.y));
        let _ = hud.set_size(PhysicalSize::new(size.width, size.height));
    } else {
        let _ = hud.set_position(PhysicalPosition::new(0, 0));
        let _ = hud.set_size(PhysicalSize::new(final_width, final_height));
    }
    hud.show().map_err(|err| err.to_string())?;
    #[cfg(windows)]
    {
        if let Ok(hwnd) = hud.hwnd() {
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::SetWindowPos(
                    hwnd.0 as _,
                    windows_sys::Win32::UI::WindowsAndMessaging::HWND_TOPMOST,
                    final_x,
                    final_y,
                    final_width as i32,
                    final_height as i32,
                    windows_sys::Win32::UI::WindowsAndMessaging::SWP_NOACTIVATE
                        | windows_sys::Win32::UI::WindowsAndMessaging::SWP_SHOWWINDOW,
                );
            }
        } else {
            let top_hwnd = find_hud_hwnd();
            if !top_hwnd.is_null() {
                unsafe {
                    windows_sys::Win32::UI::WindowsAndMessaging::ShowWindow(
                        top_hwnd,
                        windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNOACTIVATE,
                    );
                }
            }
        }
    }
    let _ = hud.set_always_on_top(true);
    let _ = hud.set_focusable(false);
    let _ = hud.set_ignore_cursor_events(true);
    Ok(())
}

#[tauri::command]
fn show_hud_overlay(
    app: tauri::AppHandle,
    overlay: serde_json::Value,
    width: f64,
    height: f64,
    x: Option<f64>,
    y: Option<f64>,
) -> Result<(), String> {
    show_hud_window(app.clone(), width, height, x, y)?;

    let Some(hud) = app.get_webview_window("hud") else {
        return Err("hud window not found".to_string());
    };
    hud.emit("hud:show", overlay).map_err(|err| err.to_string())
}

#[tauri::command]
fn clear_hud_overlay(app: tauri::AppHandle) -> Result<(), String> {
    let Some(hud) = app.get_webview_window("hud") else {
        return Ok(());
    };
    hud.emit("hud:clear", ()).map_err(|err| err.to_string())
}

#[tauri::command]
fn hide_hud_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(hud) = app.get_webview_window("hud") else {
        return Ok(());
    };

    #[cfg(windows)]
    {
        let top_hwnd = find_hud_hwnd();
        if !top_hwnd.is_null() {
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::ShowWindow(
                    top_hwnd,
                    windows_sys::Win32::UI::WindowsAndMessaging::SW_HIDE,
                );
            }
        }
    }

    hide_companion_window(&hud)
}

fn ensure_tasks_window(app: &tauri::AppHandle) -> Result<tauri::WebviewWindow, String> {
    let Some(tasks) = app.get_webview_window("tasks") else {
        return Err("tasks window not found".to_string());
    };
    apply_current_capture_protection(&tasks);
    Ok(tasks)
}

fn ensure_search_window(app: &tauri::AppHandle) -> Result<tauri::WebviewWindow, String> {
    let Some(search) = app.get_webview_window("search") else {
        return Err("search window not found".to_string());
    };
    apply_current_capture_protection(&search);
    Ok(search)
}

fn ensure_gamepath_window(app: &tauri::AppHandle) -> Result<tauri::WebviewWindow, String> {
    let Some(gamepath) = app.get_webview_window("gamepath") else {
        return Err("gamepath window not found".to_string());
    };
    apply_current_capture_protection(&gamepath);
    Ok(gamepath)
}

fn ensure_tools_window(app: &tauri::AppHandle) -> Result<tauri::WebviewWindow, String> {
    let Some(tools) = app.get_webview_window("tools") else {
        return Err("tools window not found".to_string());
    };
    apply_current_capture_protection(&tools);
    Ok(tools)
}

fn ensure_standby_window(app: &tauri::AppHandle) -> Result<tauri::WebviewWindow, String> {
    let Some(standby) = app.get_webview_window("standby") else {
        return Err("standby window not found".to_string());
    };
    apply_current_capture_protection(&standby);
    Ok(standby)
}

fn standby_window_position(app: &tauri::AppHandle, width: u32, height: u32) -> (i32, i32) {
    let main = app.get_webview_window("main");
    let monitor = main
        .as_ref()
        .and_then(|window| window.current_monitor().ok().flatten())
        .or_else(|| app.primary_monitor().ok().flatten());

    if let Some(monitor) = monitor {
        let position = monitor.position();
        let size = monitor.size();
        let scale_factor = monitor.scale_factor().max(1.0);
        let mon_right = position.x + i32::try_from(size.width).unwrap_or(1920);
        let mon_top = position.y;
        let mon_height = i32::try_from(size.height).unwrap_or(1080);
        let window_width = ((width.max(1) as f64) * scale_factor).round() as i32;
        let window_height = ((height.max(1) as f64) * scale_factor).round() as i32;
        let x = mon_right - window_width;
        let y = mon_top + ((mon_height - window_height) / 2).max(0);
        return (x, y);
    }

    (1824, 450)
}

fn configure_standby_window(
    app: &tauri::AppHandle,
    expanded: bool,
) -> Result<tauri::WebviewWindow, String> {
    configure_standby_window_mode(app, if expanded { "typein" } else { "collapsed" })
}

fn standby_window_size_for_mode(mode: &str) -> (u32, u32) {
    match mode {
        "detail" => (488_u32, 640_u32),
        "response" => (640_u32, 360_u32),
        "typein" => (620_u32, 320_u32),
        "thinking" => (488_u32, 100_u32),
        _ => (260_u32, 460_u32),
    }
}

fn configure_standby_window_mode(
    app: &tauri::AppHandle,
    mode: &str,
) -> Result<tauri::WebviewWindow, String> {
    let standby = ensure_standby_window(app)?;
    let (width, height) = standby_window_size_for_mode(mode);
    let (x, y) = standby_window_position(app, width, height);
    let _ = standby.set_size(LogicalSize::new(width as f64, height as f64));
    let _ = standby.set_position(PhysicalPosition::new(x, y));
    let _ = standby.set_always_on_top(true);
    let _ = standby.set_background_color(Some(Color(0, 0, 0, 0)));
    let _ = standby.set_shadow(false);
    Ok(standby)
}

fn set_standby_pointer_passthrough_state(
    app: &tauri::AppHandle,
    passthrough: bool,
) -> Result<(), String> {
    let standby = ensure_standby_window(app)?;
    let _ = standby.set_ignore_cursor_events(passthrough);
    let _ = standby.set_focusable(!passthrough);
    Ok(())
}

fn clamp_i32(value: i32, minimum: i32, maximum: i32) -> i32 {
    if maximum < minimum {
        minimum
    } else {
        value.clamp(minimum, maximum)
    }
}

fn companion_child_position(
    app: &tauri::AppHandle,
    child: &tauri::WebviewWindow,
    fallback_size: PhysicalSize<u32>,
    cascade_offset: i32,
) -> (i32, i32) {
    let Some(main) = app.get_webview_window("main") else {
        return (80 + cascade_offset, 80 + cascade_offset);
    };

    let Ok(main_pos) = main.outer_position() else {
        return (80 + cascade_offset, 80 + cascade_offset);
    };
    let main_size = main
        .outer_size()
        .unwrap_or_else(|_| PhysicalSize::new(520, 620));
    let child_size = child.outer_size().unwrap_or(fallback_size);
    let child_width = i32::try_from(child_size.width.max(1)).unwrap_or(i32::MAX / 4);
    let child_height = i32::try_from(child_size.height.max(1)).unwrap_or(i32::MAX / 4);
    let main_width = i32::try_from(main_size.width.max(1)).unwrap_or(520);
    let main_height = i32::try_from(main_size.height.max(1)).unwrap_or(620);
    let gap = 12;

    let monitor = main.current_monitor().ok().flatten();
    let (mon_left, mon_top, mon_right, mon_bottom) = if let Some(monitor) = monitor {
        let position = monitor.position();
        let size = monitor.size();
        (
            position.x,
            position.y,
            position.x + i32::try_from(size.width).unwrap_or(i32::MAX / 4),
            position.y + i32::try_from(size.height).unwrap_or(i32::MAX / 4),
        )
    } else {
        (
            main_pos.x - 240,
            main_pos.y - 240,
            main_pos.x + main_width + child_width + 240,
            main_pos.y + main_height + child_height + 240,
        )
    };

    let right_x = main_pos.x + main_width + gap;
    let left_x = main_pos.x - child_width - gap;
    let below_y = main_pos.y + main_height + gap;
    let above_y = main_pos.y - child_height - gap;
    let aligned_y = main_pos.y + cascade_offset;
    let aligned_x = main_pos.x + cascade_offset;

    let (preferred_x, preferred_y) = if right_x + child_width <= mon_right {
        (right_x, aligned_y)
    } else if left_x >= mon_left {
        (left_x, aligned_y)
    } else if below_y + child_height <= mon_bottom {
        (aligned_x, below_y)
    } else if above_y >= mon_top {
        (aligned_x, above_y)
    } else {
        (
            main_pos.x + (main_width - child_width).min(36).max(-36) + cascade_offset,
            main_pos.y + 36 + cascade_offset,
        )
    };

    let max_x = mon_right - child_width;
    let max_y = mon_bottom - child_height;
    (
        clamp_i32(preferred_x, mon_left, max_x),
        clamp_i32(preferred_y, mon_top, max_y),
    )
}

#[tauri::command]
fn show_tasks_window(app: tauri::AppHandle) -> Result<(), String> {
    park_hidden_companion_windows(&app);
    let tasks = ensure_tasks_window(&app)?;
    apply_current_capture_protection(&tasks);
    let (x, y) = companion_child_position(&app, &tasks, PhysicalSize::new(380, 560), 0);
    let _ = tasks.set_position(PhysicalPosition::new(x, y));
    let _ = tasks.set_always_on_top(true);
    tasks.show().map_err(|err| err.to_string())?;
    force_companion_window_repaint(&tasks, x, y);
    let result = tasks.set_focus().map_err(|err| err.to_string());
    let _ = emit_virtual_cursor_frames_changed(&app);
    result
}

#[tauri::command]
fn show_search_window(app: tauri::AppHandle) -> Result<(), String> {
    park_hidden_companion_windows(&app);
    let search = ensure_search_window(&app)?;
    apply_current_capture_protection(&search);
    let (x, y) = companion_child_position(&app, &search, PhysicalSize::new(960, 720), 24);
    let _ = search.set_position(PhysicalPosition::new(x, y));
    let _ = search.set_always_on_top(true);
    search.show().map_err(|err| err.to_string())?;
    force_companion_window_repaint(&search, x, y);
    let result = search.set_focus().map_err(|err| err.to_string());
    let _ = emit_virtual_cursor_frames_changed(&app);
    result
}

#[tauri::command]
fn show_gamepath_window(app: tauri::AppHandle) -> Result<(), String> {
    park_hidden_companion_windows(&app);
    let gamepath = ensure_gamepath_window(&app)?;
    apply_current_capture_protection(&gamepath);
    let (x, y) = companion_child_position(&app, &gamepath, PhysicalSize::new(520, 620), 48);
    let _ = gamepath.set_position(PhysicalPosition::new(x, y));
    let _ = gamepath.set_always_on_top(true);
    gamepath.show().map_err(|err| err.to_string())?;
    force_companion_window_repaint(&gamepath, x, y);
    let result = gamepath.set_focus().map_err(|err| err.to_string());
    let _ = emit_virtual_cursor_frames_changed(&app);
    result
}

#[tauri::command]
fn show_tools_window(app: tauri::AppHandle) -> Result<(), String> {
    park_hidden_companion_windows(&app);
    let tools = ensure_tools_window(&app)?;
    apply_current_capture_protection(&tools);
    let (x, y) = companion_child_position(&app, &tools, PhysicalSize::new(320, 430), 72);
    let _ = tools.set_position(PhysicalPosition::new(x, y));
    let _ = tools.set_always_on_top(true);
    tools.show().map_err(|err| err.to_string())?;
    force_companion_window_repaint(&tools, x, y);
    let result = tools.set_focus().map_err(|err| err.to_string());
    let _ = emit_virtual_cursor_frames_changed(&app);
    result
}

#[tauri::command]
fn show_standby_window(app: tauri::AppHandle) -> Result<(), String> {
    let standby = configure_standby_window(&app, false)?;
    let _ =
        standby.eval("window.__igpuSetStandbyExpanded && window.__igpuSetStandbyExpanded(false);");
    standby.show().map_err(|err| err.to_string())?;
    let _ = set_standby_pointer_passthrough_state(&app, false);
    let position = standby
        .outer_position()
        .unwrap_or(PhysicalPosition::new(0, 0));
    force_companion_window_repaint(&standby, position.x, position.y);
    let _ = app.emit_to(
        "standby",
        "standby:set-mode",
        serde_json::json!({ "expanded": false }),
    );
    let _ =
        standby.eval("window.__igpuSetStandbyExpanded && window.__igpuSetStandbyExpanded(false);");
    Ok(())
}

fn should_accept_standby_ctrl_g() -> bool {
    let now = Instant::now();
    let guard = STANDBY_CTRL_G_LAST_AT.get_or_init(|| Mutex::new(None));
    let Ok(mut last_at) = guard.lock() else {
        return true;
    };
    if let Some(previous) = last_at.as_ref() {
        if now.duration_since(*previous) < Duration::from_millis(STANDBY_CTRL_G_DEBOUNCE_MS) {
            return false;
        }
    }
    *last_at = Some(now);
    true
}

fn pilot_process_name(value: &str) -> String {
    let trimmed = value.trim();
    let file_name = Path::new(trimmed)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(trimmed)
        .trim();
    if file_name.is_empty() {
        "re9.exe".to_string()
    } else {
        file_name.to_string()
    }
}

fn pilot_status(
    running: bool,
    pid: Option<u32>,
    host: String,
    port: u16,
    process_name: String,
    elapsed_ms: Option<u64>,
    message: String,
) -> NitrogenPilotStatus {
    NitrogenPilotStatus {
        running,
        pid,
        host,
        port,
        process_name,
        elapsed_ms,
        message,
    }
}

fn stopped_pilot_status(message: String) -> NitrogenPilotStatus {
    pilot_status(false, None, String::new(), 0, String::new(), None, message)
}

fn current_nitrogen_pilot_status() -> NitrogenPilotStatus {
    let lock = NITROGEN_PILOT_PROCESS.get_or_init(|| Mutex::new(None));
    let Ok(mut guard) = lock.lock() else {
        return stopped_pilot_status("Pilot status lock unavailable".to_string());
    };

    let Some(process) = guard.as_mut() else {
        return stopped_pilot_status("NitroGen pilot is stopped".to_string());
    };

    match process.child.try_wait() {
        Ok(Some(status)) => {
            let host = process.host.clone();
            let port = process.port;
            let process_name = process.process_name.clone();
            *guard = None;
            pilot_status(
                false,
                None,
                host,
                port,
                process_name,
                None,
                format!("NitroGen pilot exited: {status}"),
            )
        }
        Ok(None) => pilot_status(
            true,
            Some(process.child.id()),
            process.host.clone(),
            process.port,
            process.process_name.clone(),
            Some(
                process
                    .started_at
                    .elapsed()
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64,
            ),
            "NitroGen pilot is running".to_string(),
        ),
        Err(err) => pilot_status(
            false,
            Some(process.child.id()),
            process.host.clone(),
            process.port,
            process.process_name.clone(),
            Some(
                process
                    .started_at
                    .elapsed()
                    .as_millis()
                    .min(u128::from(u64::MAX)) as u64,
            ),
            format!("Pilot status failed: {err}"),
        ),
    }
}

#[tauri::command]
fn get_nitrogen_pilot_status() -> Result<NitrogenPilotStatus, String> {
    Ok(current_nitrogen_pilot_status())
}

#[tauri::command]
fn start_nitrogen_pilot(
    host: String,
    port: u16,
    process_name: String,
) -> Result<NitrogenPilotStatus, String> {
    let host = host.trim();
    if host.is_empty() {
        return Err("NitroGen host is empty".to_string());
    }
    if port == 0 {
        return Err("NitroGen port is invalid".to_string());
    }

    let process_name = pilot_process_name(&process_name);
    if !Path::new(NITROGEN_PYTHON).exists() {
        return Err(format!("Python venv not found: {NITROGEN_PYTHON}"));
    }
    if !Path::new(NITROGEN_PLAY_SCRIPT).exists() {
        return Err(format!(
            "NitroGen play.py not found: {NITROGEN_PLAY_SCRIPT}"
        ));
    }

    let lock = NITROGEN_PILOT_PROCESS.get_or_init(|| Mutex::new(None));
    let mut guard = lock
        .lock()
        .map_err(|_| "Pilot lock unavailable".to_string())?;
    if let Some(existing) = guard.as_mut() {
        if existing
            .child
            .try_wait()
            .map_err(|err| err.to_string())?
            .is_none()
        {
            return Ok(pilot_status(
                true,
                Some(existing.child.id()),
                existing.host.clone(),
                existing.port,
                existing.process_name.clone(),
                Some(
                    existing
                        .started_at
                        .elapsed()
                        .as_millis()
                        .min(u128::from(u64::MAX)) as u64,
                ),
                "NitroGen pilot is already running".to_string(),
            ));
        }
        *guard = None;
    }

    let _ = fs::remove_file(NITROGEN_STOP_FILE);
    fs::create_dir_all(NITROGEN_LOG_DIR).map_err(|err| err.to_string())?;
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{NITROGEN_LOG_DIR}\\pilot_from_companion.out.log"))
        .map_err(|err| err.to_string())?;
    let stderr = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{NITROGEN_LOG_DIR}\\pilot_from_companion.err.log"))
        .map_err(|err| err.to_string())?;

    let child = Command::new(NITROGEN_PYTHON)
        .current_dir(NITROGEN_REPO_DIR)
        .arg(NITROGEN_PLAY_SCRIPT)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--process")
        .arg(&process_name)
        .arg("--low-latency")
        .arg("--capture-width")
        .arg("2560")
        .arg("--capture-height")
        .arg("1440")
        .arg("--env-fps")
        .arg("90")
        .arg("--realtime")
        .arg("--no-record")
        .arg("--no-debug-frames")
        .arg("--no-actions-log")
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|err| format!("Failed to start NitroGen pilot: {err}"))?;

    let pid = child.id();
    *guard = Some(NitrogenPilotProcess {
        child,
        host: host.to_string(),
        port,
        process_name: process_name.clone(),
        started_at: Instant::now(),
    });

    Ok(pilot_status(
        true,
        Some(pid),
        host.to_string(),
        port,
        process_name,
        Some(0),
        "NitroGen pilot started".to_string(),
    ))
}

#[tauri::command]
fn stop_nitrogen_pilot() -> Result<NitrogenPilotStatus, String> {
    fs::write(NITROGEN_STOP_FILE, "stop\n").map_err(|err| err.to_string())?;
    let lock = NITROGEN_PILOT_PROCESS.get_or_init(|| Mutex::new(None));
    let mut guard = lock
        .lock()
        .map_err(|_| "Pilot lock unavailable".to_string())?;
    let Some(process) = guard.as_mut() else {
        return Ok(stopped_pilot_status(
            "NitroGen pilot was not running".to_string(),
        ));
    };

    let host = process.host.clone();
    let port = process.port;
    let process_name = process.process_name.clone();
    let pid = process.child.id();
    let started_at = process.started_at;
    let mut exited_status: Option<String> = None;

    for _ in 0..30 {
        match process.child.try_wait() {
            Ok(Some(status)) => {
                exited_status = Some(status.to_string());
                break;
            }
            Ok(None) => thread::sleep(Duration::from_millis(100)),
            Err(err) => {
                return Ok(pilot_status(
                    true,
                    Some(pid),
                    host,
                    port,
                    process_name,
                    Some(started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64),
                    format!("Stop requested, but status check failed: {err}"),
                ));
            }
        }
    }

    if let Some(status) = exited_status {
        *guard = None;
        return Ok(pilot_status(
            false,
            None,
            host,
            port,
            process_name,
            None,
            format!("NitroGen pilot exited safely: {status}"),
        ));
    }

    Ok(pilot_status(
        true,
        Some(pid),
        host,
        port,
        process_name,
        Some(started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64),
        "Stop requested; waiting for NitroGen to release controls".to_string(),
    ))
}

fn open_standby_typein_window(app: &tauri::AppHandle) -> Result<(), String> {
    let standby = configure_standby_window_mode(app, "typein")?;
    standby.show().map_err(|err| err.to_string())?;
    let _ = set_standby_pointer_passthrough_state(app, false);
    let position = standby
        .outer_position()
        .unwrap_or(PhysicalPosition::new(0, 0));
    force_companion_window_repaint(&standby, position.x, position.y);
    let _ = standby.set_focus();
    let _ = standby.eval("window.focus();");
    let _ = app.emit_to(
        "standby",
        "standby:set-mode",
        serde_json::json!({ "expanded": true, "mode": "typein", "focusInput": true }),
    );
    let _ = standby.eval(
        "window.__igpuSetStandbyMode && window.__igpuSetStandbyMode('typein', { focusInput: true });",
    );
    Ok(())
}

#[tauri::command]
fn set_standby_window_expanded(app: tauri::AppHandle, expanded: bool) -> Result<(), String> {
    let standby = configure_standby_window(&app, expanded)?;
    let script = format!(
        "window.__igpuSetStandbyExpanded && window.__igpuSetStandbyExpanded({});",
        if expanded { "true" } else { "false" }
    );
    let _ = standby.eval(&script);
    standby.show().map_err(|err| err.to_string())?;
    let _ = set_standby_pointer_passthrough_state(&app, false);
    if expanded {
        let _ = standby.set_focus();
    }
    let _ = app.emit_to(
        "standby",
        "standby:set-mode",
        serde_json::json!({ "expanded": expanded }),
    );
    let _ = standby.eval(&script);
    Ok(())
}

#[tauri::command]
fn set_standby_window_mode(app: tauri::AppHandle, mode: String) -> Result<(), String> {
    let normalized = match mode.as_str() {
        "collapsed" => "collapsed",
        "typein" => "typein",
        "thinking" => "thinking",
        "response" => "response",
        "detail" => "detail",
        _ => "typein",
    };
    let expanded = normalized != "collapsed";
    let standby = configure_standby_window_mode(&app, normalized)?;
    standby.show().map_err(|err| err.to_string())?;
    let _ = set_standby_pointer_passthrough_state(&app, false);
    if expanded {
        let _ = standby.set_focus();
    }
    let _ = app.emit_to(
        "standby",
        "standby:set-mode",
        serde_json::json!({ "expanded": expanded, "mode": normalized }),
    );
    Ok(())
}

#[tauri::command]
fn set_standby_pointer_passthrough(app: tauri::AppHandle, passthrough: bool) -> Result<(), String> {
    set_standby_pointer_passthrough_state(&app, passthrough)
}

#[tauri::command]
fn hide_standby_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(standby) = app.get_webview_window("standby") else {
        return Ok(());
    };
    hide_companion_window(&standby)
}

#[tauri::command]
fn collapse_main_to_standby(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.minimize();
    }
    for label in ["tools", "tasks", "search", "gamepath"] {
        if let Some(window) = app.get_webview_window(label) {
            let _ = hide_companion_window(&window);
        }
    }
    show_standby_window(app)
}

#[tauri::command]
fn restore_main_from_standby(app: tauri::AppHandle) -> Result<(), String> {
    let _ = hide_standby_window(app.clone());
    let Some(main) = app.get_webview_window("main") else {
        return Err("main window not found".to_string());
    };
    main.show().map_err(|err| err.to_string())?;
    let _ = main.unminimize();
    let _ = main.set_always_on_top(true);
    let position = main
        .outer_position()
        .unwrap_or(PhysicalPosition::new(80, 80));
    force_companion_window_repaint(&main, position.x, position.y);
    let _ = main.set_focus();
    let _ = show_tools_window(app);
    Ok(())
}

#[tauri::command]
fn hide_tasks_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(tasks) = app.get_webview_window("tasks") else {
        return Ok(());
    };
    let result = hide_companion_window(&tasks);
    let _ = emit_virtual_cursor_frames_changed(&app);
    result
}

#[tauri::command]
fn hide_search_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(search) = app.get_webview_window("search") else {
        return Ok(());
    };
    let result = hide_companion_window(&search);
    let _ = emit_virtual_cursor_frames_changed(&app);
    result
}

#[tauri::command]
fn hide_gamepath_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(gamepath) = app.get_webview_window("gamepath") else {
        return Ok(());
    };
    let result = hide_companion_window(&gamepath);
    let _ = emit_virtual_cursor_frames_changed(&app);
    result
}

#[tauri::command]
fn hide_tools_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(tools) = app.get_webview_window("tools") else {
        return Ok(());
    };
    let result = hide_companion_window(&tools);
    let _ = emit_virtual_cursor_frames_changed(&app);
    result
}

#[tauri::command]
fn toggle_tasks_window(app: tauri::AppHandle) -> Result<(), String> {
    let tasks = ensure_tasks_window(&app)?;
    if tasks.is_visible().unwrap_or(false) {
        let result = hide_companion_window(&tasks);
        let _ = emit_virtual_cursor_frames_changed(&app);
        result
    } else {
        show_tasks_window(app)
    }
}

#[tauri::command]
fn toggle_search_window(app: tauri::AppHandle) -> Result<(), String> {
    let search = ensure_search_window(&app)?;
    if search.is_visible().unwrap_or(false) {
        let result = hide_companion_window(&search);
        let _ = emit_virtual_cursor_frames_changed(&app);
        result
    } else {
        show_search_window(app)
    }
}

#[tauri::command]
fn toggle_gamepath_window(app: tauri::AppHandle) -> Result<(), String> {
    let gamepath = ensure_gamepath_window(&app)?;
    if gamepath.is_visible().unwrap_or(false) {
        let result = hide_companion_window(&gamepath);
        let _ = emit_virtual_cursor_frames_changed(&app);
        result
    } else {
        show_gamepath_window(app)
    }
}

#[tauri::command]
fn toggle_tools_window(app: tauri::AppHandle) -> Result<(), String> {
    let tools = ensure_tools_window(&app)?;
    if tools.is_visible().unwrap_or(false) {
        let result = hide_companion_window(&tools);
        let _ = emit_virtual_cursor_frames_changed(&app);
        result
    } else {
        show_tools_window(app)
    }
}

#[cfg(windows)]
fn cursor_position() -> Option<(i32, i32)> {
    let mut point = POINT { x: 0, y: 0 };
    let ok = unsafe { GetCursorPos(&mut point) };
    if ok == 0 {
        None
    } else {
        Some((point.x, point.y))
    }
}

#[cfg(windows)]
fn begin_polling_window_drag(window: tauri::WebviewWindow) -> Result<(), String> {
    let (start_x, start_y) = cursor_position().ok_or_else(|| "GetCursorPos failed".to_string())?;
    let start_position = window.outer_position().map_err(|err| err.to_string())?;
    let app = window.app_handle().clone();

    thread::spawn(move || {
        loop {
            if !virtual_key_down(0x01) {
                break;
            }
            if let Some((cursor_x, cursor_y)) = cursor_position() {
                let next_x = start_position
                    .x
                    .saturating_add(cursor_x.saturating_sub(start_x));
                let next_y = start_position
                    .y
                    .saturating_add(cursor_y.saturating_sub(start_y));
                let _ = window.set_position(PhysicalPosition::new(next_x, next_y));
            }
            thread::sleep(Duration::from_millis(12));
        }
        let _ = emit_virtual_cursor_frames_changed(&app);
    });

    Ok(())
}

#[tauri::command]
fn begin_gamepath_window_drag(app: tauri::AppHandle) -> Result<(), String> {
    let gamepath = ensure_gamepath_window(&app)?;
    #[cfg(windows)]
    {
        begin_polling_window_drag(gamepath)
    }
    #[cfg(not(windows))]
    {
        gamepath.start_dragging().map_err(|err| err.to_string())
    }
}

#[tauri::command]
fn game_search_browser_back(app: tauri::AppHandle) -> Result<(), String> {
    let Some(webview) = app.get_webview(GAME_SEARCH_BROWSER_LABEL) else {
        return Err("game search browser not found".to_string());
    };
    webview
        .eval("history.back();")
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn game_search_browser_forward(app: tauri::AppHandle) -> Result<(), String> {
    let Some(webview) = app.get_webview(GAME_SEARCH_BROWSER_LABEL) else {
        return Err("game search browser not found".to_string());
    };
    webview
        .eval("history.forward();")
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn game_search_browser_reload(app: tauri::AppHandle) -> Result<(), String> {
    let Some(webview) = app.get_webview(GAME_SEARCH_BROWSER_LABEL) else {
        return Err("game search browser not found".to_string());
    };
    webview.reload().map_err(|err| err.to_string())
}

#[tauri::command]
fn game_search_browser_navigate(app: tauri::AppHandle, url: String) -> Result<(), String> {
    let Some(webview) = app.get_webview(GAME_SEARCH_BROWSER_LABEL) else {
        return Err("game search browser not found".to_string());
    };
    let parsed = tauri::Url::parse(&url).map_err(|err| err.to_string())?;
    match parsed.scheme() {
        "http" | "https" => webview.navigate(parsed).map_err(|err| err.to_string()),
        _ => Err("unsupported browser URL scheme".to_string()),
    }
}

#[tauri::command]
fn set_main_capture_exclusion(app: tauri::AppHandle, excluded: bool) -> Result<(), String> {
    if excluded && capture_protection_boot_reset_active() {
        return set_capture_protection_state(&app, false);
    }
    set_capture_protection_state(&app, excluded)
}

#[tauri::command]
fn set_app_capture_exclusion(app: tauri::AppHandle, excluded: bool) -> Result<(), String> {
    if excluded && capture_protection_boot_reset_active() {
        return set_app_capture_exclusion_state(&app, false);
    }
    set_app_capture_exclusion_state(&app, excluded)
}

#[tauri::command]
fn set_virtual_cursor_global_controls(app: tauri::AppHandle, enabled: bool) -> Result<(), String> {
    set_virtual_cursor_global_controls_state(&app, enabled)
}

#[tauri::command]
fn set_virtual_cursor_text_entry(active: bool) -> Result<(), String> {
    VIRTUAL_CURSOR_TEXT_ENTRY_ACTIVE.store(active, Ordering::Relaxed);
    if active {
        if let Ok(mut state) = virtual_cursor_state().lock() {
            state.moving_window = None;
        }
    }
    Ok(())
}

#[tauri::command]
fn set_virtual_cursor_window_move(
    app: tauri::AppHandle,
    label: String,
    active: bool,
) -> Result<(), String> {
    let safe_label = match label.as_str() {
        "main" => "main",
        "tasks" => "tasks",
        "search" => "search",
        "gamepath" => "gamepath",
        "tools" => "tools",
        _ => return Err("unsupported virtual cursor window".to_string()),
    };

    if active {
        let Some(window) = app.get_webview_window(safe_label) else {
            return Err(format!("{safe_label} window not found"));
        };
        if !window.is_visible().unwrap_or(false) {
            return Err(format!("{safe_label} window is hidden"));
        }
    }

    {
        let mut state = virtual_cursor_state()
            .lock()
            .map_err(|_| "virtual cursor state poisoned".to_string())?;
        if active {
            state.active_window = safe_label.to_string();
            state.moving_window = Some(safe_label.to_string());
            state.initialized = true;
        } else if state.moving_window.as_deref() == Some(safe_label) {
            state.moving_window = None;
        }
    }

    emit_virtual_cursor_render(&app)
}

#[tauri::command]
fn set_virtual_cursor_active_window(app: tauri::AppHandle, label: String) -> Result<(), String> {
    let safe_label = match label.as_str() {
        "main" => "main",
        "tasks" => "tasks",
        "search" => "search",
        "gamepath" => "gamepath",
        "tools" => "tools",
        _ => return Err("unsupported virtual cursor window".to_string()),
    };
    let frames = collect_virtual_cursor_window_frames(&app)?;
    let Some(frame) = frames.iter().find(|frame| frame.label == safe_label) else {
        return Ok(());
    };

    {
        let mut state = virtual_cursor_state()
            .lock()
            .map_err(|_| "virtual cursor state poisoned".to_string())?;
        state.active_window = safe_label.to_string();
        if state.moving_window.as_deref() != Some(safe_label) {
            state.moving_window = None;
        }
        state.initialized = true;
        let (local_width, local_height) = frame_local_size(frame);
        state.local_x = local_width / 2.0;
        state.local_y = local_height / 2.0;
        let scale = if frame.scale_factor > 0.0 {
            frame.scale_factor
        } else {
            1.0
        };
        state.screen_x = frame.x as f64 + state.local_x * scale;
        state.screen_y = frame.y as f64 + state.local_y * scale;
        clamp_virtual_cursor_to_world(&mut state, &frames);
    }

    emit_virtual_cursor_render(&app)
}

#[tauri::command]
fn set_virtual_cursor_window_position(
    app: tauri::AppHandle,
    label: String,
    x: f64,
    y: f64,
) -> Result<(), String> {
    let safe_label = match label.as_str() {
        "main" => "main",
        "tasks" => "tasks",
        "search" => "search",
        "gamepath" => "gamepath",
        "tools" => "tools",
        _ => return Err("unsupported virtual cursor window".to_string()),
    };
    let frames = collect_virtual_cursor_window_frames(&app)?;
    let Some(frame) = frames.iter().find(|frame| frame.label == safe_label) else {
        return Ok(());
    };

    {
        let mut state = virtual_cursor_state()
            .lock()
            .map_err(|_| "virtual cursor state poisoned".to_string())?;
        let (width, height) = frame_local_size(frame);
        let local_x = x.clamp(
            VIRTUAL_CURSOR_SCREEN_MARGIN,
            width - VIRTUAL_CURSOR_SCREEN_MARGIN,
        );
        let local_y = y.clamp(
            VIRTUAL_CURSOR_SCREEN_MARGIN,
            height - VIRTUAL_CURSOR_SCREEN_MARGIN,
        );
        let scale = if frame.scale_factor > 0.0 {
            frame.scale_factor
        } else {
            1.0
        };
        state.active_window = safe_label.to_string();
        if state.moving_window.as_deref() != Some(safe_label) {
            state.moving_window = None;
        }
        state.initialized = true;
        state.local_x = local_x;
        state.local_y = local_y;
        state.screen_x = frame.x as f64 + local_x * scale;
        state.screen_y = frame.y as f64 + local_y * scale;
        clamp_virtual_cursor_to_world(&mut state, &frames);
    }

    emit_virtual_cursor_render(&app)
}

#[tauri::command]
fn focus_virtual_cursor_text_entry(
    app: tauri::AppHandle,
    label: String,
    x: f64,
    y: f64,
) -> Result<(), String> {
    let safe_label = match label.as_str() {
        "main" => "main",
        "tasks" => "tasks",
        "search" => "search",
        "gamepath" => "gamepath",
        "tools" => "tools",
        _ => return Err("unsupported companion window".to_string()),
    };

    let Some(window) = app.get_webview_window(safe_label) else {
        return Err(format!("{safe_label} window not found"));
    };

    VIRTUAL_CURSOR_TEXT_ENTRY_ACTIVE.store(true, Ordering::Relaxed);
    if safe_label != "main" {
        apply_current_capture_protection(&window);
        let _ = window.set_always_on_top(true);
    }
    window.show().map_err(|err| err.to_string())?;
    let _ = window.set_focus();
    force_companion_window_focus(&window);

    let position = window.outer_position().map_err(|err| err.to_string())?;
    let scale = window.scale_factor().unwrap_or(1.0).max(0.1);
    let screen_x = (position.x as f64 + x * scale).round() as i32;
    let screen_y = (position.y as f64 + y * scale).round() as i32;
    click_screen_point(screen_x, screen_y)?;

    let _ = window.set_focus();
    force_companion_window_focus(&window);
    let _ = window.eval("window.focus();");
    Ok(())
}

#[tauri::command]
fn click_virtual_cursor_position(
    app: tauri::AppHandle,
    label: String,
    x: f64,
    y: f64,
) -> Result<(), String> {
    let safe_label = match label.as_str() {
        "main" => "main",
        "tasks" => "tasks",
        "search" => "search",
        "gamepath" => "gamepath",
        "tools" => "tools",
        _ => return Err("unsupported companion window".to_string()),
    };

    let Some(window) = app.get_webview_window(safe_label) else {
        return Err(format!("{safe_label} window not found"));
    };

    if safe_label != "main" {
        apply_current_capture_protection(&window);
        let _ = window.set_always_on_top(true);
    }
    window.show().map_err(|err| err.to_string())?;

    let position = window.outer_position().map_err(|err| err.to_string())?;
    let scale = window.scale_factor().unwrap_or(1.0).max(0.1);
    let screen_x = (position.x as f64 + x * scale).round() as i32;
    let screen_y = (position.y as f64 + y * scale).round() as i32;
    click_screen_point(screen_x, screen_y)
}

#[tauri::command]
fn focus_companion_window(app: tauri::AppHandle, label: String) -> Result<(), String> {
    let safe_label = match label.as_str() {
        "main" => "main",
        "tasks" => "tasks",
        "search" => "search",
        "gamepath" => "gamepath",
        "tools" => "tools",
        _ => return Err("unsupported companion window".to_string()),
    };

    let Some(window) = app.get_webview_window(safe_label) else {
        return Err(format!("{safe_label} window not found"));
    };

    if safe_label != "main" {
        apply_current_capture_protection(&window);
        let _ = window.set_always_on_top(true);
    }

    window.show().map_err(|err| err.to_string())?;
    let result = window.set_focus().map_err(|err| err.to_string());
    force_companion_window_focus(&window);
    result
}

fn collect_virtual_cursor_window_frames(
    app: &tauri::AppHandle,
) -> Result<Vec<VirtualCursorWindowFrame>, String> {
    let mut frames = Vec::new();
    for label in ["main", "tasks", "search", "gamepath", "tools"] {
        let Some(window) = app.get_webview_window(label) else {
            continue;
        };
        let visible = window.is_visible().unwrap_or(false);
        if !visible {
            continue;
        }
        let position = window.outer_position().map_err(|err| err.to_string())?;
        let size = window.outer_size().map_err(|err| err.to_string())?;
        if position.x < -10000 || position.y < -10000 || size.width == 0 || size.height == 0 {
            continue;
        }
        frames.push(VirtualCursorWindowFrame {
            label,
            x: position.x,
            y: position.y,
            width: size.width,
            height: size.height,
            scale_factor: window.scale_factor().unwrap_or(1.0),
            visible,
        });
    }
    Ok(frames)
}

fn pick_virtual_cursor_transfer_target(
    frames: &[VirtualCursorWindowFrame],
    source: &VirtualCursorWindowFrame,
    edge: &str,
    screen_x: f64,
    screen_y: f64,
) -> Option<VirtualCursorWindowFrame> {
    let source_left = source.x as f64;
    let source_top = source.y as f64;
    let source_right = source_left + source.width as f64;
    let source_bottom = source_top + source.height as f64;
    let source_center_x = source_left + source.width as f64 / 2.0;
    let source_center_y = source_top + source.height as f64 / 2.0;

    let candidates = frames.iter().filter(|frame| frame.label != source.label);
    let containing = candidates.clone().find(|frame| {
        let left = frame.x as f64;
        let top = frame.y as f64;
        screen_x >= left
            && screen_x <= left + frame.width as f64
            && screen_y >= top
            && screen_y <= top + frame.height as f64
    });
    if let Some(frame) = containing {
        return Some(frame.clone());
    }

    frames
        .iter()
        .filter(|frame| frame.label != source.label)
        .map(|frame| {
            let left = frame.x as f64;
            let top = frame.y as f64;
            let right = left + frame.width as f64;
            let bottom = top + frame.height as f64;
            let center_x = left + frame.width as f64 / 2.0;
            let center_y = top + frame.height as f64 / 2.0;

            let (wrong_direction, main_axis_gap, cross_axis_gap) = match edge {
                "left" => (
                    center_x > source_center_x + 24.0,
                    (source_left - right).max(0.0),
                    if screen_y < top {
                        top - screen_y
                    } else if screen_y > bottom {
                        screen_y - bottom
                    } else {
                        0.0
                    },
                ),
                "right" => (
                    center_x < source_center_x - 24.0,
                    (left - source_right).max(0.0),
                    if screen_y < top {
                        top - screen_y
                    } else if screen_y > bottom {
                        screen_y - bottom
                    } else {
                        0.0
                    },
                ),
                "up" => (
                    center_y > source_center_y + 24.0,
                    (source_top - bottom).max(0.0),
                    if screen_x < left {
                        left - screen_x
                    } else if screen_x > right {
                        screen_x - right
                    } else {
                        0.0
                    },
                ),
                "down" => (
                    center_y < source_center_y - 24.0,
                    (top - source_bottom).max(0.0),
                    if screen_x < left {
                        left - screen_x
                    } else if screen_x > right {
                        screen_x - right
                    } else {
                        0.0
                    },
                ),
                _ => (true, 0.0, 0.0),
            };

            let center_distance = ((center_x - source_center_x).powi(2)
                + (center_y - source_center_y).powi(2))
            .sqrt();
            let score = main_axis_gap * 2.0
                + cross_axis_gap
                + frame_distance_to_point(frame, screen_x, screen_y) * 0.25
                + center_distance * 0.05
                + if wrong_direction { 20_000.0 } else { 0.0 };
            (score, frame.clone())
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, frame)| frame)
}

fn emit_virtual_cursor_frames_changed(app: &tauri::AppHandle) -> Result<(), String> {
    let frames = collect_virtual_cursor_window_frames(app)?;
    if let Ok(mut state) = virtual_cursor_state().lock() {
        anchor_virtual_cursor_to_active_frame(&mut state, &frames);
    }
    let result = app
        .emit(
            "virtual-cursor-frames-changed",
            serde_json::json!({ "frames": frames }),
        )
        .map_err(|err| err.to_string());
    let _ = emit_virtual_cursor_render(app);
    result
}

#[tauri::command]
fn virtual_cursor_window_frames(
    app: tauri::AppHandle,
) -> Result<Vec<VirtualCursorWindowFrame>, String> {
    collect_virtual_cursor_window_frames(&app)
}

#[tauri::command]
fn virtual_cursor_transfer_window(
    app: tauri::AppHandle,
    label: String,
    x: f64,
    y: f64,
    source: String,
    id: String,
) -> Result<(), String> {
    match label.as_str() {
        "main" | "tasks" | "search" | "gamepath" | "tools" => {}
        _ => return Err("unsupported virtual cursor transfer target".to_string()),
    }

    let payload = serde_json::json!({
        "window": label,
        "x": x,
        "y": y,
        "source": source,
        "id": id,
    });
    let _ = app.emit("virtual-cursor-transfer", payload.clone());
    app.emit_to(label.as_str(), "virtual-cursor-transfer", payload)
        .map_err(|err| err.to_string())
}

#[tauri::command]
fn virtual_cursor_transfer_at_edge(
    app: tauri::AppHandle,
    source: String,
    edge: String,
    x_ratio: f64,
    y_ratio: f64,
    id: String,
) -> Result<Option<String>, String> {
    match source.as_str() {
        "main" | "tasks" | "search" | "gamepath" | "tools" => {}
        _ => return Err("unsupported virtual cursor source".to_string()),
    }
    match edge.as_str() {
        "left" | "right" | "up" | "down" => {}
        _ => return Err("unsupported virtual cursor edge".to_string()),
    }

    let frames = collect_virtual_cursor_window_frames(&app)?;
    let Some(source_frame) = frames.iter().find(|frame| frame.label == source.as_str()) else {
        return Ok(None);
    };
    if frames.len() <= 1 {
        return Ok(None);
    }

    let source_left = source_frame.x as f64;
    let source_top = source_frame.y as f64;
    let source_width = source_frame.width as f64;
    let source_height = source_frame.height as f64;
    let cursor_screen_x = match edge.as_str() {
        "left" => source_left - 1.0,
        "right" => source_left + source_width + 1.0,
        _ => source_left + x_ratio.clamp(0.0, 1.0) * source_width,
    };
    let cursor_screen_y = match edge.as_str() {
        "up" => source_top - 1.0,
        "down" => source_top + source_height + 1.0,
        _ => source_top + y_ratio.clamp(0.0, 1.0) * source_height,
    };

    let Some(target) = pick_virtual_cursor_transfer_target(
        &frames,
        source_frame,
        edge.as_str(),
        cursor_screen_x,
        cursor_screen_y,
    ) else {
        return Ok(None);
    };

    let target_scale = if target.scale_factor > 0.0 {
        target.scale_factor
    } else {
        1.0
    };
    let target_width = (target.width as f64 / target_scale).max(24.0);
    let target_height = (target.height as f64 / target_scale).max(24.0);
    let local_x =
        ((cursor_screen_x - target.x as f64) / target_scale).clamp(12.0, target_width - 12.0);
    let local_y =
        ((cursor_screen_y - target.y as f64) / target_scale).clamp(12.0, target_height - 12.0);
    let target_label = target.label.to_string();

    let active_payload = serde_json::json!({
        "window": target_label.clone(),
        "source": source.clone(),
    });
    let _ = app.emit("virtual-cursor-active-window", active_payload.clone());
    let _ = app.emit_to(target.label, "virtual-cursor-active-window", active_payload);

    let transfer_payload = serde_json::json!({
        "window": target_label.clone(),
        "x": local_x,
        "y": local_y,
        "source": source.clone(),
        "id": id,
    });
    let _ = app.emit("virtual-cursor-transfer", transfer_payload.clone());
    app.emit_to(target.label, "virtual-cursor-transfer", transfer_payload)
        .map_err(|err| err.to_string())?;

    Ok(Some(target_label))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .on_window_event(|window, event| match event {
            WindowEvent::Moved(_) | WindowEvent::Resized(_) => {
                if matches!(
                    window.label(),
                    "main" | "tasks" | "search" | "gamepath" | "tools"
                ) {
                    let _ = emit_virtual_cursor_frames_changed(window.app_handle());
                }
            }
            _ => {}
        })
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        if VIRTUAL_CURSOR_GLOBAL_CONTROLS_ENABLED.load(Ordering::Relaxed) {
                            if let Some(payload) = virtual_cursor_control_payload(shortcut.key) {
                                match payload.get("type").and_then(|value| value.as_str()) {
                                    Some("move") => {
                                        let dx = payload
                                            .get("dx")
                                            .and_then(|value| value.as_i64())
                                            .unwrap_or(0)
                                            as i32;
                                        let dy = payload
                                            .get("dy")
                                            .and_then(|value| value.as_i64())
                                            .unwrap_or(0)
                                            as i32;
                                        move_virtual_cursor(app, dx, dy);
                                    }
                                    Some("target_next") => {
                                        emit_virtual_cursor_action(app, "target_next");
                                    }
                                    Some("activate") => {
                                        emit_virtual_cursor_action(app, "activate");
                                    }
                                    Some("disable") => {
                                        let _ = set_virtual_cursor_enabled(app, false);
                                    }
                                    _ => {}
                                }
                                return;
                            }
                        }
                        if shortcut.matches(Modifiers::CONTROL, Code::KeyG)
                            || shortcut.matches(Modifiers::CONTROL | Modifiers::ALT, Code::KeyG)
                        {
                            if should_accept_standby_ctrl_g() {
                                let source = if shortcut
                                    .matches(Modifiers::CONTROL | Modifiers::ALT, Code::KeyG)
                                {
                                    "ctrl_alt_g"
                                } else {
                                    "ctrl_g"
                                };
                                emit_standby_hotkey(app, source);
                            }
                            return;
                        }
                        match shortcut.key {
                            Code::F4 => {
                                toggle_capture_protection_state(app.clone());
                            }
                            Code::F5 => {
                                let _ = show_search_window(app.clone());
                            }
                            Code::F6 => {
                                let _ = show_tasks_window(app.clone());
                            }
                            Code::F7 => {
                                let _ = app.emit("task-hotkey", ());
                            }
                            Code::F8 => {
                                let _ = app.emit("voice-hotkey-start", ());
                            }
                            Code::F9 => {
                                let _ = app.emit("capture-hotkey", ());
                            }
                            Code::F10 => {
                                let _ = app.emit("clear-hud-hotkey", ());
                            }
                            Code::F11 => {
                                let enabled =
                                    !VIRTUAL_CURSOR_GLOBAL_CONTROLS_ENABLED.load(Ordering::Relaxed);
                                let _ = set_virtual_cursor_enabled(app, enabled);
                            }
                            Code::ScrollLock => {
                                if should_accept_standby_ctrl_g() {
                                    emit_standby_hotkey(app, "scroll_lock");
                                }
                            }
                            Code::Pause => {
                                if should_accept_standby_ctrl_g() {
                                    emit_standby_hotkey(app, "pause");
                                }
                            }
                            _ => {}
                        }
                    } else if event.state() == ShortcutState::Released && shortcut.key == Code::F8 {
                        let _ = app.emit("voice-hotkey-stop", ());
                    }
                })
                .build(),
        )
        .setup(|app| {
            dismiss_input_experience_windows();
            let _ =
                CAPTURE_PROTECTION_BOOT_RESET_UNTIL.set(Instant::now() + Duration::from_secs(8));
            clear_capture_protection_on_all_windows(app.handle());
            if let Some(main) = app.get_webview_window("main") {
                clear_window_capture_protection(&main);
                let _ = main.eval("localStorage.setItem('protect-mode', 'off');");
                let _ = main.set_size(LogicalSize::new(720.0, 760.0));
                let _ = main.set_position(PhysicalPosition::new(80_i32, 80_i32));
                let _ = main.set_always_on_top(true);
                let _ = main.set_background_color(Some(Color(0, 0, 0, 0)));
                let _ = main.show();
                force_companion_window_repaint(&main, 80, 80);
                let _ = main.set_focus();
            }
            start_virtual_cursor_keyboard_poll(app.handle().clone());
            start_virtual_cursor_gamepad_poll(app.handle().clone());

            // 註冊全域快捷鍵
            for key in [
                Code::F4,
                Code::F5,
                Code::F6,
                Code::F7,
                Code::F8,
                Code::F9,
                Code::F10,
                Code::ScrollLock,
                Code::Pause,
            ] {
                let shortcut = Shortcut::new(None, key);
                if let Err(err) = app.global_shortcut().register(shortcut) {
                    eprintln!("global shortcut {:?} registration failed: {}", key, err);
                }
            }
            if let Err(err) = app
                .global_shortcut()
                .register(Shortcut::new(Some(Modifiers::CONTROL), Code::KeyG))
            {
                eprintln!("global shortcut Ctrl+G registration failed: {}", err);
            }
            if let Err(err) = app.global_shortcut().register(Shortcut::new(
                Some(Modifiers::CONTROL | Modifiers::ALT),
                Code::KeyG,
            )) {
                eprintln!("global shortcut Ctrl+Alt+G registration failed: {}", err);
            }

            if let Some(hud) = app.get_webview_window("hud") {
                clear_window_capture_protection(&hud);
                let _ = hud.set_ignore_cursor_events(true);
                let _ = hud.set_skip_taskbar(true);
                let _ = hud.set_focusable(false);
                let _ = hide_companion_window(&hud);
            }
            if let Some(tasks) = app.get_webview_window("tasks") {
                clear_window_capture_protection(&tasks);
                let _ = tasks.set_always_on_top(true);
                let _ = hide_companion_window(&tasks);
            }
            if let Some(search) = app.get_webview_window("search") {
                clear_window_capture_protection(&search);
                let _ = search.set_always_on_top(true);
                let _ = hide_companion_window(&search);
            }
            if let Some(gamepath) = app.get_webview_window("gamepath") {
                clear_window_capture_protection(&gamepath);
                let _ = gamepath.set_always_on_top(true);
                let _ = hide_companion_window(&gamepath);
            }
            if let Some(tools) = app.get_webview_window("tools") {
                clear_window_capture_protection(&tools);
                let _ = tools.set_always_on_top(true);
                let _ = hide_companion_window(&tools);
            }
            if let Some(standby) = app.get_webview_window("standby") {
                clear_window_capture_protection(&standby);
                let _ = standby.set_always_on_top(true);
                let _ = standby.set_ignore_cursor_events(true);
                let _ = standby.set_focusable(false);
                let _ = hide_companion_window(&standby);
            }
            let _ = show_tools_window(app.handle().clone());

            // 在 Windows 上設定視窗截圖排除
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            configure_hud_window,
            show_hud_window,
            show_hud_overlay,
            clear_hud_overlay,
            hide_hud_window,
            show_tasks_window,
            show_search_window,
            show_gamepath_window,
            show_tools_window,
            show_standby_window,
            hide_tasks_window,
            hide_search_window,
            hide_gamepath_window,
            hide_tools_window,
            hide_standby_window,
            toggle_tasks_window,
            toggle_search_window,
            toggle_gamepath_window,
            toggle_tools_window,
            set_standby_window_expanded,
            set_standby_window_mode,
            set_standby_pointer_passthrough,
            collapse_main_to_standby,
            restore_main_from_standby,
            begin_gamepath_window_drag,
            game_search_browser_back,
            game_search_browser_forward,
            game_search_browser_reload,
            game_search_browser_navigate,
            set_main_capture_exclusion,
            set_app_capture_exclusion,
            set_virtual_cursor_global_controls,
            set_virtual_cursor_text_entry,
            set_virtual_cursor_window_move,
            set_virtual_cursor_active_window,
            set_virtual_cursor_window_position,
            focus_virtual_cursor_text_entry,
            click_virtual_cursor_position,
            focus_companion_window,
            virtual_cursor_window_frames,
            virtual_cursor_transfer_window,
            virtual_cursor_transfer_at_edge,
            get_nitrogen_pilot_status,
            start_nitrogen_pilot,
            stop_nitrogen_pilot
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
