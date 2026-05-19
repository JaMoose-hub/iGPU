use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{Emitter, Manager, PhysicalPosition, PhysicalSize};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};

const GAME_SEARCH_BROWSER_LABEL: &str = "game-search-results";
static CAPTURE_PROTECTION_ENABLED: AtomicBool = AtomicBool::new(false);

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
    for label in ["main", "hud", "tasks", "search"] {
        if let Some(window) = app.get_webview_window(label) {
            set_window_display_excluded(&window, excluded);
        }
    }
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

fn toggle_capture_protection_state(app: tauri::AppHandle) {
    let enabled = !CAPTURE_PROTECTION_ENABLED.load(Ordering::Relaxed);
    let _ = set_capture_protection_state(&app, enabled);
}

fn park_window_offscreen(window: &tauri::WebviewWindow) {
    let _ = window.set_position(PhysicalPosition::new(-32000, -32000));
}

fn park_window_if_hidden(window: &tauri::WebviewWindow) {
    if !window.is_visible().unwrap_or(false) {
        park_window_offscreen(window);
    }
}

fn park_hidden_companion_windows(app: &tauri::AppHandle) {
    for label in ["hud", "tasks", "search"] {
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

#[tauri::command]
fn show_tasks_window(app: tauri::AppHandle) -> Result<(), String> {
    park_hidden_companion_windows(&app);
    let tasks = ensure_tasks_window(&app)?;
    apply_current_capture_protection(&tasks);
    let x = 80;
    let y = 80;
    let _ = tasks.set_position(PhysicalPosition::new(x, y));
    let _ = tasks.set_always_on_top(true);
    tasks.show().map_err(|err| err.to_string())?;
    force_companion_window_repaint(&tasks, x, y);
    tasks.set_focus().map_err(|err| err.to_string())
}

#[tauri::command]
fn show_search_window(app: tauri::AppHandle) -> Result<(), String> {
    park_hidden_companion_windows(&app);
    let search = ensure_search_window(&app)?;
    apply_current_capture_protection(&search);
    let x = 100;
    let y = 100;
    let _ = search.set_position(PhysicalPosition::new(x, y));
    let _ = search.set_always_on_top(true);
    search.show().map_err(|err| err.to_string())?;
    force_companion_window_repaint(&search, x, y);
    search.set_focus().map_err(|err| err.to_string())
}

#[tauri::command]
fn hide_tasks_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(tasks) = app.get_webview_window("tasks") else {
        return Ok(());
    };
    hide_companion_window(&tasks)
}

#[tauri::command]
fn hide_search_window(app: tauri::AppHandle) -> Result<(), String> {
    let Some(search) = app.get_webview_window("search") else {
        return Ok(());
    };
    hide_companion_window(&search)
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
    set_capture_protection_state(&app, excluded)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
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

            // 註冊全域快捷鍵
            for key in [
                Code::F4,
                Code::F5,
                Code::F6,
                Code::F7,
                Code::F8,
                Code::F9,
                Code::F10,
            ] {
                let shortcut = Shortcut::new(None, key);
                let _ = app.global_shortcut().register(shortcut);
            }

            if let Some(hud) = app.get_webview_window("hud") {
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
            hide_tasks_window,
            hide_search_window,
            game_search_browser_back,
            game_search_browser_forward,
            game_search_browser_reload,
            game_search_browser_navigate,
            set_main_capture_exclusion
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
