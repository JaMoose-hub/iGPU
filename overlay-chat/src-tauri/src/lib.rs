use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};
use tauri::{Emitter, Manager, PhysicalPosition, PhysicalSize};
use std::time::Duration;

// WDA_EXCLUDEFROMCAPTURE = 0x11，告訴 Windows 截圖時忽略這個視窗
#[cfg(windows)]
const WDA_EXCLUDEFROMCAPTURE: u32 = 0x00000011;
#[cfg(windows)]
const WDA_NONE: u32 = 0x00000000;

#[cfg(windows)]
fn find_hud_hwnd() -> windows_sys::Win32::Foundation::HWND {
    let title: Vec<u16> = "game-guidance-hud\0".encode_utf16().collect();
    unsafe {
        windows_sys::Win32::UI::WindowsAndMessaging::FindWindowW(
            std::ptr::null(),
            title.as_ptr(),
        )
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
    std::thread::sleep(Duration::from_millis(80));

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

    hud.hide().map_err(|err| err.to_string())
}

#[tauri::command]
fn set_main_capture_exclusion(app: tauri::AppHandle, excluded: bool) -> Result<(), String> {
    if app.get_webview_window("main").is_none() {
        return Err("main window not found".to_string());
    }

    #[cfg(windows)]
    {
        let affinity = if excluded {
            WDA_EXCLUDEFROMCAPTURE
        } else {
            WDA_NONE
        };
        for label in ["main", "hud"] {
            let Some(window) = app.get_webview_window(label) else {
                continue;
            };
            let hwnd = match window.hwnd() {
                Ok(hwnd) => hwnd,
                Err(err) if label == "main" => return Err(err.to_string()),
                Err(_) => continue,
            };
            unsafe {
                let ok = windows_sys::Win32::UI::WindowsAndMessaging::SetWindowDisplayAffinity(
                    hwnd.0 as _,
                    affinity,
                );
                if ok == 0 && label == "main" {
                    return Err("SetWindowDisplayAffinity failed for main window".to_string());
                }
            }
        }
    }

    #[cfg(not(windows))]
    {
        let _ = excluded;
    }

    Ok(())
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
            // 註冊全域快捷鍵
            for key in [Code::F8, Code::F9, Code::F10] {
                let shortcut = Shortcut::new(None, key);
                let _ = app.global_shortcut().register(shortcut);
            }

            if let Some(hud) = app.get_webview_window("hud") {
                let _ = hud.set_ignore_cursor_events(true);
                let _ = hud.set_skip_taskbar(true);
                let _ = hud.set_focusable(false);
                let _ = hud.hide();
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
            set_main_capture_exclusion
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
