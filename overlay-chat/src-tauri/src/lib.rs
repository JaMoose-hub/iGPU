use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};
use tauri::{Emitter, LogicalPosition, LogicalSize, Manager};

// WDA_EXCLUDEFROMCAPTURE = 0x11，告訴 Windows 截圖時忽略這個視窗
#[cfg(windows)]
const WDA_EXCLUDEFROMCAPTURE: u32 = 0x00000011;

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

    #[cfg(windows)]
    {
        let top_hwnd = find_hud_hwnd();
        if !top_hwnd.is_null() {
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::SetWindowDisplayAffinity(
                    top_hwnd,
                    WDA_EXCLUDEFROMCAPTURE,
                );
            }
        }
    }

    Ok(())
}

#[tauri::command]
fn show_hud_window(app: tauri::AppHandle, width: f64, height: f64) -> Result<(), String> {
    let Some(hud) = app.get_webview_window("hud") else {
        return Err("hud window not found".to_string());
    };

    let _ = hud.set_skip_taskbar(true);
    let _ = hud.set_ignore_cursor_events(false);
    let _ = hud.set_focusable(true);
    let _ = hud.set_always_on_top(true);
    let _ = hud.set_position(LogicalPosition::new(0.0, 0.0));
    let _ = hud.set_size(LogicalSize::new(width.max(1.0), height.max(1.0)));
    #[cfg(windows)]
    {
        if let Ok(hwnd) = hud.hwnd() {
            unsafe {
                windows_sys::Win32::UI::WindowsAndMessaging::SetWindowDisplayAffinity(
                    hwnd.0 as _,
                    WDA_EXCLUDEFROMCAPTURE,
                );
            }
        }
    }

    hud.show().map_err(|err| err.to_string())?;
    #[cfg(windows)]
    {
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
    let _ = hud.set_always_on_top(true);
    let _ = hud.set_focusable(false);
    let _ = hud.set_ignore_cursor_events(true);
    Ok(())
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
            #[cfg(windows)]
            {
                for label in ["main", "hud"] {
                    if let Some(window) = app.get_webview_window(label) {
                        if let Ok(hwnd) = window.hwnd() {
                            unsafe {
                                windows_sys::Win32::UI::WindowsAndMessaging::SetWindowDisplayAffinity(
                                    hwnd.0 as _,
                                    WDA_EXCLUDEFROMCAPTURE,
                                );
                            }
                        }
                    }
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            configure_hud_window,
            show_hud_window,
            hide_hud_window
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
