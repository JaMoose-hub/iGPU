use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};
use tauri::Emitter;

// WDA_EXCLUDEFROMCAPTURE = 0x11，告訴 Windows 截圖時忽略這個視窗
#[cfg(windows)]
const WDA_EXCLUDEFROMCAPTURE: u32 = 0x00000011;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        let _ = app.emit("voice-hotkey-start", ());
                    } else if event.state() == ShortcutState::Released {
                        let _ = app.emit("voice-hotkey-stop", ());
                    }
                })
                .build(),
        )
        .setup(|app| {
            // 註冊 F8 快捷鍵
            let f8 = Shortcut::new(None, Code::F8);
            let _ = app.global_shortcut().register(f8);

            // 在 Windows 上設定視窗截圖排除
            #[cfg(windows)]
            {
                use tauri::Manager;
                if let Some(window) = app.get_webview_window("main") {
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
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
