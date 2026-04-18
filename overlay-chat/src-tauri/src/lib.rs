// tauri-plugin-global-shortcut 透過 JS API 操作，Rust 端只需初始化即可

// WDA_EXCLUDEFROMCAPTURE = 0x11，告訴 Windows 截圖時忽略這個視窗
#[cfg(windows)]
const WDA_EXCLUDEFROMCAPTURE: u32 = 0x00000011;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
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
