V4 OCR 英文語料路徑修正 - 2026-10-27

你的畫面顯示無法載入 eng.traineddata。原自測失敗，所以 && 後面的車控程式沒有啟動，port 5000 沒有網頁服務。
本次也修正原套件檢查不足：不能只列出 eng，現在會真正啟動一次 OCR 確認英文語料能載入。

修正包只含 OCR 介面、安裝修復腳本與啟動修復腳本，不含模型；車控行為不變。
會尋找實際存在的 eng.traineddata，包括 Ubuntu 常見的 /usr/share/tesseract-ocr/4.00/tessdata。
用 --tessdata-dir 明確指定通過載入測試的位置，不受原本錯誤的 TESSDATA_PREFIX 影響。
如果語料缺少／損壞，安裝腳本會重新安裝 tesseract-ocr 與 tesseract-ocr-eng，可能需要網路和 sudo 密碼。
不修改使用者的 .bashrc 或永久環境變數。OCR 無法就緒時不啟動車控。

先停止舊車控（若有運行）：控制頁 STOP，再終端機 Ctrl+C。
電腦 PowerShell 傳送此 ZIP，目標為 Jetson 的 /home/tsai/car/。
注意：新照片中的瀏覽器使用 172.20.10.4，請以 Jetson hostname -I 顯示的 Wi-Fi IP 為準。

Jetson 終端機：
cd ~/car
unzip -o jetson_v4_ocr_data_fix_20261027.zip
bash repair_and_run_v4_ocr.sh

必須已經在此目錄解開原本的 V4 OCR 完整包，內有 run_v4_sign_ocr.sh、check_traffic_ocr.py 與 V4 模型。
修復後會顯示實際載入的語料位置，通過 STOP/P 自測後才開車控網頁。
等網頁程式啟動，再用 hostname -I 顯示的正確 IP 加 :5000 開啟控制頁。
啟動時仍為 MANUAL/STOP，須自行按 AUTO 才行駛。

本機 58 項軟體測試通過；Jetson 實際環境仍以修復腳本的載入測試結果為準。
官方路徑說明：https://tesseract-ocr.github.io/tessdoc/FAQ

APT 圖示索引 404 修正：先使用現有套件索引安裝 OCR；必要時僅在該次更新略過 DEP-11 圖示／軟體中心 metadata。保留套件驗證，不改套件來源、不升級 Ubuntu。
