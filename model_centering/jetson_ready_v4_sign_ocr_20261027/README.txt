V4 + STOP / P OCR TEST - 2026-10-27

【功能】
沿用原本 V4 模型與出彎收方向調整，沒有重新訓練模型，也沒有讀入被拒絕的標註。
AUTO 模式下：
1. 紅色八邊形且 OCR 連續兩次讀到 STOP：原地停住，保持停止，按 AUTO 才繼續。
2. 藍色牌、白色 P 形狀通過且 OCR 連續兩次讀到 P：逐步靠向右黑線。
3. 以車身／前視鏡頭中心對準右黑線為停車目標，停妥後保持停止。
4. 停妥按 AUTO：起步、回到車道中央，確認回正後再交回 V4 跟隨。
5. STOP 辨識、手動 STOP、影像過期、OCR 故障、右線遺失都優先停止。
6. ESP32 重新連線不會自行解除 STOP 路牌或停妥狀態。

【電腦 PowerShell 傳送】
scp "D:\畢業專題\model_centering\jetson_ready_v4_sign_ocr_20261027.zip" tsai@172.20.10.2:/home/tsai/car/

【Jetson 安裝與啟動】
先在舊控制頁按 STOP，再到舊程式終端機 Ctrl+C，釋放攝影機、串口及 port 5000。
cd ~/car
unzip -o jetson_ready_v4_sign_ocr_20261027.zip
bash install_ocr_deps.sh
python3 check_traffic_ocr.py
bash run_v4_sign_ocr.sh --record traffic_v4_test.avi

install_ocr_deps.sh：已有 Tesseract 與 eng 語料時直接略過安裝；缺少時用 apt 安裝，可能需要網路與 sudo 密碼。
原 V4 的 OpenCV、NumPy、Pillow、PyTorch、Flask、pyserial 環境仍需存在。
不需要另外 pip 安裝 pytesseract：保留你提供的辨識函式，以本地 Tesseract 命令列介面執行 OCR。
官方引擎文件：https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html

check_traffic_ocr.py 不開攝影機、不開串口、不發馬達命令。
它產生 STOP/P 圖樣測試本機 OCR；第二次應顯示 stop_confirmed 和 park_confirmed 都為 true。
若自測失敗，先處理 OCR／語料問題，不要直接測 AUTO。
也可用 python3 check_traffic_ocr.py --image 路牌照片.jpg 檢查實際路牌。

重新整理 http://172.20.10.2:5000，確認標題 V4 + STOP / P OCR TEST。
等影像/FPS 出現，再按 AUTO。AUTO 速度預設 0.12，ESP32 程式與腳位不變。

【停車判斷與限制】
停車以影像中的右線位置、車道寬度比例及方向對齊判斷，沒有固定右轉／左轉幾秒的路線。
需要先觀察到左右車道線以建立寬度，再持續觀察同一條右線；左線出畫後可沿已確認的右線追蹤。
右線真的看不到時立即停等，連續遺失則停止此動作；不會盲目使用舊右線繼續行駛。
對齊位置需連續三張新畫面確認。15 秒是動作未完成的中止保護，不代表停車成功。
車身中心停在右線上會跨在右線兩側；鏡頭須固定並對準車身中心，畫面中心預設 x=160。
本版只有前視影像，沒有量測車身／輪胎的實際幾何位置，也沒有額外障礙物或後方感測。
必須用實車確認「畫面中心對準右線」是否等於你要的車身位置，以及進出停車位置的路徑。
P/回中央時，若 XY 模型輸出超出原訓練視野，可在多張新畫面確認右線後用線位置修正；仍保留影像過期與遺失車道的停止保護。
辨識錯誤、光線、鏡頭偏移或車道線辨識錯誤仍可能影響行駛，這是低速測試版。

【畫面與操作】
STOP OCR HOLD：STOP 路牌保持停止，按 AUTO 繼續。
PARK ON RIGHT LINE：朝右線靠近。
PARKED ON RIGHT LINE - AUTO TO RETURN：已停妥，按 AUTO 回車道。
RETURN TO CENTER：回到車道中央。
WAIT OCR / OCR ERROR：OCR 尚未就緒、結果過期或引擎出錯，保持停止。
PARK WAIT RIGHT CORRIDOR / MANEUVER ABORT：右線不足或動作未完成，保持停止。
同一面路牌不會一直重複觸發；離開畫面且連續三次確認消失後才重新啟用。
要原地重測同一張路牌：按 STOP，再按「重新辨識路牌」，再按 AUTO。
手動模式保留操作權，路牌動作只在 AUTO 啟用。

【錄影與回傳】
啟動指令就開始錄影，不必等 AUTO。原始錄影為 640x480，模型／預覽採 320x240。
OCR 用獨立執行緒和最新影像，不會讓 Tesseract 的等待直接卡住車道推論；Jetson 實際 FPS 仍需量測。
測完先按 STOP，再 Ctrl+C，讓 traffic_v4_test.avi 和 traffic_v4_test.avi.csv 完成寫入。
CSV 包含 OCR 結果、停車狀態、右線位置、車道比例、轉向與影像時間。
固定 30 FPS 的 AVI 播放時間不代表實際行駛時間，分析以 CSV 單調時鐘為準。

電腦 PowerShell 下載回來：
scp tsai@172.20.10.2:/home/tsai/car/traffic_v4_test.avi "$env:USERPROFILE\Downloads\"
scp tsai@172.20.10.2:/home/tsai/car/traffic_v4_test.avi.csv "$env:USERPROFILE\Downloads\"
兩個檔案一起回傳對話；若有第三視角影片也可一起提供。

【已驗證與未驗證】
PC 通過 58 項控制／OCR 流程／網頁介面測試，包含 STOP 保持、同牌抑制、P 停車及 AUTO 回中央。
你提供的九個辨識函式本體保留一致，沒有執行原程式的獨立攝影機迴圈。
V4 模型檔內容不變；Python 3.6 語法、legacy CPU 權重與套件完整性已核對。
PC 沒有 Tesseract 執行檔，因此 OCR 流程測試的文字輸出使用模擬引擎，實際引擎要在 Jetson 用上述自測確認。
尚未完成 Jetson 實際路牌或實車停車／回車道測試。

語料載入路徑修正：引擎會自動尋找並實際載入英文語料，使用明確 --tessdata-dir，避免版本路徑錯置。修復與啟動可用 bash repair_and_run_v4_ocr.sh。

APT 圖示索引 404 修正：先使用現有套件索引安裝 OCR；必要時僅在該次更新略過 DEP-11 圖示／軟體中心 metadata。保留套件驗證，不改套件來源、不升級 Ubuntu。

2026-10-27 OCR text fix: adapter removes edge-connected STOP background, normalizes text to black on white with compact border and 48px letter height. OCR alphabet is A-Z; exact STOP/P and two fresh confirmations unchanged. Real Tesseract 5.5.3 fixture test passes; Jetson native self-check required.
