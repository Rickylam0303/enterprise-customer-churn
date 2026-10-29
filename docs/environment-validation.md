# 獨立環境驗證紀錄

2026-10-29：在相同 Windows 主機，以既有 Python 3.11.15 建立 `include-system-site-packages=false` 的獨立 venv，以從套件索引下載的 wheels 安裝目前 `requirements.lock.txt`。

- 鎖定的 139 個 distribution 均與已安裝版本逐項相符，包含 pip 26.1.1、setuptools 82.0.1。
- `python -m pip check`：通過。
- 23 個回歸測試：通過；包括使用臨時小資料的模型訓練、校準與匯出。
- `python -m src.reporting --config config.yaml --check`：通過；驗證既有正式 run、模型／輸出／來源 hash 與 README 指標。
- 正式 run 仍為 `20261027T175000746489Z-71d14d34`；本輪未另做完整 594,194 筆資料重訓，不改寫原 run 的 137 套件快照。

## 修正的安裝問題

原 `pip freeze` 快照預設漏掉安裝工具，導致乾淨安裝自動取得另一版 setuptools。本輪在目前鎖定檔補上已使用的 pip／setuptools 版本；模型依賴版本沒有變更。

第一個較深路徑的 venv 在安裝 Jupyter extension 時觸發 Windows 長路徑錯誤。改用較短路徑後完整安裝成功。建議使用 `%USERPROFILE%\venvs\churn` 這類短路徑。測試環境實際位於 `C:\Users\6c10r\Documents\Codex\churn-env-1028`；它是驗證產物，沒有取代原本的工作環境。

安裝曾因使用者中斷而暫停，之後使用已下載的 wheels 離線續裝完成；版本與依賴檢查是在續裝完成後執行。

這驗證了同主機、相同 Python interpreter 的全新套件環境；不代表全新電腦／OS、其他平台、Power BI Desktop 或線上部署已通過驗證。

完整版本與命令結果保存於聊天交付的 `clean-environment-verification.json` 及 `clean-*.txt`。Power BI 尚餘步驟見 [驗收清單](power-bi-final-validation.md)。
