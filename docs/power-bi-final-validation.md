# Power BI 最後整合與驗收

2026-10-29 狀態：review PBIX 已在 Desktop 開啟並顯示歷史快取。套用原先組合查詢時確認發生 Formula.Firewall；同查詢內改用單一 Folder.Files 仍失敗。目前提供分離來源與轉換的 staging 方案，但尚未成功刷新。新增 Blank query 的 WebView 無法由 Windows 操作工具定位（重試仍回報 point is over msedgewebview2.exe, not target window PBIDesktop.exe）；使用者選擇 Power BI 稍後處理。原報表、原始 CSV 及已保存 review 檔均保留；Desktop 中的試驗尚未保存。

## 套用目前來源

1. 在 Power BI Desktop 開啟 `power-bi/Executive Overview.review.pbix`，另存工作副本。
2. Transform data → New Source → Blank Query，貼上 `ProjectFilesSource.m` 並命名為 `ProjectFilesSource`；取消 Enable load。然後選取現有 `train` 查詢 → Advanced Editor，以 `power-bi/queries/LoadTrain.m` 全文取代。確認兩份 M 的 ProjectRoot 均指向含有 `config.yaml` 的專案資料夾。如使用自訂輸出位置，亦須修改 M 中的明確檔案路徑。
3. 選擇與信任本機檔案來源，Close & Apply。不要以舊的 `bi_train_predictions.csv` 或 final in-sample 分數代替 holdout 匯出。
4. 依 `power-bi/measures/ChurnMeasures.dax` 逐個建立／更新 measure；不要把整份檔案貼成單一 measure。先建立被其他 measure 引用的基礎 measure。表格名稱應為 `train`。
5. 歷史流失卡使用 `Historical Churned Customers`；預測高風險卡使用 `Predicted High-Risk Customers`。確認標題與實際 measure 一致。
6. 行動表格按 Contract × tenure_bucket × InternetService 分組，使用 `Valid Segment Churn Rate`、`Segment Churn Rank`，visual filter 設為 `Actionable Segment = 1`。若保留 review 副本的固定 count filter，最低樣本數須與 config 一致（本輪 1000）。先排除小樣本，再限制 Top N。
7. 如需獨立分群表，可新建 Blank Query，貼上 `LoadActionSegments.m`；此表每列是一個合資格分群，不是客戶。不要建立會令客戶數重複計算的多對多關係。

## Refresh 後驗收（先清除所有 slicer/filter）

正式 run：`20261027T175000746489Z-71d14d34`。請與同次執行的 `outputs/bi_kpis.json` 核對：

| 項目 | 預期 |
|---|---:|
| 客戶列數／不同 id 數 | 118,839／118,839 |
| prediction_scope 唯一值 | holdout |
| model_version 唯一值 | c05a64379a94 |
| Historical Churned Customers | 26,763 |
| Predicted High-Risk Customers（≥0.6） | 21,901 |
| Historical Churn Rate | 22.5204% |
| Mean Calibrated Churn Score | 22.6008% |
| 非空分群／合資格分群 | 44／22 |
| 行動分群最低列數 | ≥1,000 |

核對 tenure=0 被歸入第一組、SeniorCitizen=0/1 顯示正確，以及套用 slicer 後分母與 KPI 一同變動。套用小樣本過濾後，行動表不應出現小於 1000 的分群；切換 slicer 可能令合資格分群數減少。

完成後保存已刷新 PBIX、記錄 Desktop 版本／刷新時間／run_id，並保存兩頁截圖。只有完成這些檢查後，才可把 README 中「Desktop rendering/refresh has not been verified」改為已驗證。新 run 的核對值應取自新 `bi_kpis.json`，不要沿用本表舊數字。

## 範圍

目前可啟動 Desktop，但新增查詢的 WebView 操作受到工具視窗定位錯誤阻擋，尚未驗證 staging 方案。以上驗收仍需在 Desktop 完成；保留既有隱私設定。架構依據參考 [Microsoft Data Privacy Firewall 文件](https://learn.microsoft.com/en-us/power-query/dataprivacyfirewall)：資料來源讀取與引用轉換應分離；這不保證此檔案已通過引擎驗證。網站部署更新也未包含於本次本機修正；目前線上 App 不應被視為已同步。
