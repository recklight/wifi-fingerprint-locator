# 變更紀錄

本檔案記錄 `wifi-fingerprint-locator` 的所有重要變更。

格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
版本號依循 [語意化版本](https://semver.org/lang/zh-TW/)。

## [Unreleased]

---

## [0.1.0] - 2026-10-17

### 新增

- 首次釋出。
- 英文版 README（`README.md`，GitHub 預設顯示），繁體中文版在 `README.zh-TW.md`。
- 流程圖分英文與繁體中文兩版：英文版是 `docs/workflow.html` 與 `docs/images/workflow.png`，繁體中文版是 `docs/workflow.zh-TW.html` 與 `docs/images/workflow.zh-TW.png`。
- 向量化的 Intel 5300 beamforming report 擷取檔讀寫器（`beamform_report`）：位元層級解碼、天線排列還原、截斷與長度不符記錄的計數或嚴格模式報錯，以及寫檔用的編碼器。
- 依 RSSI、AGC 與雜訊底限把 CSI 換算成 $\sqrt{\mathrm{SNR}}$ 單位（`csi_scaling`），RSSI 全為零時回傳 NaN 而不是無限大。
- 線性相位校正（`phase_sanitization`）：端點斜率與最小平方兩種直線估計，使用 802.11n 實際的分組子載波索引，也可切換成等距近似以便對照。
- 資料清理政策（`channel_cleaning`）：傳送串流數混雜、0 值項目內插或剔除、非有限值剔除，全部計數並寫進 manifest。
- 六種指紋表示法登錄表（`fingerprints`）：振幅、dB 振幅、校正後相位、振幅加相位、實部加虛部，以及校正相位後的實虛部消融組。
- 四種切分策略（`splits`）：封包隨機、位置區塊、留位置（打散的分組 k 折）與時間區塊，皆可重現。
- 帶驗證集早停並還原最佳權重的 MLP 位置迴歸（`position_regressor`），支援 L-BFGS、Adam 與自行實作的 Levenberg-Marquardt（含解析 Jacobian 與記憶體上限，懲罰係數與 scikit-learn 的 `alpha` 同尺度），以及以驗證誤差挑選 L2 懲罰係數。位置區塊與留位置切分下，早停的驗證集是從訓練折整個留出的參考點。
- kNN、加權 kNN 與最近參考點三種指紋比對基準模型。
- 歐氏定位誤差指標（平均、中位數、百分位數、逐軸 MAE、格點命中率與 UAR）與經驗 CDF，只在測試資料上計算，並同時報告封包層級與擷取檔層級。
- `compare` 指令：表示法 × 切分策略 × 折的比較矩陣，輸出 CSV、JSON 與 CDF 圖。
- 具多徑、取樣時間偏移、載波相位偏移、8 位元量化、串流數變動與 0 值項目的合成勘測資料產生器，輸出與真實資料相同的磁碟格式。
- typer CLI：`synthesize`、`index`、`inspect`、`prepare`、`evaluate`、`train`、`predict`、`compare`。
