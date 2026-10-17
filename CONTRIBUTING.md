# 參與開發

---

## 開發環境

需要 Python 3.10 以上。建議使用虛擬環境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 安裝本專案

```bash
pip install -e ".[dev]"
wifi-fingerprint-locator --help
```

---

## 提交前的檢查

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

CI 會執行相同的檢查，在 Linux 與 Windows 上跑 Python 3.10–3.13，另外有一個 job 用最舊的相依版本。

### Python 3.10 相容性

本專案宣告 `requires-python = ">=3.10"`，請避開 3.11 以後才有的語法與標準函式庫：

| 不要用 | 請用 |
| --- | --- |
| `tomllib` | 本專案的設定檔是 YAML，不需要 TOML |
| `typing.Self` | 類別名稱字串或 `TypeVar` |
| `enum.StrEnum` | `class Foo(str, Enum)` |
| `datetime.UTC` | `datetime.timezone.utc` |
| `except*`、`ExceptionGroup` | 一般的 `except` 與單一例外 |
| `asyncio.TaskGroup` | `asyncio.gather` |
| PEP 695（`class Foo[T]`、`type X = ...`） | `typing.TypeVar`、一般的型別註記 |
| `itertools.batched` | 明確的切片迴圈 |
| NumPy 2.0 才有或已移除的名稱（`np.trapezoid`、`np.float_`、`np.NaN` ...） | `scipy.integrate.trapezoid`、`np.float64`、`np.nan` |

語法閘門（任何 Python 版本都能跑）：

```bash
python -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8'), str(p), feature_version=(3, 10)) for p in pathlib.Path('.').rglob('*.py') if not {'.venv', 'build', 'dist', '__pycache__'} & set(p.parts)]"
```

`tests/test_python310_compat.py` 會自動掃描上述語法、標準函式庫與 NumPy 名稱，以及相依版本下限。

---

## 程式碼風格

- 每個模組開頭加 `from __future__ import annotations`。
- 公開函式要有 type hints 與英文 numpy 風格 docstring；實作文獻方法的公開函式要有 `References` 段。
- 識別字與 docstring 一律使用美式拼法（normalize、synthesize、color）。
- 用 `logging` 輸出進度與警告，不用 `print`。
- 陣列運算盡量向量化，避免逐元素迴圈。
- 所有參數一律放在 `config.py` 的 dataclass，不要在程式裡寫死路徑或魔術數字。

---

## 資料

**本 repository 不包含任何實驗資料。** 測試只用 `synthetic.py` 產生的資料；請不要提交 CSI 擷取檔、`.mat`、`.npy`、模型檔或任何含個人資訊的檔案。

---

## 提交訊息

使用 Conventional Commits（`feat`／`fix`／`docs`／`test`／`refactor`）。有功能變動時，請同步更新 `CHANGELOG.md` 的 `[Unreleased]`。
