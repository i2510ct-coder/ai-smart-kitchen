import calendar
import datetime
import html
import json
import os
import sqlite3

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

# ===== supabase ライブラリ（未インストールでもローカル SQLite で動くようにする） =====
try:
    from supabase import create_client

    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

# ===== pyzbar は zbar 共有ライブラリが無いと import に失敗することがある =====
try:
    from pyzbar import pyzbar

    PYZBAR_AVAILABLE = True
    PYZBAR_IMPORT_ERROR = None
except Exception as e:
    PYZBAR_AVAILABLE = False
    PYZBAR_IMPORT_ERROR = str(e)


# ===== シークレット読み込み（Streamlit Cloud の Secrets → 環境変数/.env の順） =====
def get_secret(key: str):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


# ===== 設定 =====
DB_PATH = "inventory.db"
MODEL_TEXT = "gemini-2.5-flash"
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
RAKUTEN_APP_ID = get_secret("RAKUTEN_APP_ID")
APP_PASSWORD = get_secret("APP_PASSWORD")  # 設定されていればパスワード入力を要求
SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")  # service_role キー（Secrets にだけ置く）
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY and SUPABASE_AVAILABLE)

FOOD_TYPES = ["一般食品", "防災食", "調味料"]
TYPE_EMOJI = {"一般食品": "🥬", "防災食": "🥫", "調味料": "🧂"}
STATUS_STOCK = "在庫"
STATUS_CONSUMED = "消費済み"
STATUS_DISCARDED = "廃棄"


# ===== Gemini 構造化出力用のスキーマ =====
class InventoryItemSchema(BaseModel):
    name: str = Field(description="食材名（日本語）")
    price: int = Field(description="数値（不明なら 0）")
    food_type: str = Field(
        description="'一般食品' または '防災食' または '調味料' のいずれか"
    )
    expiry_date: str = Field(
        description="'YYYY-MM-DD' 形式の賞味期限（日付が読めない・不明な場合は空文字）"
    )


# ===== DB（Supabase の設定があればそちら。無ければローカル SQLite） =====
ITEM_KEYS = ["id", "food_type", "name", "expiry_date", "price", "status", "closed_at"]


@st.cache_resource
def get_supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    """SQLite を使う場合だけテーブルを作る（Supabase は SQL エディタで作成済みの前提）。"""
    if USE_SUPABASE:
        return
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_type TEXT NOT NULL,
            name TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL,
            closed_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    c.execute("PRAGMA table_info(inventory)")
    cols = [r[1] for r in c.fetchall()]
    if "closed_at" not in cols:
        c.execute("ALTER TABLE inventory ADD COLUMN closed_at TEXT NOT NULL DEFAULT ''")
    conn.commit()
    conn.close()


def _normalize_row(r: dict) -> dict:
    return {
        "id": r.get("id"),
        "food_type": r.get("food_type") or "一般食品",
        "name": r.get("name") or "",
        "expiry_date": r.get("expiry_date") or "",
        "price": float(r.get("price") or 0),
        "status": r.get("status") or STATUS_STOCK,
        "closed_at": r.get("closed_at") or "",
    }


@st.cache_data(ttl=30, show_spinner=False)
def _load_all_rows():
    """全件を1回で取得してキャッシュする（画面1回の描画で何度も通信しないため）。
    書き込みのたびに _load_all_rows.clear() で破棄する。"""
    if USE_SUPABASE:
        res = get_supabase_client().table("inventory").select("*").order("id").execute()
        return [_normalize_row(r) for r in (res.data or [])]

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, food_type, name, expiry_date, price, status, closed_at "
        "FROM inventory ORDER BY id"
    )
    rows = [_normalize_row(dict(zip(ITEM_KEYS, r))) for r in c.fetchall()]
    conn.close()
    return rows


def insert_many(rows):
    """rows: food_type, name, expiry_date, price, status, closed_at を持つ dict のリスト"""
    if not rows:
        return
    payload = [
        {
            "food_type": r["food_type"],
            "name": r["name"],
            "expiry_date": r["expiry_date"],
            "price": float(r.get("price", 0) or 0),
            "status": r.get("status", STATUS_STOCK),
            "closed_at": r.get("closed_at", "") or "",
        }
        for r in rows
    ]
    if USE_SUPABASE:
        get_supabase_client().table("inventory").insert(payload).execute()
    else:
        conn = get_conn()
        conn.executemany(
            "INSERT INTO inventory (food_type, name, expiry_date, price, status, closed_at) "
            "VALUES (:food_type, :name, :expiry_date, :price, :status, :closed_at)",
            payload,
        )
        conn.commit()
        conn.close()
    _load_all_rows.clear()


def insert_item(food_type, name, expiry_date, price, status=STATUS_STOCK, closed_at=""):
    insert_many(
        [
            {
                "food_type": food_type,
                "name": name,
                "expiry_date": expiry_date,
                "price": price,
                "status": status,
                "closed_at": closed_at,
            }
        ]
    )


def update_status(item_id, new_status):
    closed_at = (
        datetime.datetime.now().isoformat(timespec="seconds")
        if new_status != STATUS_STOCK
        else ""
    )
    if USE_SUPABASE:
        get_supabase_client().table("inventory").update(
            {"status": new_status, "closed_at": closed_at}
        ).eq("id", item_id).execute()
    else:
        conn = get_conn()
        conn.execute(
            "UPDATE inventory SET status = ?, closed_at = ? WHERE id = ?",
            (new_status, closed_at, item_id),
        )
        conn.commit()
        conn.close()
    _load_all_rows.clear()


def update_item(item_id, name, food_type, expiry_date, price):
    if USE_SUPABASE:
        get_supabase_client().table("inventory").update(
            {
                "name": name,
                "food_type": food_type,
                "expiry_date": expiry_date,
                "price": float(price),
            }
        ).eq("id", item_id).execute()
    else:
        conn = get_conn()
        conn.execute(
            "UPDATE inventory SET name = ?, food_type = ?, expiry_date = ?, price = ? WHERE id = ?",
            (name, food_type, expiry_date, price, item_id),
        )
        conn.commit()
        conn.close()
    _load_all_rows.clear()


def delete_all_items():
    if USE_SUPABASE:
        get_supabase_client().table("inventory").delete().gt("id", 0).execute()
    else:
        conn = get_conn()
        conn.execute("DELETE FROM inventory")
        conn.commit()
        conn.close()
    _load_all_rows.clear()


def fetch_items(status_filter=None, food_type_filter=None):
    rows = _load_all_rows()
    if status_filter:
        rows = [r for r in rows if r["status"] == status_filter]
    if food_type_filter:
        rows = [r for r in rows if r["food_type"] == food_type_filter]
    return rows


# ===== CSV バックアップ =====
CSV_COLUMNS = ["food_type", "name", "expiry_date", "price", "status", "closed_at"]


def export_items_csv() -> bytes:
    df = pd.DataFrame(fetch_items(), columns=ITEM_KEYS)[CSV_COLUMNS]
    return df.to_csv(index=False).encode("utf-8-sig")  # Excel でも文字化けしない


def import_items_from_csv(file):
    """CSV を読み込んで在庫に追加する。戻り値: (追加した件数, 読み飛ばした件数)"""
    df = pd.read_csv(file, dtype=str).fillna("")
    required = ["food_type", "name", "expiry_date", "price", "status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {', '.join(missing)}")

    valid_status = {STATUS_STOCK, STATUS_CONSUMED, STATUS_DISCARDED}
    rows, skipped = [], 0
    for _, r in df.iterrows():
        try:
            price = float(r["price"] or 0)
        except ValueError:
            skipped += 1
            continue
        if (
            r["food_type"] not in FOOD_TYPES
            or r["status"] not in valid_status
            or not r["name"].strip()
            or parse_expiry(r["expiry_date"]) is None
        ):
            skipped += 1
            continue
        rows.append(
            {
                "food_type": r["food_type"],
                "name": r["name"].strip(),
                "expiry_date": r["expiry_date"].strip(),
                "price": price,
                "status": r["status"],
                "closed_at": r["closed_at"] if "closed_at" in df.columns else "",
            }
        )
    insert_many(rows)
    return len(rows), skipped


# ===== 日付・判定ロジック =====
def parse_expiry(expiry_date_str: str):
    """'YYYY-MM-DD' または 'YYYY-MM' を date に変換。失敗したら None。
    'YYYY-MM' はその月の末日として扱う。"""
    s = (expiry_date_str or "").strip()
    try:
        if len(s) == 7:
            d = datetime.datetime.strptime(s + "-01", "%Y-%m-%d").date()
            last_day = calendar.monthrange(d.year, d.month)[1]
            return d.replace(day=last_day)
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def calc_days_left(expiry_date_str: str):
    """残り日数。日付が壊れている場合は None（0日と誤判定しない）。"""
    expiry = parse_expiry(expiry_date_str)
    if expiry is None:
        return None
    return (expiry - datetime.date.today()).days


def judge_threshold(food_type: str, days_left) -> str:
    if days_left is None:
        return "不明"
    limit = 30 if food_type in ("防災食", "調味料") else 2
    if days_left < 0:
        return "赤"
    elif days_left <= limit:
        return "黄"
    return "緑"


def is_near_expiry(food_type: str, days_left) -> bool:
    if days_left is None:
        return False
    if food_type == "一般食品":
        return days_left <= 2
    if food_type in ("防災食", "調味料"):
        return days_left <= 30
    return False


def level_css_class(level: str) -> str:
    return {"赤": "danger", "黄": "warning", "緑": "safe"}.get(level, "unknown")


def days_label(days_left) -> str:
    return "期限不明" if days_left is None else f"{days_left} 日"


def sort_key_days(item) -> int:
    d = calc_days_left(item["expiry_date"])
    return 10**9 if d is None else d


# ===== Gemini =====
def get_gemini_client():
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY が設定されていません。Secrets（または .env）を確認してください。")
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


def analyze_image_with_gemini(image_bytes: bytes, mime_type: str = "image/jpeg"):
    client = get_gemini_client()
    if client is None:
        return None

    prompt = f"""
画像に写っているパッケージや賞味期限の表示から、食材名・価格・区分・賞味期限を読み取ってください。
「5月20日」などの表現は、今年（現在{datetime.date.today().year}年）または来年の日付として YYYY-MM-DD に変換してください。
"""
    valid_mime_types = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
    if mime_type not in valid_mime_types:
        mime_type = "image/jpeg"

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    try:
        res = client.models.generate_content(
            model=MODEL_TEXT,
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InventoryItemSchema,
            ),
        )
        return json.loads(res.text)
    except Exception as e:
        st.error(f"画像解析に失敗しました。手動で入力してください。エラー: {e}")
        return None


def analyze_audio_with_gemini(audio_bytes: bytes, mime_type: str = "audio/wav"):
    client = get_gemini_client()
    if client is None:
        return None

    prompt = f"""
音声の内容（日本語）から、発話されている食材名、価格、区分、賞味期限を特定してください。
「5月20日」などの表現は、今年（現在{datetime.date.today().year}年）または来年の日付として YYYY-MM-DD に変換してください。
"""
    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    try:
        res = client.models.generate_content(
            model=MODEL_TEXT,
            contents=[prompt, audio_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=InventoryItemSchema,
            ),
        )
        return json.loads(res.text)
    except Exception as e:
        st.error(f"音声解析に失敗しました。手動で入力してください。エラー: {e}")
        return None


def generate_recipe_with_gemini(selected_items):
    """レシピ生成。失敗してもアプリが落ちないよう、エラー文を返す。"""
    client = get_gemini_client()
    if client is None:
        return None

    context = "\n".join(
        f"- 食材名: {i['name']} / 区分: {i['food_type']} / 期限: {i['expiry_date']}"
        for i in selected_items
    )
    prompt = f"""
あなたは家庭の冷蔵庫管理を支援するプロの料理家です。
以下の食材を「できるだけ無駄なく」使うレシピを日本語で提案してください。

【利用可能な食材】
{context}

要件：
- 2〜3品程度の献立案を出してください。
- 各レシピについて、
  - レシピ名
  - 何人分か
  - 必要な材料（上記以外に必要なものも含めて）
  - 作り方の手順
  - 美味しく作るコツ
を箇条書きでわかりやすく説明してください。
"""
    try:
        res = client.models.generate_content(model=MODEL_TEXT, contents=prompt)
        if not res.text:
            st.error("レシピを生成できませんでした。もう一度お試しください。")
            return None
        return res.text
    except Exception as e:
        st.error(f"レシピ生成に失敗しました。時間をおいて再度お試しください。エラー: {e}")
        return None


# ===== バーコード =====
def read_barcode_from_image(image_bytes: bytes):
    if not PYZBAR_AVAILABLE:
        st.error(
            "バーコード読み取り機能が使えません（zbar ライブラリが見つかりません）。\n\n"
            "Streamlit Cloud: リポジトリに packages.txt を置き `libzbar0` と書いてください。\n"
            "Mac: `brew install zbar` / Linux: `sudo apt-get install libzbar0`\n\n"
            f"詳細エラー: {PYZBAR_IMPORT_ERROR}"
        )
        return None

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        st.warning("画像の読み込みに失敗しました。もう一度撮影してください。")
        return None

    barcodes = pyzbar.decode(img)

    if not barcodes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_large = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        barcodes = pyzbar.decode(gray_large)

    if not barcodes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        barcodes = pyzbar.decode(thresh)

    if barcodes:
        return barcodes[0].data.decode("utf-8")
    return None


def get_product_name_from_barcode(barcode: str):
    # 1. Open Food Facts
    try:
        url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
        res = requests.get(url, timeout=5, headers={"User-Agent": "AI-SmartKitchen/1.0"})
        data = res.json()
        if data.get("status") == 1:
            product = data.get("product", {})
            name = (
                product.get("product_name_ja")
                or product.get("product_name")
                or product.get("generic_name")
            )
            if name:
                return name
    except Exception:
        pass

    # 2. 楽天市場（フォールバック）
    if not RAKUTEN_APP_ID:
        return None
    try:
        res = requests.get(
            "https://app.rakuten.co.jp/services/api/IchibaItem/Search/20170901",
            params={
                "applicationId": RAKUTEN_APP_ID,
                "keyword": barcode,
                "hits": 1,
                "format": "json",
            },
            timeout=5,
        )
        items = res.json().get("Items", [])
        if items:
            return items[0]["Item"]["itemName"]
    except Exception:
        pass
    return None


# ===== デモデータ =====
def seed_demo_data():
    """在庫（期限が近い/切れ/余裕あり）と、過去の消費・廃棄の履歴を投入する。"""
    today = datetime.date.today()
    now = datetime.datetime.now()

    stock = [
        ("一般食品", "牛乳", today + datetime.timedelta(days=1), 198),
        ("一般食品", "卵", today + datetime.timedelta(days=2), 250),
        ("一般食品", "ほうれん草", today + datetime.timedelta(days=-1), 180),
        ("一般食品", "鶏むね肉", today + datetime.timedelta(days=5), 480),
        ("一般食品", "豆腐", today + datetime.timedelta(days=4), 98),
        ("防災食", "アルファ米", today + datetime.timedelta(days=20), 400),
        ("防災食", "缶詰パン", today + datetime.timedelta(days=300), 450),
        ("調味料", "しょうゆ", today + datetime.timedelta(days=200), 280),
        ("調味料", "みりん", today + datetime.timedelta(days=25), 320),
    ]
    rows = []
    for ft, name, d, price in stock:
        fmt = "%Y-%m-%d" if ft == "一般食品" else "%Y-%m"
        rows.append(
            {"food_type": ft, "name": name, "expiry_date": d.strftime(fmt),
             "price": price, "status": STATUS_STOCK, "closed_at": ""}
        )

    # (区分, 名前, 金額, 何日前に消費/廃棄したか, ステータス)
    history = [
        ("一般食品", "キャベツ", 198, 5, STATUS_CONSUMED),
        ("一般食品", "ヨーグルト", 150, 12, STATUS_CONSUMED),
        ("一般食品", "豚こま肉", 520, 20, STATUS_CONSUMED),
        ("一般食品", "トマト", 280, 28, STATUS_DISCARDED),
        ("一般食品", "鮭の切り身", 600, 40, STATUS_CONSUMED),
        ("一般食品", "もやし", 40, 45, STATUS_DISCARDED),
        ("一般食品", "納豆", 100, 55, STATUS_CONSUMED),
        ("一般食品", "きゅうり", 120, 70, STATUS_DISCARDED),
        ("一般食品", "鶏もも肉", 550, 75, STATUS_CONSUMED),
        ("一般食品", "ハム", 230, 85, STATUS_CONSUMED),
    ]
    for ft, name, price, days_ago, status in history:
        closed = now - datetime.timedelta(days=days_ago)
        rows.append(
            {"food_type": ft, "name": name, "expiry_date": closed.strftime("%Y-%m-%d"),
             "price": price, "status": status,
             "closed_at": closed.isoformat(timespec="seconds")}
        )
    insert_many(rows)


# ===== パスワード =====
def require_password():
    """APP_PASSWORD が設定されている場合のみ、共通パスワードを要求する。"""
    if not APP_PASSWORD or st.session_state.get("authed"):
        return
    st.title("🥕 AIスマートキッチン")
    pw = st.text_input("パスワード", type="password")
    if pw:
        if pw == APP_PASSWORD:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
    st.stop()


# ===== UI =====
def main():
    st.set_page_config(page_title="AIスマートキッチン", page_icon="🥕", layout="wide")

    init_db()
    require_password()

    try:
        _load_all_rows()
    except Exception as e:
        st.error(
            "データベースに接続できませんでした。Supabase のプロジェクトが一時停止していないか、"
            "Secrets の SUPABASE_URL / SUPABASE_KEY が正しいか確認してください。\n\n"
            f"詳細: {e}"
        )
        st.stop()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@400;500;700&display=swap');
        html, body, .stApp, .stApp * { font-family: 'Zen Maru Gothic', 'Hiragino Maru Gothic ProN', 'Meiryo', sans-serif; }
        [data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined, .material-icons {
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important; }
        .stApp { background: linear-gradient(180deg, #fff8f0 0%, #fff0f3 100%); }
        section[data-testid="stSidebar"] { background: #ffeef2; }
        .hero { background: linear-gradient(135deg, #ffd6e0, #ffe9b8); border-radius: 28px;
                padding: 22px 30px; margin-bottom: 14px; box-shadow: 0 6px 18px rgba(255,150,170,.28); }
        .main-title { font-size: 30px; font-weight: 700; color: #e0566f; }
        .sub-title { font-size: 14px; color: #8a6d5b; margin-top: 4px; }
        .danger { color: #e0405a; font-weight: 700; }
        .warning { color: #f08a24; font-weight: 700; }
        .safe { color: #35a37b; font-weight: 700; }
        .unknown { color: #8a8a8a; font-weight: 700; }
        .badge { display: inline-block; padding: 2px 12px; border-radius: 999px; background: #fff;
                 border: 2px solid currentColor; font-size: 13px; }
        h1, h2, h3 { color: #e0566f !important; }
        hr { border: none; border-top: 2px dashed #ffd6e0 !important; }
        .stButton > button, .stDownloadButton > button { border-radius: 999px; border: 2px solid #ffb3c1;
                 background: #fff; color: #e0566f; font-weight: 700; }
        .stButton > button:hover, .stDownloadButton > button:hover { background: #ffe3e9; border-color: #ff8fa3; color: #c93a55; }
        .stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
                 background: linear-gradient(135deg, #ff8fa3, #ffb36b); color: #fff; border: none;
                 box-shadow: 0 4px 12px rgba(255,143,163,.45); }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background: #fff; border-radius: 999px; padding: 6px 18px; border: 2px solid #ffd6e0; }
        .stTabs [aria-selected="true"] { background: #ffd6e0; }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
        [data-testid="stMetric"] { background: #fff; border-radius: 22px; padding: 12px 18px;
                 box-shadow: 0 4px 12px rgba(255,150,170,.18); }
        [data-testid="stExpander"] { background: #fff; border-radius: 18px; border: 2px solid #ffe0e7; }
        div[data-testid="stAlert"] { border-radius: 18px; }
        input, textarea, [data-baseweb="select"] > div { border-radius: 14px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----- サイドバー -----
    with st.sidebar:
        if USE_SUPABASE:
            st.success("💾 保存先：Supabase（再起動しても消えません）")
        else:
            st.warning("💾 保存先：ローカルSQLite（クラウドでは再起動で消えます）")
            if SUPABASE_URL and SUPABASE_KEY and not SUPABASE_AVAILABLE:
                st.error("supabase ライブラリが未インストールです。requirements.txt を確認してください。")

        st.subheader("🛠 デモ用ツール")
        if st.button("デモデータを投入", use_container_width=True):
            seed_demo_data()
            st.session_state["toast_msg"] = "デモデータを投入しました"
            st.rerun()
        with st.expander("全データを削除"):
            confirm = st.checkbox("本当に全て削除する")
            if st.button("削除を実行", disabled=not confirm):
                delete_all_items()
                st.session_state["toast_msg"] = "全データを削除しました"
                st.rerun()

        st.subheader("💾 バックアップ")
        st.download_button(
            "CSVでダウンロード",
            data=export_items_csv(),
            file_name=f"inventory_{datetime.date.today():%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        with st.expander("CSVから読み込む（追加されます）"):
            csv_file = st.file_uploader("CSVファイル", type=["csv"], key="csv_import")
            if csv_file is not None and st.button("読み込みを実行", key="csv_import_run"):
                try:
                    ok, skipped = import_items_from_csv(csv_file)
                    st.session_state["toast_msg"] = f"{ok}件を読み込みました（読み飛ばし {skipped}件）"
                    st.rerun()
                except Exception as e:
                    st.error(f"読み込みに失敗しました: {e}")

    st.markdown(
        '<div class="hero"><div class="main-title">🥕 AIスマートキッチン 🍅</div>'
        '<div class="sub-title">冷蔵庫の「もったいない」を見える化して、おいしく使い切ろう 🍳</div></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("toast_msg"):
        st.toast(st.session_state.pop("toast_msg"), icon="✅")

    if not GEMINI_API_KEY:
        st.error(
            "⚠️ GEMINI_API_KEY が読み込まれていません。Secrets（または .env）を確認してください"
            "（画像解析・音声解析・レシピ提案が動きません）。"
        )
    if not PYZBAR_AVAILABLE:
        st.warning(f"⚠️ バーコード読み取り機能が無効です: {PYZBAR_IMPORT_ERROR}")

    st.write("---")

    all_items = fetch_items(status_filter=STATUS_STOCK)

    near_items = []
    risk_amount = 0
    for item in all_items:
        days_left = calc_days_left(item["expiry_date"])
        if is_near_expiry(item["food_type"], days_left):
            near_items.append((item, days_left))
            risk_amount += item["price"]

    discarded_items = fetch_items(status_filter=STATUS_DISCARDED)
    total_discarded_price = sum(i["price"] for i in discarded_items)

    consumed_items = fetch_items(status_filter=STATUS_CONSUMED)
    total_consumed_price = sum(i["price"] for i in consumed_items)
    safe_stock_price = sum(
        i["price"]
        for i in all_items
        if not is_near_expiry(i["food_type"], calc_days_left(i["expiry_date"]))
    )
    # 「もったいない」= 期限が近い在庫 + 捨てた金額 / それ以外 = 食べた金額 + まだ余裕のある在庫
    bad_weight = risk_amount + total_discarded_price
    good_weight = total_consumed_price + safe_stock_price
    total_weight = bad_weight + good_weight
    if total_weight > 0:
        waste_share = bad_weight / total_weight  # 全体のうち「もったいない」の割合
        angle = min(20, waste_share * 40)  # 0度（水平）〜20度。右が下がる
    else:
        angle = 0

    # JS の innerHTML に入る文字列は必ずエスケープする
    near_names = [html.escape(item["name"]) for item, _ in near_items[:4]]
    safe_names = [
        html.escape(item["name"])
        for item in all_items
        if not is_near_expiry(item["food_type"], calc_days_left(item["expiry_date"]))
    ][:4]

    if total_weight == 0:
        mood = "🍽️ まだ食材がないよ。まずは登録してみよう"
    elif angle < 3:
        mood = "🌸 いい感じ！この調子で食べきろう"
    elif angle < 14:
        mood = "⏰ ちょっと心配…早めに使ってあげてね"
    else:
        mood = "🚨 たいへん！今日は使い切りメニューにしよう"

    balance_data = json.dumps(
        {
            "angle": round(angle, 1),
            "mood": mood,
            "risk": int(risk_amount),
            "discarded": int(total_discarded_price),
            "near_count": len(near_items),
            "near_names": near_names,
            "safe_names": safe_names,
            "shake": angle > 14,
        }
    )

    home_col1, home_col2 = st.columns([1, 1])

    with home_col1:
        st.subheader("📢 期限接近アラート")
        if near_items:
            st.warning(f"⏰ もうすぐ期限の食材があるよ！このままだと {int(risk_amount)} 円がもったいない…")
            for item, days_left in sorted(near_items, key=lambda x: x[1]):
                level = judge_threshold(item["food_type"], days_left)
                cls = level_css_class(level)
                st.markdown(
                    f'<span class="{cls}">[{html.escape(item["food_type"])}] '
                    f'{html.escape(item["name"])}：あと {days_left} 日'
                    f'（期限 {html.escape(item["expiry_date"])}）</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("🎉 いまのところ安心！期限が近い食材はありません")

    with home_col2:
        st.subheader("⚖️ フードロス天秤")
        st.caption("水平ならバッチリ！期限が近い・捨てた食材が増えると、右（ごみ箱側）が下がるよ")
        html_balance = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic:wght@500;700&display=swap" rel="stylesheet">
<style>
body{font-family:'Zen Maru Gothic','Hiragino Maru Gothic ProN','Meiryo',sans-serif;margin:0;padding:10px;background:linear-gradient(135deg,#fff0f3,#fff4d6);border-radius:24px;color:#5a3e36;}
@keyframes sway{0%{transform:rotate(var(--a))}50%{transform:rotate(calc(var(--a) + var(--amp)))}100%{transform:rotate(var(--a))}}
.sway{animation:sway 2.4s ease-in-out infinite;}
.mood{display:block;width:fit-content;margin:0 auto 4px;background:#fff;border:2px solid #ffc2cf;border-radius:999px;padding:4px 16px;font-size:13px;font-weight:700;color:#e0566f;}
.wrap{position:relative;height:170px;display:flex;align-items:center;justify-content:center;}
.wrap::after{content:"";position:absolute;left:50%;top:calc(50% + 6px);width:30px;height:38px;background:#ff8fa3;transform:translateX(-50%);clip-path:polygon(50% 0%,0% 100%,100% 100%);}
.seesaw{width:260px;height:16px;background:linear-gradient(90deg,#ffb3c1,#ffd59e,#ffb3c1);border-radius:999px;position:relative;z-index:1;box-shadow:0 3px 8px rgba(255,150,170,.4);transition:transform 1.2s cubic-bezier(0.34,1.56,0.64,1);}
.end{position:absolute;bottom:8px;font-size:32px;line-height:1;}
.endL{left:-6px;}.endR{right:-6px;}
.lists{display:flex;justify-content:space-between;gap:8px;font-size:11px;line-height:1.7;min-height:36px;}
.lists>div{flex:1;}
.lists .r{text-align:right;}
.info-row{display:flex;justify-content:space-around;margin-top:8px;font-size:11px;gap:6px;}
.info-item{text-align:center;background:#fff;border-radius:16px;padding:8px 6px;flex:1;box-shadow:0 2px 8px rgba(255,150,170,.2);}
.val{font-weight:700;font-size:17px;}
.green{color:#35a37b;}.red{color:#e0405a;}.gray{color:#8a6d5b;}
</style></head><body>
<div class="mood" id="mood"></div>
<div class="wrap">
  <div id="ss" class="seesaw"><div class="end endL">😋</div><div class="end endR" id="rface">🗑️</div></div>
</div>
<div class="lists">
  <div class="green" id="safe-list"></div>
  <div class="red r" id="near-list"></div>
</div>
<div class="info-row">
  <div class="info-item"><div class="val red" id="risk">0</div><div>あぶない金額（円）</div></div>
  <div class="info-item"><div class="val red" id="disc">0</div><div>捨てた金額（円）</div></div>
  <div class="info-item"><div class="val gray" id="near">0件</div><div>もうすぐ期限</div></div>
</div>
<script>
const d=""" + balance_data + """;
const ss=document.getElementById('ss');
ss.style.setProperty('--a',d.angle+'deg');
ss.style.setProperty('--amp',(d.shake?4:1.5)+'deg');
ss.style.animationDuration=d.shake?'0.7s':'2.4s';
setTimeout(()=>{ss.style.transform='rotate('+d.angle+'deg)';},300);
setTimeout(()=>{ss.classList.add('sway');},1700);
document.getElementById('mood').innerText=d.mood;
document.getElementById('rface').innerText=d.angle<4?'🗑️':(d.angle<10?'😟':'🥺');
document.getElementById('risk').innerText=d.risk.toLocaleString();
document.getElementById('disc').innerText=d.discarded.toLocaleString();
document.getElementById('near').innerText=d.near_count+'件';
document.getElementById('near-list').innerHTML=d.near_names.map(n=>'⚠️'+n).join('<br>');
document.getElementById('safe-list').innerHTML=d.safe_names.map(n=>'✅'+n).join('<br>');
</script></body></html>"""
        components.html(html_balance, height=410)

    st.write("---")

    tab_register, tab_stock, tab_recipe, tab_history = st.tabs(
        ["➕ 食材登録", "📦 在庫一覧", "🍳 レシピ提案", "📊 履歴・フードロス"]
    )

    # ===== 食材登録タブ =====
    with tab_register:
        st.header("食材登録")

        prefill = st.session_state.get("prefill", {})
        for k in ("cam_key", "expiry_cam_key", "form_key"):
            if k not in st.session_state:
                st.session_state[k] = 0

        # ── 1. 商品名 ──
        st.subheader("1. 商品名")
        name_col, barcode_col = st.columns([5, 1])
        with name_col:
            name = st.text_input(
                "食材名",
                value=prefill.get("name", ""),
                placeholder="例：卵",
                label_visibility="collapsed",
                key=f"name_{st.session_state['form_key']}",
            )
        with barcode_col:
            if st.button("📷 バーコード", key="open_barcode_cam", disabled=not PYZBAR_AVAILABLE):
                st.session_state["show_barcode_cam"] = not st.session_state.get("show_barcode_cam", False)

        if st.session_state.get("show_barcode_cam"):
            img = st.camera_input(
                "バーコードを撮影（できるだけ大きく・明るく・ピントを合わせて）",
                key=f"camera_{st.session_state['cam_key']}",
            )
            bc1, bc2 = st.columns(2)
            with bc1:
                if img and st.button("バーコードを読み取る", key="cam_barcode"):
                    with st.spinner("バーコードを検出中..."):
                        barcode = read_barcode_from_image(img.getvalue())
                    if barcode:
                        st.info(f"検出したバーコード番号：{barcode}")
                        with st.spinner("商品名を検索中..."):
                            found_name = get_product_name_from_barcode(barcode)
                        if found_name:
                            prefill["name"] = found_name
                            st.session_state["prefill"] = prefill
                            st.session_state["show_barcode_cam"] = False
                            st.session_state["form_key"] += 1
                            st.session_state["toast_msg"] = f"商品名を読み取りました：{found_name}"
                            st.rerun()
                        else:
                            st.warning(
                                f"バーコード番号（{barcode}）は読み取れましたが、商品名が見つかりませんでした。手動で入力してください。"
                            )
                    else:
                        st.warning(
                            "バーコードを検出できませんでした。バーコードが画面全体に大きく、水平に、ピントが合った状態で写るように再撮影してください。"
                        )
            with bc2:
                if st.button("📷 再撮影", key="cam_reset"):
                    st.session_state["cam_key"] += 1
                    st.rerun()

        st.write("")

        # ── 2. 区分・金額 ──
        st.subheader("2. 区分・金額")
        current_type = prefill.get("food_type", "一般食品")
        if current_type not in FOOD_TYPES:
            current_type = "一般食品"
        type_col, price_col = st.columns(2)
        with type_col:
            food_type = st.selectbox("区分", FOOD_TYPES, index=FOOD_TYPES.index(current_type))
        with price_col:
            price = st.number_input(
                "購入金額（円）", min_value=0, value=int(prefill.get("price", 0) or 0)
            )

        st.write("")

        # ── 3. 賞味期限 ──
        st.subheader("3. 賞味期限")
        today = datetime.date.today()
        if food_type == "一般食品":
            default_date_str = (today + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            if prefill.get("expiry_date"):
                default_date_str = prefill["expiry_date"][:10]
            expiry_fmt = "%Y-%m-%d"
            expiry_hint = "YYYY-MM-DD"
            expiry_label = "賞味期限（YYYY-MM-DD）"
        else:
            default_date_str = (today + datetime.timedelta(days=30)).strftime("%Y-%m")
            if prefill.get("expiry_date"):
                default_date_str = prefill["expiry_date"][:7]
            expiry_fmt = "%Y-%m"
            expiry_hint = "YYYY-MM"
            expiry_label = "賞味期限（YYYY-MM）"

        expiry_col, expiry_cam_col = st.columns([5, 1])
        with expiry_col:
            # 区分を変えたときに書式が合うよう、key に区分を含めてリセットする
            expiry_date_str = st.text_input(
                expiry_label,
                value=default_date_str,
                placeholder=f"例：{default_date_str}",
                label_visibility="collapsed",
                key=f"expiry_{st.session_state['form_key']}_{food_type}",
            )
        with expiry_cam_col:
            if st.button("📷 撮影", key="open_expiry_cam"):
                st.session_state["show_expiry_cam"] = not st.session_state.get("show_expiry_cam", False)

        if st.session_state.get("show_expiry_cam"):
            expiry_img = st.camera_input(
                "賞味期限部分を撮影（数字がはっきり写るように近づけてください）",
                key=f"expiry_camera_{st.session_state['expiry_cam_key']}",
            )
            ec1, ec2 = st.columns(2)
            with ec1:
                if expiry_img and st.button("賞味期限を読み取る", key="read_expiry"):
                    with st.spinner("画像を解析中..."):
                        data = analyze_image_with_gemini(expiry_img.getvalue(), mime_type="image/jpeg")
                    if data and data.get("expiry_date"):
                        prefill["expiry_date"] = data["expiry_date"]
                        st.session_state["prefill"] = prefill
                        st.session_state["show_expiry_cam"] = False
                        st.session_state["form_key"] += 1
                        st.session_state["toast_msg"] = f"賞味期限を読み取りました：{data['expiry_date']}"
                        st.rerun()
                    elif data:
                        st.warning(
                            "画像は解析できましたが、賞味期限の日付が見つかりませんでした。日付部分がはっきり写るように近づけて再撮影してください。"
                        )
                    else:
                        st.warning("賞味期限を読み取れませんでした。手動で入力してください。")
            with ec2:
                if st.button("📷 再撮影", key="expiry_cam_reset"):
                    st.session_state["expiry_cam_key"] += 1
                    st.rerun()

        st.write("")

        # ── 音声入力 ──
        with st.expander("🎙 音声で一括入力"):
            st.write("例：「卵 200円 一般食品 5月20日」などと話してください。")
            audio = st.audio_input("音声を録音", key=f"audio_{st.session_state['form_key']}")
            if audio is not None:
                if st.button("音声から情報を解析", key="audio_analyze"):
                    with st.spinner("音声を解析中..."):
                        mime_type = getattr(audio, "type", None) or "audio/wav"
                        data = analyze_audio_with_gemini(audio.getvalue(), mime_type=mime_type)
                    if data:
                        st.session_state["prefill"] = data
                        st.session_state["form_key"] += 1
                        st.session_state["toast_msg"] = f"音声から読み取りました：{data.get('name', '')}"
                        st.rerun()

        # ── 画像アップロード ──
        with st.expander("🖼 画像アップロードで一括入力"):
            st.write("パッケージ画像をアップロードします。")
            up = st.file_uploader(
                "画像ファイルを選択",
                type=["png", "jpg", "jpeg"],
                key=f"upload_{st.session_state['form_key']}",
            )
            if up is not None:
                if st.button("画像から情報を解析", key="upload_analyze"):
                    with st.spinner("画像を解析中..."):
                        actual_mime = up.type or "image/jpeg"
                        data = analyze_image_with_gemini(up.read(), mime_type=actual_mime)
                    if data:
                        st.session_state["prefill"] = data
                        st.session_state["form_key"] += 1
                        st.session_state["toast_msg"] = (
                            f"解析結果：{data.get('name', '')} / {data.get('food_type', '')} / {data.get('expiry_date', '')}"
                        )
                        st.rerun()
                    else:
                        st.warning(
                            "画像から情報を読み取れませんでした。パッケージ全体・商品名・賞味期限表示がはっきり写った写真で試してください。"
                        )

        st.write("")

        # ── 登録ボタン（st.stop() は使わず、他のタブの描画を止めない） ──
        if st.button("✅ この内容で登録する", key="manual_register", type="primary"):
            error_msg = None
            if not name.strip():
                error_msg = "食材名を入力してください。"
            else:
                try:
                    datetime.datetime.strptime(expiry_date_str.strip(), expiry_fmt)
                except ValueError:
                    error_msg = f"賞味期限の形式が正しくありません。{expiry_hint} で入力してください。"

            if error_msg:
                st.error(error_msg)
            else:
                insert_item(
                    food_type=food_type,
                    name=name.strip(),
                    expiry_date=expiry_date_str.strip(),
                    price=price,
                    status=STATUS_STOCK,
                )
                st.session_state.pop("prefill", None)
                st.session_state["show_barcode_cam"] = False
                st.session_state["show_expiry_cam"] = False
                st.session_state["form_key"] += 1
                st.session_state["toast_msg"] = "在庫に登録しました"
                st.rerun()

    # ===== 在庫一覧タブ =====
    with tab_stock:
        st.header("在庫一覧")
        all_stock = fetch_items(status_filter=STATUS_STOCK)
        total_stock_price = sum(i["price"] for i in all_stock)
        st.metric("現在の在庫合計金額", f"{int(total_stock_price):,} 円")

        if "stock_sort" not in st.session_state:
            st.session_state["stock_sort"] = "id"
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            if st.button("登録順", key="sort_id", use_container_width=True):
                st.session_state["stock_sort"] = "id"
        with sort_col2:
            if st.button("賞味期限順", key="sort_expiry", use_container_width=True):
                st.session_state["stock_sort"] = "expiry"
        sort_label = "登録順" if st.session_state["stock_sort"] == "id" else "賞味期限順"
        st.caption(f"現在の並び順：{sort_label}")
        st.write("---")

        food_tab1, food_tab2, food_tab3 = st.tabs(FOOD_TYPES)

        def render_stock_table(food_type_label):
            items = fetch_items(status_filter=STATUS_STOCK, food_type_filter=food_type_label)
            if st.session_state.get("stock_sort") == "expiry":
                # 文字列ではなく「残り日数」で並べる（YYYY-MM と YYYY-MM-DD が混在しても正しい）
                items = sorted(items, key=sort_key_days)
            if not items:
                st.info("在庫はありません。")
                return

            for item in items:
                days_left = calc_days_left(item["expiry_date"])
                level = judge_threshold(item["food_type"], days_left)
                cls = level_css_class(level)
                edit_key = f"editing_{item['id']}"

                if st.session_state.get(edit_key):
                    with st.container():
                        st.markdown(f"**{item['name']}** を編集中")
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            new_name = st.text_input("食材名", value=item["name"], key=f"ename_{item['id']}")
                            cur = item["food_type"] if item["food_type"] in FOOD_TYPES else "一般食品"
                            new_type = st.selectbox(
                                "区分",
                                FOOD_TYPES,
                                index=FOOD_TYPES.index(cur),
                                key=f"etype_{item['id']}",
                            )
                        with ec2:
                            new_expiry = st.text_input(
                                "賞味期限（YYYY-MM-DD または YYYY-MM）",
                                value=item["expiry_date"],
                                key=f"eexpiry_{item['id']}",
                            )
                            new_price = st.number_input(
                                "金額（円）",
                                min_value=0,
                                value=int(item["price"]),
                                key=f"eprice_{item['id']}",
                            )
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            if st.button("保存", key=f"save_{item['id']}"):
                                expiry_clean = new_expiry.strip()
                                if not new_name.strip():
                                    st.error("食材名を入力してください。")
                                elif len(expiry_clean) not in (7, 10) or parse_expiry(expiry_clean) is None:
                                    # 登録時と同じく YYYY-MM-DD / YYYY-MM の両方を受け付ける
                                    st.error("賞味期限は YYYY-MM-DD または YYYY-MM 形式で入力してください。")
                                else:
                                    update_item(item["id"], new_name.strip(), new_type, expiry_clean, new_price)
                                    st.session_state[edit_key] = False
                                    st.rerun()
                        with sc2:
                            if st.button("キャンセル", key=f"cancel_{item['id']}"):
                                st.session_state[edit_key] = False
                                st.rerun()
                    st.write("---")

                else:
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 4])
                    with col1:
                        st.write(f"**{TYPE_EMOJI.get(item['food_type'], '🍽️')} {item['name']}**")
                        st.markdown(
                            f"<span class='badge {cls}'>あと {days_label(days_left)}</span>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.write(f"区分：{item['food_type']}")
                    with col3:
                        st.write(f"期限：{item['expiry_date']}")
                    with col4:
                        st.write(f"金額：{int(item['price'])} 円")
                    with col5:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if st.button("消費", key=f"consume_{item['id']}"):
                                update_status(item["id"], STATUS_CONSUMED)
                                st.rerun()
                        with c2:
                            if st.button("廃棄", key=f"discard_{item['id']}"):
                                update_status(item["id"], STATUS_DISCARDED)
                                st.rerun()
                        with c3:
                            if st.button("編集", key=f"edit_{item['id']}"):
                                st.session_state[edit_key] = True
                                st.rerun()
                    st.write("---")

        with food_tab1:
            render_stock_table("一般食品")
        with food_tab2:
            render_stock_table("防災食")
        with food_tab3:
            render_stock_table("調味料")

    # ===== レシピ提案タブ =====
    with tab_recipe:
        st.header("レシピ提案（使う食材を選択）")
        items = fetch_items(status_filter=STATUS_STOCK)
        if not items:
            st.info("在庫がありません。まずは食材を登録してください。")
        else:
            items_sorted = sorted(items, key=sort_key_days)

            def expiry_mark(item):
                level = judge_threshold(item["food_type"], calc_days_left(item["expiry_date"]))
                return {"赤": "🔴", "黄": "🟡"}.get(level, "")

            # ラベルが重複しても選択がずれないよう、ID付きの辞書で管理する
            label_to_item = {}
            for item in items_sorted:
                label = (
                    f"{expiry_mark(item)}[{item['food_type']}] {item['name']}"
                    f"（期限 {item['expiry_date']}）#{item['id']}"
                )
                label_to_item[label] = item

            selected = st.multiselect(
                "レシピに使いたい食材を選んでください", list(label_to_item.keys())
            )

            if st.button("レシピを生成する"):
                if not selected:
                    st.warning("少なくとも1つは食材を選んでください。")
                else:
                    selected_items = [label_to_item[l] for l in selected]
                    with st.spinner("レシピを生成中..."):
                        recipe_text = generate_recipe_with_gemini(selected_items)
                    if recipe_text:
                        st.subheader("提案レシピ")
                        st.write(recipe_text)

    # ===== 履歴・フードロス可視化タブ =====
    with tab_history:
        st.header("履歴・フードロス可視化")

        consumed = fetch_items(status_filter=STATUS_CONSUMED)
        discarded = fetch_items(status_filter=STATUS_DISCARDED)

        total_consumed = sum(i["price"] for i in consumed)
        total_discarded = sum(i["price"] for i in discarded)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("消費済み合計", f"{int(total_consumed):,} 円")
            st.write(f"件数：{len(consumed)} 件")
        with col2:
            st.metric("廃棄合計", f"{int(total_discarded):,} 円")
            st.write(f"件数：{len(discarded)} 件")

        all_history = consumed + discarded
        if all_history:
            df = pd.DataFrame(all_history)
            # 「消費・廃棄した月」で集計（closed_at が無い古いデータは期限の月で代用）
            df["month"] = df.apply(
                lambda r: (r["closed_at"] or r["expiry_date"])[:7], axis=1
            )

            st.write("---")
            st.subheader("月別：消費・廃棄の推移")
            df_trend = (
                df.pivot_table(
                    index="month", columns="status", values="price", aggfunc="sum", fill_value=0
                )
                .reindex(columns=[STATUS_CONSUMED, STATUS_DISCARDED], fill_value=0)
                .reset_index()
            )
            fig_line = px.line(
                df_trend,
                x="month",
                y=[STATUS_CONSUMED, STATUS_DISCARDED],
                color_discrete_map={STATUS_CONSUMED: "#5cc9a0", STATUS_DISCARDED: "#ff7a90"},
                markers=True,
            )
            fig_line.update_layout(
                yaxis=dict(rangemode="nonnegative"),
                xaxis_title="月",
                yaxis_title="金額（円）",
                legend_title_text="",
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("まだ消費・廃棄の履歴がありません。")

        st.write("---")
        st.subheader("履歴一覧")

        def render_history(title, items):
            st.markdown(f"**{title}**")
            if not items:
                st.write("なし")
                return
            for item in sorted(items, key=lambda x: x["closed_at"] or "", reverse=True):
                closed = f" / 記録日 {item['closed_at'][:10]}" if item["closed_at"] else ""
                st.write(
                    f"- [{item['food_type']}] {item['name']} / 期限 {item['expiry_date']}"
                    f" / 金額 {int(item['price'])} 円{closed}"
                )

        render_history("😊消費済み", consumed)
        render_history("😱廃棄", discarded)


if __name__ == "__main__":
    main()