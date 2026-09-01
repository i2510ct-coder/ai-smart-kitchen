import datetime
import cv2
import numpy as np
import json
import os
import sqlite3
import time
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ===== pyzbar は zbar 共有ライブラリが無いと import 自体に失敗することがあるので、
#       try/except で捕まえて分かりやすいメッセージを出せるようにする =====
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
    PYZBAR_IMPORT_ERROR = None
except Exception as e:
    PYZBAR_AVAILABLE = False
    PYZBAR_IMPORT_ERROR = str(e)

# ===== 設定 =====
DB_PATH = "inventory.db"
MODEL_TEXT = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RAKUTEN_APP_ID = os.getenv("RAKUTEN_APP_ID")


# ===== Gemini 構造化出力用のスキーマ定義 =====
class InventoryItemSchema(BaseModel):
    name: str = Field(description="食材名（日本語）")
    price: int = Field(description="数値（不明なら 0）")
    food_type: str = Field(
        description="'一般食品' または '防災食' または '調味料' のいずれか"
    )
    expiry_date: str = Field(
        description="'YYYY-MM-DD' 形式の賞味期限（日付が読めない・不明な場合は空文字）"
    )


# ===== DB 初期化 =====
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_type TEXT NOT NULL,
            name TEXT NOT NULL,
            expiry_date TEXT NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def get_conn():
    return sqlite3.connect(DB_PATH)


# ===== 共通ロジック =====
def calc_days_left(expiry_date_str: str) -> int:
    today = datetime.date.today()
    try:
        if len(expiry_date_str) == 7:  # YYYY-MM形式
            expiry = datetime.datetime.strptime(expiry_date_str + "-01", "%Y-%m-%d").date()
            import calendar
            last_day = calendar.monthrange(expiry.year, expiry.month)[1]
            expiry = expiry.replace(day=last_day)
        else:
            expiry = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
        return (expiry - today).days
    except Exception:
        return 0


def judge_threshold(food_type: str, days_left: int) -> str:
    if food_type == "一般食品":
        if days_left < 0:
            return "赤"
        elif days_left <= 2:
            return "黄"
        else:
            return "緑"
    elif food_type in ["防災食", "調味料"]:
        if days_left < 0:
            return "赤"
        elif days_left <= 30:
            return "黄"
        else:
            return "緑"
    else:
        if days_left < 0:
            return "赤"
        elif days_left <= 2:
            return "黄"
        else:
            return "緑"


def is_near_expiry(food_type: str, days_left: int) -> bool:
    if food_type == "一般食品":
        return days_left <= 2
    elif food_type in ["防災食", "調味料"]:
        return days_left <= 30
    return False


# ===== Gemini クライアント =====
def get_gemini_client():
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY が設定されていません。.env ファイルを確認してください。")
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


def analyze_image_with_gemini(image_bytes: bytes, mime_type: str = "image/jpeg"):
    """
    画像を解析して食材情報を抽出する。
    mime_type は呼び出し側で実際のファイル形式に合わせて渡すこと。
    st.camera_input() は JPEG を返すので "image/jpeg" のままでOK。
    st.file_uploader() は PNG の場合があるので up.type を渡す必要がある。
    """
    client = get_gemini_client()
    if client is None:
        return None

    prompt = f"""
画像に写っているパッケージや賞味期限の表示から、食材名・価格・区分・賞味期限を読み取ってください。
「5月20日」などの表現は、今年（現在{datetime.date.today().year}年）または来年の日付として YYYY-MM-DD に変換してください。
"""

    # mime_type が想定外の値だった場合の保険（例: "image/jpg" のような表記ゆれ対策）
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
        data = json.loads(res.text)
        return data
    except Exception as e:
        st.error(f"画像解析に失敗しました。手動で入力してください。エラー: {e}")
        return None


def read_barcode_from_image(image_bytes: bytes):
    """
    バーコードを画像から読み取る。
    zbar 共有ライブラリが無い環境ではここで分かりやすいエラーを出す。
    通常のデコードで失敗した場合、グレースケール化・拡大した画像でも
    もう一度試すことで検出率を上げる。
    """
    if not PYZBAR_AVAILABLE:
        st.error(
            "バーコード読み取り機能が使えません（zbar ライブラリが見つかりません）。\n\n"
            "Windows: pip install pyzbar で通常は同梱されますが、失敗する場合は "
            "Visual C++ 再頒布可能パッケージが必要です。\n"
            "Mac: `brew install zbar` を実行してください。\n"
            "Linux: `sudo apt-get install libzbar0` を実行してください。\n\n"
            f"詳細エラー: {PYZBAR_IMPORT_ERROR}"
        )
        return None

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        st.warning("画像の読み込みに失敗しました。もう一度撮影してください。")
        return None

    # 1回目：そのままの画像でデコード
    barcodes = pyzbar.decode(img)

    # 2回目：グレースケール化＋拡大して再挑戦（ピント甘め・小さいバーコード対策）
    if not barcodes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_large = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        barcodes = pyzbar.decode(gray_large)

    # 3回目：二値化して再挑戦
    if not barcodes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        barcodes = pyzbar.decode(thresh)

    if barcodes:
        return barcodes[0].data.decode("utf-8")
    return None


def get_product_name_from_barcode(barcode: str):
    """
    バーコード(JANコード)から商品名を推測する。
    楽天市場のキーワード検索APIはJANコード検索に強くないため、
    まず Open Food Facts (食品専用・無料・APIキー不要・JANコード直接検索対応) を試し、
    ヒットしなければ楽天にフォールバックする。
    """
    import requests

    # --- 1. Open Food Facts（JANコード直接検索。食品バーコードに強い） ---
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

    # --- 2. 楽天市場キーワード検索（フォールバック） ---
    if not RAKUTEN_APP_ID:
        return None
    url = "https://app.rakuten.co.jp/services/api/IchibaItem/Search/20170901"
    params = {
        "applicationId": RAKUTEN_APP_ID,
        "keyword": barcode,
        "hits": 1,
        "format": "json",
    }
    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        items = data.get("Items", [])
        if items:
            return items[0]["Item"]["itemName"]
    except Exception:
        pass
    return None


# 音声から「卵 200円 一般食品 2026-05-20」などを抽出
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
        data = json.loads(res.text)
        return data
    except Exception as e:
        st.error(f"音声解析に失敗しました。手動で入力してください。エラー: {e}")
        return None


# レシピ提案
def generate_recipe_with_gemini(selected_items):
    client = get_gemini_client()
    if client is None:
        return "Gemini API キーが設定されていません。"

    context_lines = []
    for item in selected_items:
        context_lines.append(
            f"- 食材名: {item['name']} / 区分: {item['food_type']} / 期限: {item['expiry_date']}"
        )
    context = "\n".join(context_lines)

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

    res = client.models.generate_content(model=MODEL_TEXT, contents=prompt)
    return res.text


# ===== DB 操作 =====
def insert_item(food_type, name, expiry_date, price, status="在庫"):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO inventory (food_type, name, expiry_date, price, status) VALUES (?, ?, ?, ?, ?)",
        (food_type, name, expiry_date, price, status),
    )
    conn.commit()
    conn.close()


def update_status(item_id, new_status):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE inventory SET status = ? WHERE id = ?", (new_status, item_id)
    )
    conn.commit()
    conn.close()


def update_item(item_id, name, food_type, expiry_date, price):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE inventory SET name = ?, food_type = ?, expiry_date = ?, price = ? WHERE id = ?",
        (name, food_type, expiry_date, price, item_id),
    )
    conn.commit()
    conn.close()


def fetch_items(status_filter=None, food_type_filter=None):
    conn = get_conn()
    c = conn.cursor()
    query = "SELECT id, food_type, name, expiry_date, price, status FROM inventory WHERE 1=1"
    params = []
    if status_filter:
        query += " AND status = ?"
        params.append(status_filter)
    if food_type_filter:
        query += " AND food_type = ?"
        params.append(food_type_filter)
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()

    items = []
    for r in rows:
        items.append(
            {
                "id": r[0],
                "food_type": r[1],
                "name": r[2],
                "expiry_date": r[3],
                "price": r[4],
                "status": r[5],
            }
        )
    return items


# ===== UI =====
def main():
    st.set_page_config(
        page_title="AIスマートキッチン",
        page_icon="🥕",
        layout="wide",
    )

    init_db()

    st.markdown(
        """
        <style>
        body { background-color: #f7f5f0; }
        .main-title { font-size: 32px; font-weight: 700; color: #0b3c5d; }
        .sub-title { font-size: 16px; color: #2c7873; }
        .danger { color: #b22222; font-weight: 600; }
        .warning { color: #ff8c00; font-weight: 600; }
        .safe { color: #2e8b57; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-title">AIスマートキッチン</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">家庭内のフードロスを「見える化」して、賢くおいしく使い切る。</div>',
        unsafe_allow_html=True,
    )

    if not GEMINI_API_KEY:
        st.error("⚠️ GEMINI_API_KEY が読み込まれていません。.env ファイルの場所と中身を確認してください（画像解析・音声解析・OCR・レシピ提案が全て動きません）。")
    if not PYZBAR_AVAILABLE:
        st.warning(f"⚠️ バーコード読み取り機能が無効です: {PYZBAR_IMPORT_ERROR}")

    st.write("---")

    all_items = fetch_items(status_filter="在庫")

    near_items = []
    risk_amount = 0
    for item in all_items:
        days_left = calc_days_left(item["expiry_date"])
        if is_near_expiry(item["food_type"], days_left):
            near_items.append((item, days_left))
            risk_amount += item["price"]

    import streamlit.components.v1 as components

    discarded_items = fetch_items(status_filter="廃棄")
    total_discarded_price = sum(i["price"] for i in discarded_items)

    tilt_score = 0
    for item in all_items:
        days_left = calc_days_left(item["expiry_date"])
        level = judge_threshold(item["food_type"], days_left)
        if level == "赤":
            tilt_score += item["price"] * 3
        elif level == "黄":
            tilt_score += item["price"] * 1
    tilt_score += total_discarded_price * 2
    angle = min(45, tilt_score / 500)

    near_names = [item["name"] for item, _ in near_items[:4]]
    safe_names = [item["name"] for item in all_items
                  if not is_near_expiry(item["food_type"], calc_days_left(item["expiry_date"]))][:4]

    balance_data = json.dumps({
        "angle": round(angle, 1),
        "risk": int(risk_amount),
        "discarded": int(total_discarded_price),
        "near_count": len(near_items),
        "near_names": near_names,
        "safe_names": safe_names,
        "shake": angle > 10,
    })

    home_col1, home_col2 = st.columns([1, 1])

    with home_col1:
        st.subheader("📢 期限接近アラート")
        if near_items:
            st.warning(
                f"期限が近い食材があります。想定廃棄リスク額：{int(risk_amount)} 円"
            )
            for item, days_left in near_items:
                level = judge_threshold(item["food_type"], days_left)
                cls = (
                    "danger"
                    if level == "赤"
                    else ("warning" if level == "黄" else "safe")
                )
                st.markdown(
                    f'<span class="{cls}">[{item["food_type"]}] {item["name"]}：あと {days_left} 日（期限 {item["expiry_date"]}）</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("期限が近い食材はありません。いい感じです！")

    with home_col2:
        st.subheader("⚖️ フードロス天秤")
        st.caption("期限切れ間近な食材が多いほど廃棄側に傾きます。")
        html_balance = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<style>
body{font-family:'Helvetica Neue',Arial,sans-serif;margin:0;padding:4px 8px;background:linear-gradient(135deg,#fff7ed,#fef3c7);border-radius:12px;}
@keyframes shake{0%{transform:rotate(var(--a))}25%{transform:rotate(calc(var(--a) + 5deg))}75%{transform:rotate(calc(var(--a) - 5deg))}100%{transform:rotate(var(--a))}}
.shaking{animation:shake 0.6s ease-in-out infinite;}
.wrap{position:relative;height:160px;display:flex;align-items:center;justify-content:center;}
.seesaw{width:380px;height:12px;background:linear-gradient(90deg,#334155,#64748b,#334155);border-radius:6px;position:relative;transition:transform 1.2s cubic-bezier(0.34,1.56,0.64,1);}
.seesaw::after{content:"";position:absolute;bottom:-22px;left:50%;width:22px;height:22px;background:#1e293b;transform:translateX(-50%);clip-path:polygon(50% 0%,0% 100%,100% 100%);}
.icon-left{position:absolute;left:0px;bottom:0;text-align:center;width:80px;}
.icon-right{position:absolute;right:0px;bottom:0;text-align:center;width:80px;}
.lbl{font-size:12px;font-weight:bold;margin-top:2px;}
.food-list{font-size:10px;margin-top:4px;line-height:1.6;}
.info-row{display:flex;justify-content:space-around;margin-top:12px;font-size:12px;background:rgba(255,255,255,0.6);border-radius:8px;padding:8px 4px;}
.info-item{text-align:center;}
.val{font-weight:bold;font-size:16px;}
.green{color:#16a34a;}.red{color:#dc2626;}.gray{color:#475569;}
</style></head><body>
<div class="wrap">
  <div class="icon-left"><div style="font-size:30px">💰</div><div class="lbl green">食べた！</div><div class="food-list green" id="safe-list"></div></div>
  <div id="ss" class="seesaw"></div>
  <div class="icon-right"><div style="font-size:30px">🗑️</div><div class="lbl red">廃棄</div><div class="food-list red" id="near-list"></div></div>
</div>
<div class="info-row">
  <div class="info-item"><div class="val red" id="risk">0</div><div>リスク額（円）</div></div>
  <div class="info-item"><div class="val red" id="disc">0</div><div>廃棄済み（円）</div></div>
  <div class="info-item"><div class="val gray" id="near">0件</div><div>期限接近中</div></div>
</div>
<script>
const d=""" + balance_data + """;
const ss=document.getElementById('ss');
ss.style.setProperty('--a',d.angle+'deg');
setTimeout(()=>{ss.style.transform='rotate('+d.angle+'deg)';if(d.shake){ss.classList.add('shaking');}},300);
document.getElementById('risk').innerText=d.risk.toLocaleString();
document.getElementById('disc').innerText=d.discarded.toLocaleString();
document.getElementById('near').innerText=d.near_count+'件';
document.getElementById('near-list').innerHTML=d.near_names.map(n=>'⚠️'+n).join('<br>');
document.getElementById('safe-list').innerHTML=d.safe_names.map(n=>'✅'+n).join('<br>');
</script></body></html>"""
        components.html(html_balance, height=260)

    st.write("---")

    tab_register, tab_stock, tab_recipe, tab_history = st.tabs(
        ["➕ 食材登録", "📦 在庫一覧", "🍳 レシピ提案", "📊 履歴・フードロス"]
    )

    # ===== 食材登録タブ =====
    with tab_register:
        st.header("食材登録")

        # 直前の読み取り結果をトースト表示（rerun後に一度だけ）
        if st.session_state.get("toast_msg"):
            st.toast(st.session_state.pop("toast_msg"), icon="✅")

        prefill = st.session_state.get("prefill", {})
        if "cam_key" not in st.session_state:
            st.session_state["cam_key"] = 0
        if "expiry_cam_key" not in st.session_state:
            st.session_state["expiry_cam_key"] = 0
        if "form_key" not in st.session_state:
            st.session_state["form_key"] = 0

        # ── 商品名欄 ──────────────────────────────
        st.subheader("1. 商品名")
        name_col, barcode_col = st.columns([5, 1])
        with name_col:
            name = st.text_input("食材名", value=prefill.get("name", ""), placeholder="例：卵", label_visibility="collapsed", key=f"name_{st.session_state['form_key']}")
        with barcode_col:
            if st.button("📷 バーコード", key="open_barcode_cam", disabled=not PYZBAR_AVAILABLE):
                st.session_state["show_barcode_cam"] = not st.session_state.get("show_barcode_cam", False)

        if st.session_state.get("show_barcode_cam"):
            img = st.camera_input("バーコードを撮影（できるだけ大きく・明るく・ピントを合わせて）", key=f"camera_{st.session_state['cam_key']}")
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
                            st.session_state["form_key"] += 1  # ウィジェットを作り直して新しい値を反映させる
                            st.session_state["toast_msg"] = f"商品名を読み取りました：{found_name}"
                            st.rerun()
                        else:
                            st.warning(
                                f"バーコード番号（{barcode}）は読み取れましたが、商品名が見つかりませんでした。手動で入力してください。"
                            )
                    else:
                        st.warning("バーコードを検出できませんでした。バーコードが画面全体に大きく、水平に、ピントが合った状態で写るように再撮影してください。")
            with bc2:
                if st.button("📷 再撮影", key="cam_reset"):
                    st.session_state["cam_key"] += 1
                    st.rerun()

        st.write("")

        # ── 区分・金額 ────────────────────────────
        st.subheader("2. 区分・金額")
        current_type = prefill.get("food_type", "一般食品")
        if current_type not in ["一般食品", "防災食", "調味料"]:
            current_type = "一般食品"
        type_col, price_col = st.columns(2)
        with type_col:
            food_type = st.selectbox(
                "区分",
                ["一般食品", "防災食", "調味料"],
                index=["一般食品", "防災食", "調味料"].index(current_type),
            )
        with price_col:
            price = st.number_input("購入金額（円）", min_value=0, value=int(prefill.get("price", 0)))

        st.write("")

        # ── 賞味期限 ──────────────────────────────
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
            expiry_date_str = st.text_input(expiry_label, value=default_date_str, placeholder=f"例：{default_date_str}", label_visibility="collapsed", key=f"expiry_{st.session_state['form_key']}")
        with expiry_cam_col:
            if st.button("📷 撮影", key="open_expiry_cam"):
                st.session_state["show_expiry_cam"] = not st.session_state.get("show_expiry_cam", False)

        if st.session_state.get("show_expiry_cam"):
            expiry_img = st.camera_input("賞味期限部分を撮影（数字がはっきり写るように近づけてください）", key=f"expiry_camera_{st.session_state['expiry_cam_key']}")
            ec1, ec2 = st.columns(2)
            with ec1:
                if expiry_img and st.button("賞味期限を読み取る", key="read_expiry"):
                    with st.spinner("画像を解析中..."):
                        # st.camera_input は JPEG を返すので mime_type はデフォルトのままでOK
                        data = analyze_image_with_gemini(expiry_img.getvalue(), mime_type="image/jpeg")
                    if data and data.get("expiry_date"):
                        prefill["expiry_date"] = data["expiry_date"]
                        st.session_state["prefill"] = prefill
                        st.session_state["show_expiry_cam"] = False
                        st.session_state["form_key"] += 1  # ウィジェットを作り直して新しい値を反映させる
                        st.session_state["toast_msg"] = f"賞味期限を読み取りました：{data['expiry_date']}"
                        st.rerun()
                    elif data:
                        st.warning("画像は解析できましたが、賞味期限の日付が見つかりませんでした。日付部分がはっきり写るように近づけて再撮影してください。")
                    else:
                        st.warning("賞味期限を読み取れませんでした。手動で入力してください。（上に表示されたエラー内容も確認してください）")
            with ec2:
                if st.button("📷 再撮影", key="expiry_cam_reset"):
                    st.session_state["expiry_cam_key"] += 1
                    st.rerun()

        st.write("")

        # ── 音声入力────────────────────────
        with st.expander("🎙 音声で一括入力"):
            st.write("例：「卵 200円 一般食品 5月20日」などと話してください。")
            audio = st.audio_input("音声を録音", key=f"audio_{st.session_state['form_key']}")
            if audio is not None:
                if st.button("音声から情報を解析", key="audio_analyze"):
                    with st.spinner("音声を解析中..."):
                        mime_type = getattr(audio, "type", "audio/wav")
                        data = analyze_audio_with_gemini(audio.getvalue(), mime_type=mime_type)
                    if data:
                        st.session_state["prefill"] = data
                        st.session_state["form_key"] += 1  # ウィジェットを作り直して新しい値を反映させる
                        st.session_state["toast_msg"] = f"音声から読み取りました：{data.get('name', '')}"
                        st.rerun()

        # ── 画像アップロード────────────────
        with st.expander("🖼 画像アップロードで一括入力"):
            st.write("レシートやパッケージ画像をアップロードします。")
            up = st.file_uploader("画像ファイルを選択", type=["png", "jpg", "jpeg"], key=f"upload_{st.session_state['form_key']}")
            if up is not None:
                if st.button("画像から情報を解析", key="upload_analyze"):
                    with st.spinner("画像を解析中..."):
                        # アップロードされたファイルの実際の形式(up.type)をそのまま渡す
                        # （固定で "image/jpeg" にしていたのが PNG アップロード時の失敗の原因だった）
                        actual_mime = up.type or "image/jpeg"
                        data = analyze_image_with_gemini(up.read(), mime_type=actual_mime)
                    if data:
                        st.session_state["prefill"] = data
                        st.session_state["form_key"] += 1  # ウィジェットを作り直して新しい値を反映させる
                        st.session_state["toast_msg"] = f"解析結果：{data.get('name', '')} / {data.get('food_type', '')} / {data.get('expiry_date', '')}"
                        st.rerun()
                    else:
                        st.warning("画像から情報を読み取れませんでした。パッケージ全体・商品名・賞味期限表示がはっきり写った写真で試してください。")

        st.write("")

        # ── 登録ボタン ────────────────────────────
        if st.button("✅ この内容で登録する", key="manual_register", type="primary"):
            if not name:
                st.error("食材名を入力してください。")
                st.stop()
            try:
                datetime.datetime.strptime(expiry_date_str, expiry_fmt)
            except ValueError:
                st.error(f"賞味期限の形式が正しくありません。{expiry_hint} で入力してください。")
                st.stop()
            insert_item(food_type=food_type, name=name, expiry_date=expiry_date_str, price=price, status="在庫")
            st.success("在庫に登録しました。")
            st.balloons()
            st.session_state.pop("prefill", None)
            st.session_state["show_barcode_cam"] = False
            st.session_state["show_expiry_cam"] = False
            st.session_state["form_key"] += 1  # 入力欄を空の状態にリセットする
            st.rerun()

    # ===== 在庫一覧タブ =====
    with tab_stock:
        st.header("在庫一覧")
        all_stock = fetch_items(status_filter="在庫")
        total_stock_price = sum(i["price"] for i in all_stock)
        st.metric("現在の在庫合計金額", f"{int(total_stock_price):,} 円")

        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            if st.button("登録順", key="sort_id", use_container_width=True):
                st.session_state["stock_sort"] = "id"
        with sort_col2:
            if st.button("賞味期限順", key="sort_expiry", use_container_width=True):
                st.session_state["stock_sort"] = "expiry"
        if "stock_sort" not in st.session_state:
            st.session_state["stock_sort"] = "id"
        sort_label = "登録順" if st.session_state["stock_sort"] == "id" else "賞味期限順"
        st.caption(f"現在の並び順：{sort_label}")
        st.write("---")
        food_tab1, food_tab2, food_tab3 = st.tabs(
            ["一般食品", "防災食", "調味料"]
        )

        def render_stock_table(food_type_label):
            items = fetch_items(
                status_filter="在庫", food_type_filter=food_type_label
            )
            if st.session_state.get("stock_sort") == "expiry":
                items = sorted(items, key=lambda x: x["expiry_date"])
            if not items:
                st.info("在庫はありません。")
                return

            for item in items:
                days_left = calc_days_left(item["expiry_date"])
                level = judge_threshold(item["food_type"], days_left)
                cls = (
                    "danger"
                    if level == "赤"
                    else ("warning" if level == "黄" else "safe")
                )

                edit_key = f"editing_{item['id']}"

                if st.session_state.get(edit_key):
                    with st.container():
                        st.markdown(f"**{item['name']}** を編集中")
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            new_name = st.text_input("食材名", value=item["name"], key=f"ename_{item['id']}")
                            current_type = item["food_type"] if item["food_type"] in ["一般食品", "防災食", "調味料"] else "一般食品"
                            new_type = st.selectbox("区分", ["一般食品", "防災食", "調味料"], index=["一般食品", "防災食", "調味料"].index(current_type), key=f"etype_{item['id']}")
                        with ec2:
                            new_expiry = st.text_input("賞味期限（YYYY-MM-DD）", value=item["expiry_date"], key=f"eexpiry_{item['id']}")
                            new_price = st.number_input("金額（円）", min_value=0, value=int(item["price"]), key=f"eprice_{item['id']}")
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            if st.button("保存", key=f"save_{item['id']}"):
                                try:
                                    datetime.datetime.strptime(new_expiry, "%Y-%m-%d")
                                    update_item(item["id"], new_name, new_type, new_expiry, new_price)
                                    st.session_state[edit_key] = False
                                    st.rerun()
                                except ValueError:
                                    st.error("賞味期限はYYYY-MM-DD形式で入力してください。")
                        with sc2:
                            if st.button("キャンセル", key=f"cancel_{item['id']}"):
                                st.session_state[edit_key] = False
                                st.rerun()
                    st.write("---")

                else:
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 4])
                    with col1:
                        st.write(f"**{item['name']}**")
                        st.markdown(
                            f"あと <span class='{cls}'>{days_left} 日</span>",
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
                                update_status(item["id"], "消費済み")
                                st.rerun()
                        with c2:
                            if st.button("廃棄", key=f"discard_{item['id']}"):
                                update_status(item["id"], "廃棄")
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
        items = fetch_items(status_filter="在庫")
        if not items:
            st.info("在庫がありません。まずは食材を登録してください。")
        else:
            items_sorted = sorted(
                items, key=lambda x: calc_days_left(x["expiry_date"])
            )
            def expiry_mark(item):
                days_left = calc_days_left(item["expiry_date"])
                level = judge_threshold(item["food_type"], days_left)
                if level == "赤":
                    return "🔴"
                elif level == "黄":
                    return "🟡"
                return ""

            options = [
                f"{expiry_mark(item)}[{item['food_type']}] {item['name']}（期限 {item['expiry_date']}）"
                for item in items_sorted
            ]
            selected = st.multiselect(
                "レシピに使いたい食材を選んでください", options
            )

            if st.button("レシピを生成する"):
                if not selected:
                    st.warning("少なくとも1つは食材を選んでください。")
                else:
                    selected_items = []
                    for label, item in zip(options, items_sorted):
                        if label in selected:
                            selected_items.append(item)
                    with st.spinner("レシピを生成中..."):
                        recipe_text = generate_recipe_with_gemini(
                            selected_items
                        )
                    st.subheader("提案レシピ")
                    st.write(recipe_text)

    # ===== 履歴・フードロス可視化タブ =====
    with tab_history:
        st.header("履歴・フードロス可視化")
        import pandas as pd
        import plotly.express as px

        consumed = fetch_items(status_filter="消費済み")
        discarded = fetch_items(status_filter="廃棄")

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
            df["month"] = df["expiry_date"].str[:7]

            st.write("---")
            st.subheader("月別：消費・廃棄の推移")
            df_trend = df.pivot_table(
                index="month", columns="status", values="price", aggfunc="sum", fill_value=0
            ).reindex(columns=["消費済み", "廃棄"], fill_value=0)
            df_trend_reset = df_trend.reset_index()
            fig_line = px.line(
                df_trend_reset, x="month",
                y=["消費済み", "廃棄"],
                color_discrete_map={"消費済み": "#2e8b57", "廃棄": "#b22222"},
                markers=True,
            )
            fig_line.update_layout(yaxis=dict(rangemode="nonnegative"), xaxis_title="月", yaxis_title="金額（円）")
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
            for item in items:
                st.write(
                    f"- [{item['food_type']}] {item['name']} / 期限 {item['expiry_date']} / 金額 {int(item['price'])} 円"
                )

        render_history("😊消費済み", consumed)
        render_history("😱廃棄", discarded)


if __name__ == "__main__":
    main()