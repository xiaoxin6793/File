# modules/database.py
import sqlite3
import datetime
import json
import pandas as pd
import streamlit as st
from .config import DB_NAME, STANDARD_COLUMNS, PERCENT_COLUMNS, MAPPING_LOGIC
from .utils import hash_password, force_plain_str, parse_to_float, clean_percent_to_float

# 缓存数据库连接
@st.cache_resource
def get_db_conn():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_conn()
    conn.execute('''CREATE TABLE IF NOT EXISTS product_info 
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   p_name TEXT UNIQUE, p_manager TEXT, p_strategy TEXT, p_risk TEXT,
                   p_all_data TEXT, p_update_time TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS users 
                  (username TEXT PRIMARY KEY, password TEXT, role TEXT, created_at TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS net_values 
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   product_id INTEGER,
                   nv_date TEXT,
                   unit_nv REAL,
                   cum_nv REAL,
                   UNIQUE(product_id, nv_date))''')
    
    cursor = conn.execute("SELECT count(*) FROM users")
    if cursor.fetchone()[0] == 0:
        conn.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                     ("admin", hash_password("888888"), "admin", datetime.datetime.now().strftime("%Y-%m-%d")))
        conn.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                     ("staff", hash_password("123456"), "staff", datetime.datetime.now().strftime("%Y-%m-%d")))
    conn.commit()

def check_login(username, password):
    conn = get_db_conn()
    hashed_pw = hash_password(password)
    cursor = conn.execute("SELECT role FROM users WHERE username=? AND password=?", (username, hashed_pw))
    row = cursor.fetchone()
    return row['role'] if row else None

def add_user(username, password, role):
    conn = get_db_conn()
    try:
        conn.execute("INSERT INTO users VALUES (?, ?, ?, ?)", 
                     (username, hash_password(password), role, datetime.datetime.now().strftime("%Y-%m-%d")))
        conn.commit()
        return True, "成功"
    except sqlite3.IntegrityError:
        return False, "用户名已存在"
    except Exception as e:
        return False, str(e)

def delete_user(username):
    if username == 'admin': return False
    conn = get_db_conn()
    conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    return True

def get_all_users():
    conn = get_db_conn()
    return pd.read_sql("SELECT username, role, created_at FROM users", conn)

# --- 核心修复：保存前强制重算费率合计 ---
def save_product_to_db(name, data_dict):
    conn = get_db_conn()
    
    # 1. 基础清洗
    clean_dict = {str(k): force_plain_str(v) for k, v in data_dict.items() if k}
    
    # 2. 【修复点】强制重新计算 "对应管理类费率合计" 并写入
    # 逻辑：无论字典里原来的合计是多少，都以分项之和为准，确保全量视图修改分项后，合计能自动更新
    try:
        # 兼容两种字段名（标准名 or 别名）
        m = clean_percent_to_float(clean_dict.get('管理费') or clean_dict.get('管理费率'))
        t = clean_percent_to_float(clean_dict.get('托管费') or clean_dict.get('托管费率'))
        # 服务费优先取合计，没有则取细项
        s = clean_percent_to_float(clean_dict.get('服务费') or clean_dict.get('服务费合计') or clean_dict.get('销售服务费'))
        o = clean_percent_to_float(clean_dict.get('其他（如有）') or clean_dict.get('其他如有'))
        
        total_fee = m + t + s + o
        if total_fee > 0:
            # 强制覆盖/写入合计字段
            clean_dict['对应管理类费率合计'] = f"{total_fee:.2f}%"
            # 同时更新长别名，防止数据不一致
            clean_dict['对应管理类费率合计（管理费率+托管费率+服务费合计+其他如有）'] = f"{total_fee:.2f}%"
    except Exception as e:
        print(f"Fee Recalc Error during save: {e}")

    # 3. 序列化并保存
    js = json.dumps(clean_dict, ensure_ascii=False)
    update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    manager = clean_dict.get('管理人') or clean_dict.get('管理人名称', '')
    strategy = clean_dict.get('策略', '')
    risk = clean_dict.get('风险评级') or clean_dict.get('产品风险类别', '')

    conn.execute('''INSERT OR REPLACE INTO product_info 
                 (p_name, p_manager, p_strategy, p_risk, p_all_data, p_update_time) 
                 VALUES (?,?,?,?,?,?)''', 
                 (str(name), manager, strategy, risk, js, update_time))
    conn.commit()

# --- 获取标准表格数据 (映射 + 计算 + 百分比) ---
def get_standard_dataframe(db_rows):
    flat_list = []
    for row in db_rows:
        try:
            if not row['p_all_data']: continue
            raw_dict = json.loads(row['p_all_data'])
            
            # 1. 临时字典，用于处理映射
            processed = raw_dict.copy()
            for alias, standard in MAPPING_LOGIC.items():
                if alias in processed and processed[alias]:
                    if not processed.get(standard):
                        processed[standard] = processed[alias]

            # 2. 提取标准列
            record = {}
            record['id'] = row['id'] 
            record['产品名称'] = row['p_name'] 

            for col in STANDARD_COLUMNS:
                if col == "产品名称": continue
                val = processed.get(col, "")
                is_pct = (col in PERCENT_COLUMNS)
                record[col] = force_plain_str(val, is_percent=is_pct)
            
            flat_list.append(record)
        except Exception as e:
            continue
            
    if not flat_list:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    
    df = pd.DataFrame(flat_list)
    df = df.reindex(columns=STANDARD_COLUMNS).fillna("")
    return df

def save_net_values(product_id, df):
    conn = get_db_conn()
    col_map = {}
    for c in df.columns:
        c_str = str(c).strip()
        if "日期" in c_str or "date" in c_str.lower(): col_map[c] = "nv_date"
        elif "单位" in c_str: col_map[c] = "unit_nv"
        elif "累计" in c_str: col_map[c] = "cum_nv"
    
    df = df.rename(columns=col_map)
    if "cum_nv" not in df.columns and "unit_nv" in df.columns:
        df["cum_nv"] = df["unit_nv"]
    
    if "nv_date" not in df.columns or "unit_nv" not in df.columns:
        return False, "缺少'日期'或'单位净值'列"

    count = 0
    try:
        data_to_insert = []
        for _, row in df.iterrows():
            try:
                d_str = pd.to_datetime(row['nv_date']).strftime("%Y-%m-%d")
                u_val = float(row['unit_nv'])
                c_val = float(row['cum_nv'])
                data_to_insert.append((product_id, d_str, u_val, c_val))
            except: continue
            
        if data_to_insert:
            conn.executemany("INSERT OR REPLACE INTO net_values (product_id, nv_date, unit_nv, cum_nv) VALUES (?,?,?,?)", data_to_insert)
            conn.commit()
            count = len(data_to_insert)
            return True, f"成功导入 {count} 条净值"
        else:
            return False, "无有效数据"
    except Exception as e:
        return False, str(e)

def get_net_values_df(product_id):
    conn = get_db_conn()
    df = pd.read_sql("SELECT nv_date as '日期', unit_nv as '单位净值', cum_nv as '累计净值' FROM net_values WHERE product_id=? ORDER BY nv_date ASC", conn, params=(product_id,))
    return df