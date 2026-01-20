import streamlit as st
import pandas as pd
import sqlite3
import datetime
import json
import io
import re
import hashlib
import altair as alt
import numpy as np

# --- 1. 常量配置 ---
DB_NAME = "fund_data.db"

PERCENT_COLUMNS = [
    "对应管理类费率合计", "管理费", "托管费", "服务费", "其他（如有）", 
    "业绩报酬计提比例", "申购费率", "赎回费率", "巨额赎回认定比例", 
    "行政服务费", "销售服务费", "投资顾问费", "申购费", "赎回费"
]

STANDARD_COLUMNS = [
    "产品名称", "策略", "产品类型", "开放日", "可买份额类型", 
    "申购费", "赎回费", "对应管理类费率合计", "管理费", "托管费", 
    "服务费", "其他（如有）", "业绩基准", "业绩报酬计提比例", 
    "管理人", "风险评级", "申购起点", "托管人", "锁定期", 
    "投资经理", "申购确认日", "赎回回款日期", "公司实控人", "基金备案编号"
]

MAPPING_LOGIC = {
    "产品全称": "产品名称", "产品类型": "产品类型", "策略": "策略",
    "管理人名称": "管理人", "产品风险类别": "风险评级",
    "首次认购/申购起点（不含认/申购费）": "申购起点", "开放日": "开放日",
    "封闭期": "锁定期", "申购费率": "申购费", "赎回费率": "赎回费",
    "对应管理类费率合计（管理费率+托管费率+服务费合计+其他如有）": "对应管理类费率合计",
    "管理费率": "管理费", "托管费率": "托管费", "服务费合计": "服务费",
    "其他如有":"其他（如有）", "业绩基准": "业绩基准",
    "业绩报酬计提比例": "业绩报酬计提比例", "可买份额类型": "可买份额类型",
    "申购确认日": "申购确认日", "赎回回款日期": "赎回回款日期",
    "投资经理": "投资经理", "公司实控人": "公司实控人",
    "基金备案编号": "基金备案编号", "托管人": "托管人"
}

# --- 2. 核心工具函数 ---
def hash_password(password):
    return hashlib.sha256(str(password).encode()).hexdigest()

def to_percent_str(val):
    if val is None or pd.isna(val) or str(val).strip() == "": return ""
    if "%" in str(val): return str(val).strip()
    try:
        num = float(val)
        res = "{:.10f}".format(num * 100).rstrip('0').rstrip('.')
        return f"{res}%" if res != "0" else "0%"
    except:
        return str(val)

def force_plain_str(val, is_percent=False):
    if val is None or pd.isna(val): return ""
    val_str = str(val).strip()
    if val_str.lower() in ['nan', 'none', '']: return ""
    if is_percent: return to_percent_str(val_str)
    if 'e' in val_str.lower():
        try: return "{:.10f}".format(float(val)).rstrip('0').rstrip('.')
        except: return val_str
    if val_str.endswith('.0'): return val_str[:-2]
    return val_str

def parse_to_float(val):
    if not val or str(val).lower() in ["none", "nan", ""]: return 0.0
    val_str = str(val)
    res = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
    if not res: return 0.0
    num = float(res[0])
    return num / 100 if "%" in val_str else num

# --- 3. 数据库管理 ---
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

def save_product_to_db(name, data_dict):
    conn = get_db_conn()
    clean_dict = {str(k): force_plain_str(v) for k, v in data_dict.items() if k}
    js = json.dumps(clean_dict, ensure_ascii=False)
    update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute('''INSERT OR REPLACE INTO product_info 
                 (p_name, p_manager, p_strategy, p_risk, p_all_data, p_update_time) 
                 VALUES (?,?,?,?,?,?)''', 
                 (str(name), clean_dict.get('管理人',''), clean_dict.get('策略',''), 
                  clean_dict.get('风险评级',''), js, update_time))
    conn.commit()

def get_standard_dataframe(db_rows):
    flat_list = []
    for row in db_rows:
        try:
            raw_dict = json.loads(row['p_all_data'])
            record = {col: force_plain_str(raw_dict.get(col, ""), is_percent=(col in PERCENT_COLUMNS)) for col in STANDARD_COLUMNS}
            m, t, s, o = [parse_to_float(record.get(k, 0)) for k in ["管理费", "托管费", "服务费", "其他（如有）"]]
            total = m + t + s + o
            record["对应管理类费率合计"] = force_plain_str(total, is_percent=True) if total > 0 else "0%"
            record["产品名称"] = record.get("产品名称") or row['p_name']
            flat_list.append(record)
        except: continue
    return pd.DataFrame(flat_list, columns=STANDARD_COLUMNS)

# --- 4. 净值专用函数 ---
def save_net_values(product_id, df):
    conn = get_db_conn()
    col_map = {}
    for c in df.columns:
        if "日期" in str(c) or "date" in str(c).lower(): col_map[c] = "nv_date"
        elif "单位" in str(c): col_map[c] = "unit_nv"
        elif "累计" in str(c): col_map[c] = "cum_nv"
    
    df = df.rename(columns=col_map)
    if "cum_nv" not in df.columns and "unit_nv" in df.columns:
        df["cum_nv"] = df["unit_nv"]
    
    if "nv_date" not in df.columns or "unit_nv" not in df.columns:
        return False, "缺少'日期'或'单位净值'列"

    count = 0
    for _, row in df.iterrows():
        try:
            d_str = pd.to_datetime(row['nv_date']).strftime("%Y-%m-%d")
            u_val = float(row['unit_nv'])
            c_val = float(row['cum_nv'])
            conn.execute("INSERT OR REPLACE INTO net_values (product_id, nv_date, unit_nv, cum_nv) VALUES (?,?,?,?)",
                         (product_id, d_str, u_val, c_val))
            count += 1
        except: continue
    conn.commit()
    return True, f"成功导入 {count} 条净值"

def get_net_values_df(product_id):
    conn = get_db_conn()
    df = pd.read_sql("SELECT nv_date as '日期', unit_nv as '单位净值', cum_nv as '累计净值' FROM net_values WHERE product_id=? ORDER BY nv_date ASC", conn, params=(product_id,))
    return df

# --- 5. 界面组件封装 ---

def ui_entry_tab():
    st.subheader("📤 录入与上传")
    if 'entry_df' not in st.session_state:
        st.session_state.entry_df = pd.DataFrame(columns=STANDARD_COLUMNS)
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        uploaded_files = st.file_uploader("上传要素表 (.xlsx)", type=["xlsx"], accept_multiple_files=True)
    with c2:
        st.write("")
        if st.button("➕ 手动增加一行", width='stretch'):
            new_row = pd.DataFrame([{col: "" for col in STANDARD_COLUMNS}])
            st.session_state.entry_df = pd.concat([st.session_state.entry_df, new_row], ignore_index=True)
    with c3:
        st.write("")
        if st.button("🧹 清空列表", width='stretch'):
            st.session_state.entry_df = pd.DataFrame(columns=STANDARD_COLUMNS)
            st.session_state.processed_files = []
            st.rerun()

    if uploaded_files:
        new_data_list = []
        for f in uploaded_files:
            if f.name not in st.session_state.processed_files:
                try:
                    try:
                        raw_xlsx = pd.read_excel(f, header=None, dtype=str).fillna("")
                    except:
                        f.seek(0)
                        raw_xlsx = pd.read_excel(f, header=None, dtype=str).fillna("")

                    parsed = pd.DataFrame()
                    if "项目" in str(raw_xlsx.iloc[0:15, 0].values):
                        keys = raw_xlsx[0].str.replace('*', '').str.replace('\n', '').str.strip()
                        val_col = 3 if raw_xlsx.shape[1] > 3 else raw_xlsx.shape[1]-1
                        data_dict = {k: force_plain_str(v) for k, v in zip(keys, raw_xlsx[val_col]) if k and k != "项目"}
                        parsed = pd.DataFrame([data_dict])
                    else:
                        f.seek(0)
                        parsed = pd.read_excel(f, dtype=str).fillna("").map(force_plain_str)
                        parsed = parsed.loc[:, ~parsed.columns.astype(str).str.contains('^Unnamed')]

                    for old_k, new_k in MAPPING_LOGIC.items():
                        if old_k in parsed.columns: parsed[new_k] = parsed[old_k]
                    
                    for col in STANDARD_COLUMNS:
                        if col not in parsed.columns: parsed[col] = ""
                    parsed = parsed[STANDARD_COLUMNS]

                    new_data_list.append(parsed)
                    st.session_state.processed_files.append(f.name)
                except Exception as e:
                    st.error(f"文件 {f.name} 读取失败: {e}")

        if new_data_list:
            combined_new = pd.concat(new_data_list, ignore_index=True)
            st.session_state.entry_df = pd.concat([st.session_state.entry_df, combined_new], ignore_index=True).fillna("")
            st.success(f"成功导入 {len(combined_new)} 条新记录")
            st.rerun()

    if not st.session_state.entry_df.empty:
        st.info("💡 提示：您可以直接在下方表格修改数据，确认无误后点击“确认同步”。")
        
        edited_df = st.data_editor(
            st.session_state.entry_df, 
            num_rows="dynamic", 
            key="editor_main", 
            width='stretch' 
        )
        
        if not edited_df.equals(st.session_state.entry_df):
            st.session_state.entry_df = edited_df

        if st.button("🚀 确认同步至数据库", width='stretch'):
            count = 0
            for _, row in edited_df.iterrows():
                name = row.get('产品名称') or row.get('产品全称')
                if name and str(name).strip() != "":
                    save_product_to_db(name, row.to_dict())
                    count += 1
            st.success(f"成功同步 {count} 条数据")
            st.session_state.entry_df = pd.DataFrame(columns=STANDARD_COLUMNS)
            st.session_state.processed_files = [] 
            st.rerun()

def ui_card_edit_tab():
    """产品卡片编辑 (集成净值模块 - 增强版)"""
    st.subheader("🔍 产品卡片管理")
    conn = get_db_conn()
    db_df = pd.read_sql("SELECT * FROM product_info ORDER BY p_update_time DESC", conn)
    s_key = st.text_input("搜索产品名称...", placeholder="输入名称或策略进行过滤")
    f_df = db_df[db_df['p_name'].str.contains(s_key)] if s_key else db_df

    if f_df.empty:
        st.caption("没有找到相关产品")
    
    for _, r in f_df.iterrows():
        with st.expander(f"📦 {r['p_name']} (更新: {r['p_update_time'].split(' ')[0]})"):

            # --- 分区 1: 净值与走势管理 (独立盒子) ---
            st.markdown("##### 📈 净值与走势管理")
            
            with st.container(border=True):
                # 1.1 净值导入区
                c_imp, c_info = st.columns([1, 2])
                with c_imp:
                    nv_file = st.file_uploader(f"导入净值序列 ({r['p_name']})", type=["xlsx", "csv"], key=f"nv_up_{r['id']}")
                    if nv_file:
                        try:
                            if nv_file.name.endswith('.csv'):
                                df_nv = pd.read_csv(nv_file)
                            else:
                                df_nv = pd.read_excel(nv_file)
                            
                            ok, msg = save_net_values(r['id'], df_nv)
                            if ok: st.success(msg)
                            else: st.error(msg)
                        except Exception as e:
                            st.error(f"解析失败: {e}")
                
                with c_info:
                     st.info("支持 .xlsx/.csv 格式，需包含'日期'、'单位净值'、'累计净值'列。")

                # 1.2 净值展示与分析区
                nv_df = get_net_values_df(r['id'])
                
                if not nv_df.empty:
                    # 确保日期格式
                    nv_df["日期"] = pd.to_datetime(nv_df["日期"])
                    
                    # 导航栏 (Tabs)
                    tab_chart, tab_table = st.tabs(["📊 收益走势分析", "📋 历史净值表"])
                    
                    with tab_chart:
                        # --- 恢复：最新状态展示 (始终显示最新一条数据) ---
                        last_row = nv_df.iloc[-1]
                        st.caption("🔹 最新状态 (截止数据末尾)")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("最新净值日期", last_row['日期'].strftime('%Y-%m-%d'))
                        m2.metric("最新单位净值", f"{last_row['单位净值']:.4f}")
                        m3.metric("最新累计净值", f"{last_row['累计净值']:.4f}")
                        st.divider()

                        # --- A. 区间筛选器 (实现横轴缩放与区间计算) ---
                        min_date = nv_df["日期"].min().date()
                        max_date = nv_df["日期"].max().date()
                        
                        st.markdown("###### 📅 分析区间选择")
                        
                        # Fix: 增加 key 参数，避免 StreamlitDuplicateElementId 错误
                        date_range = st.slider(
                            "拖动滑块选择时间段",
                            min_value=min_date,
                            max_value=max_date,
                            value=(min_date, max_date),
                            format="YYYY-MM-DD",
                            label_visibility="collapsed",
                            key=f"date_slider_{r['id']}" # <--- 关键修改
                        )
                        
                        # 根据滑块筛选数据
                        mask = (nv_df["日期"].dt.date >= date_range[0]) & (nv_df["日期"].dt.date <= date_range[1])
                        filtered_df = nv_df.loc[mask].sort_values("日期")

                        if len(filtered_df) > 1:
                            # --- B. 核心指标计算 ---
                            # 1. 基础数据准备
                            start_nav = filtered_df["累计净值"].iloc[0]
                            end_nav = filtered_df["累计净值"].iloc[-1]
                            days_span = (filtered_df["日期"].iloc[-1] - filtered_df["日期"].iloc[0]).days
                            
                            # 2. 收益率计算
                            interval_ret = (end_nav / start_nav) - 1 # 区间收益
                            
                            # 年化收益 (复利公式)
                            if days_span > 0:
                                annual_ret = (1 + interval_ret) ** (365 / days_span) - 1
                            else:
                                annual_ret = 0

                            # 3. 波动率计算 (自动推断数据频率)
                            # 计算平均间隔天数
                            if len(filtered_df) > 2:
                                avg_diff = filtered_df["日期"].diff().dt.days.mean()
                            else:
                                avg_diff = 1 # 默认值

                            if avg_diff <= 2: freq = 252       # 日频
                            elif avg_diff <= 10: freq = 52     # 周频
                            else: freq = 12                    # 月频
                            
                            pct_change = filtered_df["累计净值"].pct_change().dropna()
                            volatility = pct_change.std() * np.sqrt(freq)

                            # 4. 最大回撤
                            # 累计最大值序列
                            roll_max = filtered_df["累计净值"].cummax()
                            drawdown = (filtered_df["累计净值"] - roll_max) / roll_max
                            max_drawdown = drawdown.min()

                            # 5. 比率计算 (假设无风险利率为 0)
                            sharpe = annual_ret / volatility if volatility != 0 and not pd.isna(volatility) else 0
                            calmar = annual_ret / abs(max_drawdown) if max_drawdown != 0 else 0

                            # --- C. 指标展示 (两行布局) ---
                            st.caption(f"🔹 区间分析 ({date_range[0]} 至 {date_range[1]})")
                            k1, k2, k3, k4, k5, k6 = st.columns(6)
                            k1.metric("区间收益", f"{interval_ret:.2%}", help="期末累计净值 / 期初累计净值 - 1")
                            k2.metric("年化收益率", f"{annual_ret:.2%}", help="((1+区间收益)^(365/天数) - 1)")
                            k3.metric("年化波动率", f"{volatility:.2%}", help=f"收益率标准差 * sqrt({freq})")
                            k4.metric("最大回撤", f"{max_drawdown:.2%}", help="区间内最大跌幅")
                            k5.metric("夏普比率", f"{sharpe:.2f}", help="年化收益 / 年化波动")
                            k6.metric("卡玛比率", f"{calmar:.2f}", help="年化收益 / 最大回撤")
                            
                            st.divider()

                            # --- D. 绘图 (Altair) ---
                            # 红色渐变背景
                            gradient = alt.Gradient(
                                gradient='linear',
                                stops=[alt.GradientStop(color='rgba(255, 0, 0, 0.5)', offset=0), 
                                       alt.GradientStop(color='rgba(255, 255, 255, 0)', offset=1)],
                                x1=1, x2=1, y1=0, y2=1
                            )

                            base = alt.Chart(filtered_df).encode(
                                x=alt.X('日期:T', axis=alt.Axis(title=None, format='%Y-%m-%d'))
                            )

                            line = base.mark_line(color='#d62728', strokeWidth=3).encode(
                                y=alt.Y('累计净值:Q', scale=alt.Scale(zero=False), axis=alt.Axis(title='累计净值'))
                            )

                            area = base.mark_area(opacity=0.5).encode(
                                y='累计净值:Q',
                                color=alt.value(gradient)
                            )
                            
                            # 组合图表
                            chart = (area + line).properties(height=400).interactive()
                            st.altair_chart(chart, use_container_width=True)

                        else:
                            st.warning("所选区间数据不足（至少需要2个数据点），请扩大选择范围。")

                    with tab_table:
                        st.dataframe(nv_df, width='stretch', height=300)
                else:
                    st.info("暂无历史净值数据，请先在左侧上传 Excel 或 CSV 文件。")
            
            # --- 分区 2: 基础要素编辑 ---
            st.markdown("##### 📝 基础要素信息")
            raw_data = json.loads(r['p_all_data'])
            display_df = pd.DataFrame([[k, force_plain_str(v, (k in PERCENT_COLUMNS))] for k, v in raw_data.items()], columns=["项", "值"])
            new_details = st.data_editor(display_df, num_rows="dynamic", key=f"c_{r['id']}", width='stretch')
            
            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("💾 保存要素修改", key=f"s_{r['id']}"):
                    save_product_to_db(r['p_name'], dict(new_details.values))
                    st.success("已更新"); st.rerun()
            with c2:
                if st.button("🗑️ 删除产品", key=f"d_{r['id']}"):
                    conn.execute("DELETE FROM product_info WHERE id=?", (r['id'],)); conn.commit(); st.rerun()
            
            st.divider()

def ui_standard_table_tab():
    """标准视图 (管理员)"""
    st.subheader("📊 在库标准表 (全量管理)")
    conn = get_db_conn()
    rows = conn.execute("SELECT * FROM product_info").fetchall()
    std_df = get_standard_dataframe(rows)
    
    st.info("提示：此视图下的修改会直接覆盖数据库。")
    edited_std = st.data_editor(std_df, num_rows="dynamic", width='stretch', key="std_admin")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("📝 提交表格更改", width='stretch'):
            for _, row in edited_std.iterrows():
                if row['产品名称']: save_product_to_db(row['产品名称'], row.to_dict())
            st.success("同步成功"); st.rerun()
    with c2:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            edited_std.to_excel(writer, index=False)
        st.download_button("📥 导出Excel", output.getvalue(), file_name=f"磐松数据_{datetime.date.today()}.xlsx", width='stretch')

def ui_user_management_tab():
    """用户管理 (管理员)"""
    st.subheader("👥 团队账号管理")
    users_df = get_all_users()
    st.dataframe(users_df, width='stretch') 
    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ➕ 新增成员")
        with st.form("add_user_form"):
            new_user = st.text_input("用户名")
            new_pass = st.text_input("初始密码", type="password")
            new_role = st.selectbox("角色", ["staff", "admin"], help="admin: 全权; staff: 仅录入/查看")
            if st.form_submit_button("创建", width='stretch'):
                if new_user and new_pass:
                    ok, msg = add_user(new_user, new_pass, new_role)
                    if ok: st.success("创建成功"); st.rerun()
                    else: st.error(msg)
    with c2:
        st.markdown("##### ❌ 删除成员")
        target = st.selectbox("选择账号", users_df['username'].tolist())
        if st.button("删除账号", type="primary"):
            if target == 'admin': st.error("无法删除超级管理员")
            elif target == st.session_state.get('username'): st.error("不能删除自己")
            else:
                delete_user(target)
                st.success("已删除"); st.rerun()