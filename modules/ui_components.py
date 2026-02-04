# modules/ui_components.py
import streamlit as st
import pandas as pd
import json
import io
import re
import altair as alt
import numpy as np
import datetime
from .config import STANDARD_COLUMNS, MAPPING_LOGIC, PERCENT_COLUMNS, FULL_TEMPLATE_COLUMNS
from .utils import force_plain_str, to_percent_str, is_percent_col
from .database import (
    save_product_to_db, get_db_conn, get_net_values_df, 
    save_net_values, get_standard_dataframe, get_all_users, 
    add_user, delete_user
)

# --- 尝试导入排序库 ---
try:
    from streamlit_sortables import sort_items
except ImportError:
    sort_items = None

# --- 辅助函数：确保数据是 DataFrame ---
def ensure_dataframe(data, columns=None):
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(columns=columns if columns else [])

def ui_entry_tab():
    """数据录入页面 (稳定无冲突版)"""
    st.subheader("📤 录入与上传")
    
    # 1. 初始化
    if 'entry_df' not in st.session_state:
        st.session_state.entry_df = pd.DataFrame(columns=FULL_TEMPLATE_COLUMNS)
    else:
        st.session_state.entry_df = ensure_dataframe(st.session_state.entry_df, FULL_TEMPLATE_COLUMNS)
        
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'editor_version' not in st.session_state:
        st.session_state.editor_version = 0

    # 2. 启动时同步
    current_key = f"editor_main_{st.session_state.editor_version}"
    if current_key in st.session_state:
        prev_data = st.session_state[current_key]
        if isinstance(prev_data, pd.DataFrame):
            st.session_state.entry_df = prev_data

    # 3. 界面布局
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        uploaded_files = st.file_uploader("上传要素表", type=["xlsx", "csv"], accept_multiple_files=True)
    with c2:
        st.write("")
        st.write("")
        st.info("💡 提示：点击下方表格底部的 ➕ 号可直接添加新行")
    with c3:
        st.write("")
        st.write("")
        if st.button("🧹 清空列表", width='stretch'):
            st.session_state.entry_df = pd.DataFrame(columns=FULL_TEMPLATE_COLUMNS)
            st.session_state.processed_files = []
            st.session_state.editor_version += 1
            st.rerun()

    # 4. 文件解析逻辑
    if uploaded_files:
        new_data_list = []
        for f in uploaded_files:
            if f.name not in st.session_state.processed_files:
                try:
                    parsed = pd.DataFrame()
                    if f.name.lower().endswith('.csv'):
                        try:
                            try: raw_df = pd.read_csv(f, header=None, dtype=str).fillna("")
                            except: f.seek(0); raw_df = pd.read_csv(f, header=None, dtype=str, encoding='gbk').fillna("")
                        except Exception as e: st.error(f"CSV失败: {e}"); continue
                    else:
                        try: raw_df = pd.read_excel(f, header=None, dtype=str).fillna("")
                        except: f.seek(0); raw_df = pd.read_excel(f, header=None, dtype=str).fillna("")

                    ordered_columns = []
                    if "项目" in str(raw_df.iloc[0:15, 0].values):
                        keys = raw_df[0].str.replace('*', '').str.replace('\n', '').str.strip()
                        val_col = 3 if raw_df.shape[1] > 3 else raw_df.shape[1]-1
                        data_dict = {}
                        key_counter = {}
                        for k, v in zip(keys, raw_df[val_col]):
                            k_str = str(k).strip()
                            if k_str and k_str.lower() != 'nan' and k_str != "项目":
                                if k_str in key_counter: key_counter[k_str] += 1; unique_key = f"{k_str}_{key_counter[k_str]}"
                                else: key_counter[k_str] = 1; unique_key = k_str
                                data_dict[unique_key] = force_plain_str(v, is_percent=is_percent_col(k_str))
                                ordered_columns.append(unique_key)
                        parsed = pd.DataFrame([data_dict])[ordered_columns]
                    else:
                        f.seek(0)
                        if f.name.lower().endswith('.csv'): parsed = pd.read_csv(f, dtype=str).fillna("").map(force_plain_str)
                        else: parsed = pd.read_excel(f, dtype=str).fillna("").map(force_plain_str)
                        parsed = parsed.loc[:, ~parsed.columns.astype(str).str.contains('^Unnamed')]
                        for col in parsed.columns:
                            if is_percent_col(col): parsed[col] = parsed[col].apply(to_percent_str)
                        ordered_columns = parsed.columns.tolist()

                    for old, new in MAPPING_LOGIC.items():
                        if old in parsed.columns: parsed[new] = parsed[old]
                    for col in FULL_TEMPLATE_COLUMNS:
                        if col not in parsed.columns: parsed[col] = ""
                    
                    final_order = []
                    seen = set()
                    for k in ordered_columns + list(parsed.columns):
                        if k in parsed.columns and k not in seen:
                            final_order.append(k); seen.add(k)
                    parsed = parsed[final_order]
                    new_data_list.append(parsed)
                    st.session_state.processed_files.append(f.name)
                except Exception as e: st.error(f"解析失败: {e}")

        if new_data_list:
            combined = pd.concat(new_data_list, ignore_index=True)
            curr = ensure_dataframe(st.session_state.entry_df, FULL_TEMPLATE_COLUMNS)
            st.session_state.entry_df = combined if curr.empty else pd.concat([curr, combined], ignore_index=True).fillna("")
            st.session_state.editor_version += 1
            st.success(f"已导入 {len(combined)} 条"); st.rerun()

    # 5. 编辑器配置
    st.info("💡 提示：含'费'或'率'字段自动转百分比；重复字段显示时隐藏后缀。")
    my_config = {}
    df_show = ensure_dataframe(st.session_state.entry_df, FULL_TEMPLATE_COLUMNS)
    for col in df_show.columns:
        if re.search(r'_\d+$', str(col)):
            my_config[col] = st.column_config.TextColumn(label=re.sub(r'_\d+$', '', str(col)), width="medium")
    
    edited_df = st.data_editor(
        df_show, 
        num_rows="dynamic", 
        key=f"editor_main_{st.session_state.editor_version}", 
        width='stretch', 
        column_config=my_config
    )

    if st.button("🚀 确认同步至数据库", width='stretch'):
        c = 0
        for _, r in edited_df.iterrows():
            if (r.get('产品名称') or r.get('产品全称')) and str(r.get('产品名称') or r.get('产品全称')).strip():
                save_product_to_db(r.get('产品名称') or r.get('产品全称'), r.to_dict()); c += 1
        st.success(f"成功同步 {c} 条"); 
        st.session_state.entry_df = pd.DataFrame(columns=FULL_TEMPLATE_COLUMNS)
        st.session_state.processed_files = []
        st.session_state.editor_version += 1; st.rerun()

def ui_card_edit_tab():
    """产品卡片管理 (含折叠侧边栏 + 红色图表 + 单产品净值保护)"""
    
    st.subheader("🔍 产品卡片管理")
    conn = get_db_conn()
    db_df = pd.read_sql("SELECT * FROM product_info ORDER BY p_update_time DESC", conn)
    
    # 筛选区
    c_search, c_filter1, c_filter2 = st.columns([2, 1, 1])
    with c_search: s_key = st.text_input("搜索...", placeholder="输入名称/代码")
    all_strategies = [x for x in db_df['p_strategy'].unique() if x]
    all_risks = [x for x in db_df['p_risk'].unique() if x]
    with c_filter1: sel_strategies = st.multiselect("策略", all_strategies)
    with c_filter2: sel_risks = st.multiselect("风险", all_risks)

    f_df = db_df.copy()
    if s_key: f_df = f_df[f_df['p_name'].str.contains(s_key, case=False, regex=False) | f_df['p_all_data'].str.contains(s_key, case=False, regex=False)]
    if sel_strategies: f_df = f_df[f_df['p_strategy'].isin(sel_strategies)]
    if sel_risks: f_df = f_df[f_df['p_risk'].isin(sel_risks)]

    if f_df.empty: st.warning("无符合条件产品"); return

    # --- 侧边栏逻辑 ---
    with st.sidebar:
        nav_container = st.container()
        sort_container = st.container()

        if sort_items:
            # --- 下方：拖拽排序 (保留折叠框！) ---
            with sort_container:
                st.markdown("---") 
                with st.expander("⇅ 调整排序", expanded=False):
                    sorted_names = sort_items(f_df['p_name'].tolist(), direction='vertical')
            
            # 应用排序结果
            f_df['p_name'] = pd.Categorical(f_df['p_name'], categories=sorted_names, ordered=True)
            f_df = f_df.sort_values('p_name')
            
            # --- 2. 上方：快速跳转 ---
            with nav_container:
                st.markdown("### 🚀 快速导航")
                nav_md = ""
                for _, r in f_df.iterrows():
                    nav_md += f"- [{r['p_name']}](#product-{r['id']})\n"
                st.markdown(nav_md)
        else:
            with nav_container:
                st.markdown("### 🚀 快速导航")
                st.warning("安装 `streamlit-sortables` 可启用拖拽")
                nav_md = ""
                for _, r in f_df.iterrows():
                    nav_md += f"- [{r['p_name']}](#product-{r['id']})\n"
                st.markdown(nav_md)

    # --- 主视图渲染 ---
    for _, r in f_df.iterrows():
        st.markdown(f"<div id='product-{r['id']}'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            t1, t2 = st.columns([3, 1])
            t1.markdown(f"### 📦 {r['p_name']}")
            t2.caption(f"更新: {r['p_update_time'].split(' ')[0]}")
            
            raw_data = json.loads(r['p_all_data'])
            def get_val(keys):
                for k in keys: 
                    if k in raw_data: return raw_data[k]
                return "-"

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("策略", get_val(["策略", "p_strategy"]))
            k2.metric("开放日", get_val(["开放日"]))
            k3.metric("申购确认", get_val(["申购确认日", "申购确定日"]))
            k4.metric("赎回确认", get_val(["赎回确认日", "赎回确定日"]))
            k5.metric("赎回款到账", get_val(["赎回回款日期", "赎回款到账日"]))
            st.divider()

            st.markdown("##### 📈 净值与走势")
            c_imp, c_info = st.columns([1, 2])
            with c_imp:
                nv_file = st.file_uploader(f"上传净值 ({r['id']})", type=["xlsx","csv"], key=f"up_{r['id']}")
                if nv_file:
                    try:
                        if nv_file.name.endswith('.csv'): df_nv = pd.read_csv(nv_file)
                        else: df_nv = pd.read_excel(nv_file)
                        ok, msg = save_net_values(r['id'], df_nv)
                        if ok: st.success("导入成功")
                        else: st.error(msg)
                    except: st.error("文件解析失败")
            
            nv_df = get_net_values_df(r['id'])
            if not nv_df.empty:
                nv_df["日期"] = pd.to_datetime(nv_df["日期"])
                t_chart, t_dd, t_data = st.tabs(["📊 走势分析", "📉 回撤分析", "📋 历史数据"])
                
                with t_chart:
                    last = nv_df.iloc[-1]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("日期", last['日期'].strftime('%Y-%m-%d'))
                    m2.metric("单位净值", f"{last['单位净值']:.4f}")
                    m3.metric("累计净值", f"{last['累计净值']:.4f}")
                    st.divider()
                    
                    d_min, d_max = nv_df["日期"].min().date(), nv_df["日期"].max().date()
                    dr = st.slider("区间", d_min, d_max, (d_min, d_max), format="YYYY-MM-DD", key=f"sld_{r['id']}", label_visibility="collapsed")
                    
                    sub_df = nv_df[(nv_df["日期"].dt.date >= dr[0]) & (nv_df["日期"].dt.date <= dr[1])].sort_values("日期").copy()
                    
                    if len(sub_df) > 1:
                        s_nav, e_nav = sub_df["累计净值"].iloc[0], sub_df["累计净值"].iloc[-1]
                        days = (sub_df["日期"].iloc[-1] - sub_df["日期"].iloc[0]).days
                        ret = (e_nav / s_nav) - 1
                        ann_ret = (1 + ret) ** (365/days) - 1 if days > 0 else 0
                        
                        avg_diff = sub_df["日期"].diff().dt.days.mean() if len(sub_df)>2 else 1
                        freq = 252 if avg_diff <= 2 else (52 if avg_diff <= 10 else 12)
                        
                        pct = sub_df["累计净值"].pct_change().dropna()
                        vol = pct.std() * np.sqrt(freq)
                        roll_max = sub_df["累计净值"].cummax()
                        mdd = ((sub_df["累计净值"] - roll_max) / roll_max).min()
                        sharpe = ann_ret / vol if vol != 0 else 0
                        calmar = ann_ret / abs(mdd) if mdd != 0 else 0
                        
                        sub_df["回撤"] = (sub_df["累计净值"] - roll_max) / roll_max
                        
                        kk1, kk2, kk3, kk4, kk5, kk6 = st.columns(6)
                        kk1.metric("区间收益", f"{ret:.2%}")
                        kk2.metric("年化收益", f"{ann_ret:.2%}")
                        kk3.metric("年化波动", f"{vol:.2%}")
                        kk4.metric("最大回撤", f"{mdd:.2%}")
                        kk5.metric("夏普", f"{sharpe:.2f}")
                        kk6.metric("卡玛", f"{calmar:.2f}")
                        st.divider()
                        
                        # --- 红色系图表 (Red Gradient) ---
                        grad = alt.Gradient(
                            gradient='linear', 
                            stops=[
                                alt.GradientStop(color='rgba(214, 39, 40, 0.5)', offset=0), 
                                alt.GradientStop(color='rgba(214, 39, 40, 0)', offset=1)
                            ], 
                            x1=1, x2=1, y1=0, y2=1
                        )
                        
                        base_chart = alt.Chart(sub_df).encode(
                            x=alt.X('日期:T', axis=alt.Axis(format='%Y-%m-%d', title=None))
                        )
                        area = base_chart.mark_area(opacity=1).encode(
                            y=alt.Y('累计净值:Q', scale=alt.Scale(zero=False)),
                            color=alt.value(grad) 
                        )
                        line = base_chart.mark_line(color='#d62728', strokeWidth=2).encode( 
                            y='累计净值:Q'
                        )
                        chart = (area + line).properties(height=350).interactive()
                        st.altair_chart(chart, use_container_width=True)

                    else: st.caption("数据不足")
                
                with t_dd:
                    if len(sub_df) > 1:
                        st.caption(f"📉 最大回撤走势 (区间最低: {mdd:.2%})")
                        
                        dd_base = alt.Chart(sub_df).encode(
                            x=alt.X('日期:T', axis=alt.Axis(format='%Y-%m-%d', title=None))
                        )
                        dd_area = dd_base.mark_area(opacity=1).encode(
                            y=alt.Y('回撤:Q', axis=alt.Axis(format='%'), title='回撤幅度'),
                            color=alt.value(grad) 
                        )
                        dd_line = dd_base.mark_line(color='#d62728', strokeWidth=2).encode( 
                            y='回撤:Q'
                        )
                        dd_chart = (dd_area + dd_line).properties(height=350).interactive()
                        st.altair_chart(dd_chart, use_container_width=True)
                    else:
                        st.caption("数据不足以计算回撤")

                with t_data:
                    show_df = nv_df.copy(); show_df['日期'] = show_df['日期'].dt.strftime('%Y-%m-%d')
                    st.dataframe(show_df, width='stretch', height=400)
            else: st.caption("暂无数据")
            
            st.write("")
            st.markdown("##### 📝 基础要素")
            disp = pd.DataFrame([[k, force_plain_str(v, k in PERCENT_COLUMNS)] for k,v in raw_data.items()], columns=["项", "值"])
            new_info = st.data_editor(disp, key=f"edt_{r['id']}", width='stretch', height=(len(disp)+1)*35+3, hide_index=True)
            
            b1, b2 = st.columns([1, 6])
            with b1:
                if st.button("💾 保存", key=f"sv_{r['id']}", type="primary", width='stretch'):
                    backup_nv = get_net_values_df(r['id'])
                    new_name = r['p_name']
                    edited_dict = dict(new_info.values)
                    if '产品全称' in edited_dict and edited_dict['产品全称']: new_name = edited_dict['产品全称']
                    elif '产品名称' in edited_dict and edited_dict['产品名称']: new_name = edited_dict['产品名称']
                    
                    save_product_to_db(r['p_name'], edited_dict)
                    
                    if not backup_nv.empty:
                        try:
                            new_row = conn.execute("SELECT id FROM product_info WHERE p_name=?", (new_name,)).fetchone()
                            if new_row:
                                save_net_values(new_row[0], backup_nv)
                        except Exception as e: st.error(f"净值关联修复失败: {e}")
                    
                    st.success("已保存"); st.rerun()
            with b2:
                if st.button("🗑️ 删除", key=f"del_{r['id']}"):
                    conn.execute("DELETE FROM product_info WHERE id=?", (r['id'],)); conn.commit(); st.rerun()
        st.write("")

def ui_standard_table_tab():
    """标准视图 (双重保护 - 基础信息合并 + 净值备份恢复)"""
    st.subheader("📊 在库标准表")
    conn = get_db_conn(); rows = conn.execute("SELECT * FROM product_info").fetchall()
    
    edited = st.data_editor(get_standard_dataframe(rows), num_rows="dynamic", width='stretch', key="std_admin")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📝 提交更改", width='stretch'):
            for _, r in edited.iterrows(): 
                p_name = r.get('产品名称')
                if p_name: 
                    # --- 1. 读取旧数据 & 备份净值 ---
                    existing_data = {}
                    backup_nv = pd.DataFrame() 
                    
                    try:
                        cur = conn.execute("SELECT id, p_all_data FROM product_info WHERE p_name=?", (p_name,))
                        row = cur.fetchone()
                        if row:
                            current_id = row[0]
                            if row[1]: existing_data = json.loads(row[1])
                            # 关键：备份该 ID 下的净值
                            backup_nv = get_net_values_df(current_id)
                    except Exception as e:
                        print(f"Error reading/backing up data: {e}")

                    # --- 2. 合并修改 (防止基础要素丢失) ---
                    new_data = r.to_dict()
                    existing_data.update(new_data)

                    # --- 3. 保存完整数据 (可能导致 ID 变更) ---
                    save_product_to_db(p_name, existing_data)
                    
                    # --- 4. 恢复净值到新 ID (防止净值丢失) ---
                    if not backup_nv.empty:
                        try:
                            # 获取新 ID
                            new_row = conn.execute("SELECT id FROM product_info WHERE p_name=?", (p_name,)).fetchone()
                            if new_row:
                                save_net_values(new_row[0], backup_nv)
                        except Exception as e:
                            st.error(f"净值自动恢复失败 ({p_name}): {e}")
            
            st.success("同步成功"); st.rerun()
            
    with c2:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as w: edited.to_excel(w, index=False)
        st.download_button("📥 导出Excel", out.getvalue(), f"Data_{datetime.date.today()}.xlsx", width='stretch')

def ui_user_management_tab():
    """用户管理 (增加严格校验：用户名仅英文，密码禁中文)"""
    st.subheader("👥 账号管理")
    st.dataframe(get_all_users(), width='stretch')
    st.divider(); c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### ➕ 新增")
        with st.form("new_u"):
            u = st.text_input("用户")
            p = st.text_input("密码", type="password")
            r = st.selectbox("角色", ["staff", "admin"])
            if st.form_submit_button("创建", width='stretch'):
                if u and p: 
                    # --- 校验逻辑 ---
                    if not re.match(r'^[a-zA-Z0-9_]+$', u):
                        st.error("用户名只能包含英文字母、数字和下划线")
                    elif len(p) < 6:
                        st.error("密码长度至少需 6 位")
                    elif not re.match(r'^[\x21-\x7E]+$', p): # ASCII only
                        st.error("密码不支持中文字符")
                    else:
                        ok, m = add_user(u, p, r)
                        if ok: st.success("成功"); st.rerun()
                        else: st.error(m)
                else:
                    st.error("请输入用户名和密码")
    with c2:
        st.markdown("##### ❌ 删除")
        target = st.selectbox("账号", get_all_users()['username'].tolist())
        if st.button("删除", type="primary"):
            if target == 'admin': st.error("无法删除超管")
            elif target == st.session_state.get('username'): st.error("无法自删")
            else: delete_user(target); st.success("已删除"); st.rerun()