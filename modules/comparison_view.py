# modules/comparison_view.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import datetime
from scipy import stats
from .database import get_db_conn

# --- 1. 数据获取与处理 ---
def get_multi_product_data(product_ids):
    """根据多个ID获取净值数据，并转为宽表 (Index: Date, Cols: ProductName)"""
    if not product_ids:
        return pd.DataFrame()
    
    conn = get_db_conn()
    placeholders = ','.join(['?'] * len(product_ids))
    query = f"""
        SELECT t1.nv_date, t1.cum_nv, t2.p_name 
        FROM net_values t1
        JOIN product_info t2 ON t1.product_id = t2.id
        WHERE t1.product_id IN ({placeholders})
        ORDER BY t1.nv_date ASC
    """
    
    try:
        df = pd.read_sql(query, conn, params=tuple(product_ids))
        if df.empty:
            return pd.DataFrame()
        
        df['nv_date'] = pd.to_datetime(df['nv_date'])
        # 透视表：日期为索引，产品名为列
        pivot_df = df.pivot(index='nv_date', columns='p_name', values='cum_nv')
        
        # 【核心修改】只做前向填充(处理非交易日)，但不使用dropna()，保留起始的空值以便"全部区间"显示空白
        pivot_df = pivot_df.ffill() 
        return pivot_df
    except Exception as e:
        st.error(f"数据读取错误: {e}")
        return pd.DataFrame()

def calculate_financial_metrics(series, freq=252):
    """计算单条净值曲线的核心指标 (自动去除NaN)"""
    # 【核心修改】先去除空值，确保只计算该产品有效存续期的数据
    clean_series = series.dropna()
    
    if len(clean_series) < 2:
        return {k: 0 for k in ["区间收益", "年化收益", "夏普比率", "卡玛比率", "最大回撤", "年波动率", "最大回撤修复天数"]}

    # 1. 基础数据
    start_nav = clean_series.iloc[0]
    end_nav = clean_series.iloc[-1]
    days = (clean_series.index[-1] - clean_series.index[0]).days
    
    # 2. 收益率
    interval_ret = (end_nav / start_nav) - 1
    if days > 0:
        annual_ret = (1 + interval_ret) ** (365 / days) - 1
    else:
        annual_ret = 0

    # 3. 波动率 & 夏普
    pct_change = clean_series.pct_change().fillna(0)
    volatility = pct_change.std() * np.sqrt(freq)
    risk_free = 0.00  # 假设无风险利率 0%
    sharpe = (annual_ret - risk_free) / volatility if volatility != 0 else 0

    # 4. 最大回撤 & 修复天数
    roll_max = clean_series.cummax()
    drawdown = (clean_series - roll_max) / roll_max
    max_dd = drawdown.min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

# 5. 比率计算 (假设无风险利率为 0)
# sharpe = annual_ret / volatility if volatility != 0 and not pd.isna(volatility) else 0
# calmar = annual_ret / abs(max_drawdown) if max_drawdown != 0 else 0
#                            volatility = pct_change.std() * np.sqrt(freq)

    # 修复天数计算
    repair_days = 0
    if max_dd < 0:
        idx_min = drawdown.idxmin()
        peak_val = roll_max.loc[idx_min]
        sub_series = clean_series.loc[idx_min:] 
        recover_points = sub_series[sub_series >= peak_val]
        if not recover_points.empty:
            repair_days = (recover_points.index[0] - idx_min).days
        else:
            repair_days = "未修复" 

    return {
        "区间收益": interval_ret,
        "年化收益": annual_ret,
        "年波动率": volatility,
        "最大回撤": max_dd,
        "夏普比率": sharpe,
        "卡玛比率": calmar,
        "最大回撤修复天数": repair_days
    }

def calculate_beta_alpha(target_series, benchmark_series, freq=252):
    """计算相对指标 (Alpha/Beta)"""
    # 对齐索引 (只计算两者都有数据的日期的交集)
    common_idx = target_series.dropna().index.intersection(benchmark_series.dropna().index)
    
    if len(common_idx) < 10: # 数据太少不计算
        return 0, 0

    y = target_series.loc[common_idx].pct_change().fillna(0)
    x = benchmark_series.loc[common_idx].pct_change().fillna(0)
    
    # 线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    beta = slope
    alpha = intercept * freq 
    return beta, alpha

# --- 2. 界面主函数 ---
def ui_comparison_tab():
    st.subheader("⚔️ 产品对比分析")
    
    conn = get_db_conn()
    all_products = pd.read_sql("SELECT id, p_name FROM product_info", conn)
    
    # --- A. 顶部控制栏 ---
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_names = st.multiselect(
                "选择对比产品 (建议 2-5 个)", 
                options=all_products['p_name'].tolist(),
                default=all_products['p_name'].tolist()[:2] if len(all_products) >=2 else None
            )
        
        selected_ids = all_products[all_products['p_name'].isin(selected_names)]['id'].tolist()
        raw_df = get_multi_product_data(selected_ids)
        
        if raw_df.empty:
            st.info("请选择产品，或所选产品暂无共同时间段的净值数据。")
            return

        with c2:
            # --- 核心修改：日期逻辑计算 ---
            # 1. 全部区间 (Union): 全局最早到全局最晚
            global_min = raw_df.index.min().date()
            global_max = raw_df.index.max().date()
            
            # 2. 最大共同区间 (Intersection): 所有列都有值的区间的交集
            # 逻辑：找到每个产品的第一天，取最大值作为共同起点；找到每个产品的最后一天，取最小值作为共同终点
            try:
                common_start = raw_df.apply(lambda x: x.first_valid_index()).max().date()
                common_end = raw_df.apply(lambda x: x.last_valid_index()).min().date()
            except:
                common_start, common_end = global_min, global_max # 容错回退

            # 3. 下拉框选项 (最大共同区间排第一作为默认)
            time_range_opt = st.selectbox(
                "分析时段", 
                ["最大共同区间", "全部区间", "今年以来", "最近一月", "最近三月", "最近一年", "自定义"]
            )
            
            start_date, end_date = common_start, common_end # 默认值
            today = datetime.date.today()
            
            if time_range_opt == "最大共同区间":
                start_date, end_date = common_start, common_end
                if start_date > end_date:
                    st.warning("⚠️ 选中的产品没有共同存续时间段，已自动切换为全部区间。")
                    start_date, end_date = global_min, global_max
            elif time_range_opt == "全部区间":
                start_date, end_date = global_min, global_max
            elif time_range_opt == "今年以来":
                start_date = datetime.date(today.year, 1, 1)
                end_date = global_max
            elif time_range_opt == "最近一月":
                start_date = today - datetime.timedelta(days=30)
                end_date = global_max
            elif time_range_opt == "最近三月":
                start_date = today - datetime.timedelta(days=90)
                end_date = global_max
            elif time_range_opt == "最近一年":
                start_date = today - datetime.timedelta(days=365)
                end_date = global_max
            elif time_range_opt == "自定义":
                d_range = st.date_input("选择日期范围", [common_start, common_end])
                if len(d_range) == 2:
                    start_date, end_date = d_range[0], d_range[1]

            # 边界修正
            start_date = max(start_date, global_min)
            end_date = min(end_date, global_max)

    # --- 数据切片 ---
    mask = (raw_df.index.date >= start_date) & (raw_df.index.date <= end_date)
    sliced_df = raw_df.loc[mask]

    # 如果切片后全是空的（针对共同区间没数据的情况）
    if sliced_df.dropna(how='all').empty:
        st.warning("该时段内无有效数据。")
        return

    # --- 归一化数据 ---
    # 【修改点】归一化时，如果某产品在起点是NaN，它整条线应该是NaN（或者是从它有数据的第一天开始归一化）
    # 这里采用简单逻辑：每列除以该列在该区间内第一个非空值
    normalized_df = sliced_df.copy()
    for col in normalized_df.columns:
        first_valid = normalized_df[col].first_valid_index()
        if first_valid:
            base_val = normalized_df.loc[first_valid, col]
            if base_val != 0:
                normalized_df[col] = normalized_df[col] / base_val

    # --- 模块 1: 收益率曲线 ---
    st.markdown("##### 1. 📈 累计收益率曲线")
    
    chart_data = normalized_df.reset_index().melt('nv_date', var_name='产品', value_name='累计净值(归一化)')
    
    chart_yield = alt.Chart(chart_data).mark_line().encode(
        x=alt.X('nv_date:T', title=None, axis=alt.Axis(format='%Y-%m-%d')),
        y=alt.Y('累计净值(归一化):Q', title='累计收益趋势 (各产品起点=1)', scale=alt.Scale(zero=False)),
        color='产品:N',
        tooltip=['nv_date', '产品', alt.Tooltip('累计净值(归一化)', format='.4f')]
    ).properties(height=350).interactive()
    
    st.altair_chart(chart_yield, use_container_width=True)

    # --- 模块 2: 回撤走势 (核心修复位置) ---
    st.markdown("##### 2. 📉 动态回撤分析")
    
    # 动态回撤计算（需容忍NaN）
    drawdown_df = sliced_df.copy()
    for col in drawdown_df.columns:
        # 只对非空部分计算回撤
        mask_valid = drawdown_df[col].notna()
        if mask_valid.any():
            roll_max = drawdown_df.loc[mask_valid, col].cummax()
            drawdown_df.loc[mask_valid, col] = (drawdown_df.loc[mask_valid, col] - roll_max) / roll_max

    # 修复1: 增加 .dropna()，过滤掉没有数据的行，让面积图从真正有数据那天开始渲染
    chart_dd_data = drawdown_df.reset_index().melt('nv_date', var_name='产品', value_name='回撤').dropna()
    
    # 修复2: 增加 stack=None，防止面积图堆叠导致数值错误累加
    dd_area = alt.Chart(chart_dd_data).mark_area(opacity=0.3).encode(
        x=alt.X('nv_date:T', title=None),
        y=alt.Y('回撤:Q', axis=alt.Axis(format='%'), stack=None), # 这里增加了 stack=None
        y2=alt.value(0), 
        color='产品:N'
    )

    dd_line = alt.Chart(chart_dd_data).mark_line(strokeWidth=1.5).encode(
        x=alt.X('nv_date:T'),
        y=alt.Y('回撤:Q'),
        color='产品:N',
        tooltip=['nv_date', '产品', alt.Tooltip('回撤', format='.2%')]
    )
    
    st.altair_chart((dd_area + dd_line).properties(height=300).interactive(), use_container_width=True)

    # --- 模块 3: 指标对比表格 ---
    st.markdown("##### 3. 📊 核心指标对比")
    
    metrics_list = []
    # 找数据最全的作为基准，或者默认第一个
    benchmark_col = sliced_df.columns[0] 
    benchmark_series = sliced_df[benchmark_col]

    for col in sliced_df.columns:
        series = sliced_df[col] # 这里可能包含NaN
        m = calculate_financial_metrics(series)
        beta, alpha = calculate_beta_alpha(series, benchmark_series)
        
        row = {
            "产品名称": col,
            "区间收益": f"{m['区间收益']:.2%}",
            "年化收益": f"{m['年化收益']:.2%}",
            "夏普比率": f"{m['夏普比率']:.2f}",
            "卡玛比率": f"{m['卡玛比率']:.2f}",
            "最大回撤": f"{m['最大回撤']:.2%}",
            "阿尔法(α)": f"{alpha:.2%}", 
            "贝塔(β)": f"{beta:.2f}",    
            "年波动率": f"{m['年波动率']:.2%}",
            "修复天数": f"{m['最大回撤修复天数']} 天" if isinstance(m['最大回撤修复天数'], (int, float)) else m['最大回撤修复天数']
        }
        metrics_list.append(row)
    
    metrics_df = pd.DataFrame(metrics_list)
    cols_order = ["产品名称", "区间收益", "年化收益", "最大回撤", "夏普比率", "卡玛比率", "阿尔法(α)", "贝塔(β)", "年波动率", "修复天数"]
    
    st.dataframe(metrics_df[cols_order], hide_index=True, use_container_width=True)
    st.caption(f"* 注：统计指标基于该时段内各产品的【有效存续期】计算；Alpha/Beta 暂以【{benchmark_col}】为基准。")

    # --- 模块 4: 相关性热力图 ---
    st.markdown("##### 4. 🔗 相关性矩阵 (红高绿低)")
    
    # corr() 自动处理NaN (Pairwise)
    corr_matrix = sliced_df.pct_change().corr().reset_index()
    corr_melt = corr_matrix.melt('p_name', var_name='产品B', value_name='相关系数')
    
    base = alt.Chart(corr_melt).encode(
        x='p_name:O',
        y='产品B:O'
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('相关系数:Q', 
                        scale=alt.Scale(scheme='redyellowgreen', domain=[-1, 1], reverse=True),
                        title="相关性"),
        tooltip=['p_name', '产品B', alt.Tooltip('相关系数', format='.2f')]
    )

    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('相关系数:Q', format='.2f'),
        color=alt.condition(
            alt.datum.相关系数 > 0.5,
            alt.value('white'),
            alt.value('black')
        )
    )

    st.altair_chart((heatmap + text).properties(height=400), use_container_width=True)