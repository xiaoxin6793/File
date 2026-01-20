import streamlit as st
import main_system as sys
import time

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="Sealand System", 
    page_icon="🛡️",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 初始化数据库 (调用 main_system 中的函数)
sys.init_db()

# --- 2. 状态管理 ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = ""

# --- 3. 核心 CSS 注入 (支持暗黑/明亮模式自适应) ---
def load_adaptive_css():
    st.markdown("""
    <style>
        :root {
            --bg-color: #FFFFFF;
            --bg-gradient-end: #F7F9FC;
            --text-main: #111827;
            --text-sub: #6B7280;
            --input-bg: #FFFFFF;
            --input-border: #E5E7EB;
            --btn-bg: #111827;
            --btn-text: #FFFFFF;
            --btn-hover: #374151;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-color: #0E1117;
                --bg-gradient-end: #161B22;
                --text-main: #F9FAFB;
                --text-sub: #9CA3AF;
                --input-bg: #262730;
                --input-border: #374151;
                --btn-bg: #FFFFFF;
                --btn-text: #000000;
                --btn-hover: #E5E7EB;
            }
        }
        
        .stApp {
            background: linear-gradient(180deg, var(--bg-color) 0%, var(--bg-gradient-end) 100%);
            background-attachment: fixed;
            color: var(--text-main);
        }

        .stTextInput input {
            background-color: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            color: var(--text-main) !important;
            border-radius: 8px !important;
            padding: 12px 15px !important;
        }
        .stTextInput input:focus {
            border-color: var(--text-main) !important;
            box-shadow: 0 0 0 1px var(--text-main) !important;
        }
        
        /* 隐藏输入框自带的 Label */
        .stTextInput label { display: none !important; }

        div.stButton > button {
            background-color: var(--btn-bg) !important;
            color: var(--btn-text) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 20px !important;
            font-weight: 600 !important;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            background-color: var(--btn-hover) !important;
            transform: scale(1.01);
        }

        .main-title { color: var(--text-main) !important; font-weight: 700; font-size: 26px; }
        .sub-title { color: var(--text-sub) !important; font-size: 14px; }
        .footer-text { color: var(--text-sub) !important; font-size: 12px; text-align: center; margin-top: 40px; }
    </style>
    """, unsafe_allow_html=True)

load_adaptive_css()

# --- 4. 登录页面 ---
def show_login_page():
    # 隐藏侧边栏和顶部菜单，只在登录页生效
    st.markdown("""
        <style>
            header {visibility: hidden !important;}
            [data-testid="stSidebar"] {display: none;}
            .block-container {
                max-width: 500px !important;
                padding-top: 4rem !important;
                margin: auto;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <div style="font-size: 60px; margin-bottom: 10px;">🛡️</div>
            <div class="main-title">Log in to SeaLand</div>
            <div class="sub-title">国海证卷产品数据管理系统</div>
        </div>
    """, unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("user", placeholder="用户名 / Email")
        password = st.text_input("pwd", type="password", placeholder="密码")
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        
        submit = st.form_submit_button("Sign In", use_container_width=True)

        if submit:
            if not username or not password:
                st.warning("请输入完整信息")
            else:
                # 调用 main_system 中的 check_login
                role = sys.check_login(username, password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.user_role = role
                    st.session_state.username = username
                    st.success("验证成功")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("认证失败，请检查账号密码")

    st.markdown("""
        <div class="footer-text">
            System Auto-Adapt (Dark/Light)<br>
            管理员: admin 
        </div>
    """, unsafe_allow_html=True)

# --- 5. 主功能界面 ---
def show_main_interface():
    # 恢复侧边栏和宽屏布局
    st.markdown("""
        <style>
            header {visibility: visible !important;}
            [data-testid="stSidebar"] {display: block;}
            .block-container {
                max-width: 100% !important;
                padding-top: 4rem !important; 
                padding-left: 5rem !important;
                padding-right: 5rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("SeaLand System")
        st.caption(f"当前用户: {st.session_state.username}")
        st.markdown("---")
        if st.button("Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    # 根据权限显示不同的 Tab，功能全部来自 main_system
    if st.session_state.user_role == 'admin':
        t1, t2, t3, t4 = st.tabs(["📤 数据录入", "🔍 产品管理", "📊 全量视图", "👥 账号"])
        with t1: sys.ui_entry_tab()
        with t2: sys.ui_card_edit_tab()
        with t3: sys.ui_standard_table_tab()
        with t4: sys.ui_user_management_tab()
            
    elif st.session_state.user_role == 'staff':
        t1, t2 = st.tabs(["📤 数据录入", "🔍 产品管理"])
        with t1: sys.ui_entry_tab()
        with t2: sys.ui_card_edit_tab()

# --- 6. 路由逻辑 ---
if __name__ == "__main__":
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_main_interface()