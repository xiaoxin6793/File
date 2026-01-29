import streamlit as st
import time

# --- 核心修改：导入路径调整 ---
# 1. 从数据库模块导入逻辑函数
from modules.database import check_login, init_db
# 2. 从UI模块导入界面函数 
from modules import ui_components as sys

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="Sealand System", 
    page_icon="🛡️",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 1: 直接调用导入的 init_db ---
init_db()

# --- 2. 状态管理 ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = ""

# --- 3. 核心 CSS 注入 (支持暗黑/明亮模式自适应 + Tab美化 + 悬浮按钮) ---
def load_adaptive_css():
    st.markdown("""
    <style>
        /* --- 全局变量定义 --- */
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
            --tab-bg-inactive: #F3F4F6;
            --tab-text-inactive: #4B5563;
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
                --tab-bg-inactive: #1F2937;
                --tab-text-inactive: #9CA3AF;
            }
        }
        
        /* 页面背景 */
        .stApp {
            background: linear-gradient(180deg, var(--bg-color) 0%, var(--bg-gradient-end) 100%);
            background-attachment: fixed;
            color: var(--text-main);
        }

        /* 输入框样式 */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: var(--input-bg) !important;
            border: 1px solid var(--input-border) !important;
            color: var(--text-main) !important;
            border-radius: 8px !important;
        }
        
        /* 按钮样式 */
        div.stButton > button {
            background-color: var(--btn-bg) !important;
            color: var(--btn-text) !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        div.stButton > button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }

        /* Tab 导航栏美化 */
        div[data-baseweb="tab-highlight"] { display: none !important; }
        div[data-baseweb="tab-list"] { gap: 10px; padding-bottom: 10px; }
        button[data-baseweb="tab"] {
            height: 45px;
            border-radius: 10px !important;
            padding: 0 24px !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            background-color: var(--tab-bg-inactive) !important;
            color: var(--tab-text-inactive) !important;
            border: 1px solid transparent !important;
            transition: all 0.2s;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--btn-bg) !important;
            color: var(--btn-text) !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: scale(1.02);
        }
        button[data-baseweb="tab"]:hover {
            background-color: var(--input-border) !important;
            color: var(--text-main) !important;
        }

        /* 侧边栏链接 */
        section[data-testid="stSidebar"] a {
            text-decoration: none;
            color: var(--text-sub) !important;
            display: block;
            padding: 8px 12px;
            margin-bottom: 4px;
            border-radius: 6px;
            border-left: 3px solid transparent;
            transition: background 0.2s;
            font-size: 14px;
        }
        section[data-testid="stSidebar"] a:hover {
            background-color: var(--tab-bg-inactive);
            color: var(--text-main) !important;
            border-left: 3px solid var(--btn-bg);
        }

        .main-title { color: var(--text-main) !important; font-weight: 800; font-size: 28px; }
        .sub-title { color: var(--text-sub) !important; font-size: 15px; font-weight: 500; }
        .footer-text { color: var(--text-sub) !important; font-size: 12px; text-align: center; margin-top: 40px; opacity: 0.6; }
        
        /* --- 🔝 悬浮按钮样式 (新增) --- */
        #myBtn {
            display: none; /* 默认隐藏 */
            position: fixed;
            bottom: 40px;
            right: 40px;
            z-index: 999999;
            border: none;
            outline: none;
            background-color: var(--btn-bg);
            color: var(--btn-text);
            cursor: pointer;
            padding: 0;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            font-size: 24px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            text-align: center;
            line-height: 50px;
        }

        #myBtn:hover {
            background-color: var(--btn-hover);
            transform: scale(1.15);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }
    </style>

    <div id="back-to-top-container">
        <button onclick="topFunction()" id="myBtn" title="回到顶部">↑</button>
    </div>

    <script>
        // 找到 Streamlit 真正滚动的那个容器
        var scrollContainer = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
        var mybutton = window.parent.document.getElementById("myBtn");

        if (!scrollContainer) {
            scrollContainer = window;
        }
        
        // 监听滚动事件
        scrollContainer.onscroll = function() { scrollFunction() };

        function scrollFunction() {
            var scrollTop = scrollContainer.scrollTop || document.documentElement.scrollTop;
            // 下滑超过 100px 显示
            if (scrollTop > 100) {
                mybutton.style.display = "block";
            } else {
                mybutton.style.display = "none";
            }
        }

        // 点击回到顶部
        window.parent.topFunction = function() {
            scrollContainer.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
    </script>
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
            <div class="sub-title">国海证券产品数据管理系统</div>
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
                role = check_login(username, password)
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

    # 根据权限显示不同的 Tab，功能全部来自
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