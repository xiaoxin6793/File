# modules/utils.py
import hashlib
import pandas as pd
import re
from .config import PERCENT_COLUMNS

def hash_password(password):
    """密码加密"""
    return hashlib.sha256(str(password).encode()).hexdigest()

def to_percent_str(val):
    """
    统一格式化为百分比字符串 (保留2位小数)
    例如: 1.5 -> "1.50%"
    """
    if val is None or pd.isna(val) or str(val).strip() == "": return ""
    s_val = str(val).strip()
    
    # 如果已经包含%，直接返回，避免重复处理
    if "%" in s_val: return s_val
    
    try:
        num = float(s_val)
        # 策略：保留原值，仅添加 % 符号 (符合一般录入习惯，如录入 1.5 代表 1.5%)
        return f"{num:.2f}%"
    except:
        return s_val

def force_plain_str(val, is_percent=False):
    """
    强制转换为纯文本，处理科学计数法和百分比
    """
    if val is None or pd.isna(val): return ""
    val_str = str(val).strip()
    
    if val_str.lower() in ['nan', 'none', '']: return ""
    
    # 如果是百分比字段，进行百分比格式化
    if is_percent:
        return to_percent_str(val_str)
        
    # 处理科学计数法 (1.23e+5 -> 123000)
    if 'e' in val_str.lower():
        try: return "{:.10f}".format(float(val)).rstrip('0').rstrip('.')
        except: return val_str
        
    if val_str.endswith('.0'): return val_str[:-2]
    return val_str

def parse_to_float(val):
    """
    通用解析：会将 1.5% 解析为 0.015 (用于常规逻辑)
    """
    if not val or str(val).lower() in ["none", "nan", ""]: return 0.0
    val_str = str(val)
    res = re.findall(r"[-+]?\d*\.\d+|\d+", val_str)
    if not res: return 0.0
    num = float(res[0])
    return num / 100 if "%" in val_str else num

def clean_percent_to_float(val):
    """
    【新增】专门用于费率加法计算
    解析数值但不除以100，例如 "1.5%" -> 1.5
    用于 database.py 中的 (管理费+托管费...) 计算
    """
    if not val or str(val).lower() in ["none", "nan", ""]: return 0.0
    val_str = str(val).strip()
    
    # 移除 % 和空格
    clean_str = val_str.replace('%', '').replace(' ', '')
    
    # 提取数字
    res = re.findall(r"[-+]?\d*\.\d+|\d+", clean_str)
    if not res: return 0.0
    
    try:
        return float(res[0])
    except:
        return 0.0

def is_percent_col(col_name):
    """智能判断列名是否暗示其为百分比数据"""
    c_str = str(col_name)
    
    # 1. 优先：检查配置白名单
    if c_str in PERCENT_COLUMNS:
        return True
        
    # 2. 其次：核心关键词匹配
    keywords = ["费率", "比例", "收益率", "折扣", "占比", "税率", "费", "业绩报酬"]
    if any(k in c_str for k in keywords):
        return True
        
    # 3. 最后：用户特指的模糊词
    special_words = ["其他如有", "其他（如有）"]
    if any(s in c_str for s in special_words):
        return True
        
    return False