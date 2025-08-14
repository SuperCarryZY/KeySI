#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试UI功能的简单脚本
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ui_imports():
    """测试UI模块的导入"""
    try:
        print("正在测试UI模块导入...")
        
        # 测试基本导入
        import dash
        from dash import dcc, html
        print("✓ Dash导入成功")
        
        # 测试其他必要模块
        import numpy as np
        import pandas as pd
        print("✓ NumPy和Pandas导入成功")
        
        # 测试NLP相关模块
        try:
            from nltk.tokenize import word_tokenize
            print("✓ NLTK导入成功")
        except ImportError:
            print("⚠ NLTK导入失败（可选模块）")
        
        print("✅ 所有必要模块导入成功！")
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_ui_creation():
    """测试UI创建"""
    try:
        print("\n正在测试UI创建...")
        
        # 模拟关键字数据
        global keywords, GLOBAL_KEYWORDS, GLOBAL_OUTPUT_DICT
        keywords = ["test1", "test2", "test3"]
        GLOBAL_KEYWORDS = keywords
        GLOBAL_OUTPUT_DICT = {"cluster1": keywords}
        
        # 导入并测试UI创建
        from finaluimodified import create_layout, app
        
        # 测试布局创建
        layout = create_layout()
        print("✓ UI布局创建成功")
        
        # 测试应用实例
        if app and hasattr(app, 'layout'):
            print("✓ Dash应用实例创建成功")
        else:
            print("❌ Dash应用实例创建失败")
            return False
        
        print("✅ UI创建测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ UI创建测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始UI功能测试...")
    print("=" * 50)
    
    # 测试1: 模块导入
    if not test_ui_imports():
        print("❌ 模块导入测试失败，无法继续")
        return False
    
    # 测试2: UI创建
    if not test_ui_creation():
        print("❌ UI创建测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！UI模块可以正常使用")
    print("\n要启动完整的Web应用，请运行:")
    print("python finaluimodified.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
