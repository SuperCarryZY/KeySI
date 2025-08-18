#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的UI功能测试
"""

def test_basic_imports():
    """测试基本导入"""
    try:
        import dash
        print("✓ Dash导入成功")
        
        import plotly.graph_objs as go
        print("✓ Plotly导入成功")
        
        import numpy as np
        print("✓ NumPy导入成功")
        
        import pandas as pd
        print("✓ Pandas导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_ui_creation():
    """测试UI创建"""
    try:
        from dash import Dash, html, dcc
        
        # 创建简单的Dash应用
        app = Dash(__name__)
        
        # 创建基本布局
        layout = html.Div([
            html.H3("测试UI"),
            dcc.Graph(
                id='test-plot',
                figure={
                    'data': [{'x': [1, 2, 3], 'y': [1, 4, 2], 'type': 'scatter'}],
                    'layout': {'title': '测试图表'}
                }
            )
        ])
        
        app.layout = layout
        print("✓ 基本UI创建成功")
        return True
        
    except Exception as e:
        print(f"❌ UI创建失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 40)
    print("简化UI功能测试")
    print("=" * 40)
    
    # 测试基本导入
    imports_ok = test_basic_imports()
    
    # 测试UI创建
    ui_ok = test_ui_creation()
    
    print("\n" + "=" * 40)
    print("测试结果总结")
    print("=" * 40)
    
    if imports_ok and ui_ok:
        print("🎉 所有测试通过，UI功能正常")
        print("可以运行主应用: py finaluimodified.py")
    else:
        print("💥 部分测试失败，需要检查依赖安装")
    
    return imports_ok and ui_ok

if __name__ == "__main__":
    main()
