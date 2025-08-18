#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键词2D可视化工具 - 简化运行脚本
"""

import os
import sys
from keyword_2d_visualizer import Keyword2DVisualizer

def main():
    print("🔍 关键词2D位置可视化工具")
    print("=" * 50)
    
    # 设置工作目录
    os.chdir("/Users/yanzhu/Box Sync/Yan/KeySI/CSV")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 列出可用的CSV文件
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        print("❌ 当前目录没有找到CSV文件")
        return
    
    print("\n📁 可用的CSV文件:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    # 选择文件
    print("\n请选择要处理的文件:")
    print("输入文件编号（多个文件用空格分隔，如: 1 2）")
    print("或直接输入文件名（如: risk_factors.csv）")
    
    file_input = input("请输入: ").strip()
    
    selected_files = []
    if file_input.replace(' ', '').isdigit():
        # 输入的是数字
        indices = [int(x) - 1 for x in file_input.split()]
        for idx in indices:
            if 0 <= idx < len(csv_files):
                selected_files.append(csv_files[idx])
    else:
        # 输入的是文件名
        file_names = file_input.split()
        for file_name in file_names:
            if file_name in csv_files:
                selected_files.append(file_name)
            else:
                print(f"⚠️  文件 '{file_name}' 不存在，跳过")
    
    if not selected_files:
        print("❌ 没有选择有效的文件")
        return
    
    print(f"\n✅ 选择的文件: {selected_files}")
    
    # 输入目标关键词
    print("\n🎯 请输入要特别标记的关键词:")
    print("多个关键词用空格分隔（如: cancer covid patient）")
    print("直接回车跳过")
    
    keywords_input = input("关键词: ").strip()
    target_keywords = keywords_input.split() if keywords_input else []
    
    if target_keywords:
        print(f"✅ 目标关键词: {target_keywords}")
    
    # 选择文本列
    print("\n📊 文本列设置:")
    print("默认文本在第1列（索引为1）")
    text_column_input = input("请输入文本列索引（直接回车使用默认值1）: ").strip()
    text_column = int(text_column_input) if text_column_input.isdigit() else 1
    
    print(f"✅ 文本列索引: {text_column}")
    
    # 创建可视化器
    print("\n🚀 开始处理...")
    visualizer = Keyword2DVisualizer(device="cpu")
    
    # 处理文件
    all_file_keywords = {}
    for file_path in selected_files:
        print(f"\n📖 处理文件: {file_path}")
        file_keywords = visualizer.process_csv_file(file_path, text_column)
        all_file_keywords.update(file_keywords)
    
    if not all_file_keywords:
        print("❌ 没有成功处理任何文件")
        return
    
    # 显示处理结果摘要
    print("\n📈 处理结果摘要:")
    for filename, keywords in all_file_keywords.items():
        print(f"  {filename}: {len(keywords)} 个关键词")
        if keywords:
            top_keywords = [kw for kw, score in sorted(keywords, key=lambda x: x[1], reverse=True)[:5]]
            print(f"    前5个关键词: {', '.join(top_keywords)}")
    
    # 可视化
    print("\n🎨 生成2D可视化...")
    output_path = "keyword_2d_visualization.png"
    visualizer.visualize_keywords_2d(all_file_keywords, target_keywords, output_path)
    
    print(f"\n✅ 完成！可视化结果已保存到: {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc() 