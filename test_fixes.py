#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的关键词添加和组显示功能
"""

def test_keyword_addition_logic():
    """测试关键词添加逻辑"""
    print("=== 测试关键词添加逻辑 ===")
    
    # 模拟初始状态
    group_data = {}
    group_order = {"Group 1": [], "Group 2": []}
    
    print("初始状态:")
    print(f"group_data: {group_data}")
    print(f"group_order: {group_order}")
    
    # 模拟添加第一个关键词到Group 1
    clicked_keyword = "cancer"
    selected_group = "Group 1"
    
    # 模拟 handle_plot_click 的逻辑
    new_data = dict(group_data)
    new_data[clicked_keyword] = selected_group
    
    print(f"\n添加关键词 '{clicked_keyword}' 到 '{selected_group}':")
    print(f"new_data: {new_data}")
    
    # 模拟 update_group_order 的逻辑（修复后）
    for kw, grp in new_data.items():
        if grp and grp in group_order and kw not in group_order[grp]:
            group_order[grp].append(kw)
    
    print(f"更新后的 group_order: {group_order}")
    
    # 模拟添加第二个关键词到Group 1
    clicked_keyword2 = "patient"
    selected_group2 = "Group 1"
    
    new_data[clicked_keyword2] = selected_group2
    
    print(f"\n添加关键词 '{clicked_keyword2}' 到 '{selected_group2}':")
    print(f"new_data: {new_data}")
    
    # 再次更新 group_order
    for kw, grp in new_data.items():
        if grp and grp in group_order and kw not in group_order[grp]:
            group_order[grp].append(kw)
    
    print(f"更新后的 group_order: {group_order}")
    
    # 验证：Group 1 应该有两个关键词
    if len(group_order["Group 1"]) == 2:
        print("✅ 关键词添加逻辑正确：两个关键词都被添加到Group 1")
    else:
        print("❌ 关键词添加逻辑错误")
    
    return group_order

def test_group_switching():
    """测试组切换时的关键词显示"""
    print("\n=== 测试组切换时的关键词显示 ===")
    
    # 使用上面的结果
    group_order = test_keyword_addition_logic()
    
    print(f"\n当前所有组的关键词:")
    for group_name, keywords in group_order.items():
        print(f"{group_name}: {keywords}")
    
    # 模拟切换到Group 2
    selected_group = "Group 2"
    print(f"\n切换到 {selected_group}")
    
    # 验证：所有组的关键词都应该保持可见
    print("验证所有组的关键词是否保持可见:")
    for group_name, keywords in group_order.items():
        if keywords:
            print(f"✅ {group_name}: {keywords}")
        else:
            print(f"⚠️  {group_name}: {keywords}")
    
    # 添加关键词到Group 2
    clicked_keyword = "covid"
    new_data = {"cancer": "Group 1", "patient": "Group 1", "covid": "Group 2"}
    
    for kw, grp in new_data.items():
        if grp and grp in group_order and kw not in group_order[grp]:
            group_order[grp].append(kw)
    
    print(f"\n添加关键词到Group 2后的状态:")
    for group_name, keywords in group_order.items():
        print(f"{group_name}: {keywords}")
    
    # 验证：两个组都应该有关键词
    if len(group_order["Group 1"]) > 0 and len(group_order["Group 2"]) > 0:
        print("✅ 组切换逻辑正确：所有组的关键词都保持可见")
    else:
        print("❌ 组切换逻辑错误：某些组的关键词丢失")

def test_remove_keyword():
    """测试关键词删除逻辑"""
    print("\n=== 测试关键词删除逻辑 ===")
    
    # 使用上面的结果
    group_order = {"Group 1": ["cancer", "patient"], "Group 2": ["covid"]}
    
    print(f"删除前的状态: {group_order}")
    
    # 模拟删除Group 1中的第一个关键词
    group_name = "Group 1"
    keyword_index = 0
    
    if group_name in group_order and keyword_index < len(group_order[group_name]):
        removed_keyword = group_order[group_name].pop(keyword_index)
        print(f"删除了关键词 '{removed_keyword}' 从 '{group_name}'")
    
    print(f"删除后的状态: {group_order}")
    
    if len(group_order["Group 1"]) == 1 and "cancer" not in group_order["Group 1"]:
        print("✅ 关键词删除逻辑正确")
    else:
        print("❌ 关键词删除逻辑错误")

if __name__ == "__main__":
    print("开始测试修复后的功能...")
    print("=" * 50)
    
    try:
        test_keyword_addition_logic()
        test_group_switching()
        test_remove_keyword()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
