#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BertTopic安装助手脚本
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔄 {description}...")
    print(f"执行命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description}成功")
            if result.stdout:
                print(f"输出: {result.stdout.strip()}")
        else:
            print(f"❌ {description}失败")
            if result.stderr:
                print(f"错误: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False
    
    return True

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python版本 {version.major}.{version.minor}.{version.micro} 符合要求")
        return True
    else:
        print(f"❌ Python版本 {version.major}.{version.minor}.{version.micro} 过低")
        print("需要Python 3.8或更高版本")
        return False

def install_berttopic():
    """安装BertTopic和相关依赖"""
    print("\n📦 开始安装BertTopic...")
    
    # 升级pip
    if not run_command("python -m pip install --upgrade pip", "升级pip"):
        print("⚠ pip升级失败，继续安装...")
    
    # 安装BertTopic核心依赖
    dependencies = [
        ("umap-learn", "UMAP降维库"),
        ("hdbscan", "层次密度聚类"),
        ("bertopic", "BertTopic主题建模")
    ]
    
    success_count = 0
    for package, description in dependencies:
        if run_command(f"python -m pip install {package}", f"安装{description}"):
            success_count += 1
        else:
            print(f"⚠ {description}安装失败")
    
    return success_count >= 2  # 至少需要UMAP和BertTopic

def test_berttopic_import():
    """测试BertTopic导入"""
    print("\n🧪 测试BertTopic导入...")
    
    try:
        import bertopic
        print("✅ BertTopic导入成功")
        print(f"版本: {bertopic.__version__}")
        return True
    except ImportError as e:
        print(f"❌ BertTopic导入失败: {e}")
        return False

def create_requirements_file():
    """创建requirements文件"""
    print("\n📝 创建requirements文件...")
    
    requirements_content = """# BertTopic和相关依赖
bertopic>=0.15.0
umap-learn>=0.5.3
hdbscan>=0.8.29

# 基础依赖
dash>=2.14.0
plotly>=5.17.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
keybert>=0.7.0
nltk>=3.8
rank-bm25>=0.2.2
rapidfuzz>=3.0.0
joblib>=1.3.0
scipy>=1.10.0
matplotlib>=3.7.0
"""
    
    try:
        with open("requirements_complete.txt", "w", encoding="utf-8") as f:
            f.write(requirements_content)
        print("✅ requirements_complete.txt 创建成功")
        return True
    except Exception as e:
        print(f"❌ 创建requirements文件失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("BertTopic安装助手")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        print("\n💥 Python版本不符合要求，安装终止")
        return False
    
    # 安装BertTopic
    if install_berttopic():
        print("\n🎉 BertTopic安装完成！")
        
        # 测试导入
        if test_berttopic_import():
            print("\n🚀 现在可以运行主应用了！")
            print("运行命令: python finaluimodified.py")
        else:
            print("\n⚠ BertTopic安装可能有问题，但系统仍可使用t-SNE备选方案")
    else:
        print("\n💥 BertTopic安装失败")
        print("系统将使用t-SNE作为备选降维方案")
    
    # 创建requirements文件
    create_requirements_file()
    
    print("\n" + "=" * 60)
    print("安装完成！")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()
