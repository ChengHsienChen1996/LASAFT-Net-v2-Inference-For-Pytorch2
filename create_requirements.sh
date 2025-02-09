#!/bin/bash

# 確保 pipreqs 和 pip-tools 已安裝
check_dependencies() {
    echo "檢查必要的套件..."
    pip install pipreqs pip-tools
}

# 使用 pipreqs 生成初始的 requirements.txt
generate_initial_requirements() {
    echo "使用 pipreqs 生成初始 requirements.txt..."
    pipreqs . --force
}

# 移除版本號
remove_versions() {
    echo "移除版本號..."
    # 使用 sed 移除版本號，保留套件名稱
    sed -i -E 's/([A-Za-z0-9\-_]+)==.*/\1/g' requirements.txt
    mv requirements.txt requirements.in
}

# 使用 pip-compile 生成最終的 requirements.txt
generate_final_requirements() {
    echo "使用 pip-compile 生成最終 requirements.txt..."
    pip-compile requirements.in
}

# 主要執行流程
main() {
    check_dependencies
    generate_initial_requirements
    remove_versions
    generate_final_requirements
    echo "完成! 已生成 requirements.txt 檔案"
}

# 執行主程式
main