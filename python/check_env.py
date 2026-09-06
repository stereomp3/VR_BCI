"""
VR-BCI 系統環境安裝與相容性檢查工具 (Environment & Dependency Checker)
用於檢查目前 Python 版本、PyTorch CUDA 加速、神經生理與機器學習依賴套件。
"""

import sys
import os
import importlib
import platform

# 確保支援 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def print_section(title):
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def check_module(name, display_name=None, min_version=None):
    display_name = display_name or name
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "已安裝 (無版本號)")
        status = "✅ [PASS]"
        note = f"v{ver}"
        is_ok = True
        return is_ok, f"{status} {display_name:<20}: {note}"
    except ImportError as e:
        status = "❌ [FAIL]"
        return False, f"{status} {display_name:<20}: 未安裝 ({e})"


def main():
    print("=" * 75)
    print("🔍 VR-BCI 專案環境完整性檢測工具")
    print("=" * 75)

    all_passed = True

    # 1. 系統與 Python 版本
    print_section("1. 系統與 Python 直譯器")
    py_ver = platform.python_version()
    py_major, py_minor, py_micro = sys.version_info[:3]
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"

    print(f"• 作業系統      : {os_name}")
    print(f"• Python 直譯器 : {sys.executable}")
    print(f"• Python 版本   : v{py_ver}", end=" ")

    if py_major == 3 and py_minor in (10, 11):
        print("✅ [建議版本 3.11.8 相容]")
    else:
        print("⚠️ [建議使用 Python 3.11.8 以獲得最佳穩定性]")

    # 2. PyTorch 與 CUDA GPU 支援
    print_section("2. PyTorch 與 GPU / CUDA 加速檢測")
    torch_ok, msg = check_module("torch", "PyTorch")
    print(msg)

    if torch_ok:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_ver = torch.version.cuda
            cudnn_ver = torch.backends.cudnn.version()
            print(f"✅ [PASS] CUDA 支援         : 是 (CUDA v{cuda_ver})")
            print(f"✅ [PASS] GPU 裝置名稱      : {gpu_name} (共 {gpu_count} 張 GPU)")
            print(f"✅ [PASS] 顯示卡記憶體 (VRAM): {vram_gb:.2f} GB")
            print(f"✅ [PASS] cuDNN 版本        : v{cudnn_ver}")

            # CUDA Smoke Test
            try:
                x = torch.randn(100, 100, device="cuda")
                y = torch.matmul(x, x)
                torch.cuda.synchronize()
                print("✅ [PASS] GPU 矩陣運算測試  : 成功 (CUDA 核心運作正常)")
            except Exception as e:
                print(f"❌ [FAIL] GPU 矩陣運算失敗  : {e}")
                all_passed = False
        else:
            print("⚠️ [WARN] CUDA 支援         : 否 (未檢測到可用 GPU，將使用 CPU 運算)")
            print("   👉 若您具備 NVIDIA 顯卡，請安裝對應 CUDA 版本的 PyTorch：")
            print("      pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121")
    else:
        all_passed = False

    # 3. 核心數據處理與機器學習庫
    print_section("3. 核心數據分析與機器學習依賴")
    core_packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-Learn"),
        ("statsmodels", "Statsmodels"),
    ]
    for pkg, disp in core_packages:
        ok, msg = check_module(pkg, disp)
        print(msg)
        if not ok:
            all_passed = False

    # 4. 腦波與 BCI 深度學習專用套件
    print_section("4. 腦波 (EEG) 與 BCI 模型庫")
    eeg_packages = [
        ("mne", "MNE-Python (腦波前處理)"),
        ("captum", "Captum (Saliency Map 運算)"),
        ("braindecode", "Braindecode (BCI 模型庫)"),
        ("pylsl", "PyLSL (Lab Streaming Layer)"),
    ]
    for pkg, disp in eeg_packages:
        ok, msg = check_module(pkg, disp)
        print(msg)
        if not ok:
            all_passed = False

    # 5. XBrainLab 專用可選套件
    print_section("5. 專案特徵視覺化庫")
    xb_ok, xb_msg = check_module("XBrainLab", "XBrainLab (可視化延伸)")
    print(xb_msg)
    if not xb_ok:
        print("   ℹ️ [提示] 若需要使用 XBrainLab 繪製 Saliency Map 空間拓撲圖，可將 XBrainLab 加入 PYTHONPATH。")

    # 6. 總結
    print_section("檢測結果總結")
    if all_passed:
        print("🎉 恭喜！所有核心依賴與執行環境檢測全數通過，您可以正常運行 VR-BCI 系統與分析管線！")
    else:
        print("⚠️ 檢測到部分依賴套件缺失，請參考上方提示或執行：")
        print("   pip install -r python/requirements.txt")

    print("=" * 75 + "\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
