python
"""
简化版 Pretty Printer 初始化：
- 自动注册 libstdc++
- 显式注册项目自带的 Eigen 打印脚本
"""

import gdb
import glob
import os
import sys

if not hasattr(gdb, "_miner_pretty_printers"):
    gdb._miner_pretty_printers = True

    def _register_libstdcxx():
        msys_root = os.environ.get("MSYS2_HOME", r"C:/msys64")
        search_roots = [os.path.join(msys_root, subdir, "share") for subdir in ("ucrt64", "mingw64", "clang64")]
        search_roots.append("/usr/share")

        printer_path = None
        for root in search_roots:
            pattern = os.path.join(root, "gcc-*", "python")
            for path in sorted(glob.glob(pattern), reverse=True):
                if os.path.isdir(path):
                    printer_path = path
                    break
            if printer_path:
                break

        if not printer_path:
            gdb.write("警告: 未找到 libstdc++ Pretty Printer 路径，调试输出可能不完整。\n")
            return

        if printer_path not in sys.path:
            sys.path.insert(0, printer_path)

        try:
            from libstdcxx.v6.printers import register_libstdcxx_printers
        except ImportError as exc:
            gdb.write("警告: 无法导入 libstdc++ Pretty Printer: {}\n".format(exc))
            return

        def _register_current(objfile):
            if objfile is None:
                return
            try:
                register_libstdcxx_printers(objfile)
            except Exception as exc:
                gdb.write("libstdc++ Pretty Printer 注册失败: {}\n".format(exc))

        gdb.events.new_objfile.connect(lambda event: _register_current(getattr(event, "new_objfile", None)))
        _register_current(gdb.current_objfile())
        gdb.write("已启用 libstdc++ Pretty Printer，路径: {}\n".format(printer_path))

    def _register_eigen():
        eigen_dir = os.path.normpath(r"D:/workspace/Miner/lib/Eigen/debug/gdb")
        if not os.path.isdir(eigen_dir):
            gdb.write("提示: 未找到 Eigen Pretty Printer 路径: {}\n".format(eigen_dir))
            return

        if eigen_dir not in sys.path:
            sys.path.insert(0, eigen_dir)

        try:
            import printers
            printers.register_eigen_printers(gdb)
            gdb.write("已启用 Eigen Pretty Printer，路径: {}\n".format(eigen_dir))
        except Exception as exc:
            gdb.write("Eigen Pretty Printer 注册失败: {}\n".format(exc))

    _register_libstdcxx()
    _register_eigen()
end

