"""
doctor.py — Environment Diagnostics & Auto-Fix for 2048 DQN Training

Usage: python doctor.py          # check only
       python doctor.py --fix    # auto-fix issues
"""
import os
import sys
import subprocess
import shutil
import platform
import warnings

# -- ANSI Colors -------------------------------------------
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[1m"; D = "\033[0m"
OK = f"{G}OK{D}"; WARN = f"{Y}WARN{D}"; FAIL = f"{R}FAIL{D}"; FIX = f"{C}FIX{D}"
CHECK = "+"; CROSS = "x"; ARROW = "->"


class Doctor:
    def __init__(self, auto_fix=False):
        self.auto_fix = auto_fix
        self.results = []
        self.fixes_applied = []
        self.all_ok = True

    def _check(self, name, condition, detail="", critical=True):
        status = OK if condition else (FAIL if critical else WARN)
        self.results.append((name, status, detail))
        if not condition and critical:
            self.all_ok = False

    def _can_fix(self, issue):
        return self.auto_fix

    def run(self):
        self._header()
        self._check_python()
        self._check_pytorch_cuda()
        self._check_packages()
        self._check_cpp_compiler()
        self._check_disk()
        self._check_env_vars()
        self._check_system_limits()
        self._report()

    def _header(self):
        print(f"\n{B}2048 DQN V4 — Environment Doctor{D}\n")
        print(f"  Platform: {platform.platform()}")
        print(f"  Python:   {sys.version}")
        print(f"  CWD:      {os.getcwd()}")
        print(f"  Fix mode: {'ON' if self.auto_fix else 'OFF (use --fix)'}")
        print()

    def _check_python(self):
        ver = sys.version_info
        self._check("Python >= 3.9", ver >= (3, 9),
                    f"Python {ver.major}.{ver.minor}.{ver.micro}")

    def _check_pytorch_cuda(self):
        try:
            import torch
            ver = torch.__version__
            self._check("PyTorch installed", True, f"v{ver}")

            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                cuda_ver = torch.version.cuda
                self._check("CUDA available", True,
                            f"GPU: {name} | VRAM: {mem_gb:.1f}GB | CUDA {cuda_ver}")
                self._check("CUDA >= 11.8", cuda_ver and float(cuda_ver) >= 11.8,
                            f"CUDA {cuda_ver}", critical=False)

                # cuDNN
                try:
                    cudnn = torch.backends.cudnn.version()
                    self._check("cuDNN available", cudnn > 0,
                                f"v{cudnn}", critical=False)
                except Exception:
                    self._check("cuDNN available", False, "not detected", critical=False)

                # TF32
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    self._check("TF32 support", True, "", critical=False)
            else:
                self._check("CUDA available", False, "GPU training unavailable")
                if self.auto_fix:
                    print(f"  {Y}No GPU detected. Install PyTorch with CUDA:{D}")
                    print(f"    pip install torch --index-url https://download.pytorch.org/whl/cu124")

        except ImportError:
            self._check("PyTorch installed", False, "pip install torch")
            if self.auto_fix:
                self._pip_install("torch")

    def _check_packages(self):
        pkgs = {
            "numpy": "numpy",
            "tqdm": "tqdm",
            "matplotlib": "matplotlib",
            "pybind11": "pybind11",
        }
        for pkg, install_name in pkgs.items():
            try:
                __import__(pkg)
                self._check(f"Package: {pkg}", True, "", critical=False)
            except ImportError:
                self._check(f"Package: {pkg}", False, "missing", critical=False)
                if self.auto_fix:
                    self._pip_install(install_name)
                    self.fixes_applied.append(f"installed {install_name}")

        # Optional
        for pkg, desc in [("rich", "TUI"), ("pynvml", "GPU monitoring"),
                          ("triton", "torch.compile (Linux)")]:
            try:
                __import__(pkg)
                self._check(f"Optional: {pkg}", True, desc, critical=False)
            except ImportError:
                if pkg == "triton" and platform.system() != "Linux":
                    continue
                self._check(f"Optional: {pkg}", False, desc, critical=False)
                if self.auto_fix and pkg != "triton":
                    self._pip_install(pkg)
                    self.fixes_applied.append(f"installed {pkg}")
                elif self.auto_fix and pkg == "triton" and platform.system() == "Linux":
                    self._pip_install("triton")

    def _check_cpp_compiler(self):
        # Check if C++ extension is already compiled
        ext_file = None
        for f in os.listdir("."):
            if f.startswith("game2048_cpp") and (f.endswith(".pyd") or f.endswith(".so")):
                ext_file = f
                break

        if ext_file:
            self._check("C++ engine", True, f"compiled ({ext_file})")
            return

        # Check compiler availability
        if platform.system() == "Windows":
            has_cl = shutil.which("cl") is not None
            has_gcc = shutil.which("g++") is not None or shutil.which("gcc") is not None
            if has_cl:
                self._check("C++ compiler", True, "MSVC", critical=False)
            elif has_gcc:
                self._check("C++ compiler", True, "MinGW (use --compiler=mingw32)", critical=False)
            else:
                self._check("C++ compiler", False, "not found. Install MSVC or MinGW", critical=False)
        else:
            has_gcc = shutil.which("g++") is not None
            if has_gcc:
                self._check("C++ compiler", True, "g++", critical=False)
            else:
                self._check("C++ compiler", False, "install build-essential", critical=False)

        # Try to compile if fix mode
        if self.auto_fix:
            try:
                cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
                print(f"  {C}Compiling C++ engine...{D}")
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.fixes_applied.append("compiled C++ engine")
            except Exception:
                pass

    def _check_disk(self):
        path = os.getcwd()
        try:
            stat = shutil.disk_usage(path)
            free_gb = stat.free / 1024**3
            self._check("Disk space >= 10GB", free_gb >= 10,
                        f"{free_gb:.1f}GB free", critical=False)
            self._check("Disk space >= 50GB", free_gb >= 50,
                        f"{free_gb:.1f}GB free (recommended)", critical=False)
        except Exception:
            self._check("Disk check", False, "could not determine", critical=False)

    def _check_env_vars(self):
        fixes = []

        # OMP_NUM_THREADS
        omp = os.environ.get("OMP_NUM_THREADS", "")
        if omp:
            self._check("OMP_NUM_THREADS", True, str(omp), critical=False)
        else:
            self._check("OMP_NUM_THREADS", False, "not set", critical=False)
            if self.auto_fix:
                n = min(os.cpu_count() or 8, 16)
                os.environ["OMP_NUM_THREADS"] = str(n)
                fixes.append(f"OMP_NUM_THREADS={n}")

        # MKL
        mkl = os.environ.get("MKL_NUM_THREADS", "")
        if not mkl and self.auto_fix:
            n = min(os.cpu_count() or 8, 8)
            os.environ["MKL_NUM_THREADS"] = str(n)
            fixes.append(f"MKL_NUM_THREADS={n}")

        # CUDA cache
        cuda_cache = os.environ.get("CUDA_CACHE_PATH", "")
        if not cuda_cache and self.auto_fix and platform.system() != "Windows":
            os.environ["CUDA_CACHE_PATH"] = os.path.expanduser("~/.cuda_cache")
            fixes.append("CUDA_CACHE_PATH set")

        # PYTORCH_CUDA_ALLOC_CONF
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and self.auto_fix:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            fixes.append("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

        if fixes:
            self.fixes_applied.extend(fixes)

    def _check_system_limits(self):
        if platform.system() == "Windows":
            return
        try:
            import resource
            nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
            self._check("Open file limit >= 4096", nofile[0] >= 4096,
                        f"{nofile[0]}", critical=False)
        except Exception:
            pass

        # Swap
        try:
            result = subprocess.run(["free", "-g"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    self._check("Swap available", True,
                                f"{parts[2]}GB", critical=False)
        except Exception:
            pass

    def _pip_install(self, pkg):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    def _report(self):
        print(f"\n{B}{'-'*60}{D}")
        print(f"{B}  Results{D}\n")
        for name, status, detail in self.results:
            detail_str = f"  ({detail})" if detail else ""
            print(f"  [{status}] {name}{detail_str}")

        if self.fixes_applied:
            print(f"\n{B}Fixes applied:{D}")
            for f in self.fixes_applied:
                print(f"  {G}+{D} {f}")

        print(f"\n{B}{'-'*60}{D}")
        if self.all_ok:
            print(f"{G}  All checks passed. Ready to train!{D}\n")
        else:
            print(f"{Y}  Some issues found. Run with --fix to auto-resolve.{D}\n")

        # Final recommendations
        print(f"{B}Recommendations:{D}")
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if mem >= 40:
                    print(f"  -> n_envs=128, batch_size=2048 (GPU has {mem:.0f}GB)")
                elif mem >= 20:
                    print(f"  -> n_envs=64, batch_size=1024 (GPU has {mem:.0f}GB)")
                elif mem >= 10:
                    print(f"  -> n_envs=32, batch_size=512 (GPU has {mem:.0f}GB)")
                else:
                    print(f"  -> n_envs=16, batch_size=256 (limited VRAM)")
        except Exception:
            pass

        print(f"  -> Run: bash train_v4.sh  to start training")
        print()


def main():
    auto_fix = "--fix" in sys.argv or "--auto-fix" in sys.argv
    doctor = Doctor(auto_fix=auto_fix)
    doctor.run()
    return 0 if doctor.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
