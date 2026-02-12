import platform
import sys
import importlib

PKGS = ["scanpy", "anndata", "numpy", "pandas", "scipy", "liana", "torch"]

def main():
    print("python:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    for p in PKGS:
        try:
            m = importlib.import_module(p)
            ver = getattr(m, "__version__", "unknown")
            print(f"{p}: {ver}")
        except Exception as e:
            print(f"{p}: NOT IMPORTABLE ({e})")

if __name__ == "__main__":
    main()

