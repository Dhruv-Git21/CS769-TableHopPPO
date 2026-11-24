#!/usr/bin/env bash
# Prints hardware/software facts the orchestrator needs to pick a compute tier.
echo "== date =="; date
echo "== OS =="; uname -a
echo "== CPU =="; (nproc; grep -m1 'model name' /proc/cpuinfo) 2>/dev/null
echo "== RAM =="; (free -h 2>/dev/null || vm_stat 2>/dev/null | head -5)
echo "== disk =="; df -h . | tail -1
echo "== GPU =="; (nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "no nvidia-smi / no NVIDIA GPU")
echo "== python =="; python3 --version 2>&1
python3 - <<'EOF' 2>&1
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          "mps", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"gpu{i}", p.name, round(p.total_memory/2**30, 1), "GiB")
except Exception as e:
    print("torch not importable:", e)
EOF
echo "== tools =="; for t in git latexmk pdflatex bibtex pandoc soffice libreoffice pdftoppm zip; do printf "%s: " $t; (command -v $t || echo MISSING); done
echo "== network =="; (curl -sS -m 8 -o /dev/null -w "huggingface.co %{http_code}\n" https://huggingface.co || echo "huggingface unreachable"); (curl -sS -m 8 -o /dev/null -w "github.com %{http_code}\n" https://github.com || echo "github unreachable")
