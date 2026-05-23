"""Internal: GPU/PyTorch compatibility check for train_v4.bat"""
import subprocess, sys

def check_gpu():
    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
    return result.returncode == 0

def check_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cc = f'{props.major}{props.minor}'
            return {'ok': True, 'cc': cc, 'pt_ver': torch.__version__,
                    'cuda_ver': torch.version.cuda or '0'}
        return {'ok': False, 'reason': 'CPU-only PyTorch'}
    except ImportError:
        return {'ok': False, 'reason': 'PyTorch not installed'}

has_gpu = check_gpu()
if not has_gpu:
    print('NO_GPU')
    sys.exit(0)

info = check_cuda()
if not info['ok']:
    print(f"NEED_CUDA:{info['reason']}")
elif int(info['cc']) >= 120:
    pt_ver = tuple(int(x) for x in info['pt_ver'].split('.')[:2])
    cu_ver = float(info['cuda_ver'])
    if pt_ver < (2, 7) or cu_ver < 12.8:
        print(f"BLACKWELL:{info['cc']}:{info['pt_ver']}")
    else:
        print(f"OK:{info['cc']}")
else:
    print(f"OK:{info['cc']}")
