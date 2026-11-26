#!/usr/bin/env python
"""Launcher: disable TF32 (A800/Ampere) so all matmul/conv run in true fp32, log the flags, then run the target script
unchanged with its own argv.  Usage: launch_fp32.py <script.py> [args...]"""
import runpy, sys
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print('[launch_fp32] torch %s cuda %s device %s | matmul.allow_tf32=%s cudnn.allow_tf32=%s' % (
    torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0),
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32), flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(script)))
runpy.run_path(script, run_name='__main__')
