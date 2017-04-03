import os
import sys
import subprocess
import multiprocessing

msbuild = r'C:\Program Files (x86)\MSBuild\14.0\Bin\MSBuild.exe'
assert os.path.exists(msbuild), 'Expect MSBuild.exe installed at %s' % msbuild
cpu_count = multiprocessing.cpu_count()

if not 'PYTHON_ROOT' in os.environ:
    python_exe = sys.executable
    assert bool(python_exe), 'python.exe not found'
    os.environ['PYTHON_ROOT'] = os.path.split(sys.executable)[0]
    print('set PYTHON_ROOT to %s' % os.environ['PYTHON_ROOT'])

subprocess.call([msbuild, 'caffe.sln', '/t:Build', '/p:Configuration=Release', '/maxcpucount:%d' % max(1, cpu_count - 4)], shell=True)
