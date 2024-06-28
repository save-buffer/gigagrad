project('gigagrad', 'cpp', default_options : ['cpp_std=c++20'])

gigagrad_deps = []
gigagrad_sources = [
    'src/graph.cpp',
    'src/codegen.cpp',
    'src/backend_scalar_c.cpp',
    'src/training.cpp',
    'src/debug_util.cpp',
    'src/optimizations.cpp',
  ]

if host_machine.system() == 'darwin'
  fs = import('fs')
  if not fs.is_dir('metal-cpp')
    run_command('wget', 'https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip')
    run_command('unzip', 'metal-cpp_macOS14.2_iOS17.2.zip')
    run_command('python3', 'metal-cpp/SingleHeader/MakeSingleHeader.py', 'metal-cpp/Foundation/Foundation.hpp', 'metal-cpp/QuartzCore/QuartzCore.hpp', 'metal-cpp/Metal/Metal.hpp')
  endif
  gigagrad_deps += dependency('appleframeworks', modules : ['foundation', 'quartz', 'metal'])
  gigagrad_sources += 'src/backend_metal.cpp'
  add_global_arguments('-DHAS_METAL', language : 'cpp')
endif

gigagrad = static_library('gigagrad', gigagrad_sources, dependencies : gigagrad_deps)

test_deps = [dependency('catch2-with-main')]

executable('gigagrad-test', 'test/graph-test.cpp', dependencies : test_deps, link_with : gigagrad)
executable('gigagrad-emnist', 'test/gigagrad-emnist.cpp', link_with : gigagrad)
executable('gigagrad-matmul-benchmark', 'test/matmul_benchmark.cpp', link_with : gigagrad)
