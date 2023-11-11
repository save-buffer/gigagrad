project('gigagrad', 'cpp', default_options : ['cpp_std=c++20'])

gigagrad_sources = ['src/graph.cpp', 'src/codegen.cpp', 'src/backend_scalar_c.cpp', 'src/training.cpp']
gigagrad = library('gigagrad', gigagrad_sources)

test_deps = [dependency('catch2-with-main')]

executable('gigagrad-test', 'test/graph-test.cpp', dependencies : test_deps, link_with : gigagrad)
executable('gigagrad-emnist', 'test/gigagrad-emnist.cpp', link_with : gigagrad)
