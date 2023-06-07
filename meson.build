project('gigagrad', 'cpp', default_options : ['cpp_std=c++20'])
gigagrad = library('gigagrad', 'src/graph.cpp')

testdeps = [dependency('catch2-with-main')]
executable('gigagrad-test', 'test/graph-test.cpp', dependencies : testdeps, link_with : gigagrad)