project('gigagrad', 'cpp', default_options : ['cpp_std=c++20'])

gigagrad_sources = ['src/graph.cpp', 'src/codegen.cpp', 'src/backend.cpp', 'src/backend_scalar_c.cpp']
gigagrad = library('gigagrad', gigagrad_sources)

test_deps = [dependency('catch2-with-main')]
vizserver_deps = [dependency('cpp-httplib')]
executable('gigagrad-test', 'test/graph-test.cpp', dependencies : test_deps, link_with : gigagrad)
executable('gigagrad-viz-server', 'src/gigagrad_viz_server.cpp',
           dependencies : vizserver_deps,
           link_with : gigagrad)
