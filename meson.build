project('gigagrad', 'cpp', default_options : ['cpp_std=c++20'])
gigagrad = library('gigagrad', 'src/graph.cpp')

test_deps = [dependency('catch2-with-main')]
vizserver_deps = [dependency('cpp-httplib'), dependency('nlohmann_json')]
executable('gigagrad-test', 'test/graph-test.cpp', dependencies : test_deps, link_with : gigagrad)
executable('gigagrad-viz-server', 'src/gigagrad_viz_server.cpp',
           dependencies : vizserver_deps,
           link_with : gigagrad)
