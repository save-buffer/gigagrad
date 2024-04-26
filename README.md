# Gigagrad
A small deep learning library that goes gigafast (not yet though). Gigagrad makes heavy use of C++'s operator overloading to
provide an ergonomic way to define neural networks, without all the runtime overhead of Python. Eventually, Gigagrad will be
able to generate executables or static libraries containing neural networks. Gigagrad's implementation takes inspiration from Tinygrad
and Pytorch.

# Building
This project uses the [Meson](https://mesonbuild.com/Getting-meson.html) build system. You can 
install it with `pip3` with `pip3 install meson`. Next, building Gigagrad is as simple as
```
    meson setup build
    cd build
    meson compile
```
From there, you can run tests such as `./gigagrad-test`.

# Usage
```c++
// Declare a network
gg::nn::Module network;

// Add an input vector of length 4
auto x = network.AddInput(4);

// Add a weight of length 4
auto w = network.AddWeight(4);

// L1 is now the elementwise difference between w and x
auto L1 = w - x;

// Compile the training context. Currently, we only have a scalar C backend
gg::TrainingContext ctx = gg::CompileTrainingGraph<gg::codegen::BackendScalarC>(network, L1);

// Set input data
float x_data[] = { 1.0, 2.0, 3.0, 4.0 };
float w_data[] = { -0.1, 0.1, -0.001, 0.0001 };
float training_example_data[] = { 0.0, 0.0, 0.0, 0.0 };
x.data() = x_data;
w.data() = w_data;
ctx.training_example = training_example_data;

// Run twenty iterations of gradient descent, and print the loss!
for(int i = 0; i < 20; i++)
{
    ctx.Execute();
    printf("%.6f\n", ctx.loss[0]);
}
// Print your learned weight: W = { 0.98, 1.97, 2.96, 3.94 }
printf("W = { %.2f, %.2f, %.2f, %.2f }\n", w_data[0], w_data[1], w_data[2], w_data[3]);
```

# Backends
- [x] Scalar C (useful for debugging)
- [ ] OpenMP with SIMD
- [ ] CUDA
- [ ] TensTorrent Metallium
- [ ] Intel OneAPI
- [ ] Vulkan
