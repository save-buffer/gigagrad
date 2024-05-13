#pragma once
#include "backend.h"
#include "codegen.h"

namespace gigagrad
{
namespace codegen
{

struct BackendMetal : public Backend
{
    using GraphEvalFn = void (*)(void **);
    virtual ~BackendMetal();
    virtual void LowerProgram(Program &&program);
    virtual void *InitBuffers();
    virtual void *GetBuffer(size_t idx);
    virtual void Execute();

    MTL::Buffer* Allocate(size_t size);
    void LoadLibrary();
    void RunKernel(std::string kernel_name);

    void *handle;
    Program program;
    std::vector<void *> buffers;
    GraphEvalFn eval_fn;
};

}
}
