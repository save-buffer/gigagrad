#pragma once
#include "backend.h"
#include "codegen.h"

namespace gigagrad
{
namespace codegen
{

struct BackendScalarC : public Backend
{
    using GraphEvalFn = void (*)(void **);
    virtual ~BackendScalarC();
    virtual void LowerProgram(Program &&program);
    virtual void *InitBuffers();
    virtual void *GetBuffer(size_t idx);
    virtual void Execute();

    void *handle;
    Program program;
    std::vector<void *> buffers;
    GraphEvalFn eval_fn;
};

}
}
