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
    virtual ~BackendScalarC() = default;
    virtual void LowerProgram(Program &&program);
    virtual void *InitBuffers();
    virtual void Execute();

    Program program;
    std::vector<void *> buffers;
    GraphEvalFn eval_fn;
};

}
}
