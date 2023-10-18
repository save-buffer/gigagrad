#include "backend.h"
#include "backend_scalar_c.h"
#include "codegen.h"

namespace gigagrad
{
namespace codegen
{

GraphEvalFn LowerProgram(const char *prefix, Backend backend, const Program &program)
{
    switch(backend)
    {
    case Backend::ScalarC:
        return internal::Lower_ScalarC(prefix, program);
    default:
        throw std::domain_error("Invalid backend");
    }
}
}
}
