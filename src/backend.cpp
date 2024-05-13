#include "backend.h"
#include "backend_scalar_c.h"
#include "backend_metal.h"
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
    case Backend::Metal:
        return internal::Lower_Metal(prefix, program);
    default:
        throw std::domain_error("Invalid backend");
    }
}
}
}
