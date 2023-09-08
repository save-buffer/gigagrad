#pragma once
#include "backend.h"
#include "codegen.h"

namespace Gigagrad
{
namespace Codegen
{
namespace Internal
{
GraphEvalFn Lower_ScalarC(const char *prefix, const Program &program);
}
}
}
