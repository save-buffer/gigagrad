#pragma once
#include "backend.h"

namespace gigagrad
{

void Eval(codegen::GraphEvalFn fn, std::vector<codegen::BufferDescriptor> buffer_descs);

}
