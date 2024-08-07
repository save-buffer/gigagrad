#pragma once
#include "codegen.h"

namespace gigagrad
{
namespace codegen
{
std::vector<Instruction> TileLoops(const std::vector<Instruction> &insns, size_t tile_by);
std::vector<Instruction> EliminateCommonSubexpressions(const std::vector<Instruction> &insns);
std::vector<Instruction> SimplifyAddressExpressions(const std::vector<Instruction> &insns);
}
}
