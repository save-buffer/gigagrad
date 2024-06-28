#include "optimizations.h"

#include <algorithm>

using namespace gigagrad;
using namespace gigagrad::codegen;

size_t OffsetInsnRef(size_t old_ref, const std::vector<size_t> &insertion_map)
{
    return old_ref + std::upper_bound(insertion_map.begin(), insertion_map.end(), old_ref) - insertion_map.begin();
}

static void OffsetVariableReferences(LoadIntImmediateInsn &insn, const std::vector<size_t> &insertion_map)
{
}

static void OffsetVariableReferences(IntArithmeticInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.x = OffsetInsnRef(insn.x, insertion_map);
    insn.y = OffsetInsnRef(insn.y, insertion_map);
}

static void OffsetVariableReferences(BeginLoopInsn &insn, const std::vector<size_t> &insertion_map)
{
}

static void OffsetVariableReferences(EndLoopInsn &insn, const std::vector<size_t> &insertion_map)
{
}

static void OffsetVariableReferences(LoadInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.idx = OffsetInsnRef(insn.idx, insertion_map);
}

static void OffsetVariableReferences(StoreInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.offset = OffsetInsnRef(insn.offset, insertion_map);
    insn.value = OffsetInsnRef(insn.value, insertion_map);
}

static void OffsetVariableReferences(LoadImmediateInsn &insn, const std::vector<size_t> &insertion_map)
{
}

static void OffsetVariableReferences(UnaryInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.x = OffsetInsnRef(insn.x, insertion_map);
}

static void OffsetVariableReferences(BinaryInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.x = OffsetInsnRef(insn.x, insertion_map);
    insn.y = OffsetInsnRef(insn.y, insertion_map);
}

static void OffsetVariableReferences(AccumulateInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.accumulator = OffsetInsnRef(insn.accumulator, insertion_map);
    insn.x = OffsetInsnRef(insn.x, insertion_map);
}

static void OffsetVariableReferences(Instruction &insn, const std::vector<size_t> &insertion_map)
{
    std::visit([&](auto &&x) { OffsetVariableReferences(x, insertion_map); }, insn);
}

std::vector<Instruction> gigagrad::codegen::TileLoops(const std::vector<Instruction> &insns, size_t tile_by)
{
    std::vector<Instruction> result;
    std::vector<size_t> loops;
    for(size_t iinsn = 0; iinsn < insns.size(); iinsn++)
    {
        if(std::holds_alternative<BeginLoopInsn>(insns[iinsn]))
            if(std::get<BeginLoopInsn>(insns[iinsn]).range % tile_by == 0)
                loops.push_back(iinsn);
    }

    std::vector<size_t> insertion_map;
    for(size_t i = 0; i < loops.size(); i++)
    {
        size_t loop_idx = loops[i];
        result.push_back(insns[loop_idx]);
        std::get<BeginLoopInsn>(result.back()).step = tile_by;
        insertion_map.push_back(0);
    }
    size_t iloop_seen = 0;
    for(size_t iinsn = 0; iinsn < insns.size(); iinsn++)
    {
        const Instruction &insn = insns[iinsn];
        result.push_back(insn);
        if(std::holds_alternative<BeginLoopInsn>(insn)
           && std::get<BeginLoopInsn>(insn).range % tile_by == 0)
        {
            std::get<BeginLoopInsn>(result.back()).range = tile_by;
            result.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, iloop_seen, result.size() - 1});
            insertion_map.push_back(iinsn);

            iloop_seen += 1;
        }
        else
        {
            OffsetVariableReferences(result.back(), insertion_map);
        }
    }
    for(size_t i = 0; i < loops.size(); i++)
    {
        result.push_back(EndLoopInsn{});
    }
    return result;
}
