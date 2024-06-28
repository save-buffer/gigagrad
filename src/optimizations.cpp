#include "optimizations.h"

#include <algorithm>
#include <unordered_map>

using namespace gigagrad;
using namespace gigagrad::codegen;

size_t OffsetInsnRef(size_t old_ref, const std::vector<size_t> &insertion_map)
{
    return old_ref + std::upper_bound(insertion_map.begin(), insertion_map.end(), old_ref) - insertion_map.begin();
}

static void OffsetVariableReferences(Nop &insn, const std::vector<size_t> &insertion_map) {}
static void OffsetVariableReferences(LoadIntImmediateInsn &insn, const std::vector<size_t> &insertion_map) {}
static void OffsetVariableReferences(IntArithmeticInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.x = OffsetInsnRef(insn.x, insertion_map);
    insn.y = OffsetInsnRef(insn.y, insertion_map);
}

static void OffsetVariableReferences(BeginLoopInsn &insn, const std::vector<size_t> &insertion_map) {}
static void OffsetVariableReferences(EndLoopInsn &insn, const std::vector<size_t> &insertion_map) {}
static void OffsetVariableReferences(LoadInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.idx = OffsetInsnRef(insn.idx, insertion_map);
}

static void OffsetVariableReferences(StoreInsn &insn, const std::vector<size_t> &insertion_map)
{
    insn.offset = OffsetInsnRef(insn.offset, insertion_map);
    insn.value = OffsetInsnRef(insn.value, insertion_map);
}

static void OffsetVariableReferences(LoadImmediateInsn &insn, const std::vector<size_t> &insertion_map) {}
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

struct UnionFind
{
    UnionFind(size_t n) : parent(n)
    {
        std::iota(parent.begin(), parent.end(), 0);
    }

    void Union(size_t a, size_t b)
    {
        size_t pa = this->Find(a);
        size_t pb = this->Find(b);
        if(pa < pb)
            parent[b] = pa;
        else
            parent[a] = pb;
    }

    size_t Find(size_t x)
    {
        while(x != parent[x])
        {
            size_t p = parent[x];
            parent[x] = parent[p];
            x = p;
        }
        return x;
    }

    std::vector<size_t> parent;
};

static void ReplaceReferencesWithParent(Nop &, UnionFind &uf) {}
static void ReplaceReferencesWithParent(LoadIntImmediateInsn &, UnionFind &uf) {}
static void ReplaceReferencesWithParent(IntArithmeticInsn &insn, UnionFind &uf)
{
    insn.x = uf.Find(insn.x);
    insn.y = uf.Find(insn.y);
}
static void ReplaceReferencesWithParent(BeginLoopInsn &, UnionFind &uf) {}
static void ReplaceReferencesWithParent(EndLoopInsn &, UnionFind &uf) {}
static void ReplaceReferencesWithParent(LoadInsn &insn, UnionFind &uf)
{
    insn.idx = uf.Find(insn.idx);
}

static void ReplaceReferencesWithParent(StoreInsn &insn, UnionFind &uf)
{
    insn.offset = uf.Find(insn.offset);
}

static void ReplaceReferencesWithParent(LoadImmediateInsn &insn, UnionFind &uf) {}
static void ReplaceReferencesWithParent(UnaryInsn &insn, UnionFind &uf)
{
    insn.x = uf.Find(insn.x);
}

static void ReplaceReferencesWithParent(BinaryInsn &insn, UnionFind &uf)
{
    insn.x = uf.Find(insn.x);
    insn.y = uf.Find(insn.y);
}

static void ReplaceReferencesWithParent(AccumulateInsn &insn, UnionFind &uf)
{
    insn.accumulator = uf.Find(insn.accumulator);
    insn.x = uf.Find(insn.x);
}

static void ReplaceReferencesWithParent(Instruction &insn, UnionFind &uf)
{
    std::visit([&](auto &&x) { ReplaceReferencesWithParent(x, uf); }, insn);
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

std::vector<Instruction> gigagrad::codegen::EliminateCommonSubexpressions(const std::vector<Instruction> &insns)
{
    std::vector<Instruction> result;
    std::unordered_map<int64_t, size_t> value_map;
    UnionFind uf(insns.size());
    for(size_t iinsn = 0; iinsn < insns.size(); iinsn++)
    {
        if(std::holds_alternative<LoadIntImmediateInsn>(insns[iinsn]))
        {
            const LoadIntImmediateInsn &li = std::get<LoadIntImmediateInsn>(insns[iinsn]);
            if(value_map.contains(li.value))
            {
                uf.Union(iinsn, value_map[li.value]);
                result.push_back(Nop{});
            }
            else
            {
                value_map[li.value] = iinsn;
                result.push_back(insns[iinsn]);
            }
        }
        else
        {
            Instruction insn = insns[iinsn];
            ReplaceReferencesWithParent(insn, uf);
            result.push_back(insn);
        }
    }
    return result;
}
