#include "optimizations.h"

#include <algorithm>
#include <unordered_map>

using namespace gigagrad;
using namespace gigagrad::codegen;

template <typename Fn> static void VisitVariableReferences(Nop &insn, Fn fn) {}
template <typename Fn> static void VisitVariableReferences(LoadIntImmediateInsn &insn, Fn fn) {}
template <typename Fn>
static void VisitVariableReferences(IntArithmeticInsn &insn, Fn fn)
{
    fn(insn.x);
    fn(insn.y);
}
template <typename Fn> static void VisitVariableReferences(BeginLoopInsn &insn, Fn fn) {}
template <typename Fn> static void VisitVariableReferences(EndLoopInsn &insn, Fn fn) {}
template <typename Fn>
static void VisitVariableReferences(LoadInsn &insn, Fn fn)
{
    fn(insn.idx);
}

template <typename Fn>
static void VisitVariableReferences(StoreInsn &insn, Fn fn)
{
    fn(insn.offset);
    fn(insn.value);
}

template <typename Fn> static void VisitVariableReferences(LoadImmediateInsn &insn, Fn fn) {}
template <typename Fn>
static void VisitVariableReferences(UnaryInsn &insn, Fn fn)
{
    fn(insn.x);
}

template <typename Fn>
static void VisitVariableReferences(BinaryInsn &insn, Fn fn)
{
    fn(insn.x);
    fn(insn.y);
}

template <typename Fn>
static void VisitVariableReferences(AccumulateInsn &insn, Fn fn)
{
    fn(insn.accumulator);
    fn(insn.x);
}

template <typename Fn>
static void VisitVariableReferences(Instruction &insn, Fn fn)
{
    std::visit([&](auto &&x) { VisitVariableReferences(x, fn); }, insn);
}
static void OffsetVariableReferences(Instruction &insn, const std::vector<size_t> &insertion_map)
{
    VisitVariableReferences(
        insn,
        [&](size_t &old_ref)
        {
            old_ref += std::upper_bound(
                insertion_map.begin(),
                insertion_map.end(),
                old_ref) - insertion_map.begin();
        });
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

static void ReplaceReferencesWithParent(Instruction &insn, UnionFind &uf)
{
    VisitVariableReferences(
        insn,
        [&](size_t &old_ref)
        {
            old_ref = uf.Find(old_ref);
        });
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
