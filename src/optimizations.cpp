#include "optimizations.h"

#include <algorithm>
#include <cassert>
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

static std::vector<Instruction> EliminateCommonConstants(const std::vector<Instruction> &insns)
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

static Instruction SimplifyAdd(
    size_t me,
    size_t x,
    size_t y,
    const std::vector<Instruction> &insns,
    UnionFind &uf)
{
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x])
       && std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
    {
        int64_t value = std::get<LoadIntImmediateInsn>(insns[x]).value + std::get<LoadIntImmediateInsn>(insns[y]).value;
        return LoadIntImmediateInsn{value};
    }

    if(std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
        std::swap(x, y);

    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x]))
        if(std::get<LoadIntImmediateInsn>(insns[x]).value == 0)
        {
            uf.Union(me, y);
            return Nop{};
        }
    return insns[me];
}

static Instruction SimplifySub(
    size_t me,
    size_t x,
    size_t y,
    const std::vector<Instruction> &insns,
    UnionFind &uf)
{
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x])
       && std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
    {
        int64_t value = std::get<LoadIntImmediateInsn>(insns[x]).value - std::get<LoadIntImmediateInsn>(insns[y]).value;
        return LoadIntImmediateInsn{value};
    }

    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x]))
        if(std::get<LoadIntImmediateInsn>(insns[x]).value == 0)
        {
            uf.Union(me, y);
            return Nop{};
        }
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
        if(std::get<LoadIntImmediateInsn>(insns[y]).value == 0)
        {
            uf.Union(me, x);
            return Nop{};
        }

    return IntArithmeticInsn{IntArithmeticInsn::Op::SUB, x, y};
}

static Instruction SimplifyMul(
    size_t me,
    size_t x,
    size_t y,
    const std::vector<Instruction> &insns,
    UnionFind &uf)
{
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x])
       && std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
    {
        int64_t value = std::get<LoadIntImmediateInsn>(insns[x]).value * std::get<LoadIntImmediateInsn>(insns[y]).value;
        return LoadIntImmediateInsn{value};
    }

    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x]))
    {
        if(std::get<LoadIntImmediateInsn>(insns[x]).value == 0)
            return LoadIntImmediateInsn{0};
        if(std::get<LoadIntImmediateInsn>(insns[x]).value == 1)
        {
            uf.Union(me, y);
            return Nop{};
        }
    }
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
    {
        if(std::get<LoadIntImmediateInsn>(insns[y]).value == 0)
            return LoadIntImmediateInsn{0};
        if(std::get<LoadIntImmediateInsn>(insns[y]).value == 1)
        {
            uf.Union(me, x);
            return Nop{};
        }
    }
    return IntArithmeticInsn{IntArithmeticInsn::Op::MUL, x, y};
}

static Instruction SimplifyDiv(
    size_t me,
    size_t x,
    size_t y,
    const std::vector<Instruction> &insns,
    UnionFind &uf)
{
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x])
       && std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
    {
        int64_t value = std::get<LoadIntImmediateInsn>(insns[x]).value / std::get<LoadIntImmediateInsn>(insns[y]).value;
        return LoadIntImmediateInsn{value};
    }

    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x]))
    {
        if(std::get<LoadIntImmediateInsn>(insns[x]).value == 0)
            return LoadIntImmediateInsn{0};
    }
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
    {
        if(std::get<LoadIntImmediateInsn>(insns[y]).value == 1)
        {
            uf.Union(me, x);
            return Nop{};
        }
    }

    return IntArithmeticInsn{IntArithmeticInsn::Op::DIV, x, y};
}

static Instruction SimplifyMod(
    size_t me,
    size_t x,
    size_t y,
    const std::vector<Instruction> &insns,
    UnionFind &uf)
{
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x])
       && std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
    {
        int64_t value = std::get<LoadIntImmediateInsn>(insns[x]).value % std::get<LoadIntImmediateInsn>(insns[y]).value;
        return LoadIntImmediateInsn{value};
    }

    if(std::holds_alternative<LoadIntImmediateInsn>(insns[x]))
    {
        if(std::get<LoadIntImmediateInsn>(insns[x]).value == 0)
            return LoadIntImmediateInsn{0};
    }
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[y]))
    {
        if(std::get<LoadIntImmediateInsn>(insns[y]).value == 1)
            return LoadIntImmediateInsn{0};
    }

    return IntArithmeticInsn{IntArithmeticInsn::Op::MOD, x, y};
}

static Instruction SimplifyIntArith(
    size_t me,
    size_t x,
    size_t y,
    const std::vector<Instruction> &insns,
    UnionFind &uf)
{
    IntArithmeticInsn::Op op = std::get<IntArithmeticInsn>(insns[me]).op;
    switch(op)
    {
    case IntArithmeticInsn::Op::ADD:
        return SimplifyAdd(me, x, y, insns, uf);
    case IntArithmeticInsn::Op::SUB:
        return SimplifySub(me, x, y, insns, uf);
    case IntArithmeticInsn::Op::MUL:
        return SimplifyMul(me, x, y, insns, uf);
    case IntArithmeticInsn::Op::DIV:
        return SimplifyDiv(me, x, y, insns, uf);
    case IntArithmeticInsn::Op::MOD:
        return SimplifyMod(me, x, y, insns, uf);
    }
    throw std::runtime_error("This should be unreachable!");
}

static std::vector<Instruction> PropagateConstants(const std::vector<Instruction> &insns)
{
    UnionFind uf(insns.size());
    std::vector<Instruction> result;
    for(size_t iinsn = 0; iinsn < insns.size(); iinsn++)
    {
        const Instruction &insn = insns[iinsn];
        if(std::holds_alternative<IntArithmeticInsn>(insn))
        {
            const IntArithmeticInsn &arith = std::get<IntArithmeticInsn>(insn);
            Instruction simplified = SimplifyIntArith(iinsn, arith.x, arith.y, insns, uf);
            ReplaceReferencesWithParent(simplified, uf);
            result.push_back(simplified);
        }
        else
        {
            Instruction updated = insn;
            ReplaceReferencesWithParent(updated, uf);
            result.push_back(updated);
        }
    }
    return result;
}

static bool IsConstant(const Instruction &insn)
{
    return std::holds_alternative<LoadIntImmediateInsn>(insn);
}

static int64_t GetConstant(const Instruction &insn)
{
    return std::get<LoadIntImmediateInsn>(insn).value;
}

static bool HasOp(const Instruction &insn, IntArithmeticInsn::Op op)
{
    return std::holds_alternative<IntArithmeticInsn>(insn)
        && std::get<IntArithmeticInsn>(insn).op == op;
}

static const IntArithmeticInsn &GetOp(const Instruction &insn)
{
    return std::get<IntArithmeticInsn>(insn);
}

static void CanonicalizeAndSimplifyAdd(
    size_t iinsn,
    Instruction insn,
    std::vector<Instruction> &insns,
    std::vector<size_t> &insertion_map)
{
    const IntArithmeticInsn &arith = GetOp(insn);
    if(IsConstant(insns[arith.x]) && IsConstant(insns[arith.y]))
    {
        int64_t val = GetConstant(insns[arith.x]) + GetConstant(insns[arith.y]);
        insns.push_back(LoadIntImmediateInsn{val});
    }
    else if(IsConstant(insns[arith.y]))
    {
        if(GetConstant(insns[arith.y]) == 0)
        {
            insns.push_back(insns[arith.x]);
        }
        else
        {
            IntArithmeticInsn updated = arith;
            std::swap(updated.x, updated.y);
            insns.push_back(updated);
        }
    }
    else if(HasOp(insns[arith.y], IntArithmeticInsn::Op::ADD))
    {
        const IntArithmeticInsn &rhs = GetOp(insns[arith.y]);
        size_t t1 = arith.x;
        size_t t2 = rhs.x;
        size_t t3 = rhs.y;
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, t1, t2});
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, insns.size() - 1, t3});
        insertion_map.push_back(iinsn);
    }
    else if(HasOp(insns[arith.x], IntArithmeticInsn::Op::ADD))
    {
        const IntArithmeticInsn &lhs = GetOp(insns[arith.x]);
        if(IsConstant(insns[lhs.x]) && IsConstant(insns[arith.y]))
        {
            int64_t val = GetConstant(insns[lhs.x]) + GetConstant(insns[arith.y]);
            insns.push_back(LoadIntImmediateInsn{val});
            insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, insns.size() - 1, lhs.y});
            insertion_map.push_back(iinsn);
        }
        else
        {
            insns.push_back(arith);
        }
    }
    else
    {
        insns.push_back(arith);
    }
}

static void CanonicalizeAndSimplifyMul(
    size_t iinsn,
    Instruction insn,
    std::vector<Instruction> &insns,
    std::vector<size_t> &insertion_map)
{
    const IntArithmeticInsn &arith = GetOp(insn);
    if(IsConstant(insns[arith.x]) && IsConstant(insns[arith.y]))
    {
        int64_t val = GetConstant(insns[arith.x]) * GetConstant(insns[arith.y]);
        insns.push_back(LoadIntImmediateInsn{val});
    }
    else if(IsConstant(insns[arith.y]))
    {
        if(GetConstant(insns[arith.y]) == 0)
        {
            insns.push_back(LoadIntImmediateInsn{0});
        }
        else if(GetConstant(insns[arith.y]) == 1)
        {
            insns.push_back(insns[arith.x]);
        }
        else if(HasOp(insns[arith.x], IntArithmeticInsn::Op::ADD))
        {
            if(IsConstant(insns[GetOp(insns[arith.x]).x]))
            {
                auto c1 = GetOp(insns[arith.x]).x;
                auto c2 = arith.y;
                auto t = GetOp(insns[arith.x]).y;
                int64_t val = GetConstant(insns[c1]) * GetConstant(insns[c2]);
                insns.push_back(LoadIntImmediateInsn{val});
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, c2, t});
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, insns.size() - 2, insns.size() - 1});
                insertion_map.push_back(iinsn);
                insertion_map.push_back(iinsn);
            }
            else
            {
                auto t1 = GetOp(insns[arith.x]).x;
                auto t2 = GetOp(insns[arith.x]).y;
                auto c = arith.y;
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, c, t1});
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, c, t2});
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, insns.size() - 2, insns.size() - 1});
                insertion_map.push_back(iinsn);
                insertion_map.push_back(iinsn);
            }
        }
        else
        {
            IntArithmeticInsn updated = arith;
            std::swap(updated.x, updated.y);
            insns.push_back(updated);
        }
    }
    else if(IsConstant(insns[arith.x]))
    {
        if(GetConstant(insns[arith.x]) == 0)
        {
            insns.push_back(LoadIntImmediateInsn{0});
        }
        else if(GetConstant(insns[arith.x]) == 1)
        {
            insns.push_back(insns[arith.y]);
        }
        else if(HasOp(insns[arith.y], IntArithmeticInsn::Op::ADD))
        {
            if(IsConstant(insns[GetOp(insns[arith.y]).x]))
            {
                auto c1 = arith.x;
                auto c2 = GetOp(insns[arith.y]).x;
                auto t = GetOp(insns[arith.y]).y;
                int64_t val = GetConstant(insns[c1]) * GetConstant(insns[c2]);
                insns.push_back(LoadIntImmediateInsn{val});
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, c1, t});
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, insns.size() - 2, insns.size() - 1});
                insertion_map.push_back(iinsn);
                insertion_map.push_back(iinsn);
            }
            else
            {
                auto t1 = GetOp(insns[arith.y]).x;
                auto t2 = GetOp(insns[arith.y]).y;
                auto c = arith.y;
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, c, t1});
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, c, t2});
                insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, insns.size() - 2, insns.size() - 1});
                insertion_map.push_back(iinsn);
                insertion_map.push_back(iinsn);
            }
        }
        else
        {
            insns.push_back(arith);
        }
    }
    else if(HasOp(insns[arith.y], IntArithmeticInsn::Op::MUL))
    {
        const IntArithmeticInsn &rhs = GetOp(insns[arith.y]);
        auto t1 = arith.x;
        auto t2 = rhs.x;
        auto t3 = rhs.y;
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, t1, t2});
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, insns.size() - 1, t3});
        insertion_map.push_back(iinsn);
    }
    else if(HasOp(insns[arith.x], IntArithmeticInsn::Op::MUL))
    {
        const IntArithmeticInsn &lhs = GetOp(insns[arith.x]);
        if(IsConstant(insns[lhs.x]) && IsConstant(insns[arith.y]))
        {
            int64_t val = GetConstant(insns[lhs.x]) * GetConstant(insns[arith.y]);
            insns.push_back(LoadIntImmediateInsn{val});
            insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, insns.size() - 1, lhs.y});
            insertion_map.push_back(iinsn);
        }
        else
        {
            insns.push_back(arith);
        }
    }
    else if(HasOp(insns[arith.x], IntArithmeticInsn::Op::ADD))
    {
        auto t1 = GetOp(insns[arith.x]).x;
        auto t2 = GetOp(insns[arith.x]).y;
        auto t3 = arith.y;
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, t1, t3});
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, t2, t3});
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, insns.size() - 2, insns.size() - 1});
        insertion_map.push_back(iinsn);
        insertion_map.push_back(iinsn);
    }
    else if(HasOp(insns[arith.y], IntArithmeticInsn::Op::ADD))
    {
        auto t1 = arith.x;
        auto t2 = GetOp(insns[arith.y]).x;
        auto t3 = GetOp(insns[arith.y]).y;
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, t1, t2});
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::MUL, t1, t3});
        insns.push_back(IntArithmeticInsn{IntArithmeticInsn::Op::ADD, insns.size() - 2, insns.size() - 1});
        insertion_map.push_back(iinsn);
        insertion_map.push_back(iinsn);
    }
    else
    {
        insns.push_back(arith);
    }
}

static void CanonicalizeAndSimplify(
    size_t iinsn,
    Instruction insn,
    std::vector<Instruction> &insns,
    std::vector<size_t> &insertion_map)
{
    const IntArithmeticInsn &arith = GetOp(insn);
    switch(arith.op)
    {
    case IntArithmeticInsn::Op::ADD:
        return CanonicalizeAndSimplifyAdd(iinsn, insn, insns, insertion_map);
    case IntArithmeticInsn::Op::MUL:
        return CanonicalizeAndSimplifyMul(iinsn, insn, insns, insertion_map);
    default:
        insns.push_back(insn);
        return;
#if 0
    case IntArithmeticInsn::Op::DIV:
        return CanonicalizeAndSimplifyDiv(iinsn, insns, uf);
    case IntArithmeticInsn::Op::MOD:
        return CanonicalizeAndSimplifyMod(iinsn, insns, uf);
    default:
        // SUB is currently not emitted, so unimplemented.
        return;
#endif
    }
}

std::vector<Instruction> RemoveUnused(const std::vector<Instruction> &insns)
{
    std::vector<bool> used(insns.size(), false);
    std::vector<Instruction> result = insns;
    for(ssize_t iinsn = insns.size() - 1; iinsn >= 0; iinsn--)
    {
        bool is_int_insn = std::holds_alternative<IntArithmeticInsn>(insns[iinsn])
            || std::holds_alternative<LoadIntImmediateInsn>(insns[iinsn]);
        if(is_int_insn && !used[iinsn])
            result[iinsn] = Nop{};
        else
            VisitVariableReferences(
                result[iinsn],
                [&](size_t &s) { used[s] = true; });
    }
    return result;
}

struct AddressExpression
{
    struct Term
    {
        std::variant<size_t, dim_t, IntArithmeticInsn::Op> t;
        size_t left = 0;
        size_t right = 0;
    };

    bool IsConst(size_t i) const
    {
        return std::holds_alternative<dim_t>(terms[i].t);
    }

    dim_t GetConst(size_t i) const
    {
        return std::get<dim_t>(terms[i].t);
    }

    size_t GetLeft(size_t i) const
    {
        assert(IsOp(i));
        return terms[i].left;
    }

    size_t GetRight(size_t i) const
    {
        assert(IsOp(i));
        return terms[i].right;
    }

    bool IsOp(size_t i) const
    {
        return std::holds_alternative<IntArithmeticInsn::Op>(terms[i].t);
    }

    bool IsOp(size_t i, char op) const
    {
        assert(op == static_cast<char>(IntArithmeticInsn::Op::ADD)
               || op == static_cast<char>(IntArithmeticInsn::Op::SUB)
               || op == static_cast<char>(IntArithmeticInsn::Op::MUL)
               || op == static_cast<char>(IntArithmeticInsn::Op::DIV)
               || op == static_cast<char>(IntArithmeticInsn::Op::MOD));
        return std::holds_alternative<IntArithmeticInsn::Op>(terms[i].t)
            && op == static_cast<char>(std::get<IntArithmeticInsn::Op>(terms[i].t));
    }

    // Op must be +, -, *, /, or %. `left` and `right` must be either 'c' (denoting constant)
    // or 't' (denoting term)
    bool Matches(size_t i, char op, char left, char right)
    {
        assert(op == '+' || op == '-' || op == '*' || op == '/' || op == '%');
        assert(left == 'c' || left == 't');
        assert(right == 'c' || right == 't');

        if(!IsOp(i, op))
            return false;

        size_t l = GetLeft(i);
        size_t r = GetRight(i);
        if((left == 'c' && !IsConst(l)) || (left == 't' && IsConst(l)))
            return false;
        if((right == 'c' && !IsConst(r)) || (right == 't' && IsConst(r)))
            return false;
        return true;
    }

    bool MatchesL(size_t i, char op, char opl, char t1, char t2, char t3)
    {
        assert(op == '+' || op == '-' || op == '*' || op == '/' || op == '%');
        assert(opl == '+' || opl == '-' || opl == '*' || opl == '/' || opl == '%');
        assert(t1 == 'c' || t1 == 't');
        assert(t2 == 'c' || t2 == 't');
        assert(t3 == 'c' || t3 == 't');

        if(!IsOp(i, op))
            return false;

        size_t l = GetLeft(i);
        if(!IsOp(l, opl))
            return false;

        size_t ll = GetLeft(l);
        size_t lr = GetRight(l);
        size_t r = GetRight(i);
        if((t1 == 'c' && !IsConst(ll)) || (t1 == 't' && IsConst(ll)))
            return false;
        if((t2 == 'c' && !IsConst(lr)) || (t2 == 't' && IsConst(lr)))
            return false;
        if((t3 == 'c' && !IsConst(r)) || (t3 == 't' && IsConst(r)))
            return false;
        return true;
    }

    bool MatchesR(size_t i, char op, char opr, char t1, char t2, char t3)
    {
        assert(op == '+' || op == '-' || op == '*' || op == '/' || op == '%');
        assert(opr == '+' || opr == '-' || opr == '*' || opr == '/' || opr == '%');
        assert(t1 == 'c' || t1 == 't');
        assert(t2 == 'c' || t2 == 't');
        assert(t3 == 'c' || t3 == 't');

        if(!IsOp(i, op))
            return false;

        size_t r = GetRight(i);
        if(!IsOp(r, opr))
            return false;

        size_t l = GetLeft(i);
        size_t rl = GetLeft(r);
        size_t rr = GetRight(r);
        if((t1 == 'c' && !IsConst(l)) || (t1 == 't' && IsConst(l)))
            return false;
        if((t2 == 'c' && !IsConst(rl)) || (t2 == 't' && IsConst(rl)))
            return false;
        if((t3 == 'c' && !IsConst(rr)) || (t3 == 't' && IsConst(rr)))
            return false;
        return true;
    }

    size_t WalkAndSimplify(const std::vector<Instruction> &insns, size_t iinsn);
    size_t Output(std::vector<Instruction> &output, const std::vector<size_t> &input_to_output, size_t me);
    size_t Output(std::vector<Instruction> &output, const std::vector<size_t> &input_to_output);

    std::vector<Term> terms;
};

size_t AddressExpression::WalkAndSimplify(const std::vector<Instruction> &insns, size_t iinsn)
{
    size_t me = terms.size();
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[iinsn]))
    {
        terms.push_back({ std::get<LoadIntImmediateInsn>(insns[iinsn]).value });
        std::get<LoadIntImmediateInsn>(insns[iinsn]).Print(iinsn);
    }
    else if(std::holds_alternative<BeginLoopInsn>(insns[iinsn]))
    {
        terms.push_back({ iinsn });
        std::get<BeginLoopInsn>(insns[iinsn]).Print(iinsn);
    }
    else
    {
        auto &arith = std::get<IntArithmeticInsn>(insns[iinsn]);
        IntArithmeticInsn::Op op = arith.op;
        terms.push_back({ op });
        terms[me].left = WalkAndSimplify(insns, arith.x);
        terms[me].right = WalkAndSimplify(insns, arith.y);
        arith.Print(iinsn);
        if(Matches(me, '+', 'c', 'c'))
        {
            terms[me].t = GetConst(GetLeft(me)) + GetConst(GetRight(me));
        }
        if(Matches(me, '+', 't', 'c'))
        {
            std::swap(terms[me].left, terms[me].right);
        }
        if(Matches(me, '*', 'c', 'c'))
        {
            terms[me].t = GetConst(GetLeft(me)) * GetConst(GetRight(me));
        }
        if(Matches(me, '*', 't', 'c'))
        {
            std::swap(terms[me].left, terms[me].right);
        }
        if(MatchesR(me, '+', '+', 't', 't', 't'))
        {
            size_t t1 = GetLeft(me);
            size_t t2 = GetLeft(GetRight(me));
            size_t t3 = GetRight(GetRight(me));
            terms[me].t = IntArithmeticInsn::Op::ADD;
            terms[me].left = terms.size();
            terms[me].right = t3;
            terms.push_back({ IntArithmeticInsn::Op::ADD, t1, t2 });
        }
        if(MatchesR(me, '*', '*', 't', 't', 't'))
        {
            size_t t1 = GetLeft(me);
            size_t t2 = GetLeft(GetRight(me));
            size_t t3 = GetRight(GetRight(me));
            terms[me].t = IntArithmeticInsn::Op::MUL;
            terms[me].left = terms.size();
            terms[me].right = t3;
            terms.push_back({ IntArithmeticInsn::Op::MUL, t1, t2 });
        }
        if(MatchesL(me, '+', '+', 'c', 't', 'c'))
        {
            dim_t c1 = GetConst(GetLeft(GetLeft(me)));
            size_t t = GetRight(GetLeft(me));
            dim_t c2 = GetConst(GetRight(me));
            terms[me].t = IntArithmeticInsn::Op::ADD;
            terms[me].left = terms.size();
            terms[me].right = t;
            terms.push_back({ c1 + c2 });
        }
        if(MatchesL(me, '*', '*', 'c', 't', 'c'))
        {
            dim_t c1 = GetConst(GetLeft(GetLeft(me)));
            size_t t = GetRight(GetLeft(me));
            dim_t c2 = GetConst(GetRight(me));
            terms[me].t = IntArithmeticInsn::Op::MUL;
            terms[me].left = terms.size();
            terms[me].right = t;
            terms.push_back({ c1 * c2 });
        }
        if(MatchesL(me, '*', '+', 'c', 't', 'c'))
        {
            dim_t c1 = GetConst(GetLeft(GetLeft(me)));
            size_t t = GetRight(GetLeft(me));
            size_t c2_idx = GetRight(me);
            dim_t c2 = GetConst(c2_idx);
            terms[me].t = IntArithmeticInsn::Op::ADD;
            terms[me].left = terms.size();
            terms[me].right = terms.size() + 1;
            terms.push_back({ c1 * c2 });
            terms.push_back({ IntArithmeticInsn::Op::MUL, c2_idx, t });
        }
        if(MatchesR(me, '*', '+', 'c', 'c', 't'))
        {
            size_t c1_idx = GetLeft(me);
            dim_t c1 = GetConst(c1_idx);
            dim_t c2 = GetConst(GetLeft(GetRight(me)));
            size_t t = GetRight(GetRight(me));
            terms[me].t = IntArithmeticInsn::Op::ADD;
            terms[me].left = terms.size();
            terms[me].right = terms.size() + 1;
            terms.push_back({ c1 * c2 });
            terms.push_back({ IntArithmeticInsn::Op::MUL, c1_idx, t });
        }
        if(MatchesL(me, '*', '+', 't', 't', 'c'))
        {
            size_t t1 = GetLeft(GetLeft(me));
            size_t t2 = GetRight(GetLeft(me));
            size_t c = GetRight(me);
            terms[me].t = IntArithmeticInsn::Op::ADD;
            terms[me].left = terms.size();
            terms[me].right = terms.size() + 1;
            terms.push_back({ IntArithmeticInsn::Op::MUL, c, t1 });
            terms.push_back({ IntArithmeticInsn::Op::MUL, c, t2 });
        }
        if(MatchesR(me, '*', '+', 'c', 't', 't'))
        {
            size_t c = GetLeft(me);
            size_t t1 = GetLeft(GetRight(me));
            size_t t2 = GetRight(GetRight(me));
            terms[me].t = IntArithmeticInsn::Op::ADD;
            terms[me].left = terms.size();
            terms[me].right = terms.size() + 1;
            terms.push_back({ IntArithmeticInsn::Op::MUL, c, t1 });
            terms.push_back({ IntArithmeticInsn::Op::MUL, c, t2 });
        }
        if(MatchesL(me, '*', '+', 't', 't', 't'))
        {
            size_t t1 = GetLeft(GetLeft(me));
            size_t t2 = GetRight(GetLeft(me));
            size_t t3 = GetRight(me);
            terms[me].t = IntArithmeticInsn::Op::ADD;
            terms[me].left = terms.size();
            terms[me].right = terms.size() + 1;
            terms.push_back({ IntArithmeticInsn::Op::MUL, t1, t3 });
            terms.push_back({ IntArithmeticInsn::Op::MUL, t2, t3 });
        }
        if(MatchesR(me, '*', '+', 't', 't', 't'))
        {
            size_t t1 = GetLeft(me);
            size_t t2 = GetLeft(GetRight(me));
            size_t t3 = GetRight(GetRight(me));
            terms[me].t = IntArithmeticInsn::Op::ADD;
            terms[me].left = terms.size();
            terms[me].right = terms.size() + 1;
            terms.push_back({ IntArithmeticInsn::Op::MUL, t1, t2 });
            terms.push_back({ IntArithmeticInsn::Op::MUL, t1, t3 });
        }
    }
    return me;
}

size_t AddressExpression::Output(
    std::vector<Instruction> &insns,
    const std::vector<size_t> &input_to_output,
    size_t me)
{
    if(std::holds_alternative<size_t>(terms[me].t))
    {
        size_t loop_id_old = std::get<size_t>(terms[me].t);
        size_t loop_id_new = input_to_output[loop_id_old];
        return loop_id_new;
    }
    else if(std::holds_alternative<dim_t>(terms[me].t))
    {
        insns.push_back(LoadIntImmediateInsn{GetConst(me)});
        return insns.size() - 1;
    }
    else
    {
        size_t left = Output(insns, input_to_output, GetLeft(me));
        size_t right = Output(insns, input_to_output, GetRight(me));
        IntArithmeticInsn::Op op = std::get<IntArithmeticInsn::Op>(terms[me].t);
        insns.push_back(IntArithmeticInsn{op, left, right});
        return insns.size() - 1;
    }
}

size_t AddressExpression::Output(std::vector<Instruction> &insns, const std::vector<size_t> &input_to_output)
{
    return Output(insns, input_to_output, 0);
}

AddressExpression WalkAndSimplifyAddrExpression(const std::vector<Instruction> &insns, size_t idx)
{
    AddressExpression expr;
    expr.WalkAndSimplify(insns, idx);
    return expr;
}

std::vector<Instruction> gigagrad::codegen::SimplifyAddressExpressions(const std::vector<Instruction> &insns)
{
    std::vector<Instruction> result;
    std::vector<size_t> input_to_output(insns.size());
    for(size_t iinsn = 0; iinsn < insns.size(); iinsn++)
    {
        if(std::holds_alternative<LoadInsn>(insns[iinsn]))
        {
            LoadInsn load = std::get<LoadInsn>(insns[iinsn]);
            AddressExpression expr = WalkAndSimplifyAddrExpression(insns, load.idx);
            load.idx = expr.Output(result, input_to_output);
            input_to_output[iinsn] = result.size();
            result.push_back(load);
        }
        else if(std::holds_alternative<StoreInsn>(insns[iinsn]))
        {
            StoreInsn store = std::get<StoreInsn>(insns[iinsn]);
            AddressExpression expr = WalkAndSimplifyAddrExpression(insns, store.offset);
            store.offset = expr.Output(result, input_to_output);
            store.value = input_to_output[store.value];
            input_to_output[iinsn] = result.size();
            result.push_back(store);
        }
        else if(!std::holds_alternative<LoadIntImmediateInsn>(insns[iinsn])
                && !std::holds_alternative<IntArithmeticInsn>(insns[iinsn]))
        {
            Instruction insn = insns[iinsn];
            VisitVariableReferences(
                insn,
                [&](size_t &x) { x = input_to_output[x]; });
            input_to_output[iinsn] = result.size();
            result.push_back(insn);
        }
    }
    return result;
}

#if 0
std::vector<Instruction> gigagrad::codegen::SimplifyAddressExpressions(const std::vector<Instruction> &insns)
{
    std::vector<size_t> insertion_map;
    std::vector<Instruction> result;
    for(size_t iinsn = 0; iinsn < insns.size(); iinsn++)
    {
        Instruction insn = insns[iinsn];
        OffsetVariableReferences(insn, insertion_map);
        if(std::holds_alternative<IntArithmeticInsn>(insn))
            CanonicalizeAndSimplify(iinsn, insn, result, insertion_map);
        else
            result.push_back(insn);
    }
    result = RemoveUnused(result);
    return result;
}
#endif

std::vector<Instruction> gigagrad::codegen::EliminateCommonSubexpressions(const std::vector<Instruction> &insns)
{
    auto result = EliminateCommonConstants(insns);
    return result;
}
