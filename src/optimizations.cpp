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
        dim_t min = 0;
        dim_t max = 0;
        dim_t multiple = 0;
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
        if((left == 'c' && !IsConst(l)))// || (left == 't' && IsConst(l)))
            return false;
        if((right == 'c' && !IsConst(r)))// || (right == 't' && IsConst(r)))
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

    size_t Canonicalize(size_t me);
    size_t WalkAndSimplify(const std::vector<Instruction> &insns, size_t iinsn);
    size_t Output(std::vector<Instruction> &output, const std::vector<size_t> &input_to_output, size_t me);
    size_t Output(std::vector<Instruction> &output, const std::vector<size_t> &input_to_output);

    std::vector<Term> terms;
};

size_t AddressExpression::Canonicalize(size_t me)
{
    if(Matches(me, '+', 'c', 'c'))
    {
        terms[me].t = dim_t{GetConst(GetLeft(me)) + GetConst(GetRight(me))};
    }
    if(Matches(me, '+', 't', 'c'))
    {
        std::swap(terms[me].left, terms[me].right);
    }
    if(Matches(me, '*', 'c', 'c'))
    {
        terms[me].t = dim_t{GetConst(GetLeft(me)) * GetConst(GetRight(me))};
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
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
        terms[me].left = Canonicalize(GetLeft(me));
        terms[me].right = Canonicalize(GetRight(me));
    }
    if(Matches(me, '+', 'c', 't'))
    {
        if(GetConst(GetLeft(me)) == 0)
            terms[me] = terms[GetRight(me)];
    }
    if(Matches(me, '*', 'c', 't'))
    {
        if(GetConst(GetLeft(me)) == 0)
            terms[me].t = dim_t{0};
        else if(GetConst(GetLeft(me)) == 1)
            me = GetRight(me);
    }
    if(Matches(me, '/', 't', 'c'))
    {
        if(GetConst(GetRight(me)) == 1)
            me = GetLeft(me);
    }
    if(Matches(me, '%', 't', 'c'))
    {
        if(GetConst(GetRight(me)) == 1)
            terms[me].t = dim_t{0};
    }
    return me;
}

size_t AddressExpression::WalkAndSimplify(const std::vector<Instruction> &insns, size_t iinsn)
{
    size_t me = terms.size();
    terms.push_back({});
    if(std::holds_alternative<LoadIntImmediateInsn>(insns[iinsn]))
    {
        dim_t value = std::get<LoadIntImmediateInsn>(insns[iinsn]).value;
        terms[me].t = value;
        terms[me].min = value;
        terms[me].max = value;
        terms[me].multiple = value;

        std::get<LoadIntImmediateInsn>(insns[iinsn]).Print(iinsn);
    }
    else if(std::holds_alternative<BeginLoopInsn>(insns[iinsn]))
    {
        const BeginLoopInsn &loop = std::get<BeginLoopInsn>(insns[iinsn]);
        terms[me].t = size_t{iinsn};
        terms[me].min = 0;
        terms[me].max = loop.range;
        terms[me].multiple = loop.step;
        std::get<BeginLoopInsn>(insns[iinsn]).Print(iinsn);
    }
    else
    {
        auto &arith = std::get<IntArithmeticInsn>(insns[iinsn]);
        IntArithmeticInsn::Op op = arith.op;
        terms[me].t = op;
        terms[me].left = WalkAndSimplify(insns, arith.x);
        terms[me].right = WalkAndSimplify(insns, arith.y);
        arith.Print(iinsn);
        me = Canonicalize(me);
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
        assert(GetConst(me) != 1);
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

std::vector<Instruction> gigagrad::codegen::EliminateCommonSubexpressions(const std::vector<Instruction> &insns)
{
    auto result = EliminateCommonConstants(insns);
    return result;
}
