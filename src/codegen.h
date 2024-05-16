#pragma once

#include <cinttypes>
#include <cstdio>
#include <numeric>
#include <unordered_map>

#include "graph.h"

namespace gigagrad
{
    namespace codegen
    {

        struct LoadIntImmediateInsn
        {
            int64_t value;

            void Print(size_t iinsn)
            {
                std::printf("v%zu = %" PRIi64 "\n", iinsn, value);
            }
        };

        struct IntArithmeticInsn
        {
            enum class Op : char
            {
                ADD = '+',
                SUB = '-',
                MUL = '*',
                DIV = '/',
                MOD = '%',
            };

            Op op;
            size_t x;
            size_t y;

            void Print(size_t iinsn)
            {
                std::printf("v%zu = v%zu %c v%zu\n", iinsn, x, (char)op, y);
            }
        };

        struct BeginLoopInsn
        {
            dim_t range;
            dim_t stride;

            void Print(size_t iinsn)
            {
                std::printf("v%zu = LOOP [0..%zd, %zd]\n", iinsn, range, stride);
            }
        };

        struct EndLoopInsn
        {
            void Print(size_t)
            {
                std::printf("END LOOP\n");
            }
        };

        struct LoadInsn
        {
            size_t input;
            size_t idx;

            void Print(size_t iinsn)
            {
                std::printf("v%zu = LOAD I%zu[v%zu]\n", iinsn, input, idx);
            }
        };

        struct StoreInsn
        {
            size_t offset;
            size_t value;

            void Print(size_t iinsn)
            {
                std::printf("Output[v%zu] = v%zu\n", offset, value);
            }
        };

        struct LoadImmediateInsn
        {
            float value;

            void Print(size_t iinsn)
            {
                std::printf("v%zu = %f\n", iinsn, value);
            }
        };

        struct UnaryInsn
        {
            UnaryOpType type;
            size_t x;
            NumericDataType dtype;

            void Print(size_t iinsn)
            {
                auto op_str = type == UnaryOpType::NOP    ? "NOP"
                              : type == UnaryOpType::EXP  ? "EXP"
                              : type == UnaryOpType::LOG  ? "LOG"
                              : type == UnaryOpType::CAST ? "CAST"
                              : type == UnaryOpType::SIN  ? "SIN"
                                                          : "INVALID";
                std::printf("v%zu = %s(v%zu)\n", iinsn, op_str, x);
            }
        };

        struct BinaryInsn
        {
            BinaryOpType type;
            size_t x;
            size_t y;
            NumericDataType dtype;

            void Print(size_t iinsn)
            {
                auto op_str = type == BinaryOpType::ADD   ? "+"
                              : type == BinaryOpType::SUB ? "-"
                              : type == BinaryOpType::MUL ? "*"
                              : type == BinaryOpType::DIV ? "/"
                              : type == BinaryOpType::POW ? "^"
                              : type == BinaryOpType::CMP ? "=="
                              : type == BinaryOpType::MAX ? "max"
                                                          : "INVALID";
                std::printf("v%zu = v%zu %s v%zu\n", iinsn, x, op_str, y);
            }
        };

        struct AccumulateInsn
        {
            ReduceOpType type;
            size_t accumulator;
            size_t x;

            void Print(size_t iinsn)
            {
                auto op_str = type == ReduceOpType::MAX   ? "MAX"
                              : type == ReduceOpType::SUM ? "SUM"
                                                          : "INVALID";
                std::printf("v%zu <- %s(v%zu, v%zu)\n", accumulator, op_str, accumulator, x);
            }
        };

        using Instruction = std::variant<
            LoadIntImmediateInsn,
            IntArithmeticInsn,
            BeginLoopInsn,
            EndLoopInsn,
            LoadInsn,
            StoreInsn,
            LoadImmediateInsn,
            UnaryInsn,
            BinaryInsn,
            AccumulateInsn>;

        struct FunctionBuilder
        {
            explicit FunctionBuilder(GraphNodeHandle node, size_t max_seen_size = 1)
                : node(node)
            {
                const Shape &shape = node.shape();
                size_t output_size = std::accumulate(
                    shape.begin(),
                    shape.end(),
                    dim_t{1},
                    std::multiplies{});

                this->output_size = std::max(
                    max_seen_size,
                    output_size);
            }

            size_t Loop(dim_t range, dim_t stride)
            {
                insns.emplace_back(BeginLoopInsn{range, stride});
                return insns.size() - 1;
            }

            size_t EndLoop()
            {
                insns.emplace_back(EndLoopInsn{});
                return insns.size() - 1;
            }

            size_t Input(size_t program_input_idx)
            {
                // TODO: Make this not O(n), probably using unordered_map from buffer index
                //       to input index
                auto input = std::find(inputs.begin(), inputs.end(), program_input_idx);
                if (input == inputs.end())
                {
                    inputs.push_back(program_input_idx);
                    return inputs.size() - 1;
                }
                return input - inputs.begin();
            }

            size_t Load(size_t input_idx, size_t load_idx)
            {
                insns.emplace_back(LoadInsn{input_idx, load_idx});
                return insns.size() - 1;
            }

            size_t Store(size_t offset, size_t value)
            {
                insns.emplace_back(StoreInsn{offset, value});
                return insns.size() - 1;
            }

            size_t Immediate(float value)
            {
                insns.emplace_back(LoadImmediateInsn{value});
                return insns.size() - 1;
            }

            size_t IntImmediate(int64_t value)
            {
                insns.emplace_back(LoadIntImmediateInsn{value});
                return insns.size() - 1;
            }

            size_t Arithmetic(size_t x, IntArithmeticInsn::Op op, size_t y)
            {
                insns.emplace_back(IntArithmeticInsn{op, x, y});
                return insns.size() - 1;
            }

            size_t Unary(UnaryOpType type, size_t x, NumericDataType dtype)
            {
                insns.emplace_back(UnaryInsn{type, x, dtype});
                return insns.size() - 1;
            }

            size_t Binary(BinaryOpType type, size_t x, size_t y, NumericDataType dtype)
            {
                insns.emplace_back(BinaryInsn{type, x, y, dtype});
                return insns.size() - 1;
            }

            size_t Accumulate(ReduceOpType type, size_t accumulator, size_t x)
            {
                insns.emplace_back(AccumulateInsn{type, accumulator, x});
                return insns.size() - 1;
            }

            void Print()
            {
                for (ssize_t i = 0; i < std::ssize(insns); i++)
                {
                    std::visit([&](auto &&insn)
                               { insn.Print(i); }, insns[i]);
                }
            }

            GraphNodeHandle node; // Node that represents the output of the function
            size_t output_size;

            std::vector<Instruction> insns;
            std::vector<size_t> inputs; // Indices into the program inputs
            size_t output_buffer;
        };

        struct BufferDescriptor
        {
            std::variant<GraphNodeHandle, size_t> id; // Either a tensor or a function index
            size_t size_elts;
        };

        struct Program
        {
            void PushFunction(FunctionBuilder function)
            {
                functions.emplace_back(std::move(function));
                functions.back().output_buffer = AddBuffer(functions.size() - 1);
                node_function_cache[functions.back().node.node_idx] = functions.size() - 1;
            }

            size_t NumFunctions()
            {
                return functions.size();
            }

            size_t AddBuffer(GraphNodeHandle t, size_t size_elts)
            {
                if (t->Kind() != GraphNode::Kind::Tensor)
                    throw std::domain_error("Cannot AddBuffer on non-tensor");

                for (size_t iinput = 0; iinput < buffers.size(); iinput++)
                {
                    const auto &buff_id = buffers[iinput].id;
                    if (std::holds_alternative<GraphNodeHandle>(buff_id))
                        if (std::get<GraphNodeHandle>(buff_id).node_idx == t.node_idx)
                            return iinput;
                }
                buffers.push_back({t, size_elts});
                return buffers.size() - 1;
            }

            size_t AddBuffer(const size_t fn_idx)
            {
                for (size_t iinput = 0; iinput < buffers.size(); iinput++)
                {
                    const auto &buff_id = buffers[iinput].id;
                    if (std::holds_alternative<size_t>(buff_id))
                        if (std::get<size_t>(buff_id) == fn_idx)
                            return iinput;
                }
                buffers.push_back({fn_idx, functions[fn_idx].output_size});
                return buffers.size() - 1;
            }

            size_t GetOutputBufferForNodeIdx(size_t node_id)
            {
                size_t function_id = node_function_cache[node_id];
                return functions[function_id].output_buffer;
            }

            void ChangeOutputBuffer(size_t fn_idx, size_t new_output_buffer)
            {
                if (new_output_buffer >= buffers.size())
                    throw std::domain_error("Invalid output buffer");
                functions[fn_idx].output_buffer = new_output_buffer;
            }

            void Print()
            {
                for (size_t i = 0; i < functions.size(); i++)
                {
                    std::printf("BEGIN FUNCTION %zu\n", i);
                    functions[i].Print();
                    std::printf("END FUNCTION %zu\n", i);
                }
            }

            std::unordered_map<size_t, size_t> node_function_cache;
            std::vector<FunctionBuilder> functions;
            std::vector<BufferDescriptor> buffers;
        };

        void CodegenNode(codegen::Program &prog, GraphNodeHandle node, std::optional<size_t> output_buffer = std::nullopt);
        codegen::Program CodegenNode(GraphNodeHandle node);

    }
}
