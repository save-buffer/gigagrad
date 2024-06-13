#include "graph.h"

#include <cstdio>
#include <cstdint>
#include <vector>

using namespace gigagrad;

static void PrintShape(std::FILE *file, const Shape &shape)
{
    std::fprintf(file, "{ ");
    for(ssize_t ishape = 0; ishape < std::ssize(shape); ishape++)
    {
        std::fprintf(file, "%zu ", shape[ishape]);
        if(ishape != std::ssize(shape) - 1)
            std::fprintf(file, "x ");
    }
    std::fprintf(file, "}");
}

static void NodeToDotDeclaration(std::FILE *file, const Tensor &t)
{
    std::fprintf(file, "Tensor");
}

static void NodeToDotDeclaration(std::FILE *file, const Immediate &i)
{
    std::fprintf(file, "%.4f", i.value);
}

static void NodeToDotDeclaration(std::FILE *file, const UnaryOp &u)
{
    const char *op_name;
    switch(u.type)
    {
    case UnaryOpType::EXP:
        op_name = "exp";
        break;
    case UnaryOpType::LOG:
        op_name = "log";
        break;
    case UnaryOpType::SIN:
        op_name = "sin";
        break;
    case UnaryOpType::SQRT:
        op_name = "sqrt";
        break;
    default:
        op_name = "<INVALID>";
    }
    std::fprintf(file, "%s ", op_name);
}

static void NodeToDotDeclaration(std::FILE *file, const BinaryOp &b)
{
    const char *op_name;
    switch(b.type)
    {
    case BinaryOpType::ADD:
        op_name = "+";
        break;
    case BinaryOpType::SUB:
        op_name = "-";
        break;
    case BinaryOpType::MUL:
        op_name = "*";
        break;
    case BinaryOpType::DIV:
        op_name = "/";
        break;
    case BinaryOpType::POW:
        op_name = "POW";
        break;
    case BinaryOpType::CMP:
        op_name = "==";
        break;
    case BinaryOpType::MAX:
        op_name = "MAX";
        break;
    }
    std::fprintf(file, "%s ", op_name);
}

static void NodeToDotDeclaration(std::FILE *file, const ReduceOp &r)
{
    const char *op_name;
    switch(r.type)
    {
    case ReduceOpType::SUM:
        op_name = "SUM";
        break;
    case ReduceOpType::MAX:
        op_name = "MAX";
        break;
    }
    std::fprintf(file, "%s ", op_name);
}

static void NodeToDotDeclaration(std::FILE *file, const ViewOp &v)
{
    std::fprintf(file, "Offset=%zd, Strides=", v.offset);
    PrintShape(file, v.strides);
    std::fprintf(file, ", Shape=");
}

static void DrawEdges(std::FILE *file, GraphNodeHandle node, std::vector<bool> &visited);

static void DrawEdges(std::FILE *file, size_t my_idx, const Tensor &t, std::vector<bool> &visited)
{
}

static void DrawEdges(std::FILE *file, size_t my_idx, const Immediate &i, std::vector<bool> &visited)
{
}

static void DrawEdges(std::FILE *file, size_t my_idx, const UnaryOp &u, std::vector<bool> &visited)
{
    std::fprintf(file, "    n%zu -> n%zu;\n", u.x.node_idx, my_idx);
    DrawEdges(file, u.x, visited);
}

static void DrawEdges(std::FILE *file, size_t my_idx, const BinaryOp &b, std::vector<bool> &visited)
{
    std::fprintf(file, "    n%zu -> n%zu;\n", b.x.node_idx, my_idx);
    std::fprintf(file, "    n%zu -> n%zu;\n", b.y.node_idx, my_idx);
    DrawEdges(file, b.x, visited);
    DrawEdges(file, b.y, visited);
}

static void DrawEdges(std::FILE *file, size_t my_idx, const ReduceOp &r, std::vector<bool> &visited)
{
    std::fprintf(file, "    n%zu -> n%zu;\n", r.x.node_idx, my_idx);
    DrawEdges(file, r.x, visited);
}

static void DrawEdges(std::FILE *file, size_t my_idx, const ViewOp &v, std::vector<bool> &visited)
{
    std::fprintf(file, "    n%zu -> n%zu;\n", v.x.node_idx, my_idx);
    DrawEdges(file, v.x, visited);
}

static void DrawEdges(std::FILE *file, GraphNodeHandle node, std::vector<bool> &visited)
{
    size_t my_idx = node.node_idx;
    if(visited[my_idx])
        return;
    visited[my_idx] = true;
    node->Visit([&](auto &&n) { return DrawEdges(file, node.node_idx, n, visited); });
}

static void GenerateNodeDeclarations(std::FILE *file, Graph &graph)
{
    for(size_t inode = 0; inode < graph.nodes.size(); inode++)
    {
        GraphNode &n = graph.nodes[inode];
        std::fprintf(file, "    n%zu [label=\"", inode);
        n.Visit([&](auto &&v) { NodeToDotDeclaration(file, v); });
        PrintShape(file, n.shape);
        std::fprintf(file, "\"]\n");
    }
}

void GraphNodeHandle::ToDotFile(const char *filename)
{
    std::FILE *file = fopen(filename, "w+");
    if(!file)
        throw std::system_error(errno, std::generic_category());
    std::fprintf(file, "digraph gg\n{\n");
    GenerateNodeDeclarations(file, *this->graph);
    std::vector<bool> visited(this->graph->nodes.size());
    DrawEdges(file, *this, visited);
    std::fprintf(file, "}\n");
    fclose(file);
}

void Graph::ToDotFile(const char *filename)
{
    std::FILE *file = fopen(filename, "w+");
    if(!file)
        throw std::system_error(errno, std::generic_category());
    std::fprintf(file, "digraph gg\n{\n");
    GenerateNodeDeclarations(file, *this);
    std::vector<bool> visited(this->nodes.size());
    for(size_t inode = 0; inode < this->nodes.size(); inode++)
        DrawEdges(file, GraphNodeHandle{this, inode}, visited);
    std::fprintf(file, "}\n");
    fclose(file);
}
