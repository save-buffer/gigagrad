#include "graph.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace gigagrad
{

dim_t FixDim(dim_t dim, dim_t mod)
{
    auto fixed_dim = ((dim % mod) + mod) % mod;
    return fixed_dim;
}

const GraphNode &WrapInUnary(const GraphNode &x, UnaryOpType type)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(UnaryOp{graph, type, x});
}

const GraphNode &WrapInReduction(const GraphNode &x, ReduceOpType type, Dims dims, bool keepdim)
{
    Graph &graph = GetGraph(x);
    std::sort(dims.begin(), dims.end());
    return graph.AddNode(ReduceOp{graph, type, x, std::move(dims), keepdim});
}

const GraphNode &GraphNode::sum(bool keepdim) const
{
    return this->sum(Dims{}, keepdim);
}

const GraphNode &GraphNode::sum(dim_t dim, bool keepdim) const
{
    return this->sum(Dims{dim}, keepdim);
}

const GraphNode &GraphNode::sum(Dims dims, bool keepdim) const
{
    return WrapInReduction(*this, ReduceOpType::SUM, std::move(dims), keepdim);
}

const GraphNode &GraphNode::max(bool keepdim) const
{
    return this->max(Dims{}, keepdim);
}

const GraphNode &GraphNode::max(dim_t dim, bool keepdim) const
{
    return this->max(Dims{dim}, keepdim);
}

const GraphNode &GraphNode::max(Dims dims, bool keepdim) const
{
    return WrapInReduction(*this, ReduceOpType::MAX, std::move(dims), keepdim);
}

const GraphNode &GraphNode::reshape(Shape shape) const
{
    Graph &graph = GetGraph(*this);
    return graph.AddNode(ReshapeOp{graph, *this, std::move(shape)});
}

const GraphNode &GraphNode::reshape(dim_t length) const
{
    return this->reshape(Shape{length});
}

const GraphNode &GraphNode::permute(Dims dims) const
{
    Graph &graph = GetGraph(*this);
    return graph.AddNode(PermuteOp{graph, *this, std::move(dims)});
}

const GraphNode &GraphNode::transpose() const
{
    Shape shape = this->shape();
    Dims dims(shape.size());
    std::iota(std::rbegin(dims), std::rend(dims), 0);
    return this->permute(std::move(dims));
}

// Matmul is a little tricky. We abuse the broadcasting semantics as follows:
// If we have matrices X, Y of shape AxB and BxC, then we reshape X into a
// AxBx1 tensor, and reshape Y into a 1xBxC matrix. Broadcasting then turns this
// into a cube of multiplications, and then we reduce along the middle axis
// and cut out the middle axis (since it has dim 1 anyway)
const GraphNode &GraphNode::matmul(const GraphNode &y) const
{
    Shape x_shape = this->shape();
    Shape y_shape = y.shape();

    // Special case for 1-D vectors by padding them up to 2D
    if(x_shape.size() == 1)
        x_shape.insert(x_shape.begin(), 1);
    if(y_shape.size() == 1)
        y_shape.push_back(1);

    if(x_shape.size() < 2 || y_shape.size() < 2)
        throw std::domain_error("Shapes must be at least of size 2 for matmul");

    x_shape.push_back(1);
    y_shape.insert(y_shape.end() - 2, 1);
    if(*(x_shape.end() - 2) != *(y_shape.end() - 2))
        throw std::domain_error("Incompatible shapes in matmul");

    const GraphNode &x_reshaped = this->reshape(std::move(x_shape));
    const GraphNode &y_reshaped = y.reshape(std::move(y_shape));
    const GraphNode &elementwise_mul = x_reshaped * y_reshaped;
    return elementwise_mul.sum(-2, false /* keepdim */); // Sum along the middle axis
}

const GraphNode &exp(const GraphNode &x)
{
    return WrapInUnary(x, UnaryOpType::EXP);
}

const GraphNode &log(const GraphNode &x)
{
    return WrapInUnary(x, UnaryOpType::LOG);
}

const GraphNode &sin(const GraphNode &x)
{
    return WrapInUnary(x, UnaryOpType::SIN);
}

const GraphNode &sigmoid(const GraphNode &x)
{
    return 1.0 / (1.0 + exp(-x));
}

const GraphNode &operator-(const GraphNode &x)
{
    return -1 * x;
}

const GraphNode &operator+(const GraphNode &x, const GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::ADD, x, y});
}

const GraphNode &operator+(float x, const GraphNode &y)
{
    Graph &graph = GetGraph(y);
    const GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return xnode + y;
}

const GraphNode &operator+(const GraphNode &x, float y)
{
    return y + x;
}

const GraphNode &operator-(const GraphNode &x, const GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::SUB, x, y});
}

const GraphNode &operator-(float x, const GraphNode &y)
{
    return (-x) + y;
}

const GraphNode &operator-(const GraphNode &x, float y)
{
    return x + (-y);
}

const GraphNode &operator*(const GraphNode &x, const GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::MUL, x, y});
}

const GraphNode &operator*(float x, const GraphNode &y)
{
    Graph &graph = GetGraph(y);
    const GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return xnode * y;
}

const GraphNode &operator*(const GraphNode &x, float y)
{
    return y * x;
}

const GraphNode &operator/(const GraphNode &x, const GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::DIV, x, y});
}

const GraphNode &operator/(float x, const GraphNode &y)
{
    Graph &graph = GetGraph(y);
    const GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return xnode / y;
}

const GraphNode &operator/(const GraphNode &x, float y)
{
    Graph &graph = GetGraph(x);
    const GraphNode &ynode = graph.AddNode(Immediate{graph, y});
    return x / ynode;
}

const GraphNode &operator^(const GraphNode &x, float y)
{
    Graph &graph = GetGraph(x);
    const GraphNode &ynode = graph.AddNode(Immediate{graph, y});
    return graph.AddNode(BinaryOp{graph, BinaryOpType::POW, x, ynode});
}

const GraphNode &operator==(const GraphNode &x, const GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::CMP, x, y});
}

const GraphNode &operator==(float x, const GraphNode &y)
{
    Graph &graph = GetGraph(y);
    const GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return xnode == y;
}

const GraphNode &operator==(const GraphNode &x, float y)
{
    return y == x;
}

const GraphNode &operator<(const GraphNode &x, float y)
{
    return y > x;
}

const GraphNode &operator<(float x, const GraphNode &y)
{
    return y > x;
}

const GraphNode &operator<(const GraphNode &x, const GraphNode &y)
{
    return y > x;
}

const GraphNode &operator<=(const GraphNode &x, float y)
{
    return max(x - y, 0.0f) == 0.0f;
}

const GraphNode &operator<=(float x, const GraphNode &y)
{
    return max(x - y, 0.0f) == 0.0f;
}

const GraphNode &operator<=(const GraphNode &x, const GraphNode &y)
{
    return max(x - y, 0.0f) == 0.0f;
}

const GraphNode &operator>(const GraphNode &x, float y)
{
    return max(x, y) == x;
}

const GraphNode &operator>(float x, const GraphNode &y)
{
    return max(x, y) == x;
}

const GraphNode &operator>(const GraphNode &x, const GraphNode &y)
{
    return max(x, y) == x;
}

const GraphNode &operator>=(const GraphNode &x, float y)
{
    return min(x - y, 0.0f) == 0.0f;
}

const GraphNode &operator>=(float x, const GraphNode &y)
{
    return min(x - y, 0.0f) == 0.0f;
}

const GraphNode &operator>=(const GraphNode &x, const GraphNode &y)
{
    return min(x - y, 0.0f) == 0.0f;
}

const GraphNode &max(const GraphNode &x, const GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::MAX, x, y});
}

const GraphNode &max(float x, const GraphNode &y)
{
    Graph &graph = GetGraph(y);
    const GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return max(xnode, y);
}

const GraphNode &max(const GraphNode &x, float y)
{
    return max(y, x);
}

const GraphNode &sum(const GraphNode &x, bool keepdim)
{
    return x.sum(keepdim);
}

const GraphNode &min(const GraphNode &x, const GraphNode &y)
{
    return -max(-x, -y);
}

const GraphNode &min(float x, const GraphNode &y)
{
    return -max(-x, -y);
}

const GraphNode &min(const GraphNode &x, float y)
{
    return -max(-x, -y);
}

const GraphNode &pow(const GraphNode &x, float y)
{
    Graph &graph = GetGraph(x);
    const GraphNode &ynode = graph.AddNode(Immediate{graph, y});
    return graph.AddNode(BinaryOp{graph, BinaryOpType::POW, x, ynode});
}

const GraphNode &pow(float x, const GraphNode &y)
{
    Graph &graph = GetGraph(y);
    const GraphNode &xnode = graph.AddNode(Immediate{graph, x});
    return graph.AddNode(BinaryOp{graph, BinaryOpType::POW, xnode, y});
}

const GraphNode &pow(const GraphNode &x, const GraphNode &y)
{
    Graph &graph = GetGraph(x);
    return graph.AddNode(BinaryOp{graph, BinaryOpType::POW, x, y});
}

const GraphNode &sum(const GraphNode &x, dim_t axis, bool keepdim)
{
    return x.sum(axis, keepdim);
}

const GraphNode &sum(const GraphNode &x, Dims dims, bool keepdim)
{
    return x.sum(std::move(dims), keepdim);
}

const GraphNode &max(const GraphNode &x, bool keepdim)
{
    return x.max(keepdim);
}

const GraphNode &max(const GraphNode &x, dim_t axis, bool keepdim)
{
    return x.max(axis, keepdim);
}

const GraphNode &max(const GraphNode &x, Dims dims, bool keepdim)
{
    return x.max(std::move(dims), keepdim);
}

const GraphNode &min(const GraphNode &x, bool keepdim)
{
    return -max(-x, keepdim);
}

const GraphNode &min(const GraphNode &x, dim_t axis, bool keepdim)
{
    return -max(-x, axis, keepdim);
}

const GraphNode &min(const GraphNode &x, Dims dims, bool keepdim)
{
    return -max(-x, std::move(dims), keepdim);
}

const GraphNode &reshape(const GraphNode &x, Shape shape)
{
    return x.reshape(std::move(shape));
}

const GraphNode &reshape(const GraphNode &x, dim_t length)
{
    return x.reshape(length);
}

const GraphNode &permute(const GraphNode &x, Dims permutation)
{
    return x.permute(std::move(permutation));
}

const GraphNode &transpose(const GraphNode &x)
{
    return x.transpose();
}

const GraphNode &operator%(const GraphNode &x, const GraphNode &y)
{
    return x.matmul(y);
}

const GraphNode &matmul(const GraphNode &x, const GraphNode &y)
{
    return x.matmul(y);
}

const GraphNode &Graph::AddInput(Shape shape)
{
    this->inputs.emplace_back(Tensor{*this, std::move(shape)});
    return this->inputs.back();
}

const GraphNode &Graph::AddInput(dim_t dim)
{
    return this->AddInput(Shape{dim});
}

const GraphNode &Graph::AddWeight(Shape shape)
{
    this->weights.emplace_back(Tensor{*this, std::move(shape)});
    return this->weights.back();
}

const GraphNode &Graph::AddWeight(dim_t dim)
{
    return this->AddWeight(Shape{dim});
}

const GraphNode &Graph::AddNode(GraphNode node)
{
    this->nodes.emplace_back(node);
    return this->nodes.back();
}

Shape GetBroadcastedShape(Shape x, Shape y)
{
    // Ensure x.size() >= y.size()
    if(y.size() > x.size())
        std::swap(x, y);

    for(ssize_t i = 0; i < std::ssize(y); i++)
    {
        // Store the proper dimension in dim_x
        auto &dim_x = x[x.size() - i - 1];
        const auto &dim_y = y[y.size() - i - 1];
        if(dim_x == 1 && dim_y != 1)
            dim_x = dim_y;
        else if(dim_x != 1 && dim_y == 1)
            continue;
        else if(dim_x == dim_y)
            continue;
        else
            throw std::domain_error("Cannot broadcast incompatible shapes");
    }
    return x;
}

Shape VerifyWithShape(const GraphNode &node);

Shape VerifyWithShape(const Tensor &t)
{
    return t.shape;
}

Shape VerifyWithShape(const Immediate &i)
{
    return {};
}

Shape VerifyWithShape(const UnaryOp &u)
{
    return VerifyWithShape(u.x);
}

Shape VerifyWithShape(const BinaryOp &u)
{
    Shape shapex = VerifyWithShape(u.x);
    Shape shapey = VerifyWithShape(u.y);
    return GetBroadcastedShape(std::move(shapex), std::move(shapey));
}

Shape VerifyWithShape(const ReduceOp &r)
{
    Shape shape = VerifyWithShape(r.x);
    if(r.dims.empty())
    {
        if(r.keepdim)
            return Shape(shape.size(), 1); // Shape of all 1's
        return {};
    }

    if(r.dims.size() > shape.size())
        throw std::domain_error("Specified more dims to reduce on than there are dimensions in tensor");
    
    for(auto dim : r.dims)
    {
        auto fixed_dim = FixDim(dim, static_cast<dim_t>(shape.size()));
        shape[fixed_dim] = -1; // Mark it as -1 for now. We'll either remove it or change it to 1 later
    }
    if(!r.keepdim)
    {
        shape.erase(std::remove(shape.begin(), shape.end(), -1), shape.end());
    }
    else
    {
        for(auto &d : shape)
            if(d == -1)
                d = 1;
    }
    return shape;
}

Shape VerifyWithShape(const ReshapeOp &r)
{
    Shape input_shape = VerifyWithShape(r.x);
    Shape new_shape = r.shape;
    auto num_elements = std::accumulate(input_shape.begin(), input_shape.end(), dim_t{1}, std::multiplies{});
    auto num_implicit_dims = std::count(new_shape.begin(), new_shape.end(), -1);
    if(num_implicit_dims == 0)
    {
        auto new_num_elements = std::accumulate(new_shape.begin(), new_shape.end(), dim_t{1}, std::multiplies{});
        if(new_num_elements != num_elements)
            throw std::domain_error("Reshape number of elements doesn't match that of input tensor");
        return new_shape;
    }

    if(num_implicit_dims > 1)
        throw std::domain_error("Reshape can have at most one implicit dimension");

    auto num_elems_not_including_implicit_dim = std::accumulate(
        new_shape.begin(),
        new_shape.end(),
        dim_t{1},
        [](auto x, auto y)
        {
            if(y == -1)
                return x;
            return x * y;
        });
    auto remaining_dim = num_elements / num_elems_not_including_implicit_dim;
    for(auto &x : new_shape)
        if(x == -1)
            x = remaining_dim;
    return new_shape;
}

Shape VerifyWithShape(const PermuteOp &p)
{
    Shape shape = VerifyWithShape(p.x);
    if(p.dims.size() != shape.size())
        throw std::domain_error("Permute not given proper number of dimensions");
    std::vector<bool> uniqueness(shape.size(), false);
    Shape result(shape.size());
    for(size_t i = 0; i < shape.size(); i++)
    {
        // If dim is negative, we need to fix it to be between 0 and shape.size()
        auto dim = p.dims[i];
        auto fixed_dim = FixDim(dim, static_cast<dim_t>(shape.size()));
        if(uniqueness[fixed_dim])
            throw std::domain_error("Found repeated dim in permute");
        uniqueness[fixed_dim] = true;
        result[fixed_dim] = shape[i];
    }
    return result;
}

Shape VerifyWithShape(const GraphNode &node)
{
    return std::visit([](auto &&arg) { return VerifyWithShape(arg); }, node);
}

Shape GraphNode::shape() const
{
    return std::visit([](auto &&arg) { return VerifyWithShape(arg); }, *this);
}

void GraphNode::Verify() const
{
    std::visit([](auto &&arg) { VerifyWithShape(arg); }, *this);
}

}
