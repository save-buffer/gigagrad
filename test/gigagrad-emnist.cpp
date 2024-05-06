#include "src/training.h"
#include "src/backend_scalar_c.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

namespace gg = gigagrad;

constexpr gg::dim_t BatchSize = 128;

// Download dataset from
// https://www.nist.gov/itl/products-and-services/emnist-dataset

enum class DataType : uint8_t
{
    U8 = 0x08,
    I8 = 0x09,
    I16 = 0x0B,
    I32 = 0x0C,
    F32 = 0x0D,
    F64 = 0x0E,
};

size_t SizeOf(DataType dtype)
{
    switch(dtype)
    {
    case DataType::U8:
        return sizeof(uint8_t);
    case DataType::I8:
        return sizeof(int8_t);
    case DataType::I16:
        return sizeof(int16_t);
    case DataType::I32:
        return sizeof(int32_t);
    case DataType::F32:
        return sizeof(float);
    case DataType::F64:
        return sizeof(double);
    default:
        exit(1);
    }
}

struct ParsedDataFile
{
    DataType dtype;
    std::vector<size_t> shape;
    std::vector<uint8_t> data;
};

ParsedDataFile LoadDataFile(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if(!file)
    {
        fprintf(stderr, "File %s not found\n", filename);
        exit(1);
    }
    uint32_t magic = 0;
    if(fseek(file, 0, SEEK_SET))
    {
        fprintf(stderr, "Failed to seek: %d\n", ferror(file));
        exit(1);
    }
    if(fread(&magic, sizeof(uint32_t), 1, file) < 1)
    {
        fprintf(stderr, "Failed to read from file: %d\n", ferror(file));
        exit(1);
    }
    if(static_cast<uint16_t>(magic & 0xFFFF) != 0)
    {
        fprintf(stderr, "Invalid magic bytes at beginning of file %s (0x%x), exiting\n", filename, magic);
        exit(1);
    }

    uint8_t raw_dtype = (magic >> 16) & 0xFF;
    DataType dtype;
    switch(raw_dtype)
    {
    case (uint8_t)DataType::U8:
        dtype = DataType::U8;
        break;
    case (uint8_t)DataType::I8:
        dtype = DataType::I8;
        break;
    case (uint8_t)DataType::I16:
        dtype = DataType::I16;
        break;
    case (uint8_t)DataType::I32:
        dtype = DataType::I32;
        break;
    case (uint8_t)DataType::F32:
        dtype = DataType::F32;
        break;
    case (uint8_t)DataType::F64:
        dtype = DataType::F64;
        break;
    default:
        fprintf(stderr, "Invalid datatype 0x%x, exiting\n", raw_dtype);
        exit(1);
        __builtin_unreachable();
    }


    uint8_t ndim = (magic >> 24) & 0xFF;

    std::vector<size_t> shape(ndim);
    size_t offset = 4;
    size_t total_dataset_size = 1;
    for(size_t i = 0; i < ndim; i++)
    {
        uint32_t dim = 0;
        if(fseek(file, offset, SEEK_SET))
        {
            fprintf(stderr, "Failed to seek to offset %zu\n", offset);
            exit(1);
        }
        if(fread(&dim, sizeof(dim), 1, file) < 1)
        {
            fprintf(stderr, "Failed to read\n");
            exit(1);
        }
        shape[i] = static_cast<size_t>(__builtin_bswap32(dim));
        total_dataset_size *= shape[i];
        offset += sizeof(dim);
    }

    std::vector<uint8_t> data(total_dataset_size * SizeOf(dtype), 0);
    if(fseek(file, offset, SEEK_SET))
    {
        fprintf(stderr, "Failed to seek to offset %zu\n", offset);
        exit(1);
    }
    if(fread(data.data(), SizeOf(dtype), total_dataset_size, file) < total_dataset_size)
    {
        fprintf(stderr, "Failed to read dataset\n");
        exit(1);
    }

    return { dtype, std::move(shape), std::move(data) };
}

template <typename T>
std::vector<float> CastToFloatAndNormalize(const std::vector<uint8_t> &input)
{
    size_t num_elements = input.size() / sizeof(T);
    std::vector<float> result(num_elements);
    const T *input_data = reinterpret_cast<const T *>(input.data());
    for(size_t i = 0; i < num_elements; i++)
    {
        result[i] = static_cast<float>(input_data[i]) / 255.0;
    }
    return result;
}

std::vector<float> CastToFloatAndNormalize(DataType dtype, const std::vector<uint8_t> &input)
{
    switch(dtype)
    {
    case DataType::U8:
        return CastToFloatAndNormalize<uint8_t>(input);
    case DataType::I8:
        return CastToFloatAndNormalize<int8_t>(input);
    case DataType::I16:
        return CastToFloatAndNormalize<int16_t>(input);
    case DataType::I32:
        return CastToFloatAndNormalize<int32_t>(input);
    case DataType::F32:
        return CastToFloatAndNormalize<float>(input);
    case DataType::F64:
        return CastToFloatAndNormalize<double>(input);
    default:
        exit(1);
    }
}

template <typename T>
std::vector<float> ToOneHot(const std::vector<uint8_t> &input)
{
    size_t num_elements = input.size() / sizeof(T);
    const T *input_data = reinterpret_cast<const T *>(input.data());
    T max_val = 0;
    for(size_t i = 0; i < num_elements; i++)
    {
        if(input_data[i] < 0)
        {
            fprintf(stderr, "Tried to one-hot encode invalid value: %d\n", (int)input_data[i]);
            exit(1);
        }
        max_val = std::max(max_val, input_data[i]);
    }
    std::vector<float> result(max_val * num_elements, 0.0f);
    for(size_t i = 0; i < num_elements; i++)
    {
        T cur_val = input_data[i];
        result[i * max_val + cur_val] = 1.0f;
    }
    return result;
}

std::vector<float> ToOneHot(DataType dtype, const std::vector<uint8_t> &input)
{
    switch(dtype)
    {
    case DataType::U8:
        return ToOneHot<uint8_t>(input);
    case DataType::I8:
        return ToOneHot<int8_t>(input);
    case DataType::I16:
        return ToOneHot<int16_t>(input);
    case DataType::I32:
        return ToOneHot<int32_t>(input);
    default:
        fprintf(stderr, "Can't cast non-integer to one-hot\n");
        exit(1);
    }
}

struct Dataset
{
    std::vector<size_t> shape;
    std::vector<float> inputs;
    std::vector<float> labels;
};

Dataset LoadDataset(const char *directory, const char *dataset)
{
    std::string image_name = std::string(directory) + "/emnist-mnist-" + dataset + "-images-idx3-ubyte";
    std::string label_name = std::string(directory) + "/emnist-mnist-" + dataset + "-labels-idx1-ubyte";
    ParsedDataFile images = LoadDataFile(image_name.c_str());
    ParsedDataFile labels = LoadDataFile(label_name.c_str());
    std::vector<float> images_float = CastToFloatAndNormalize(images.dtype, images.data);
    std::vector<float> labels_onehot = ToOneHot(labels.dtype, labels.data);
    return { std::move(images.shape), std::move(images_float), std::move(labels_onehot) };
}

void InitializeWeights(float *weight, size_t size_elts)
{
    std::default_random_engine gen(0);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for(size_t i = 0; i < size_elts; i++)
        weight[i] = dist(gen);
}

int main(int argc, const char **argv)
{
    if(argc != 2)
    {
        fprintf(stderr, "Please specify exactly one argument: the directory of the EMNIST dataset\n");
        exit(1);
    }

    Dataset train = LoadDataset(argv[1], "train");

    gg::nn::Module network;
    auto x = network.AddInput({ BatchSize, 28 * 28, 1 });
    auto w1 = network.AddWeight({ 40, 28 * 28 });
    auto b1 = network.AddWeight({ 40, 1 });
    auto z1 = (w1 % x) + b1;
    auto a2 = z1.relu();
    auto w2 = network.AddWeight({ 10, 40 });
    auto b2 = network.AddWeight({ 10, 1 });
    auto z2 = (w2 % a2) + b2;
    auto result = z2.softmax(-2);
    gg::TrainingContext ctx = gg::CompileTrainingGraph<gg::codegen::BackendScalarC>(network, result);

    w1.data() = new float[800 * 28 * 28];
    b1.data() = new float[800 * 1];
    w2.data() = new float[10 * 800];
    b2.data() = new float[10 * 1];
    InitializeWeights(w1.data(), 800 * 28 * 28);
    InitializeWeights(b1.data(), 800 * 1);
    InitializeWeights(w2.data(), 10 * 800);
    InitializeWeights(b2.data(), 10 * 1);

    size_t num_batches = (train.shape[0] + BatchSize - 1) / BatchSize;
    for(size_t iepoch = 0; iepoch < 4; iepoch++)
    {
        for(size_t ibatch = 0; ibatch < num_batches; ibatch++)
        {
            x.data() = &train.inputs[BatchSize * 28 * 28 * ibatch];
            ctx.training_example = &train.labels[BatchSize * 10 * ibatch];
            ctx.Execute();
            printf("Epoch %zu Batch (%zu / %zu) loss: %.6f\n", iepoch, ibatch, num_batches, *ctx.loss);
        }
        printf("Epoch %zu loss: %.6f\n", iepoch, *ctx.loss);
    }
    return 0;
}
