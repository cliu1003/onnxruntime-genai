// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "adapter_loader.h"

#include <charconv>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include "ort_genai.h"

namespace benchmark {
namespace {

void SkipWhitespace(std::string_view header, size_t& pos) {
  while (pos < header.size() && std::isspace(static_cast<unsigned char>(header[pos]))) {
    ++pos;
  }
}

std::string ParseJsonString(std::string_view header, size_t& pos) {
  if (pos >= header.size() || header[pos] != '"') {
    throw std::runtime_error("Expected JSON string.");
  }
  ++pos;
  std::string value;
  while (pos < header.size()) {
    const char c = header[pos++];
    if (c == '"') {
      return value;
    }
    if (c == '\\') {
      if (pos >= header.size()) {
        throw std::runtime_error("Invalid escape sequence in JSON string.");
      }
      value.push_back(header[pos++]);
    } else {
      value.push_back(c);
    }
  }
  throw std::runtime_error("Unterminated JSON string.");
}

void SkipJsonValue(std::string_view header, size_t& pos) {
  SkipWhitespace(header, pos);
  if (pos >= header.size()) {
    throw std::runtime_error("Unexpected end of JSON.");
  }

  const char c = header[pos];
  if (c == '"') {
    ParseJsonString(header, pos);
    return;
  }
  if (c == '{') {
    size_t depth = 0;
    do {
      if (header[pos] == '{') {
        ++depth;
      } else if (header[pos] == '}') {
        --depth;
      }
      ++pos;
    } while (pos < header.size() && depth > 0);
    return;
  }
  if (c == '[') {
    size_t depth = 0;
    do {
      if (header[pos] == '[') {
        ++depth;
      } else if (header[pos] == ']') {
        --depth;
      }
      ++pos;
    } while (pos < header.size() && depth > 0);
    return;
  }

  while (pos < header.size() &&
         header[pos] != ',' && header[pos] != '}' && header[pos] != ']' &&
         !std::isspace(static_cast<unsigned char>(header[pos]))) {
    ++pos;
  }
}

double ParseJsonNumber(std::string_view json, size_t& pos) {
  SkipWhitespace(json, pos);
  const size_t start = pos;
  if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) {
    ++pos;
  }
  while (pos < json.size()) {
    const char c = json[pos];
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
      ++pos;
    } else {
      break;
    }
  }
  if (start == pos) {
    throw std::runtime_error("Expected JSON number.");
  }

  const std::string number_string(json.substr(start, pos - start));
  char* end = nullptr;
  const double value = std::strtod(number_string.c_str(), &end);
  if (end != number_string.c_str() + number_string.size()) {
    throw std::runtime_error("Failed to parse JSON number: " + number_string);
  }
  return value;
}

std::vector<int64_t> ParseInt64Array(std::string_view header, size_t& pos) {
  if (pos >= header.size() || header[pos] != '[') {
    throw std::runtime_error("Expected JSON array.");
  }
  ++pos;
  std::vector<int64_t> values;
  SkipWhitespace(header, pos);
  if (pos < header.size() && header[pos] == ']') {
    ++pos;
    return values;
  }

  while (pos < header.size()) {
    SkipWhitespace(header, pos);
    size_t number_end = pos;
    while (number_end < header.size() &&
           (std::isdigit(static_cast<unsigned char>(header[number_end])) || header[number_end] == '-')) {
      ++number_end;
    }
    int64_t value{};
    const auto number_view = header.substr(pos, number_end - pos);
    const auto [ptr, ec] = std::from_chars(number_view.data(), number_view.data() + number_view.size(), value);
    if (ec != std::errc{} || ptr != number_view.data() + number_view.size()) {
      throw std::runtime_error("Failed to parse JSON array number.");
    }
    values.push_back(value);
    pos = number_end;
    SkipWhitespace(header, pos);
    if (pos < header.size() && header[pos] == ',') {
      ++pos;
      continue;
    }
    if (pos < header.size() && header[pos] == ']') {
      ++pos;
      return values;
    }
    throw std::runtime_error("Malformed JSON array.");
  }
  throw std::runtime_error("Unterminated JSON array.");
}

LoraDequantEntry ParseLoraDequantEntry(std::string_view json, size_t& pos) {
  SkipWhitespace(json, pos);
  if (pos >= json.size() || json[pos] != '{') {
    throw std::runtime_error("Expected JSON object in lora_dequant.json.");
  }
  ++pos;

  LoraDequantEntry entry{};
  while (pos < json.size()) {
    SkipWhitespace(json, pos);
    if (pos < json.size() && json[pos] == '}') {
      ++pos;
      break;
    }

    const std::string field = ParseJsonString(json, pos);
    SkipWhitespace(json, pos);
    if (pos >= json.size() || json[pos] != ':') {
      throw std::runtime_error("Malformed lora_dequant.json entry.");
    }
    ++pos;
    SkipWhitespace(json, pos);

    if (field == "onnx_input") {
      entry.onnx_input = ParseJsonString(json, pos);
    } else if (field == "scale") {
      entry.scale = static_cast<float>(ParseJsonNumber(json, pos));
    } else if (field == "zero_point") {
      entry.zero_point = static_cast<int32_t>(ParseJsonNumber(json, pos));
    } else {
      SkipJsonValue(json, pos);
    }

    SkipWhitespace(json, pos);
    if (pos < json.size() && json[pos] == ',') {
      ++pos;
    }
  }

  if (entry.onnx_input.empty()) {
    throw std::runtime_error("lora_dequant.json entry is missing onnx_input.");
  }
  return entry;
}

LoraDequantMap ParseLoraDequantMap(std::string_view json) {
  LoraDequantMap map;
  size_t pos = 0;
  SkipWhitespace(json, pos);
  if (pos >= json.size() || json[pos] != '{') {
    throw std::runtime_error("lora_dequant.json must be a JSON object.");
  }
  ++pos;

  while (pos < json.size()) {
    SkipWhitespace(json, pos);
    if (pos < json.size() && json[pos] == '}') {
      ++pos;
      break;
    }

    const std::string key = ParseJsonString(json, pos);
    SkipWhitespace(json, pos);
    if (pos >= json.size() || json[pos] != ':') {
      throw std::runtime_error("Malformed lora_dequant.json entry.");
    }
    ++pos;
    SkipWhitespace(json, pos);

    map.emplace(key, ParseLoraDequantEntry(json, pos));

    SkipWhitespace(json, pos);
    if (pos < json.size() && json[pos] == ',') {
      ++pos;
    }
  }

  return map;
}

struct ParsedSafetensorsTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<int64_t> data_offsets;
};

std::vector<ParsedSafetensorsTensor> ParseSafetensorsHeader(std::string_view header) {
  std::vector<ParsedSafetensorsTensor> tensors;
  size_t pos = 0;
  SkipWhitespace(header, pos);
  if (pos >= header.size() || header[pos] != '{') {
    throw std::runtime_error("Safetensors header must be a JSON object.");
  }
  ++pos;

  while (pos < header.size()) {
    SkipWhitespace(header, pos);
    if (pos < header.size() && header[pos] == '}') {
      break;
    }

    const std::string name = ParseJsonString(header, pos);
    SkipWhitespace(header, pos);
    if (pos >= header.size() || header[pos] != ':') {
      throw std::runtime_error("Malformed safetensors header entry.");
    }
    ++pos;
    SkipWhitespace(header, pos);
    if (pos >= header.size() || header[pos] != '{') {
      throw std::runtime_error("Malformed safetensors tensor entry.");
    }
    ++pos;

    if (name != "__metadata__") {
      tensors.push_back(ParsedSafetensorsTensor{.name = name});
    }

    while (pos < header.size()) {
      SkipWhitespace(header, pos);
      if (pos < header.size() && header[pos] == '}') {
        ++pos;
        break;
      }

      const std::string field = ParseJsonString(header, pos);
      SkipWhitespace(header, pos);
      if (pos >= header.size() || header[pos] != ':') {
        throw std::runtime_error("Malformed safetensors tensor field.");
      }
      ++pos;
      SkipWhitespace(header, pos);

      if (name != "__metadata__") {
        auto& tensor = tensors.back();
        if (field == "dtype") {
          tensor.dtype = ParseJsonString(header, pos);
        } else if (field == "shape") {
          tensor.shape = ParseInt64Array(header, pos);
        } else if (field == "data_offsets") {
          tensor.data_offsets = ParseInt64Array(header, pos);
        } else {
          SkipJsonValue(header, pos);
        }
      } else {
        SkipJsonValue(header, pos);
      }

      SkipWhitespace(header, pos);
      if (pos < header.size() && header[pos] == ',') {
        ++pos;
      }
    }

    SkipWhitespace(header, pos);
    if (pos < header.size() && header[pos] == ',') {
      ++pos;
    }
  }

  return tensors;
}

size_t ElementCountFromShape(std::span<const int64_t> shape) {
  if (shape.empty()) {
    return 0;
  }
  return static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>{}));
}

uint16_t Float32ToFloat16(float value) {
  uint32_t bits{};
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x00001000;

  const uint32_t exponent = (bits & 0x7F800000) >> 23;
  const uint32_t mantissa = bits & 0x007FFFFF;
  return static_cast<uint16_t>((bits & 0x80000000) >> 16 |
                               (exponent > 112) * ((((exponent - 112) << 10) & 0x7C00) | mantissa >> 13) |
                               ((exponent < 113) & (exponent > 101)) * ((((0x007FF000 + mantissa) >> (125 - exponent)) + 1) >> 1) |
                               (exponent > 143) * 0x7FFF);
}

DequantizedAdapterTensor DequantizeAdapterTensor(const LoadedAdapterTensor& tensor, const LoraDequantEntry& entry) {
  const size_t element_count = ElementCountFromShape(tensor.shape);
  if (element_count != tensor.data.size()) {
    throw std::runtime_error("LoRA weight '" + tensor.name + "' shape does not match int8 data size.");
  }

  DequantizedAdapterTensor dequantized{};
  dequantized.name = entry.onnx_input;
  dequantized.shape = tensor.shape;
  dequantized.data.resize(element_count);

  const float scale = entry.scale;
  const float zero_point = static_cast<float>(entry.zero_point);
  for (size_t i = 0; i < element_count; ++i) {
    const float fp32 = (static_cast<float>(tensor.data[i]) - zero_point) * scale;
    dequantized.data[i] = Float32ToFloat16(fp32);
  }
  return dequantized;
}

}  // namespace

LoadedAdapter LoadSafetensors(const std::string& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error("Failed to open adapter safetensors file: " + path);
  }

  uint64_t header_size{};
  input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
  if (!input) {
    throw std::runtime_error("Failed to read safetensors header size: " + path);
  }

  std::string header(static_cast<size_t>(header_size), '\0');
  input.read(header.data(), static_cast<std::streamsize>(header_size));
  if (!input) {
    throw std::runtime_error("Failed to read safetensors header: " + path);
  }

  const auto parsed_tensors = ParseSafetensorsHeader(header);
  const auto data_section_offset = static_cast<std::streamoff>(sizeof(uint64_t) + header_size);
  LoadedAdapter adapter{};
  adapter.tensors.reserve(parsed_tensors.size());

  for (const auto& parsed : parsed_tensors) {
    if (parsed.dtype != "I8" && parsed.dtype != "U8" && parsed.dtype != "F16") {
      throw std::runtime_error("Unsupported safetensors dtype for LoRA weight '" + parsed.name +
                               "': " + parsed.dtype + " (only I8/U8/F16 supported)");
    }
    if (parsed.data_offsets.size() != 2) {
      throw std::runtime_error("Invalid safetensors data_offsets for tensor: " + parsed.name);
    }

    const size_t data_start = static_cast<size_t>(parsed.data_offsets[0]);
    const size_t data_end = static_cast<size_t>(parsed.data_offsets[1]);
    if (data_end < data_start) {
      throw std::runtime_error("Invalid safetensors data offsets for tensor: " + parsed.name);
    }

    const size_t byte_count = data_end - data_start;
    LoadedAdapterTensor tensor{};
    tensor.name = parsed.name;
    tensor.shape = parsed.shape;
    tensor.is_uint8 = parsed.dtype == "U8";
    tensor.is_fp16 = parsed.dtype == "F16";

    input.seekg(data_section_offset + static_cast<std::streamoff>(data_start), std::ios::beg);
    if (tensor.is_fp16) {
      if (byte_count % sizeof(uint16_t) != 0) {
        throw std::runtime_error("Invalid safetensors fp16 byte size for tensor: " + parsed.name);
      }
      tensor.f16_data.resize(byte_count / sizeof(uint16_t));
      input.read(reinterpret_cast<char*>(tensor.f16_data.data()), static_cast<std::streamsize>(byte_count));
    } else if (tensor.is_uint8) {
      tensor.u8_data.resize(byte_count);
      input.read(reinterpret_cast<char*>(tensor.u8_data.data()), static_cast<std::streamsize>(byte_count));
    } else {
      tensor.data.resize(byte_count);
      input.read(reinterpret_cast<char*>(tensor.data.data()), static_cast<std::streamsize>(byte_count));
    }
    if (!input) {
      throw std::runtime_error("Failed to read safetensors tensor data for: " + parsed.name);
    }

    adapter.tensors.push_back(std::move(tensor));
  }

  if (adapter.tensors.empty()) {
    throw std::runtime_error("No LoRA weight tensors found in safetensors file: " + path);
  }

  return adapter;
}

std::optional<LoraDequantMap> LoadLoraDequantMap(const std::string& model_path) {
  namespace fs = std::filesystem;
  const fs::path path = fs::path(model_path) / "lora_dequant.json";
  if (!fs::exists(path)) {
    return std::nullopt;
  }

  std::ifstream input{path};
  if (!input) {
    throw std::runtime_error("Failed to open lora_dequant.json: " + path.string());
  }

  const std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  return ParseLoraDequantMap(content);
}

void BindAdapterToGenerator(OgaGenerator& generator, BoundAdapter& bound_adapter, const std::string& model_path) {
  if (!bound_adapter.loaded) {
    return;
  }

  bound_adapter.ort_tensors.clear();
  bound_adapter.dequantized_tensors.clear();
  bound_adapter.ort_tensors.reserve(bound_adapter.loaded->tensors.size());

  const auto dequant_map = LoadLoraDequantMap(model_path);
  if (dequant_map.has_value()) {
    bound_adapter.dequantized_tensors.reserve(bound_adapter.loaded->tensors.size());
  }

  for (const auto& tensor : bound_adapter.loaded->tensors) {
    if (dequant_map.has_value()) {
      const auto it = dequant_map->find(tensor.name);
      if (it != dequant_map->end()) {
        bound_adapter.dequantized_tensors.push_back(DequantizeAdapterTensor(tensor, it->second));
        auto& dequantized = bound_adapter.dequantized_tensors.back();
        auto ort_tensor = OgaTensor::Create(
            dequantized.data.data(),
            dequantized.shape,
            OgaElementType_float16);
        generator.SetModelInput(dequantized.name.c_str(), *ort_tensor);
        bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
        continue;
      }
    }

    if (tensor.is_fp16) {
      auto ort_tensor = OgaTensor::Create(
          const_cast<uint16_t*>(tensor.f16_data.data()),
          tensor.shape,
          OgaElementType_float16);
      generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
      bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
      continue;
    }

    if (tensor.is_uint8) {
      auto ort_tensor = OgaTensor::Create(
          const_cast<uint8_t*>(tensor.u8_data.data()),
          tensor.shape,
          OgaElementType_uint8);
      generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
      bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
      continue;
    }

    auto ort_tensor = OgaTensor::Create(
        const_cast<int8_t*>(tensor.data.data()),
        tensor.shape,
        OgaElementType_int8);
    generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
    bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
  }
}

}  // namespace benchmark
