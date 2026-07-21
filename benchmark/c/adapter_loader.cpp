// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "adapter_loader.h"

#include <charconv>
#include <cctype>
#include <fstream>
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
    throw std::runtime_error("Expected JSON string in safetensors header.");
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
        throw std::runtime_error("Invalid escape sequence in safetensors header.");
      }
      value.push_back(header[pos++]);
    } else {
      value.push_back(c);
    }
  }
  throw std::runtime_error("Unterminated JSON string in safetensors header.");
}

void SkipJsonValue(std::string_view header, size_t& pos) {
  SkipWhitespace(header, pos);
  if (pos >= header.size()) {
    throw std::runtime_error("Unexpected end of safetensors header.");
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
         header[pos] != ',' && header[pos] != '}' && !std::isspace(static_cast<unsigned char>(header[pos]))) {
    ++pos;
  }
}

std::vector<int64_t> ParseInt64Array(std::string_view header, size_t& pos) {
  if (pos >= header.size() || header[pos] != '[') {
    throw std::runtime_error("Expected JSON array in safetensors header.");
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
      throw std::runtime_error("Failed to parse safetensors header number.");
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
    throw std::runtime_error("Malformed JSON array in safetensors header.");
  }
  throw std::runtime_error("Unterminated JSON array in safetensors header.");
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
    if (parsed.dtype != "I8") {
      throw std::runtime_error("Unsupported safetensors dtype for LoRA weight '" + parsed.name +
                               "': " + parsed.dtype + " (only I8/int8 is supported)");
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
    tensor.data.resize(byte_count);

    input.seekg(data_section_offset + static_cast<std::streamoff>(data_start), std::ios::beg);
    input.read(reinterpret_cast<char*>(tensor.data.data()), static_cast<std::streamsize>(byte_count));
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

void BindAdapterToGenerator(OgaGenerator& generator, BoundAdapter& bound_adapter) {
  if (!bound_adapter.loaded) {
    return;
  }

  bound_adapter.ort_tensors.clear();
  bound_adapter.ort_tensors.reserve(bound_adapter.loaded->tensors.size());

  for (const auto& tensor : bound_adapter.loaded->tensors) {
    auto ort_tensor = OgaTensor::Create(
        const_cast<int8_t*>(tensor.data.data()),
        tensor.shape,
        OgaElementType_int8);
    generator.SetModelInput(tensor.name.c_str(), *ort_tensor);
    bound_adapter.ort_tensors.push_back(std::move(ort_tensor));
  }
}

}  // namespace benchmark
