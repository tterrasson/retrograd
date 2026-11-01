// GGUF metadata without a model: read the entries under a prefix, and copy a
// file with some entries changed and its tensors untouched.
//
// What a training run learns besides weights - the temperatures of a decision
// head, the template it was trained with - is metadata. Writing it means
// rewriting the header of a file that already exists (a saved adapter, an
// exported model, a copy of the base model), and the tensor data of such a
// file is never touched: it is copied byte for byte after the new header.
#include "retro_runtime.hpp"

#include "json.h"

#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace retro {

namespace {

using json = common_json;

struct gguf_owner {
    gguf_context * gguf = nullptr;
    ggml_context * meta = nullptr;
    ~gguf_owner() {
        if (gguf) {
            gguf_free(gguf);
        }
        if (meta) {
            ggml_free(meta);
        }
    }
};

struct file_owner {
    FILE * file = nullptr;
    ~file_owner() {
        if (file) {
            std::fclose(file);
        }
    }
};

bool open_header(const char * path, bool with_tensors, gguf_owner & out) {
    gguf_init_params params {};
    params.no_alloc = true;
    params.ctx = with_tensors ? &out.meta : nullptr;
    out.gguf = gguf_init_from_file(path, params);
    if (!out.gguf) {
        set_error(std::string("cannot read the GGUF header of ") + path);
        return false;
    }
    return true;
}

// A value as llama.cpp prints it (gguf_kv_to_str): numbers through
// std::to_string, so an F32 has six decimals.
std::string value_text(const gguf_context * gguf, int64_t key) {
    switch (gguf_get_kv_type(gguf, key)) {
        case GGUF_TYPE_UINT8:   return std::to_string(gguf_get_val_u8(gguf, key));
        case GGUF_TYPE_INT8:    return std::to_string(gguf_get_val_i8(gguf, key));
        case GGUF_TYPE_UINT16:  return std::to_string(gguf_get_val_u16(gguf, key));
        case GGUF_TYPE_INT16:   return std::to_string(gguf_get_val_i16(gguf, key));
        case GGUF_TYPE_UINT32:  return std::to_string(gguf_get_val_u32(gguf, key));
        case GGUF_TYPE_INT32:   return std::to_string(gguf_get_val_i32(gguf, key));
        case GGUF_TYPE_UINT64:  return std::to_string(gguf_get_val_u64(gguf, key));
        case GGUF_TYPE_INT64:   return std::to_string(gguf_get_val_i64(gguf, key));
        case GGUF_TYPE_FLOAT32: return std::to_string(gguf_get_val_f32(gguf, key));
        case GGUF_TYPE_FLOAT64: return std::to_string(gguf_get_val_f64(gguf, key));
        case GGUF_TYPE_BOOL:    return gguf_get_val_bool(gguf, key) ? "true" : "false";
        case GGUF_TYPE_STRING:  return gguf_get_val_str(gguf, key);
        case GGUF_TYPE_ARRAY: {
            json items = json::array();
            const size_t n = gguf_get_arr_n(gguf, key);
            if (gguf_get_arr_type(gguf, key) == GGUF_TYPE_STRING) {
                for (size_t i = 0; i < n; ++i) {
                    items.push_back(gguf_get_arr_str(gguf, key, i));
                }
                return items.dump();
            }
            return "[" + std::to_string(n) + " items]";
        }
        default:
            return "";
    }
}

// Writes one entry of the rewrite into `out`; `source` says which type an
// integer keeps.
bool set_entry(gguf_context * out, const gguf_context * source, const std::string & key, const json & value) {
    const char * name = key.c_str();
    if (value.is_null()) {
        gguf_remove_key(out, name);
        return true;
    }
    if (value.is_string()) {
        gguf_set_val_str(out, name, value.get<std::string>().c_str());
        return true;
    }
    if (value.is_boolean()) {
        gguf_set_val_bool(out, name, value.get<bool>());
        return true;
    }
    if (value.is_number_float()) {
        gguf_set_val_f32(out, name, value.get<float>());
        return true;
    }
    if (value.is_number_integer()) {
        const int64_t existing = gguf_find_key(source, name);
        const enum gguf_type type = existing >= 0 ? gguf_get_kv_type(source, existing) : GGUF_TYPE_UINT32;
        const int64_t v = value.get<int64_t>();
        const auto fits = [&](int64_t lo, int64_t hi) {
            if (v < lo || v > hi) {
                set_error("metadata entry " + key + " does not fit its integer type");
                return false;
            }
            return true;
        };
        switch (type) {
            case GGUF_TYPE_UINT8:  if (!fits(0, UINT8_MAX)) return false;  gguf_set_val_u8(out, name, static_cast<uint8_t>(v)); return true;
            case GGUF_TYPE_INT8:   if (!fits(INT8_MIN, INT8_MAX)) return false; gguf_set_val_i8(out, name, static_cast<int8_t>(v)); return true;
            case GGUF_TYPE_UINT16: if (!fits(0, UINT16_MAX)) return false; gguf_set_val_u16(out, name, static_cast<uint16_t>(v)); return true;
            case GGUF_TYPE_INT16:  if (!fits(INT16_MIN, INT16_MAX)) return false; gguf_set_val_i16(out, name, static_cast<int16_t>(v)); return true;
            case GGUF_TYPE_UINT32: if (!fits(0, UINT32_MAX)) return false; gguf_set_val_u32(out, name, static_cast<uint32_t>(v)); return true;
            case GGUF_TYPE_INT32:  if (!fits(INT32_MIN, INT32_MAX)) return false; gguf_set_val_i32(out, name, static_cast<int32_t>(v)); return true;
            case GGUF_TYPE_UINT64: if (!fits(0, INT64_MAX)) return false; gguf_set_val_u64(out, name, static_cast<uint64_t>(v)); return true;
            case GGUF_TYPE_INT64:  gguf_set_val_i64(out, name, v); return true;
            default:
                set_error("metadata entry " + key + " holds a non-integer value and cannot take an integer");
                return false;
        }
    }
    if (value.is_array()) {
        std::vector<std::string> items;
        for (const json & item : value) {
            if (!item.is_string()) {
                set_error("metadata entry " + key + ": only arrays of strings can be written");
                return false;
            }
            items.push_back(item.get<std::string>());
        }
        std::vector<const char *> pointers;
        pointers.reserve(items.size());
        for (const std::string & item : items) {
            pointers.push_back(item.c_str());
        }
        gguf_set_arr_str(out, name, pointers.data(), pointers.size());
        return true;
    }
    set_error("metadata entry " + key + " has a value that cannot be written");
    return false;
}

bool rewrite(const char * src_path, const char * dst_path, const json & entries) {
    gguf_owner source;
    if (!open_header(src_path, /*with_tensors =*/ true, source)) {
        return false;
    }
    gguf_context_ptr out(gguf_init_empty());
    if (!out) {
        set_error("failed to allocate a GGUF context");
        return false;
    }
    gguf_set_kv(out.get(), source.gguf);
    for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (!set_entry(out.get(), source.gguf, it.key(), it.value())) {
            return false;
        }
    }
    // The data section is copied as it is, so every tensor must land at the
    // offset it has in the source: same order, same sizes, same alignment.
    if (gguf_get_alignment(out.get()) != gguf_get_alignment(source.gguf)) {
        set_error("the GGUF declares an alignment the rewrite does not reproduce");
        return false;
    }
    const int64_t n_tensors = gguf_get_n_tensors(source.gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(source.gguf, i);
        const ggml_tensor * tensor = ggml_get_tensor(source.meta, name);
        if (!tensor) {
            set_error(std::string("the GGUF header lists a tensor it does not describe: ") + name);
            return false;
        }
        gguf_add_tensor(out.get(), tensor);
        if (gguf_get_tensor_offset(out.get(), i) != gguf_get_tensor_offset(source.gguf, i)) {
            set_error(std::string("tensor ") + name + " would move in the rewritten file");
            return false;
        }
    }

    std::vector<uint8_t> header(gguf_get_meta_size(out.get()));
    gguf_get_meta_data(out.get(), header.data());

    const std::filesystem::path destination(dst_path);
    std::filesystem::path partial = destination;
    partial += ".partial";
    {
        file_owner input { std::fopen(src_path, "rb") };
        if (!input.file) {
            set_error(std::string("cannot open ") + src_path);
            return false;
        }
        file_owner output { std::fopen(partial.string().c_str(), "wb") };
        if (!output.file) {
            set_error("cannot create " + partial.string());
            return false;
        }
        const size_t data_offset = gguf_get_data_offset(source.gguf);
        if (std::fwrite(header.data(), 1, header.size(), output.file) != header.size()
                || std::fseek(input.file, static_cast<long>(data_offset), SEEK_SET) != 0) {
            set_error("failed to write the GGUF header of " + partial.string());
            return false;
        }
        std::vector<char> chunk(1 << 20);
        while (true) {
            const size_t n = std::fread(chunk.data(), 1, chunk.size(), input.file);
            if (n > 0 && std::fwrite(chunk.data(), 1, n, output.file) != n) {
                set_error("failed to copy the tensor data to " + partial.string());
                return false;
            }
            if (n < chunk.size()) {
                if (std::ferror(input.file)) {
                    set_error(std::string("failed to read the tensor data of ") + src_path);
                    return false;
                }
                break;
            }
        }
        if (std::fflush(output.file) != 0) {
            set_error("failed to flush " + partial.string());
            return false;
        }
    }
    std::error_code error;
    std::filesystem::rename(partial, destination, error);
    if (error) {
        std::filesystem::remove(partial, error);
        set_error("cannot move the rewritten GGUF into place at " + destination.string());
        return false;
    }
    return true;
}

}  // namespace

extern "C" int retro_gguf_rewrite_metadata(const char * src_path, const char * dst_path, const char * entries_json) {
    return boundary([&]() -> int {
        if (is_blank(src_path) || is_blank(dst_path)) {
            set_error("source and destination paths are required");
            return -1;
        }
        json entries = json::object();
        if (entries_json && entries_json[0] != '\0') {
            // A parse error throws, and the boundary reports its message.
            entries = json::parse(entries_json);
            if (!entries.is_object()) {
                set_error("metadata entries must be a JSON object");
                return -1;
            }
        }
        return rewrite(src_path, dst_path, entries) ? 0 : -1;
    });
}

extern "C" int retro_gguf_read_metadata(
        const char * path,
        const char * prefix,
        char * buffer,
        size_t n_buffer,
        size_t * out_n_bytes) {
    return boundary([&]() -> int {
        if (is_blank(path)) {
            set_error("path is required");
            return -1;
        }
        if (!out_n_bytes) {
            set_error("out_n_bytes is required");
            return -1;
        }
        gguf_owner file;
        if (!open_header(path, /*with_tensors =*/ false, file)) {
            return -1;
        }
        const std::string wanted = prefix ? prefix : "";
        json out = json::object();
        const int64_t n_kv = gguf_get_n_kv(file.gguf);
        for (int64_t i = 0; i < n_kv; ++i) {
            const std::string key = gguf_get_key(file.gguf, i);
            if (key.compare(0, wanted.size(), wanted) == 0) {
                out[key] = value_text(file.gguf, i);
            }
        }
        return copy_string_out(out.dump(), buffer, n_buffer, out_n_bytes);
    });
}

}  // namespace retro
