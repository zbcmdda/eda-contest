#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace delay::json {

class Value {
public:
    using Array = std::vector<Value>;
    using Object = std::unordered_map<std::string, Value>;

    Value() = default;
    explicit Value(bool value) : data_(value) {}
    explicit Value(std::int64_t value) : data_(value) {}
    explicit Value(std::string value) : data_(std::move(value)) {}
    explicit Value(Array value) : data_(std::move(value)) {}
    explicit Value(Object value) : data_(std::move(value)) {}

    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] std::int64_t as_int() const;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] const Object& as_object() const;
    [[nodiscard]] const Value& at(const std::string& key) const;

private:
    using Storage = std::variant<std::monostate, bool, std::int64_t, std::string, Array, Object>;
    Storage data_;
};

[[nodiscard]] Value parse_file(const std::string& path);

}  // namespace delay::json
