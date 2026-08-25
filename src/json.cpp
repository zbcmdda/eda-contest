#include "json.hpp"

#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace delay::json {
namespace {

[[noreturn]] void type_error(const char* expected) {
    throw std::runtime_error(std::string("JSON value is not ") + expected);
}

class Parser {
public:
    Parser(std::string input, std::string path) : input_(std::move(input)), path_(std::move(path)) {}

    Value parse() {
        skip_space();
        Value value = parse_value();
        skip_space();
        if (position_ != input_.size()) {
            fail("trailing content");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        throw std::runtime_error(path_ + ": JSON parse error at byte " +
                                 std::to_string(position_) + ": " + message);
    }

    void skip_space() {
        while (position_ < input_.size() &&
               std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    char take() {
        if (position_ == input_.size()) {
            fail("unexpected end of file");
        }
        return input_[position_++];
    }

    bool consume(char expected) {
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect_literal(const char* literal) {
        while (*literal != '\0') {
            if (take() != *literal++) {
                fail("invalid literal");
            }
        }
    }

    Value parse_value() {
        skip_space();
        if (position_ == input_.size()) {
            fail("expected value");
        }
        switch (input_[position_]) {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return Value(parse_string());
            case 't':
                expect_literal("true");
                return Value(true);
            case 'f':
                expect_literal("false");
                return Value(false);
            case 'n':
                expect_literal("null");
                return Value();
            default:
                return Value(parse_integer());
        }
    }

    Value parse_object() {
        take();
        Value::Object object;
        skip_space();
        if (consume('}')) {
            return Value(std::move(object));
        }
        while (true) {
            skip_space();
            if (position_ == input_.size() || input_[position_] != '"') {
                fail("expected object key");
            }
            std::string key = parse_string();
            skip_space();
            if (!consume(':')) {
                fail("expected ':'");
            }
            Value value = parse_value();
            if (!object.emplace(std::move(key), std::move(value)).second) {
                fail("duplicate object key");
            }
            skip_space();
            if (consume('}')) {
                return Value(std::move(object));
            }
            if (!consume(',')) {
                fail("expected ',' or '}'");
            }
        }
    }

    Value parse_array() {
        take();
        Value::Array array;
        skip_space();
        if (consume(']')) {
            return Value(std::move(array));
        }
        while (true) {
            array.push_back(parse_value());
            skip_space();
            if (consume(']')) {
                return Value(std::move(array));
            }
            if (!consume(',')) {
                fail("expected ',' or ']'");
            }
        }
    }

    static void append_utf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7f) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    std::uint32_t parse_hex4() {
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = take();
            value <<= 4;
            if (c >= '0' && c <= '9') {
                value |= static_cast<std::uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                value |= static_cast<std::uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                value |= static_cast<std::uint32_t>(c - 'A' + 10);
            } else {
                fail("invalid Unicode escape");
            }
        }
        return value;
    }

    std::string parse_string() {
        if (take() != '"') {
            fail("expected string");
        }
        std::string output;
        while (true) {
            const char c = take();
            if (c == '"') {
                return output;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                fail("control character in string");
            }
            if (c != '\\') {
                output.push_back(c);
                continue;
            }
            switch (take()) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': append_utf8(output, parse_hex4()); break;
                default: fail("invalid string escape");
            }
        }
    }

    std::int64_t parse_integer() {
        const std::size_t start = position_;
        const bool negative = consume('-');
        if (position_ == input_.size() || !std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            fail("expected integer");
        }
        std::uint64_t value = 0;
        while (position_ < input_.size() &&
               std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            const unsigned digit = static_cast<unsigned>(input_[position_] - '0');
            if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
                fail("integer overflow");
            }
            value = value * 10 + digit;
            ++position_;
        }
        if (position_ < input_.size() &&
            (input_[position_] == '.' || input_[position_] == 'e' || input_[position_] == 'E')) {
            fail("floating-point numbers are not supported");
        }
        const std::uint64_t positive_limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        if ((!negative && value > positive_limit) || (negative && value > positive_limit + 1)) {
            position_ = start;
            fail("integer out of range");
        }
        if (negative && value == positive_limit + 1) {
            return std::numeric_limits<std::int64_t>::min();
        }
        return negative ? -static_cast<std::int64_t>(value) : static_cast<std::int64_t>(value);
    }

    std::string input_;
    std::string path_;
    std::size_t position_ = 0;
};

}  // namespace

bool Value::as_bool() const {
    if (const auto* value = std::get_if<bool>(&data_)) {
        return *value;
    }
    type_error("a Boolean");
}

std::int64_t Value::as_int() const {
    if (const auto* value = std::get_if<std::int64_t>(&data_)) {
        return *value;
    }
    type_error("an integer");
}

const std::string& Value::as_string() const {
    if (const auto* value = std::get_if<std::string>(&data_)) {
        return *value;
    }
    type_error("a string");
}

const Value::Array& Value::as_array() const {
    if (const auto* value = std::get_if<Array>(&data_)) {
        return *value;
    }
    type_error("an array");
}

const Value::Object& Value::as_object() const {
    if (const auto* value = std::get_if<Object>(&data_)) {
        return *value;
    }
    type_error("an object");
}

const Value& Value::at(const std::string& key) const {
    const auto& object = as_object();
    const auto iterator = object.find(key);
    if (iterator == object.end()) {
        throw std::runtime_error("missing JSON key: " + key);
    }
    return iterator->second;
}

Value parse_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open JSON file: " + path);
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        throw std::runtime_error("cannot read JSON file: " + path);
    }
    return Parser(buffer.str(), path).parse();
}

}  // namespace delay::json
