#include <list>
#include <iostream>
#include <locale>
#include <vector>
#include <stdint.h>
#include <codecvt>
#include <set>

static std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8_converter;
static std::set<wchar_t> chinese_punc_chars = {
    8211, 8212, 8216, 8217, 8220, 8221,
    8230, 12289, 12290, 12296, 12297, 12298,
    12299, 12300, 12301, 12302, 12303, 12304,
    12305, 12308, 12309, 65281, 65288, 65289,
    65292, 65294, 65306, 65307, 65311};

bool _is_chinese_char(wchar_t text_char) {
    bool flag = (19968 <= text_char && text_char <= 40959) || (13312 <= text_char && text_char <= 19903) || (131072 <= text_char && text_char <= 173791) || (173824 <= text_char && text_char <= 183983) || (63744 <= text_char && text_char <= 64255) || (194560 <= text_char && text_char <= 195103);
    return flag;
}

bool is_punctuation_char(wchar_t text_char) {
    bool flag = (33 <= text_char && text_char <= 47) || (58 <= text_char && text_char <= 64) || (91 <= text_char && text_char <= 96) || (123 <= text_char && text_char <= 126) || (chinese_punc_chars.find(text_char) != chinese_punc_chars.end());
    return flag;
}

std::list<std::wstring> tokenize_v2(std::wstring &text) {
    size_t text_size = text.size();
    size_t chinese_or_punc_char_size = 0;
    size_t en_char_size = 0;
    std::vector<uint8_t> chinese_or_punc_char_flags(text_size, 0);
    for (size_t i = 0; i < text_size; ++i) {
        if (_is_chinese_char(text[i])) {
            ++chinese_or_punc_char_size;
            chinese_or_punc_char_flags[i] = 1;
        }
    }

    std::cout << chinese_or_punc_char_size << " " << text_size << std::endl;
    std::list<std::wstring> tokens;
    // means that all data is chinese
    if (chinese_or_punc_char_size == text_size) {
        for (size_t i = 0; i < text_size; ++i) {
            tokens.emplace_back(text.substr(i, 1));
        }
        return tokens;
    }
    size_t start_index = 0;
    for (size_t i = 0; i < text_size; ++i) {
        if (chinese_or_punc_char_flags[i]) {
            std::cout << utf8_converter.to_bytes(text.substr(i,1)) << std::endl;
            tokens.emplace_back(text.substr(i, 1));
            ++start_index;
            continue;
        }
        // the whitespace is between chinses char...
        if (text[i] !=32) {
            continue;
        }
        if (text[i] == 32 && start_index < i){
            // emplace back the substr
            tokens.emplace_back(text.substr(start_index, i - start_index));
            start_index = i + 1;
            continue;
        }
        ++start_index;
    }
    // append the remain value
    if (start_index < text_size) {
        std::cout << "remains " << utf8_converter.to_bytes(text.substr(start_index,text_size-start_index)) << std::endl;
        tokens.emplace_back(text.substr(start_index, text_size - start_index));
    }
    return tokens;
}

int main(){
    std::wstring s1 = L"你今天还好吗？(the brown fox jumps over the lazy dog!)人类";
    auto tokens = tokenize_v2(s1);
    for(auto iter=tokens.begin();iter!=tokens.end();++iter){
        std::cout << utf8_converter.to_bytes(*iter) << " ->";
    }
    std::cout << std::endl;
    return 0;
}