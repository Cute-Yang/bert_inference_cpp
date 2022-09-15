#include <string>
#include <iostream>
#include <vector>
#include <regex>
#include <list>
#include <locale>
#include <codecvt>

static std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8_converter;

constexpr uint8_t whitespace_ascii = 32;

std::vector<std::string> whitespace_tokenize(std::string &text) {
    size_t i = 0;
    size_t j = 0;
    size_t text_size = text.size();
    if (text[0] == whitespace_ascii) {
        j++;
    }
    if (text[text_size - 1] == whitespace_ascii) {
        text_size--;
    }
    std::vector<std::string> text_tokens;
    text_tokens.reserve(text_size);
    for (j;j < text_size; ++j) {
        if (text[j] == whitespace_ascii) {
            text_tokens.push_back(text.substr(i, (j - i)));
            i = j + 1;
        }
    }
    text_tokens.push_back(text.substr(i, (j - i)));
    return text_tokens;
}


void string_regexp_search(){
    std::regex pattern("lake|\\[MASK\\]");
    std::string s1 = "the lake is very beautiful[MASK],jumps over!";
    std::string::const_iterator i1 = s1.begin(),i2=s1.end();
    std::smatch result;
    // std::forward_list<std::string> match_result;
    // std::forward_list<uint32_t> match_flag;
    std::vector<std::string> split_result;
    std::vector<int> split_flag;
    int x1 = 0;
    
    while (std::regex_search(i1, i2, result, pattern)){
        auto p1 = result[0].first;
        auto p2 = result[0].second;
        split_result.push_back(s1.substr(x1,p1-i1));
        split_flag.push_back(0);
        split_result.push_back(s1.substr(p1-s1.begin(),p2-p1));
        split_flag.push_back(1);
        std::cout << result[0] << std::endl;
        x1=p2-i1;
        std::cout << x1 << std::endl;
        i1=p2;
    }
    if(i1!=i2){
        split_result.push_back(s1.substr(i1-s1.begin(),i2-i1));
        split_flag.push_back(0);
    }
    auto data_size = split_result.size();
    for(int i = 0;i<data_size;++i){
        std::cout << split_result[i] << "->" << split_flag[i] << std::endl;
    }
}

void string_regexp_search_v2(){
    std::wregex pattern(L"老板|\\[MASK\\]|游戏");
    std::wstring s1 = L"the lake is very  游戏 不好玩beautiful [MASK] ,jumps over! 老板发财哟！";
    std::wstring::const_iterator i1 = s1.begin(), i2 = s1.end();
    std::wsmatch result;
    // std::forward_list<std::string> match_result;
    // std::forward_list<uint32_t> match_flag;
    std::vector<std::wstring> split_result;
    std::vector<int> split_flag;
    int x1 = 0;

    while (std::regex_search(i1, i2, result, pattern)) {
        auto p1 = result[0].first;
        auto p2 = result[0].second;
        split_result.push_back(s1.substr(x1, p1 - i1));
        split_flag.push_back(0);
        split_result.push_back(s1.substr(p1 - s1.begin(), p2 - p1));
        split_flag.push_back(1);
        // std::cout << result[0] << std::endl;
        x1 = p2 - i1;
        std::cout << x1 << std::endl;
        i1 = p2;
    }
    if (i1 != i2) {
        split_result.push_back(s1.substr(i1 - s1.begin(), i2 - i1));
        split_flag.push_back(0);
    }
    auto data_size = split_result.size();
    for (int i = 0; i < data_size; ++i) {
        std::cout  << split_flag[i] << std::endl;
    }
}



void string_regexp_search_v3(){
    std::wstring s1 = L"狡兔跨懒狗...[MASK]做自己喜欢的";
    std::wregex pattern(L"\\[MASK\\]|喜欢|狗");
    const std::wsregex_iterator end;
    for(std::wsregex_iterator iter(s1.begin(),s1.end(),pattern);iter!=end;++iter){
        std::cout << utf8_converter.to_bytes(iter->str()) << std::endl;
    }
}


using LazySplitType = std::list<std::pair<std::wstring,bool>>;
LazySplitType split_with_pattern(std::wstring& text,std::wregex& pattern){
    LazySplitType split_result;
    std::wstring::const_iterator i1 = text.cbegin();
    std::wstring::const_iterator i2 = text.cend();
    std::wsmatch search_resut;
    std::wstring::difference_type i=0;
    while(std::regex_search(i1,i2,search_resut,pattern)){
        split_result.push_back({text.substr(i,search_resut[0].first-i1),false});
        split_result.push_back({text.substr(search_resut[0].first-text.cbegin(),search_resut[0].second-search_resut[0].first),true});
        i1=search_resut[0].second;
        i=search_resut[0].second - text.cbegin();
    }
    if(i1!=i2){
        split_result.push_back({text.substr(i1-text.cbegin(),i2-i1),false});
    }
    return split_result;
}

std::list<std::string> whitespace_tokenize_v2(std::string &text) {
    size_t i = 0;
    size_t j = 0;
    size_t index_bound = text.size() - 1;
    std::list<std::string> split_result;
    while (text[j] == whitespace_ascii) {
        ++j;
    }
    while (text[index_bound] == whitespace_ascii) {
        --index_bound;
    }
    for (j; j <= index_bound; ++j) {
        if (text[j] == whitespace_ascii) {
            if (j == i) {
                ++i;
                continue;
            }
            split_result.push_back(text.substr(i, (j - i)));
            i = j + 1;
        }
    }
    split_result.push_back(text.substr(i, (j - i)));
    return split_result;
}

int main(){
    std::string text = "the brown fox  jumps  over the      lazydog ";
    auto text_split = whitespace_tokenize(text);
    for(auto iter=text_split.begin();iter!=text_split.end();++iter){
        std::cout << *iter << "->";
    }
    std::cout << std::endl;

    auto text_split_v2 = whitespace_tokenize_v2(text);
    for (auto iter = text_split_v2.begin(); iter != text_split_v2.end(); ++iter) {
        std::cout << *iter << "->";
    }
    std::cout << std::endl;

    // string_regexp_search_v2();
    // string_regexp_search_v3();

    std::wstring s1 = L"一板一[URL]眼，就会[MASK]滋生[SEP]弱点";
    std::wregex pattern(L"\\[URL\\]|弱者|\\[MASK\\]|\\[SEP\\]");
    LazySplitType split_result = split_with_pattern(s1,pattern);

    for(auto iter=split_result.begin();iter!=split_result.end();++iter){
        std::cout << utf8_converter.to_bytes(iter->first) << " "  << iter->second << std::endl;
    }
    return 0;
}