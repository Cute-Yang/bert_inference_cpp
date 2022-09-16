#include <list>
#include <string>
#include <locale>
#include <codecvt>
#include <iostream>

static std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8_converter;

std::list<std::wstring> split_line(std::wstring &line, std::wstring &line_sep) {
    size_t sep_size = line_sep.size();
    size_t text_size = line.size();
    // record the _intercept size!
    size_t start_index = 0;
    size_t i = 0;
    std::list<std::wstring> split_result;
    // if line starts with line_sep,we should skip the line_sep
    // if (line.substr(0, sep_size) == line_sep) {
    //     i = i + sep_size;
    //     start_index = i;
    // }
    while (i < text_size) {
        if (line[i] == line_sep[0] && line_sep == line.substr(i, sep_size)) {
            // means that found a sep
            split_result.push_back(line.substr(start_index, i - start_index));
            i = i + sep_size;
            start_index = i;
            continue;
        }
        ++i;
    }
    // if the end of line is not line_sep,we should push back the remain substr!
    if (start_index < text_size) {
        split_result.push_back(line.substr(start_index, text_size - start_index));
    }else if (start_index == text_size){
        split_result.push_back({});
    } 
    return split_result;
}


int main(){
    std::wstring s1 = L"[sep]生与死[sep] 轮回不止,我们[sep]生,他们死,hh[sep]停滞不前吧![sep]";
    std::wstring sep = L"[sep]";
    auto split_result = split_line(s1,sep);
    std::cout << "the original data is " << utf8_converter.to_bytes(s1) << std::endl;
    for(auto iter=split_result.begin();iter!=split_result.end();++iter){
        std::string data = utf8_converter.to_bytes(*iter);
        std::cout << data << std::endl;
    }
    return 0;
}