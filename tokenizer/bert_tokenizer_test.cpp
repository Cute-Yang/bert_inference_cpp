#include "bert_tokenizer.h"
#include <iostream>


static lazydog::BertTokenizer tokenizer("../../datas/vocab.txt");

template<typename dataType>
void print_list_data(std::list<dataType>& datas){
    size_t data_size = datas.size();
    auto iter = datas.cbegin();
    for(int i = 0;i<data_size - 1;++i){
        std::cout << "'" << tokenizer.utf8_converter.to_bytes(*iter) << "'" << ",";
        ++iter;
    }
    std::cout << "'" <<tokenizer.utf8_converter.to_bytes(*iter) << "'" << std::endl;
}

int main(){
    std::wstring text = L"我真的好钟意你阿，baby!";
    tokenizer.read_vocab_paris_from_file();
    auto tokens = tokenizer.tokenize(text);
    print_list_data<std::wstring>(tokens);
    return 0;
}