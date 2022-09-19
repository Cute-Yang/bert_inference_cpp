#include "data_common/bert_tokenizer.h"
#include <iostream>

int main(){
    std::string vocab = "../datas/vocab.txt";
    lazydog::BertTokenizer tokenizer(vocab,127);
    tokenizer.read_vocab_paris_from_file();
    // tokenizer.add_custom_tokens(L"[decimal_number_seq]");
    std::wstring pattern = L"我的|老板";
    tokenizer.set_pattern(pattern);
    std::wstring s1 = L"这个 人老板eqreq逆天次啊了真的好生气 (the brown fox jumps over the lazy dog...)";
    auto tokens = tokenizer.tokenize(s1);
    for (auto iter = tokens.begin(); iter != tokens.end(); ++iter) {
        std::cout << tokenizer.utf8_converter.to_bytes(*iter) << "->";
    }
    std::cout << std::endl;
    // std::vector<uint32_t> token_ids = tokenizer.convert_tokens_2_ids(tokens);
    // for (size_t i = 0; i < token_ids.size(); ++i) {
    //     std::cout << token_ids[i] << " ";
    // }
    // std::cout << std::endl;
}