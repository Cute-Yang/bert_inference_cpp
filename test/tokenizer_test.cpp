#include "data_common/bert_tokenizer.h"
#include <iostream>
#include <chrono>



template<typename dataType>
void print_vector_value(std::vector<dataType> datas){
    std::cout << "[";
    size_t n = datas.size();
    for(size_t i=0;i<n-1;++i){
        std::cout << datas[i] << ",";
    }
    std::cout << datas[n-1];
    std::cout << "]" << std::endl;
}

int main(){
    std::string vocab = "../datas/vocab.txt";
    lazydog::BertTokenizer tokenizer(vocab,127,64);
    tokenizer.read_vocab_paris_from_file();
    // tokenizer.add_custom_tokens(L"[decimal_number_seq]");
    std::wstring pattern = L"我的|老板";
    tokenizer.set_pattern_with_string(pattern);
    std::wstring s1 = L"这个 人老板eqreq逆天次啊了真的好生气 (the brown fox jumps over the lazy dog...)";
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    auto tokens = tokenizer.tokenize(s1);
    decltype(t1) t2 = std::chrono::steady_clock::now();

    std::chrono::duration<double> time_used  = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    std::cout << "used time " << time_used.count() * 1000 << " ms" << std::endl;
    std::cout << "[";
    for (auto iter = tokens.begin(); iter != tokens.end(); ++iter) {
        std::cout << tokenizer.utf8_converter.to_bytes(*iter) << ",";
    }
    std::cout << "]" << std::endl;
    std::cout << "[";
    std::vector<uint32_t> token_ids = tokenizer.convert_tokens_2_ids(tokens);
    for (size_t i = 0; i < token_ids.size()-1; ++i) {
        std::cout << token_ids[i] << ",";
    }
    std::cout << token_ids[token_ids.size() - 1];
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    std::cout << "***********************input params**********************" << std::endl;
    
    uint32_t max_seq_size = tokenizer.get_max_seq_size();
    std::vector<uint32_t> input_ids(max_seq_size,0);
    std::vector<uint32_t> attention_mask(max_seq_size,0);
    tokenizer.produce_input_ids_and_attention_mask(tokens,input_ids,attention_mask);
    std::cout << "input_ids:" << std::endl;
    print_vector_value(input_ids);
    std::cout << "attention_mask:" << std::endl;
    print_vector_value(attention_mask); 
    return 0;
}