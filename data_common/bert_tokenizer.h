#ifndef __BERT_TOKENIZER_H__
#define __BERT_TOKENIZER_H__
#include <string>
#include "data_value.h"
#include <set>
#include <map>
#include <vector>
#include <cstdio>
#include <regex>
#include <list>
#include <stdint.h>
#include <locale>
#include <codecvt>


namespace lazydog {
    using input_ids_type = std::vector<uint32_t>;
    using attention_mask_type = std::vector<uint32_t>;
    using token_type = std::wstring;
    using tokens_type = std::vector<std::wstring>;

    class BertTokenizer {
        public:
            static std::set<wchar_t> chinese_punc_chars;
            static std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8_converter;
        private:
            std::string vocab_path;
            // some special tokens
            std::wstring sep_token=L"[SEP]";
            std::wstring mask_token=L"[MASK]";
            std::wstring pad_token=L"[PAD]";
            std::wstring cls_token=L"[CLS]";
            std::wstring unk_token = L"[UNK]";
            // the max size of chars for a english word,if greater than this,regard as not word
            uint32_t max_input_chars_per_word=28;
            // a map to restore the vocab,consider to use unordered_map
            std::map<std::wstring, uint32_t> token_2_id{};
            std::map<uint32_t,std::wstring> id_2_token{};
            std::map<std::wstring,uint32_t> added_token_2_id{};
            std::map<uint32_t,std::wstring> added_id_2_token{};
            std::wregex pattern;
        public:
            BertTokenizer();
            BertTokenizer(std::string vocab_path_);
            BertTokenizer(std::string vocab_path_, uint32_t max_input_chars_per_word_);
            BertTokenizer(std::string vocab_path_, std::wstring sep_token_,
                        std::wstring mask_token_, std::wstring pad_token_, std::wstring cls_token_,
                        uint32_t max_input_chars_per_word_);

            uint32_t convert_token_2_id(std::wstring& token);

            std::vector<uint32_t> convert_tokens_2_ids(std::list<std::wstring> &tokens);

            void convert_tokens_2_ids(std::list<std::wstring> &tokens, std::vector<uint32_t> &input_ids);

            std::vector<uint32_t> convert_tokens_2_ids(std::list<std::wstring> &tokens,size_t max_size);

            void convert_tokens_2_ids_with_check(std::list<std::wstring>& tokens,std::vector<uint32_t>& input_ids);
            
            void insert_one_token_2_ordered_vector(std::wstring& token);
            
            void add_custom_tokens(std::list<std::wstring>& add_tokens,bool is_special=true);

            std::list<std::wstring> _tokenize(std::wstring& text);

            std::list<std::wstring> tokenize(std::wstring& text);

            std::list<std::wstring> tokenize_without_never_split(std::wstring& text);

            bool _is_chinese_char(wchar_t single_char);

            std::wstring _run_strip_accents(std::wstring& text);

            std::list<std::wstring> _run_split_on_punc(std::list<std::wstring>& basic_text_tokens);

            bool _is_punctuation_char(wchar_t text_char);

            std::list<std::wstring> _whitespace_tokenize(std::wstring& text);

            void read_vocab_paris_from_file();
            
            std::list<std::pair<std::wstring,bool>> _split_text_with_regexp_search(std::wstring& text);

            std::list<std::pair<std::wstring,bool>> _split_text_with_trie_structure(std::wstring& text);
            
            std::list<std::wstring> wordpiece_tokenize(std::list<std::wstring>& text_tokens);

            void reset_vocab_path(std::string new_vocab_path,bool reset_vocab=true);

            void reset_max_input_chars_per_word(uint32_t new_max_input_chars_per_word);

            size_t get_vocab_size() const;

            void transfer_string_to_lower(std::wstring& text);

            void transfer_string_to_upper(std::wstring& text);

            std::wstring join_substrs(std::list<std::wstring> substrs);

            std::list<std::wstring> split_line(std::wstring& line,std::wstring& line_sep);
            
            std::list<std::wstring> split_line_v2(std::wstring& line,std::wstring& line_sep);
        };
} // namespace lazydog

#endif