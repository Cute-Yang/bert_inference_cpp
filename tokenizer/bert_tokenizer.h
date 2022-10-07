#ifndef __BERT_TOKENIZER_H__
#define __BERT_TOKENIZER_H__
#include <string>
#include "data_value.h"
#include <set>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <regex>
#include <list>
#include <stdint.h>
#include <locale>
#include <codecvt>


namespace lazydog {
    class BertTokenizer {
        public:
            static std::set<wchar_t> chinese_punc_chars;
            static std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8_converter;
        private:
            // std::wstring_convert<std::codecvt_utf8<wchar_t>> utf8_converter;
            std::string vocab_path;
            // some special tokens
            std::wstring sep_token = L"[SEP]";
            std::wstring mask_token = L"[MASK]";
            std::wstring pad_token = L"[PAD]";
            std::wstring cls_token = L"[CLS]";
            std::wstring unk_token = L"[UNK]";
            // the max size of chars for a english word,if greater than this,regard as not word
            uint32_t max_input_chars_per_word=28;
            // a map to restore the vocab,consider to use unordered_map
            std::unordered_map<std::wstring, uint32_t> token_2_id;
            std::unordered_map<uint32_t,std::wstring> id_2_token;
            std::unordered_map<std::wstring,uint32_t> added_token_2_id;
            std::unordered_map<uint32_t,std::wstring> added_id_2_token;
            std::wregex pattern;
            uint32_t max_seq_size = 32;
            uint32_t _last_size = 32 -1;
            uint32_t _intercept_size = 32 -2;
            std::vector<uint32_t> input_ids_placeholder;
            std::vector<uint32_t> attention_mask_placeholder;
        public:
            BertTokenizer();
            BertTokenizer(std::string vocab_path_);
            BertTokenizer(std::string vocab_path_, uint32_t max_input_chars_per_word_,uint32_t max_seq_size_);
            BertTokenizer(std::string vocab_path_, std::wstring sep_token_,
                        std::wstring mask_token_, std::wstring pad_token_, std::wstring cls_token_,
                        uint32_t max_input_chars_per_word_,uint32_t max_seq_size_);
            BertTokenizer(BertTokenizer& ref);
            BertTokenizer(BertTokenizer&& ref);

            uint32_t convert_token_2_id(std::wstring& token);

            std::vector<uint32_t> convert_tokens_2_ids(std::list<std::wstring> &tokens);

            void convert_tokens_2_ids(std::list<std::wstring> &tokens, std::vector<uint32_t> &input_ids);

            std::vector<uint32_t> convert_tokens_2_ids(std::list<std::wstring> &tokens,size_t max_size);

            void convert_tokens_2_ids_with_check(std::list<std::wstring>& tokens,std::vector<uint32_t>& input_ids);
            
            void insert_one_token_2_ordered_vector(std::wstring& token);
            
            void add_custom_tokens(std::list<std::wstring>& add_tokens,bool is_special);

            std::list<std::wstring> _tokenize(std::wstring& text);
            std::list<std::wstring> _tokenizer_v2(std::wstring& text);

            std::list<std::wstring> tokenize(std::wstring& text);

            std::list<std::wstring> tokenize_without_never_split(std::wstring& text);

            inline bool _is_chinese_char(wchar_t single_char);

            std::wstring _run_strip_accents(std::wstring& text);

            std::list<std::wstring> _run_split_on_punc(std::list<std::wstring>& basic_text_tokens);

            inline bool _is_punctuation_char(wchar_t text_char);

            std::list<std::wstring> _whitespace_tokenize(std::wstring& text);

            void read_vocab_paris_from_file();
            
            std::list<std::pair<std::wstring,bool>> _split_text_with_regexp_search(std::wstring& text);

            std::list<std::pair<std::wstring,bool>> _split_text_with_trie_structure(std::wstring& text);
            
            std::list<std::wstring> wordpiece_tokenize(std::list<std::wstring>& text_tokens);

            void reset_vocab_path(std::string new_vocab_path,bool reset_vocab);

            void reset_max_input_chars_per_word(uint32_t new_max_input_chars_per_word);

            inline size_t get_vocab_size() const;

            inline void transfer_string_to_lower(std::wstring& text);

            inline void transfer_string_to_upper(std::wstring& text);

            std::wstring join_substrs(std::list<std::wstring> substrs);

            std::list<std::wstring> split_line(std::wstring& line,std::wstring& line_sep);
            
            void print_list_string(std::list<std::wstring>& text_list);
            std::list<std::wstring> split_line_v2(std::wstring& line,std::wstring& line_sep);
            
            bool _is_en_char(wchar_t text_char);
            
            void set_pattern_with_string(std::wstring& pattern_){
                pattern = pattern_;
            }

            void set_pattern_with_file(std::string& pattern_file);

            void produce_input_ids_and_attention_mask(std::list<std::wstring>& text_tokens,std::vector<uint32_t>& input_ids,std::vector<uint32_t>& attention_mask);
            uint32_t get_max_seq_size() const;
        };
} // namespace lazydog

#endif