#include "sequence_builer.h"
#include <cstdio>

namespace lazydog {
    /**
     * @brief Construct a new Text Sequence Builer:: Text Sequence Builer object
     * we will use the default bert cls id and bert seq id!
     * @param max_seq_size_ 
     */
    TextSequenceBuiler::TextSequenceBuiler(uint32_t max_seq_size_):max_seq_size(max_seq_size_) {
        printf("we will use default bert_cls_id:%d,bert_seq_id:%d\n",bert_cls_id,bert_sep_id);
        // input_ids_placeholder.reserve(max_seq_size);
        input_ids_placeholder.assign(max_seq_size,0);
        input_ids_placeholder[0] = bert_cls_id;
        // attention_mask_placeholder.reserve(max_seq_size);
        attention_mask_placeholder.assign(max_seq_size,1);
        _last_size = max_seq_size -1;
        _intercept_size = max_seq_size - 2;
    }

    TextSequenceBuiler::TextSequenceBuiler(uint32_t max_seq_size_,uint32_t bert_cls_id_,uint32_t bert_sep_id_):
        max_seq_size(max_seq_size_),bert_cls_id(bert_cls_id_),bert_sep_id(bert_sep_id_){
            // input_ids_placeholder.reserve(max_seq_size);
            input_ids_placeholder.assign(max_seq_size,0);
            input_ids_placeholder[0] = bert_cls_id;
            attention_mask_placeholder.assign(max_seq_size,0);
            _last_size = max_seq_size - 1;
            _intercept_size = max_seq_size - 2;
        }

    /**
     * @brief Construct a new Text Sequence Builer:: Text Sequence Builer object
     * if the size of two seq builder is not eqaul,we will adjust the size of data!
     * @param seq_builder 
     */
    TextSequenceBuiler::TextSequenceBuiler(TextSequenceBuiler& seq_builder){
        bert_cls_id = seq_builder.bert_cls_id;
        bert_sep_id = seq_builder.bert_sep_id;
        // 
        if(max_seq_size!=seq_builder.max_seq_size){
            input_ids_placeholder.resize(seq_builder.max_seq_size);
            attention_mask_placeholder.resize(seq_builder.max_seq_size);
        }
        input_ids_placeholder.assign(seq_builder.input_ids_placeholder.begin(),seq_builder.input_ids_placeholder.end());
        attention_mask_placeholder.assign(seq_builder.attention_mask_placeholder.end(),seq_builder.attention_mask_placeholder.end());
        max_seq_size = seq_builder.max_seq_size;
        _last_size = seq_builder._last_size;
        _intercept_size = seq_builder._intercept_size;
    }
    /**
     * @brief Construct a new Text Sequence Builer:: Text Sequence Builer object
     * will swap the placeholder data ptr!
     * @param seq_builder 
     */
    TextSequenceBuiler::TextSequenceBuiler(TextSequenceBuiler&& seq_builder){
        bert_cls_id = seq_builder.bert_cls_id;
        bert_sep_id = seq_builder.bert_sep_id;
        max_seq_size = max_seq_size;
        input_ids_placeholder.swap(seq_builder.input_ids_placeholder);
        attention_mask_placeholder.swap(seq_builder.attention_mask_placeholder);
        _last_size = seq_builder._last_size;
        _intercept_size = seq_builder._intercept_size;
    }

    void TextSequenceBuiler::build_sequence(std::vector<std::uint32_t> &token_ids, std::vector<uint32_t> &input_ids, std::vector<uint32_t> &attention_mask) {
        size_t n = token_ids.size();
        // we should intercept this sequence
        if( n >=_intercept_size){
            std::copy(token_ids.begin(),token_ids.begin() + _intercept_size,input_ids.begin() +1);
            input_ids[_last_size] = bert_cls_id;
            attention_mask.assign(max_seq_size,1);
        }else{
            std::copy(token_ids.begin(),token_ids.end(),input_ids.begin()+1);
            input_ids[n+1] = bert_sep_id;
            attention_mask.assign(n+1,1);
        }
        // return {input_ids,attention_mask};
    }

    /**
     * @brief 
     * we assume that the sizeof token_ids <= (max_seq_size -2 )
     * @param token_ids 
     * @param input_ids 
     * @param attention_mask 
     */
    void TextSequenceBuiler::build_sequence_no_check(std::vector<std::uint32_t> &token_ids, std::vector<uint32_t> &input_ids, std::vector<uint32_t> &attention_mask){
        // reset the values!
        input_ids.assign(input_ids_placeholder.begin(),input_ids_placeholder.end());
        attention_mask.assign(attention_mask_placeholder.begin(),attention_mask_placeholder.end());
        std::copy(token_ids.begin(),token_ids.end(),input_ids.begin());
        // set the valid token mask -> 1
        attention_mask.assign(token_ids.size(),1);
    }

    BertInputType TextSequenceBuiler::build_sequence(std::vector<uint32_t> &token_ids) {
        std::vector<uint32_t> input_ids(input_ids_placeholder);
        std::vector<uint32_t> attention_mask(attention_mask_placeholder);
        size_t n = token_ids.size();
        if(n>=_intercept_size){
            std::copy(token_ids.begin(),token_ids.begin() + _intercept_size,input_ids.begin() + 1);
            input_ids[_last_size] = bert_sep_id;
            attention_mask.assign(max_seq_size,1);
        }else{
            std::copy(token_ids.begin(),token_ids.end(),input_ids.begin()+1);
            input_ids[n+1] = bert_sep_id;
            attention_mask.assign(n+1,1);
        }
        return {input_ids,attention_mask};
    }


}