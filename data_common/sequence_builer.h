#ifndef _SEQUENCE_BUILDER_H_
#define _SEQUENCE_BUILDER_H_
#include <stdint.h>
#include <vector>
#include <string>
#include <list>

namespace lazydog {
    using BertInputType = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;
    class TextSequenceBuiler {

    private:
        uint32_t max_seq_size;
        // some specail token id we need in current task!
        uint32_t bert_cls_id = 101;
        uint32_t bert_sep_id = 102;
        uint32_t _last_size;
        uint32_t _intercept_size;
        std::vector<uint32_t> input_ids_placeholder;
        std::vector<uint32_t> attention_mask_placeholder;

    public:
        TextSequenceBuiler(uint32_t max_seq_size_);
        TextSequenceBuiler(uint32_t max_seq_size_, uint32_t bert_cls_id_, uint32_t bert_sep_id_);
        TextSequenceBuiler(TextSequenceBuiler &seq_builder);
        TextSequenceBuiler(TextSequenceBuiler &&seq_builder);

        BertInputType build_sequence(std::vector<uint32_t> &token_ids);

        void build_sequence(std::vector<std::uint32_t> &token_ids, std::vector<uint32_t> &input_ids, std::vector<uint32_t> &attention_mask);

        BertInputType build_sequence_no_check(std::vector<uint32_t> &token_ids);
        
        void build_sequence_no_check(std::vector<std::uint32_t> &token_ids,std::vector<uint32_t>& input_ids,std::vector<uint32_t>& attention_mask);
        
        
    };
}

#endif