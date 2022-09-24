#ifndef __BERT_CLASSIFICATION__
#define __BERT_CLASSIFICATION__
#include <NvInfer.h>
#include <cuda_fp16.h>
#include<cuda_runtime_api.h>
#include<fstream>
#include <vector>
#include <cstdio>
#include "data_common/bert_tokenizer.h"
#include <unordered_map>
#include <pthread.h>


namespace lazydog {   
    typedef uint32_t input_id_type;
    typedef uint32_t attention_mask_type;
    typedef float prob_type;
    // the sizeof binding datas!
    constexpr uint32_t num_bindings = 3;
    constexpr size_t input_ids_indices = 0;
    constexpr size_t attention_mask_indices = 1;
    constexpr size_t probs_indices = 2;

    class SeaLogger: public nvinfer1::ILogger{
        public:
            Severity report_level;
        
        SeaLogger(Severity level=Severity::kINFO):
            report_level(level){};
        
        void log(Severity level,const char* msg) noexcept override {
            if(level > report_level){
                return;
            }
            switch(level){
                case Severity::kINTERNAL_ERROR:
                    printf("INTERNAL_ERROR: ");
                    break;
                case Severity::kERROR:
                    printf("ERROR: ");
                    break;
                case Severity::kWARNING:
                    printf("WARNING: ");
                    break;
                case Severity::kINFO:
                    printf("INFO: ");
                    break;
                default:
                    printf("VERBOSE: ");
            }
            printf("%s\n",msg);
        }
    };

    class MemoryBlock{
        private:
            uint32_t num_classes;
            uint32_t max_seq_size;
            uint32_t batch_size = 1;
            bool init_flag = false;
        
        public:
            std::vector<void *> host_buffers{num_bindings,nullptr};
            std::vector<void *> device_buffers{num_bindings,nullptr};
            // vector to manage our memory!
            std::vector<input_id_type> input_ids;
            std::vector<attention_mask_type> attention_mask;
            std::vector<prob_type> probs;

        public:
            MemoryBlock(uint32_t num_classes_,uint32_t max_seq_size_);

            MemoryBlock(uint32_t num_classes_,uint32_t max_seq_size_,uint32_t batch_size_);

            ~MemoryBlock();
            
            bool init_memory();
            
            void free_cuda_memory();

            void*const* get_device_buffer_ptr() const;

            void*const* get_host_buffer_ptr() const;
    };

    class BertClassifier{
        private:
            uint32_t max_cuda_stream_size = 2;
            uint32_t max_context_size = 10;
            uint32_t max_memory_block_size = 10;
            nvinfer1::IRuntime* trt_runtime = nullptr;
            nvinfer1::ICudaEngine* engine = nullptr;
            std::vector<nvinfer1::IExecutionContext*> contexts;
            std::vector<cudaStream_t*> streams;
            std::string plan_file;
            uint32_t max_seq_size;
            uint32_t num_classes;
            uint32_t batch_size = 1;

            //cache the copy byte!
            size_t input_ids_bytes;
            size_t attention_mask_bytes;
            size_t probs_bytes;

            std::vector<MemoryBlock> memory_blocks;
            BertTokenizer* tokenizer = nullptr;
            // a default logger for tensorrt!
            SeaLogger trt_logger{nvinfer1::ILogger::Severity::kINFO};
            // to avoid init for twice!
            bool trt_init_flag = false;

            bool memory_block_init_flag = false;

            std::unordered_map<pthread_t,size_t> thread_lookup_table;
        
        public:
            BertClassifier(std::string plan_file_);
            BertClassifier(uint32_t max_cuda_stream_size_,uint32_t max_context_size_,
                           uint32_t max_memory_block_size_,std::string plan_file_);
            
            BertClassifier(uint32_t max_cuda_stream_size_,uint32_t max_context_size_,
                           uint32_t max_memory_block_size_,std::string plan_file_,
                           uint32_t batch_size_);
            
            inline void set_tokenizer(BertTokenizer* token_ptr){
                if(token_ptr == nullptr){
                    printf("tokenizer can not be set to nullptr!\n");
                }
                tokenizer = token_ptr;
            }

            ~BertClassifier();

            inline uint32_t get_max_seq_size() const {
                return max_seq_size;
            }

            inline uint32_t get_num_classes() const {
                return num_classes;
            }

            inline uint32_t get_batch_size() const {
                return batch_size;
            }

            void init_memory_block();

            void init_trt();
            
            // void predict_with_check(std::wstring text);
            
            std::vector<prob_type>& predict(std::wstring& text,uint32_t indices);
            void predict(std::wstring& text,uint32_t indices,std::vector<prob_type>& prob_result);

            void free_cuda_streams(uint32_t beg,uint32_t end);

            void free_cuda_streams();

            inline void compute_data_bytes();

            void explicitly_free_memory_blocks(uint32_t beg,uint32_t end);
    };
}

#endif