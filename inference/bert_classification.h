#ifndef __BERT_CLASSIFICATION__
#define __BERT_CLASSIFICATION__
#include <NvInfer.h>
#include <cuda_fp16.h>
#include<cuda_runtime_api.h>
#include<fstream>
#include <vector>

namespace lazydog {
    typedef int input_id_type;
    typedef int attention_mask_type;
    typedef float prob_type;
    class MemoryBlock{
        private:
            uint32_t num_classes;
            uint32_t max_seq_size;
        
        public:
            std::vector<void *> host_buffers;
            std::vector<void *> device_buffers;
            std::vector<input_id_type> input_ids;
            std::vector<attention_mask_type> attention_mask;
            std::vector<prob_type> probs;

        public:
            MemoryBlock(uint32_t num_classes_,uint32_t max_seq_size_);
            
            void init_memory() noexcept;
            
            void free_cuda_memory() noexcept;

            void** get_device_buffer_ptr() noexcept;

            void** get_host_buffer_ptr() noexcept;
            
            
            
    };

    class BertClassifier{
        private:
            uint32_t max_cuda_stream_size;
            uint32_t max_context_size;
            nvinfer1::IRuntime* trt_runtime;
            std::vector<nvinfer1::IExecutionContext*> contexts;
            std::vector<cudaStream_t*> streams;
            std::string plan_file;
            uint32_t max_seq_size;
            std::vector<MemoryBlock> memory_blocks;
    };
}

#endif