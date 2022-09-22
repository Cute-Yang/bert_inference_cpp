#include "inference/bert_classification.h"

#define ck(call) check(call,__LINE__,__FILE__)

namespace lazydog{
    MemoryBlock::MemoryBlock(uint32_t num_classes_,uint32_t max_seq_size_):
        num_classes(num_classes_),max_seq_size(max_seq_size_){};
    
    bool MemoryBlock::init_memory() {
        if (init_flag){
            printf("already initialized,do not invoke this function!\n");
        }
        printf("init the host memory buffer!\n");
        input_ids.assign(max_seq_size,0);
        attention_mask.assign(max_seq_size,0);
        probs.assign(num_classes,0.0f);
        host_buffers[input_ids_indices] = input_ids.data();
        host_buffers[attention_mask_indices] = attention_mask.data();
        host_buffers[probs_indices] = attention_mask.data();
        
        printf("init the device buffer!\n");
        // for input_ids
        cudaError_t create_status;
        create_status =  cudaMalloc(&device_buffers[input_ids_indices],max_seq_size*sizeof(input_id_type));
        if(create_status!=cudaError::cudaSuccess){
            printf("failed to allocate device memory for input_ids\n");
            return false;
        }

        // for attention mask
        create_status = cudaMalloc(&device_buffers[attention_mask_indices],max_seq_size*sizeof(attention_mask_type));
        if(create_status!=cudaError_t::cudaSuccess){
            printf("failed to allocate device memory for attention_mask\n");
            cudaFree(device_buffers[probs_indices]);
            return false;
        }
        
        // for probs
        create_status = cudaMalloc(&device_buffers[probs_indices],max_seq_size*sizeof(prob_type));
        if(create_status!=cudaError_t::cudaSuccess){
            printf("failed to allocate device memory for probs\n");
            cudaFree(device_buffers[input_ids_indices]);
            cudaFree(device_buffers[attention_mask_indices]);
            return false;
        }

        init_flag = true;
        return true;
    }

    void*const* MemoryBlock::get_host_buffer_ptr() const {
        return host_buffers.data();
    }

    void*const* MemoryBlock::get_device_buffer_ptr() const {
        return device_buffers.data();
    }

    MemoryBlock::~MemoryBlock(){
        printf("free the device memory allocated!\n");
        for(size_t i=0;i<device_buffers.size();++i){
            if(device_buffers[i]!=nullptr){
                cudaFree(device_buffers[i]);
            }
        }
    }

    BertClassifier::BertClassifier(std::string plan_file_):plan_file(plan_file_){
        compute_data_bytes();
    }
    
    BertClassifier::BertClassifier(uint32_t max_cuda_stream_size_,uint32_t max_context_size_,
                                   uint32_t max_memory_block_size_,std::string plan_file_):
        max_cuda_stream_size(max_cuda_stream_size),max_context_size(max_context_size_),
        max_memory_block_size(max_memory_block_size_),plan_file(plan_file_){
            compute_data_bytes();
        }

    BertClassifier::BertClassifier(uint32_t max_cuda_stream_size_,uint32_t max_context_size_,
                                   uint32_t max_memory_block_size_,std::string plan_file_,
                                   uint32_t batch_size_):
        max_cuda_stream_size(max_cuda_stream_size),max_context_size(max_context_size_),
        max_memory_block_size(max_memory_block_size_),plan_file(plan_file_),
        batch_size(batch_size_){
            compute_data_bytes();
        }
    
    inline void BertClassifier::compute_data_bytes(){
        input_ids_bytes = batch_size * max_seq_size * sizeof(input_id_type);
        attention_mask_bytes = batch_size * max_seq_size * sizeof(attention_mask_type);
        probs_bytes = batch_size * num_classes * sizeof(prob_type);
    }

    BertClassifier::~BertClassifier(){
        printf("free the memory block and cuda stream!\n");
        free_cuda_stream();
    }

    void BertClassifier::init_memory_block(){
        if(memory_block_init_flag){
            printf("already initialize the memory block!\n");
            return;
        }
        
        memory_blocks.reserve(max_memory_block_size);
        for(uint32_t i=0;i<max_memory_block_size;++i){
            memory_blocks.emplace_back(num_classes,max_seq_size,batch_size);
        }
        for(uint32_t i=0;i<max_memory_block_size;++i){
            if(!memory_blocks[i].init_memory()){
                printf("failed to initialize memory block at %d\n",i);
            }
        }
        memory_block_init_flag = true;
    }

    void BertClassifier::init_trt(){
        trt_runtime = nvinfer1::createInferRuntime(trt_logger);
        std::ifstream reader(plan_file,std::ios::binary);
        size_t fsize = 0;
        reader.seekg(0,reader.end);
        fsize = reader.tellg();
        reader.seekg(0,reader.beg);
        std::vector<char> engine_string(fsize);
        reader.read(engine_string.data(),fsize); 
        if(engine_string.size() == 0){
            printf("failed to read engine file -> %s,maybe this file is empty!\n",plan_file.c_str());
            return;
        }
        printf("successfully read engine string\n");
        
        engine = trt_runtime->deserializeCudaEngine(engine_string.data(),fsize);
        if(engine == nullptr){
            printf("failed to deserialize plan file -> %s,maybe the file is bad!\n",plan_file);
            return;
        }
        printf("successfully create tensort engine!\n");

        contexts.assign(max_context_size,nullptr);
        for(uint32_t i=0;i<max_context_size;++i){
            contexts[i] = engine->createExecutionContext();
        }

        streams.assign(max_cuda_stream_size,nullptr);
        for(uint32_t i=0;i<max_cuda_stream_size;++i){
            if(cudaStreamCreate(streams[i])!=cudaError::cudaSuccess){
                printf("failed to create cuda stream at %d\n",i);
                if (i>0){
                    free_cuda_stream(0,i-1);
                }
                return;
            }
        }
        trt_init_flag = true;
    }

    void BertClassifier::free_cuda_stream(uint32_t beg,uint32_t end){
        if (beg > end){
            printf("beg -> %d greater than end -> %d \n",beg,end);
            return;
        }
        auto size = streams.size();

        if(beg < 0 || end >=size){
            printf("out of range!\n");
            return;
        }
        // loop free
        for(uint32_t i=beg;i<end;++i){
            if(streams[i] != nullptr){
                if (cudaStreamDestroy(*streams[i]) == cudaError::cudaSuccess){
                    printf("successfully destory stream at %d...\n",i);
                }else{
                    printf("failed to destroy the stream at %d...\n",i);
                }
            }
        }
    }

    void  BertClassifier::free_cuda_stream(){
        free_cuda_stream(0,streams.size());
    }

    const std::vector<prob_type>& BertClassifier::predict(std::wstring text,uint32_t indices){
        uint32_t memory_indices = max_memory_block_size % indices;
        uint32_t stream_indices = max_cuda_stream_size % indices;
        uint32_t context_indices = max_context_size % indices;
        auto& memory_block = memory_blocks[memory_indices];
        std::vector<input_id_type>& input_ids = memory_block.input_ids;
        std::vector<attention_mask_type>& attention_mask = memory_block.attention_mask;
        std::vector<prob_type>& probs = memory_block.probs;
        auto text_tokens = tokenizer->tokenize(text);
        tokenizer->produce_input_ids_and_attention_mask(text_tokens,input_ids,attention_mask);
        
        // copy data
        cudaMemcpy(memory_block.device_buffers[input_ids_indices],memory_block.host_buffers[input_ids_indices],input_ids_bytes,cudaMemcpyKind::cudaMemcpyHostToDevice);
        cudaMemcpy(memory_block.device_buffers[attention_mask_indices],memory_block.host_buffers[attention_mask_indices],attention_mask_bytes,cudaMemcpyKind::cudaMemcpyHostToDevice);
        contexts[context_indices]->executeV2(memory_block.device_buffers.data());

        //copy result
        cudaMemcpy(memory_block.host_buffers[probs_indices],memory_block.device_buffers[probs_indices],probs_bytes,cudaMemcpyKind::cudaMemcpyDeviceToHost);
        
        return memory_block.probs;
    }
}