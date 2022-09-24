#include "service/bert_service.h"

namespace lazydog {
    BertClassificationServer::BertClassificationServer(std::string serve_url_):
        serve_url(serve_url_){}
    
    BertClassificationServer::BertClassificationServer(std::string server_url_,uint32_t compute_thread_nums_,
                                                        uint32_t handler_thread_nums_):
        serve_url(server_url_),compute_thread_nums(compute_thread_nums_),handler_thread_nums(handler_thread_nums_){}

    BertClassificationServer::BertClassificationServer(std::string serve_url_, uint32_t compute_thread_nums_, uint32_t handler_thread_nums_,
                                                       uint32_t max_connection_nums_, uint32_t per_response_timeout_):
        serve_url(serve_url_),compute_thread_nums(compute_thread_nums_),handler_thread_nums(handler_thread_nums_),
        max_connection_nums(max_connection_nums_),peer_response_timeout(per_response_timeout_){}

    inline void  BertClassificationServer::set_gpu(int gpu_id){
        if (gpu_id < 0){
            printf("set gpu unvisible\n");
            return;
        }
        if(cudaSetDevice(gpu_id) == cudaError_t::cudaSuccess){
            printf("set gpu -> %d\n",gpu_id);
            return;
        }
        printf("failed to set gpu...\n");
    }

    void BertClassificationServer::_add_thread_2_map(pthread_t thread_id){
        printf("insert pair -> [thread_id]:%ld  [indices]:%d\n",thread_id,awake_thread_indices);
        thread_indices_map[thread_id] = awake_thread_indices++;
    }
    
    uint32_t BertClassificationServer::_get_current_thread_indices(){
        pthread_t thread_id = pthread_self();
        if(thread_indices_map.find(thread_id) == thread_indices_map.end()){
            _add_thread_2_map(thread_id);
        }
        return thread_indices_map[thread_id];
    }


    void BertClassificationServer::init_server(){
        WFGlobalSettings settings = GLOBAL_SETTINGS_DEFAULT;
        settings.compute_threads = compute_thread_nums;
        settings.handler_threads = handler_thread_nums;
        settings.endpoint_params.max_connections = max_connection_nums;
        settings.endpoint_params.response_timeout = peer_response_timeout;

        WORKFLOW_library_init(&settings);
        
        model_server = std::make_unique<WFHttpServer>(server_process);
    }

    
    void BertClassificationServer::server_process(WFHttpTask* task){
        const char* task_uri = task->get_req()->get_request_uri();
        if(strcmp(task_uri,"/welcome") == 0){
            task->get_resp()->append_output_body("<html>Welcome to lazydog text cls server</html>");
            return;
        }
        else if (strcmp(task_uri,"hello_word") == 0){
            task->get_resp()->append_output_body("<html>Hello World!</html>");
            return;
        }
        else if (strcmp(task_uri,serve_url.c_str()) == 0){
            auto* req = task->get_req();
            auto* resp = task->get_resp();
            cls_request cls_task_req;
            auto* series = series_of(task);
            auto* ctx = new series_context;
            ctx->response = resp;
            WFGoTask* predict_task = nullptr;
            if(model_inference_timeout == 0){
                predict_task = WFTaskFactory::create_go_task(serve_url,do_work,cls_task_req,ctx);
            }else {
                predict_task = WFTaskFactory::create_timedgo_task(0,model_inference_timeout*1e6,serve_url,cls_task_req,ctx);
            }
            auto&& callback = std::bind(&server_process_callback,this,task);
            predict_task->set_callback(callback);
        }
    }
    
    
}