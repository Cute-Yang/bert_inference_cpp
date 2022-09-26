#ifndef __BERT_SERVICE_H__
#define __BERT_SERVICE_H__

#include "workflow/WFTask.h"
#include "workflow/WFHttpServer.h"
#include <unordered_map>
#include <atomic>
#include "cuda_runtime_api.h"
#include <pthread.h>
#include "inference/bert_classification.h"
#include <memory>
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "rapidjson/document.h"

namespace lazydog { 
    struct series_context {
        // the req id
        bool _is_req_valid = true;
        protocol::HttpResponse* response=nullptr;
        const prob_type* predict_result;
    };
    
    struct cls_request{
        std::string text;
        bool is_valid = true;
    };

    class BertClassificationServer {
        public:
            // generate some default config!
            uint32_t max_connection_nums = 200;
            uint32_t peer_response_timeout = 15 * 1000;
            // for compute task!
            uint32_t compute_thread_nums = 0;
            uint32_t handler_thread_nums = 50;

            float prob_thresh = 0.5;

        private:
            std::unordered_map<pthread_t,uint32_t> thread_indices_map;
            std::string serve_url;
            uint32_t model_inference_timeout = 500; // ms
            std::atomic<uint32_t> awake_thread_indices = 0;
            // WFHttpServer model_server;
            std::unique_ptr<WFHttpServer> model_server;
            BertClassifier* model_classifier = nullptr;
        
        public:

            BertClassificationServer(std::string serve_url_);
            
            BertClassificationServer(std::string serve_url_,uint32_t compute_thread_nums_,uint32_t handler_thread_nums_);

            BertClassificationServer(std::string serve_url_,uint32_t compute_thread_nums_,uint32_t handler_thread_nums_,
                                     uint32_t max_connection_nums_,uint32_t per_response_timeout_);
            
            inline void set_model_inference_timeout(uint32_t timeout) {
                model_inference_timeout = timeout;
            }

            inline uint32_t get_machine_cpu_cores();
            
            inline void set_gpu(int gpu_id);

            void _add_thread_2_map(pthread_t thread_id) noexcept;

            uint32_t _get_current_thread_indices();

            inline void start(uint16_t port){
                printf("server start at localhost:%d\n",port);
                model_server->start(port);
            }

            inline void start(const char* host,uint16_t port){
                printf("server start at %s:%d\n",host,port);
                model_server->start(host,port);
            } 

            inline void shutdown() {
                model_server->shutdown();
            }

            inline void wait_finish(){
                model_server->wait_finish();
            }

            void do_work(cls_request* req,series_context* ctx);

            void server_process(WFHttpTask* task);

            void server_process_callback(WFGoTask* predict_task);
            
            void init_server();

            std::string _wrap_response_json_data(const prob_type* prob_result);

            cls_request parse_request_json(const char* request_body);
    };
}

#endif