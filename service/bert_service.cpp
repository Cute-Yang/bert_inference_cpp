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
        
        printf("create a model server ptr...\n");
        model_server = std::make_unique<WFHttpServer>(server_process);
    }

    
    void BertClassificationServer::server_process(WFHttpTask* task){
        const char* task_uri = task->get_req()->get_request_uri();
        if(strcmp(task_uri,"/welcome") == 0){
            task->get_resp()->append_output_body("<html>Welcome to lazydog text cls server</html>");
            return;
        }
        else if (strcmp(task_uri,"/hello_word") == 0){
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
                // create async task
                predict_task = WFTaskFactory::create_go_task(serve_url,do_work,cls_task_req,ctx);
            }else {
                // create async task with specify timeout!
                predict_task = WFTaskFactory::create_timedgo_task(0,model_inference_timeout*1e6,serve_url,do_work,cls_task_req,ctx);
                // _warp a class membert function -> std::function!
                predict_task->set_callback(std::bind(&BertClassificationServer::server_process_callback,this,std::placeholders::_1));
            }
            series_of(task) -> push_back(predict_task);
        }// means that get invalid uri
        else{
            task->get_resp()->append_output_body("<html>Invalid uri!</html>");
        }
    }
    
    std::string BertClassificationServer::_wrap_response_json_data(const prob_type* prob_result){
        uint32_t num_classes = model_classifier->get_num_classes();
        uint32_t batch_size = model_classifier->get_batch_size();
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buf);
        writer.StartObject();
        writer.Key("status");
        writer.Int(0);
        writer.Key("probs");
        writer.StartObject();
        for(uint32_t i=0;i<batch_size;++i){
            writer.Key(("prob_" + std::to_string(i)).c_str());
            writer.StartArray();
            for(uint32_t j=0;j<num_classes;++j){
                writer.Double(static_cast<double>(*(prob_result + i * num_classes + j)));
            }
            writer.EndArray();
        }
        writer.EndObject();
        writer.Key("error_detail");
        writer.String("");
        writer.EndObject();
        return buf.GetString();
    }

    void BertClassificationServer::server_process_callback(WFGoTask* predict_task){
        auto* context = (series_context*)series_of(predict_task) -> get_context();
        auto state = predict_task->get_state();
        // means model run failed or the request data is invalid!
        if(state == WFT_STATE_ABORTED){
            context->response->append_output_body("{\"status\":-1,probs:[],\"error_detail\":\"model_inference_error\"}");
            return;
        }

        if (!context->_is_req_valid){
            context->response->append_output_body("{\"status\":-2,probs:[],\"error_detail\":\"invalid_request\"}");
            return;
        }

        std::string response_body = _wrap_response_json_data(context->predict_result);
        // to avoid one copy!
        context->response->append_output_body(std::move(response_body));
    }

    cls_request BertClassificationServer::parse_request_json(const char* request_body){
        rapidjson::Document doc;
        doc.Parse(request_body);
        cls_request req{};
        if(doc.HasParseError() || doc.IsNull() || doc.ObjectEmpty() || !doc.IsObject()){
            req.text = "";
            req.is_valid = false;
        }else{
            if(!doc.HasMember("text") || !doc["text"].IsString()){
                req.text = "";
                req.is_valid = false;
            }else{
                req.text = doc["text"].GetString();
                // req.is_valid = true;
            }
        }
        return req;
    }
    

    void BertClassificationServer::do_work(cls_request* req,series_context* ctx){
        if(!req->is_valid){
            ctx->_is_req_valid = false;
            return;
        }
        uint32_t thread_indices = _get_current_thread_indices();
        const prob_type* data_ptr = model_classifier->predict(req->text,thread_indices);
        ctx->predict_result = data_ptr;
    }
}