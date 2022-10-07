#include "bert_service.h"
#include "bert_classification.h"
#include "bert_tokenizer.h"
#include <string>

static WFFacilities::WaitGroup wait_group(1);

void sig_handler(int signo)
{
	wait_group.done();
}

int main(){
    std::string plan_file = "../datas/wzry_bert.plan";
    std::string vocab_file = "../assets/vocab.txt";
    uint32_t num_classes = 8;
    uint32_t max_seq_size = 32;

    uint32_t compute_thread_nums = 40;
    uint32_t handler_thread_nums = 100;

    uint32_t max_context_size = compute_thread_nums;
    uint32_t max_memory_block_size = compute_thread_nums;
    uint32_t max_stream_size = 3;

    std::string serve_url = "/lazydog_up";

    lazydog::BertTokenizer tokenzier{vocab_file};
    tokenzier.read_vocab_paris_from_file();
    lazydog::BertClassifier classifier{max_stream_size,max_context_size,max_memory_block_size, plan_file};
    classifier.set_tokenizer(&tokenzier);
    classifier.init_memory_block();
    classifier.init_trt();
    lazydog::BertClassificationServer server{serve_url,compute_thread_nums,handler_thread_nums};
    server.set_model_classifier(&classifier);
    server.init_server();

    uint16_t port = 6006;

    if(server.start(port) == 0){
        printf("start serving....\n");
        wait_group.wait();
        server.stop();
    }else{
        printf("failed to start server...\n");
    }
    return 0;
}