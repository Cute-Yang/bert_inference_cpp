#include "bert_tokenizer.h"
#include "bert_classification.h"
#include "bert_service.h"

int main(){
    std::string plan_file = "datas/bert_infer.plan";
    std::string vocab = "datas/vocab.txt";
    std::string server_url = "/text_inference";
    uint32_t batch_size = 1;
    uint32_t max_seq_size = 32;
    uint16_t port = 6000;

    lazydog::BertTokenizer tokenizer(vocab);
    lazydog::BertClassifier model(plan_file);
    model.set_tokenizer(&tokenizer);

    lazydog::BertClassificationServer server(server_url);
    server.set_model(&model);

    if(server.start(port) !=0){
        getchar();
        server.stop();
    }
}