#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <time.h>

#include <thread>
#include <mutex>
#include <deque>
#include <vector>
#include <condition_variable>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <cassert>
#include <string>


using namespace caffe;  // NOLINT(build/namespaces)
using std::string;


class CThreadDemo
{
private:
    std::deque<std::pair<cv::Mat, string>> m_data;
    std::mutex m_mtx; // 全局互斥锁.
    std::condition_variable m_cv; // 全局条件变量.
    int       m_nGen;

private:
    void ProductThread(std::vector<string> v){
        // 生产者线程完成图像的预处理，并将处理后的图像送入队列
        cv::Size input_geometry_;  
        input_geometry_ = cv::Size(4096, 4096);   //模型输入图尺寸
        
        float mean_file[3] = {69.966, 69.966, 69.966};   //均值文件
        string data_path = "/home/xxs/workspace/CudaLayer/test/";
        // string img_path = "edge_0000.png";

        int count = v.size();
        printf("v.size%d\n", count);
        // int ncount = 0;

        // while (ncount < count){
        for (int i = 0; i < count; ++i)
        {
            /* code */
        
            struct timeval tv1,tv2;
            gettimeofday(&tv1,NULL);
            // while(m_data.size()<50){
                // std::cout<<files[1][0]<<std::endl;
                string img_path = v[i].substr(0,7);
                // printf("%s\n", img_path.c_str());            
                cv::Mat img = cv::imread(data_path+img_path, -1);
                CHECK(!img.empty()) << "Unable to decode image "<<std::endl;

                cv::Mat sample;
                sample = img;
                cv::Mat sample_resized;
                cv::Mat sample_float;
                cv::Mat sample_normalized;
                cv::Mat mean_;
                sample_resized = sample;

                mean_ = cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(mean_file[0],mean_file[1],mean_file[2]));

                sample_resized.convertTo(sample_float, CV_32FC3);

                CHECK(!mean_.empty()) << " 1 empty image "<<std::endl;
                CHECK(!sample_float.empty()) << " 2 empty image "<<std::endl;

                cv::subtract(sample_float, mean_, sample_normalized);

                std::unique_lock <std::mutex> lck(m_mtx);
                
                struct timeval tv3,tv4;
                gettimeofday(&tv3,NULL);
                m_data.push_back(make_pair(sample_normalized,img_path));
                gettimeofday(&tv4,NULL);

//                printf("push time %1d\n",tv4.tv_sec*1000 + tv4.tv_usec/1000 - tv3.tv_sec*1000 - tv3.tv_usec/1000 );

                printf("size =======%d \n",m_data.size());
                lck.unlock();
                m_cv.notify_all();
                // ++ ncount ;
                // printf("ncount%d\n", ncount);
                gettimeofday(&tv2,NULL);
            // }

//            printf("read time %1d\n",tv2.tv_sec*1000 + tv2.tv_usec/1000 - tv1.tv_sec*1000 - tv1.tv_usec/1000 );
                   // int gpu_num=4;


        }
    }


    
 

    void ConsumeThread(int id){
        // 消费者线程

        string model_file   = "/home/xxs/workspace/CudaLayer/test/deploy_merge.prototxt";
        string trained_file = "/home/xxs/workspace/Resnet18_cls_iter_2500_merge.caffemodel";
        //float mean_file[3]    = {69.966, 69.966, 69.966};
        //  float std_file[3] = {1.0, 1.0, 1.0};
        string label_file   = "/home/xxs/workspace/CudaLayer/multi_test/cnt_synset_word.txt";
        string txt_file = "/home/xxs/workspace/CudaLayer/test/test1.txt";
        string dir = "/home/xxs/workspace/CudaLayer/test/";
        int gpu_id = id;
        printf("\n\n\n\n%d",gpu_id);

       
   
        int num_channels_ = 3;
        shared_ptr <Net<float>> net_;
        net_.reset(new Net<float>(model_file, TEST));
        net_->CopyTrainedLayersFrom(trained_file);

        cv::Size input_geometry_;
        Blob<float>* input_layer = net_->input_blobs()[0];
        input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        
        

        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(gpu_id);
        input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
        /* Forward dimension change to all layers. */
        net_->Reshape();

        /* Forward dimension change to all layers. */
        std::vector<cv::Mat> input_channels;
        // int count = ;
        while (m_data.size() > 0){
            struct timeval tv1,tv2;
            gettimeofday(&tv1,NULL);
            // clock_t start, end;
            // start = clock();
            // double duration;
            printf("while  forward  \n");
            std::unique_lock <std::mutex> lck(m_mtx);
            while (m_data.empty()){
                printf("no pic  \n");
                m_cv.wait(lck);
            }


            struct timeval tv5,tv6;
            gettimeofday(&tv5,NULL);
           
            
            cv::Mat sample_normalized = m_data.front().first;
            string img_path = m_data.front().second;
            m_data.pop_front();
            gettimeofday(&tv6,NULL);
//            printf("pop time %1d\n",tv6.tv_sec*1000 + tv6.tv_usec/1000 - tv5.tv_sec*1000 - tv5.tv_usec/1000 );
            
            lck.unlock();
            // init net data
            Blob<float>* input_layer2 = net_->input_blobs()[0];
            int width = input_layer2->width();
            int height = input_layer2->height();
            float* input_data = input_layer2->mutable_cpu_data();
            for (int i = 0; i < input_layer2->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
            }
                cv::split(sample_normalized, input_channels);
                CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
                == net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";

            //forward
            net_->ForwardPrefilled();

            std::vector<std::vector<float> > outputs;

            Blob<float>* output_layer = net_->output_blobs()[0];
            for (int i = 0; i < output_layer->num(); ++i) {
              const float* begin = output_layer->cpu_data() + i * output_layer->channels();
              const float* end = begin + output_layer->channels(); //
              /* Copy the output layer to a std::vector */
              outputs.push_back(std::vector<float>(begin, end));
            }
            printf("%d\n", outputs.size());

            // count--;
            gettimeofday(&tv2,NULL);
            printf("GPU %d process time %1d\n",gpu_id ,tv2.tv_sec*1000 + tv2.tv_usec/1000 - tv1.tv_sec*1000 - tv1.tv_usec/1000 );
            // end = clock();
            // duration = (double)(end - start) / CLOCKS_PER_SEC;
            // printf( "process time %f seconds\n", duration );
        }

    }
    
    


public:
    CThreadDemo(){
        m_data.clear();
        m_nGen = 0;
    }
    ~CThreadDemo(){
        m_data.clear();
    }
    void Start(){
        
        vector<string> labels;
        vector<vector<string>> files;
        int c = 1;
        string txt_file = "/home/xxs/workspace/CudaLayer/test/test1.txt";
        string data_root = "/home/xxs/workspace/CudaLayer/test/";
        std::ifstream label(txt_file.c_str());
        CHECK(label) << "Unable to open labels file " << txt_file;
        string line;
        while (std::getline(label, line)){
            // std::cout<<line<<std::endl;
            labels.push_back(string(line));
        }
        int num  = labels.size();
        // int gpus= 4;
        int tmp1=  num /float(c);
        for (int i = 0; i <c ; ++i)
        {
           std::vector<string> v;
           files.push_back(v);
        }
        for (int i = 0; i < num; ++i)
        {
            for (int j = 0; j < c; ++j)
            {
                if (i%c == j)
                {
                    files[j].push_back(labels[i]);
                }
            }
        }

        std::vector<std::thread> threads;
        threads.clear();
        int gpu_id [4]= {1,2,3,4};
        for (int i = 0; i < c; i++){/* 生产者线程 */
            threads.push_back(std::thread(&CThreadDemo::ProductThread, this, files[i]));
        }

        for (int i = 0; i < 1; i++){/* 消费者线程 */
            threads.push_back(std::thread(&CThreadDemo::ConsumeThread, this, gpu_id[i]));
        }
        for (auto& t : threads){/* 等待所有线程的退出 */
            t.join();
        }
    }
};


int main(int argc, char* argv[])
{

    struct timeval tv1,tv2;
    gettimeofday(&tv1,NULL);
    CThreadDemo test;
    test.Start();
    gettimeofday(&tv2,NULL);
    printf("%1d\n",tv2.tv_sec*1000 + tv2.tv_usec/1000 - tv1.tv_sec*1000 - tv1.tv_usec/1000 );

    return 0;
}