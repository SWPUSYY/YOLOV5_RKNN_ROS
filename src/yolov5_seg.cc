#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "yolov5_seg.h"
// #include "common.h"
// #include "file_utils.h"
#include "image_utils.h"

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

unsigned char class_colors[][3] = {
        {255, 56, 56},   // 'FF3838'
        {255, 157, 151}, // 'FF9D97'
        {255, 112, 31},  // 'FF701F'
        {255, 178, 29},  // 'FFB21D'
        {207, 210, 49},  // 'CFD231'
        {72, 249, 10},   // '48F90A'
        {146, 204, 23},  // '92CC17'
        {61, 219, 134},  // '3DDB86'
        {26, 147, 52},   // '1A9334'
        {0, 212, 187},   // '00D4BB'
        {44, 153, 168},  // '2C99A8'
        {0, 194, 255},   // '00C2FF'
        {52, 69, 147},   // '344593'
        {100, 115, 255}, // '6473FF'
        {0, 24, 236},    // '0018EC'
        {132, 56, 255},  // '8438FF'
        {82, 0, 133},    // '520085'
        {203, 56, 255},  // 'CB38FF'
        {255, 149, 200}, // 'FF95C8'
        {255, 55, 199}   // 'FF37C7'
    };

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

void letterbox(const cv::Mat &image, cv::Mat &padded_image, BOX_RECT &pads, const float scale, const cv::Size &target_size, const cv::Scalar &pad_color)
{
    // 调整图像大小
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(), scale, scale);

    // 计算填充大小
    int pad_width = target_size.width - resized_image.cols;
    int pad_height = target_size.height - resized_image.rows;

    pads.left = pad_width / 2;
    pads.right = pad_width - pads.left;
    pads.top = pad_height / 2;
    pads.bottom = pad_height - pads.top;

    // 在图像周围添加填充
    cv::copyMakeBorder(resized_image, padded_image, pads.top, pads.bottom, pads.left, pads.right, cv::BORDER_CONSTANT, pad_color);
}

static unsigned char* load_model(const char* filename, int* model_size)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("fopen %s fail!\n", filename);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char* model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        fclose(fp);
        return NULL;
    }
    *model_size = model_len;
    fclose(fp);
    return model;
}

int init_yolov5_seg_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    int ret;
    int model_len = 0;
    rknn_context ctx = 0;

    // Load RKNN Model
    // model_len = read_data_from_file(model_path, &model);
    unsigned char* model = load_model(model_path, &model_len);
    if (model == NULL)
    {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Set core mask
    rknn_core_mask core_mask;
    core_mask = RKNN_NPU_CORE_0_1_2;

    ret = rknn_set_core_mask(ctx, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16)
    {
        app_ctx->is_quant = true;
    }
    // if (output_attrs[0].type == RKNN_TENSOR_INT8)
    // {
    //     app_ctx->is_quant = true;
    // }
    else
    {
        app_ctx->is_quant = false;
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }
    else
    {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_yolov5_seg_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

// int inference_yolov5_seg_model(rknn_app_context_t *app_ctx, image_buffer_t *img, object_detect_result_list *od_results)
// {
//     int ret;
//     image_buffer_t dst_img;
//     letterbox_t letter_box;
//     rknn_input inputs[app_ctx->io_num.n_input];
//     rknn_output outputs[app_ctx->io_num.n_output];
//     const float nms_threshold = NMS_THRESH;
//     const float box_conf_threshold = BOX_THRESH;
//     int bg_color = 114; // pad color for letterbox

//     if ((!app_ctx) || !(img) || (!od_results))
//     {
//         return -1;
//     }

//     memset(od_results, 0x00, sizeof(*od_results));
//     memset(&letter_box, 0, sizeof(letterbox_t));
//     memset(&dst_img, 0, sizeof(image_buffer_t));
//     memset(inputs, 0, sizeof(inputs));
//     memset(outputs, 0, sizeof(outputs));

//     // Pre Process
//     app_ctx->input_image_width = img->width;
//     app_ctx->input_image_height = img->height;
//     dst_img.width = app_ctx->model_width;
//     dst_img.height = app_ctx->model_height;
//     dst_img.format = IMAGE_FORMAT_RGB888;
//     dst_img.size = get_image_size(&dst_img);
//     dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
//     if (dst_img.virt_addr == NULL)
//     {
//         printf("malloc buffer size:%d fail!\n", dst_img.size);
//         return -1;
//     }

//     // letterbox
//     ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
//     if (ret < 0)
//     {
//         printf("convert_image_with_letterbox fail! ret=%d\n", ret);
//         goto out;
//     }

//     // Set Input Data
//     inputs[0].index = 0;
//     inputs[0].type = RKNN_TENSOR_UINT8;
//     inputs[0].fmt = RKNN_TENSOR_NHWC;
//     inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
//     inputs[0].buf = dst_img.virt_addr;

//     ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
//     if (ret < 0)
//     {
//         printf("rknn_input_set fail! ret=%d\n", ret);
//         goto out;
//     }

//     // Run
//     printf("rknn_run\n");
//     ret = rknn_run(app_ctx->rknn_ctx, nullptr);
//     if (ret < 0)
//     {
//         printf("rknn_run fail! ret=%d\n", ret);
//         goto out;
//     }

//     // Get Output
//     memset(outputs, 0, sizeof(outputs));
//     for (int i = 0; i < app_ctx->io_num.n_output; i++)
//     {
//         outputs[i].index = i;
//         outputs[i].want_float = (!app_ctx->is_quant);
//     }
//     ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
//     if (ret < 0)
//     {
//         printf("rknn_outputs_get fail! ret=%d\n", ret);
//         goto out;
//     }

//     // Post Process
//     post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

//     // Remeber to release rknn output
//     rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

// out:
//     if (dst_img.virt_addr != NULL)
//     {
//         free(dst_img.virt_addr);
//     }

//     return ret;
// }

int inference_yolov5_seg_model_syy(rknn_app_context_t* app_ctx, cv::Mat& img, object_detect_result_list* od_results)
{
    int ret;
    // image_buffer_t dst_img;

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    int img_width = img.cols;
    int img_height = img.rows;
    
    letterbox_t letter_box;
    rknn_input inputs[app_ctx->io_num.n_input];
    rknn_output outputs[app_ctx->io_num.n_output];
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;
    int bg_color = 114; // pad color for letterbox

    // if ((!app_ctx) || !(img) || (!od_results))
    // {
    //     return -1;
    // }

    memset(od_results, 0x00, sizeof(*od_results));
    memset(&letter_box, 0, sizeof(letterbox_t));
    // memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    // Pre Process
    // dst_img.width = app_ctx->model_width;
    // dst_img.height = app_ctx->model_height;
    // dst_img.format = IMAGE_FORMAT_RGB888;
    // dst_img.size = get_image_size(&dst_img);
    
    cv::Size target_size(app_ctx->model_width, app_ctx->model_height);
    cv::Mat dst_img(target_size.height, target_size.width, CV_8UC3);

    float scale_w = (float)target_size.width / img_width;
    float scale_h = (float)target_size.height / img_height;

// #if defined(DMA_ALLOC_DMA32)
//     /*
//      * Allocate dma_buf within 4G from dma32_heap,
//      * return dma_fd and virtual address.
//      */
//     ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHE_PATCH, dst_img.size, &dst_img.fd, (void **)&dst_img.virt_addr);
//     if (ret < 0) {
//         printf("alloc dma32_heap buffer failed!\n");
//         return -1;
//     }
// #else
//     dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
//     if (dst_img.virt_addr == NULL)
//     {
//         printf("malloc buffer size:%d fail!\n", dst_img.size);
//         return -1;
//     }
// #endif

    // letterbox
    // ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    // if (ret < 0)
    // {
    //     printf("convert_image_with_letterbox fail! ret=%d\n", ret);
    //     goto out;
    // }

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));

    // 图像缩放/Image scaling
    if (img_width != target_size.width || img_height != target_size.height)
    {
        // RGA是一个独立的2D硬件加速器
        // rga
        // rga_buffer_t src;
        // rga_buffer_t dst;
        // memset(&src, 0, sizeof(src));
        // memset(&dst, 0, sizeof(dst));
        // ret = resize_rga(src, dst, img, dst_img, target_size);
        // if (ret != 0)
        // {
        //     fprintf(stderr, "resize with rga error\n");
        // }

        // imwrite("dst_img.jpg", dst_img);

        // opencv
        float min_scale = std::min(scale_w, scale_h);
        scale_w = min_scale;
        scale_h = min_scale;
        letterbox(img, dst_img, pads, min_scale, target_size, cv::Scalar(128, 128, 128));

        imwrite("dst_img.jpg", dst_img);

        inputs[0].buf = dst_img.data;
    }
    else
    {
        inputs[0].buf = img.data;
    }

    letter_box.x_pad = pads.left;
    letter_box.y_pad = pads.top;
    letter_box.scale = std::min(scale_w, scale_h);

    std::cout << "letter_box.x_pad = " << letter_box.x_pad << std::endl;

    // Set Input Data
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
    // inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
    if (ret < 0)
    {
        printf("rknn_input_set fail! ret=%d\n", ret);
        // goto out;
    }

    // Run
    printf("rknn_run\n");
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        // goto out;
    }

    // Get Output
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++)
    {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        // goto out;
    }

    std::cout << " outputs size = " << outputs->size << std::endl;
    std::cout << " letter_box = " << letter_box.scale << std::endl;

    printf(" start post process \n");

    // Post Process
    post_process(app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, od_results);

    printf(" finished post process \n");

    // Remeber to release rknn output
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);

// out:
//     if (dst_img.virt_addr != NULL)
//     {
//         #if defined(DMA_ALLOC_DMA32)
//         dma_buf_free(dst_img.size, &dst_img.fd, dst_img.virt_addr);
//         #else
//         free(dst_img.virt_addr);
//         #endif
//     }

    return ret;
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// 按面积大小进行轮廓排序
bool Contour_Area(std::vector<cv::Point> contour_1, std::vector<cv::Point> contour_2){
    return cv::contourArea(contour_1) > cv::contourArea(contour_2);
}

// 按周长大小进行轮廓排序
bool Contour_arcLength(std::vector<cv::Point> contour_1, std::vector<cv::Point> contour_2){
    return cv::arcLength(contour_1, true) > cv::arcLength(contour_2, true);
}

std::vector<std::vector<cv::Point>> v_hull;

bool pointInHull(std::vector<cv::Point>& points, const std::vector<std::vector<cv::Point>> &hull)
{
    if(hull.size() != 0){
        std::vector<double> results;
        for(cv::Point p : points){
            double result = cv::pointPolygonTest(hull[0], p, false);
            // double distance = cv::pointPolygonTest(hull[0], p, true);
            results.push_back(result);
        }

        for(double result : results){
            if(result == 0 || result == 1)
                return true;
        }
    }
    return false;
}

void draw_mask_and_box(cv::Mat &orig_img, object_detect_result_list &od_results, const struct timeval& start_time, const char* type)
{
    // draw mask
    if(od_results.count < 1){
        return; 
    }
    else if (od_results.count >= 1)
    {
        int width = orig_img.cols;
        int height = orig_img.rows;
        char *img_data = (char *)orig_img.data;
        int cls_id = od_results.results[0].cls_id;

        uint8_t *seg_mask = od_results.results_seg[0].seg_mask;

        // 绘制轨道mask的底图
        cv::Mat black_img = cv::Mat::zeros(height, width, CV_8UC1);

        // 绘制黑底白mask
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                int pixel_offset = 3 * (j * width + k);
                if (seg_mask[j * width + k] != 0)
                {
                    black_img.at<uchar>(j ,k) = 255;
                }
            }
        }

        cv::imwrite("./black_img.jpg", black_img);

        // 寻找轮廓
        cv::Mat threshold_img;
        std::vector<std::vector<cv::Point>> contours;//轮廓点
        std::vector<std::vector<cv::Point>> sort_contours;//轮廓点
        std::vector<cv::Vec4i> hierarcy;
        
        // 转为二值图像
        cv::threshold(black_img, threshold_img, 100, 255, cv::THRESH_BINARY);

        // 开运算
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat open_img;
        cv::morphologyEx(threshold_img, open_img, cv::MORPH_OPEN, element);

        // cv::imwrite("open_img.jpg", open_img);

        findContours(threshold_img, contours, hierarcy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

        if(contours.size() != 0){
            // 轮廓面积排序，取最大值为轨道轮廓
            if(contours.size() == 1){
                sort_contours = std::move(contours);
            }

            else if(contours.size() > 1){
                std::sort(contours.begin(), contours.end(), Contour_Area);
                sort_contours.emplace_back(std::move(contours[0]));
            }

            // 寻找凸包
            std::vector<std::vector<cv::Point>> convex(sort_contours.size());//凸包轮廓点
            for (size_t i = 0; i < sort_contours.size(); i++)
            {
                convexHull(sort_contours[i], convex[i], false, true);
            }

            cv::Mat dst_img = cv::Mat::zeros(black_img.size(), CV_8UC3);//图像必须是3通道
            std::vector<cv::Vec4i> empty(0);

            v_hull.emplace_back(std::move(convex[0]));
            // v_hull.emplace_back(std::move(scaleConvexHullY(convex[0], 1.2f, 0.3f)));

            for (size_t k = 0; k < sort_contours.size(); k++)
            {
                cv::Scalar color = cv::Scalar(0, 0, 255);
                //绘制整个图像轮廓
                // drawContours(dst_img, sort_contours, (int)k, color, 2, cv::LINE_AA, hierarcy, 0, cv::Point(0, 0));
                //绘制凸包点
                drawContours(orig_img, v_hull, (int)k, color, 2, cv::LINE_AA, empty, 0, cv::Point(0, 0));
            }
            // cv::imwrite("./black_img.jpg", dst_img);  
            // free(seg_mask);
        }
        free(seg_mask);
    }

    
    //     for (int j = 0; j < height; j++)
    //     {
    //         for (int k = 0; k < width; k++)
    //         {
    //             int pixel_offset = 3 * (j * width + k);
    //             if (seg_mask[j * width + k] != 0)
    //             {
    //                 // orig_img.at<cv::Vec3b>(j, k)[0] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][0] * (1 - alpha) + orig_img.at<cv::Vec3b>(j, k)[0] * alpha, 0, 255); // r
    //                 // orig_img.at<cv::Vec3b>(j, k)[1] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][1] * (1 - alpha) + orig_img.at<cv::Vec3b>(j, k)[1] * alpha, 0, 255); // g
    //                 // orig_img.at<cv::Vec3b>(j, k)[2] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][2] * (1 - alpha) + orig_img.at<cv::Vec3b>(j, k)[2] * alpha, 0, 255); // b

    //                 orig_img.at<cv::Vec3b>(j, k)[0] = (255, 0, 255); // r
    //                 orig_img.at<cv::Vec3b>(j, k)[1] = (0, 255, 255); // g
    //                 orig_img.at<cv::Vec3b>(j, k)[2] = (255, 255, 0); // b
    //             }
    //         }
    //     }
    //     free(seg_mask);
    // }

    // draw boxes
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);

        if(det_result->cls_id == 0)
            continue;

        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        int box_width = x2 - x1;
        // box底边左右四分点
        cv::Point quartiles_left, quartiles_right, center_point;
        quartiles_left = cv::Point(x1 + (box_width / 4), y2);
        quartiles_right = cv::Point(x2 - (box_width / 4), y2);
        center_point = cv::Point(x1 + (box_width / 2), y2);

        std::vector<cv::Point> points;
        points.emplace_back(std::move(quartiles_left));
        points.emplace_back(std::move(quartiles_right));
        points.emplace_back(std::move(center_point));
        
        // std::cout << " quartiles_left = " << quartiles_left << std::endl;
        // std::cout << " quartiles_right = " << quartiles_right << std::endl;

        bool point_in_hull = pointInHull(points, v_hull);

        if(point_in_hull == 1){
            rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 2);
            sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
            putText(orig_img, text, cv::Point(x1, y1 - 6), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0,0,255), 1, cv::LINE_AA);

            cv::circle(orig_img, quartiles_left, 2, cv::Scalar(255, 255, 255), -1);
            cv::circle(orig_img, quartiles_right, 2, cv::Scalar(255, 255, 255), -1);
            cv::circle(orig_img, center_point, 2, cv::Scalar(255, 255, 255), -1);
        }
    }

    cv::imshow("img", orig_img);
    cv::waitKey(1);

    // 保存结果
    if(strcmp(type, "img") == 0){
        cv::imwrite("./out.jpg", orig_img);
        struct timeval stop_time;
        gettimeofday(&stop_time, NULL);
        printf("rknn run and process use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    }
    else if(strcmp(type, "video") == 0){
        // 计算FPS
        struct timeval stop_time;
        gettimeofday(&stop_time, NULL);
        float t = (__get_us(stop_time) - __get_us(start_time))/1000;
        printf("Infer time(ms): %f ms\n", t);
		putText(orig_img, cv::format("FPS: %.2f", (1.0 / (t / 1000))), cv::Point(10, 30), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
    }

    v_hull.clear();
    
    return;
}

void draw_mask_and_box_test(cv::Mat &orig_img, object_detect_result_list &od_results, const timeval &start_time)
{
    std::cout << " od_results.count = " << od_results.count << std::endl;

    cv::imwrite("orig_img.jpg", orig_img);
    // draw mask
    if(od_results.count < 1){
        return; 
    }
    else if (od_results.count >= 1)
    {
        int width = orig_img.cols;
        int height = orig_img.rows;
        char *img_data = (char *)orig_img.data;
        int cls_id = od_results.results[0].cls_id;

        uint8_t *seg_mask = od_results.results_seg[0].seg_mask;

        float alpha = 0.5f; // opacity

        // 绘制轨道mask的底图
        cv::Mat black_img = cv::Mat::zeros(height, width, CV_8UC1);

        // 绘制黑底白mask
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                int pixel_offset = 3 * (j * width + k);
                if (seg_mask[j * width + k] != 0)
                {
                    black_img.at<uchar>(j ,k) = 255;
                }
            }
        }

        cv::imwrite("black.jpg", black_img);

        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                int pixel_offset = 3 * (j * width + k);
                if (seg_mask[j * width + k] != 0)
                {
                    // orig_img.at<cv::Vec3b>(j, k)[0] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][0] * (1 - alpha) + orig_img.at<cv::Vec3b>(j, k)[0] * alpha, 0, 255); // r
                    // orig_img.at<cv::Vec3b>(j, k)[1] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][1] * (1 - alpha) + orig_img.at<cv::Vec3b>(j, k)[1] * alpha, 0, 255); // g
                    // orig_img.at<cv::Vec3b>(j, k)[2] = (unsigned char)clamp(class_colors[seg_mask[j * width + k] % 20][2] * (1 - alpha) + orig_img.at<cv::Vec3b>(j, k)[2] * alpha, 0, 255); // b

                    orig_img.at<cv::Vec3b>(j, k)[0] = (255, 0, 255); // r
                    orig_img.at<cv::Vec3b>(j, k)[1] = (0, 255, 255); // g
                    orig_img.at<cv::Vec3b>(j, k)[2] = (255, 255, 0); // b
                }
            }
        }
        free(seg_mask);
    }

    // draw boxes
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
        printf("%s @ (%d %d %d %d) %.3f\n", coco_cls_to_name(det_result->cls_id),
               det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom,
               det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 2);
        sprintf(text, "%s %.1f%%", coco_cls_to_name(det_result->cls_id), det_result->prop * 100);
        putText(orig_img, text, cv::Point(x1, y1 - 6), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0,0,255), 1, cv::LINE_AA);
    }

    cv::imshow("img", orig_img);
    cv::waitKey(1);

    cv::imwrite("./out.jpg", orig_img);
    struct timeval stop_time;
    gettimeofday(&stop_time, NULL);
    printf("rknn run and process use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    return;
}