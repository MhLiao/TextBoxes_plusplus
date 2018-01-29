#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "caffe/layers/detection_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void DetectionOutputLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* loc_data = bottom[0]->gpu_data();
  const Dtype* prior_data = bottom[2]->gpu_data();
  const int num = bottom[0]->num();

  // Decode predictions.
  Dtype* bbox_data = bbox_preds_.mutable_gpu_data();
  const int loc_count = bbox_preds_.count();
  const bool clip_bbox = false;
  DecodeBBoxesGPU<Dtype>(loc_count, loc_data, prior_data, code_type_,
      variance_encoded_in_target_, num_priors_, share_location_,
      num_loc_classes_, background_label_id_, clip_bbox, bbox_data, use_polygon_);
  // Retrieve all decoded location predictions.
  const Dtype* bbox_cpu_data;
  if (!share_location_) {
    Dtype* bbox_permute_data = bbox_permute_.mutable_gpu_data();
    if(use_polygon_){
      PermuteDataGPU<Dtype>(loc_count, bbox_data, num_loc_classes_, num_priors_,
        12, bbox_permute_data);
    }else{
      PermuteDataGPU<Dtype>(loc_count, bbox_data, num_loc_classes_, num_priors_,
        9, bbox_permute_data);
    }  
    bbox_cpu_data = bbox_permute_.cpu_data();
  } else {
    bbox_cpu_data = bbox_preds_.cpu_data();
  }

  // Retrieve all confidences.
  Dtype* conf_permute_data = conf_permute_.mutable_gpu_data();
  PermuteDataGPU<Dtype>(bottom[1]->count(), bottom[1]->gpu_data(),
      num_classes_, num_priors_, 1, conf_permute_data);
  const Dtype* conf_cpu_data = conf_permute_.cpu_data();

  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > indices;
    int num_det = 0;
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    if (share_location_) {
      if(use_polygon_){
        bbox_idx = i * num_priors_ * 12;
      }else{
        bbox_idx = i * num_priors_ * 9;
      }
    } else {
      if(use_polygon_){
        bbox_idx = conf_idx * 12;
      }else{
        bbox_idx = conf_idx * 9;
      }
    }
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      const Dtype* cur_conf_data = conf_cpu_data + conf_idx + c * num_priors_;
      const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        if(use_polygon_){
          cur_bbox_data += c * num_priors_ * 12;
        }else{
          cur_bbox_data += c * num_priors_ * 9;
        }
      }
      ApplyNMSFast(cur_bbox_data, cur_conf_data, num_priors_,
          confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]), use_polygon_);
      // ApplyNMSFastRBox(cur_bbox_data, cur_conf_data, num_priors_,
          // confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[conf_idx + label * num_priors_ + idx];
          score_index_pairs.push_back(std::make_pair(
                  score, std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                SortScorePairDescend<pair<int, int> >);
      score_index_pairs.resize(keep_top_k_);
      // Store the new indices.
      map<int, vector<int> > new_indices;
      for (int j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      all_indices.push_back(new_indices);
      num_kept += keep_top_k_;
    } else {
      all_indices.push_back(indices);
      num_kept += num_det;
    }
  }
  int top_num1 = 0;
  int top_num2 = 0;
  if(use_polygon_){
    top_num1 = 15;
    top_num2 = 12;
  }else{
    top_num1 = 12;
    top_num2 = 9;
  }
  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(top_num1);
  Dtype* top_data;
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    top_shape[2] = num;
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += top_num1;
    }
  } else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();
  }
  int count = 0;
  boost::filesystem::path output_directory(output_directory_);
  for (int i = 0; i < num; ++i) {
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors_ * top_num2;
    } else {
      bbox_idx = conf_idx * top_num2;
    }
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      vector<int>& indices = it->second;
      const Dtype* cur_conf_data =
        conf_cpu_data + conf_idx + label * num_priors_;
      const Dtype* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += label * num_priors_ * top_num2;
      }
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * top_num1] = i;
        top_data[count * top_num1 + 1] = label;
        top_data[count * top_num1 + 2] = cur_conf_data[idx];
        for (int k = 0; k < top_num2; ++k) {
          top_data[count * top_num1 + 3 + k] = cur_bbox_data[idx * top_num2 + k];
        }
        ++count;
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DetectionOutputLayer);

}  // namespace caffe
