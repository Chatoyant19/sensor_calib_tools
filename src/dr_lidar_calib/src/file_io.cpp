#include <fcntl.h>   // linux open() function
#include <unistd.h>  // linux close() function
#include <google/protobuf/io/coded_stream.h>           // CodedInputStream
#include <google/protobuf/io/zero_copy_stream.h>       // ZeroCopyInputStream,
#include <google/protobuf/io/zero_copy_stream_impl.h>  // FileInputStream, FileOutputStream
#include <google/protobuf/text_format.h>   
#include <google/protobuf/message.h>
#include <glob.h>

#include "file_io.h"
#include "sensor_extrinsic.pb.h"
#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "rapidxml_print.hpp"

namespace file_io {
bool readProtoFromTextFile(const std::string& file, google::protobuf::Message* proto) {
  int fd = open(file.c_str(), O_RDONLY);
  if (fd == -1) {
    std::cerr << "ProtoIO: failed to open file(RD): " << file << std::endl;
    return false;
  }
  google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
  bool flag = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return flag;
}

void readExtrinsicFromPbFile(const std::string& pb_file, Eigen::Matrix4d& extrinsic) {
  extrinsic = Eigen::Matrix4d::Identity();
  tutorial::SensorExtrinsic extrinsic_pb;
  if(readProtoFromTextFile(pb_file, &extrinsic_pb)) {
    Eigen::Vector3d trans;
    Eigen::Quaterniond qua;
    tutorial::Translation trans_pb = extrinsic_pb.translation()[0];
    trans = Eigen::Vector3d(trans_pb.x(), trans_pb.y(), trans_pb.z());
    tutorial::Rotation qua_pb = extrinsic_pb.rotation()[0];
    qua = Eigen::Quaterniond(qua_pb.w(), qua_pb.x(), qua_pb.y(), qua_pb.z());                
    extrinsic.block<3, 3>(0, 0) = qua.toRotationMatrix();
    extrinsic.block<3, 1>(0, 3) = trans;
  }
  else
    return;
}

// todo
void readCamInFromXmlFile(const std::string& xml_file, std::string& model_type, 
                          cv::Mat& camera_intrinsic, cv::Mat& camera_distort,
                          int& img_height, int& img_width) {
  rapidxml::file<> fdoc(xml_file.c_str());
	rapidxml::xml_document<> doc;
	doc.parse<0>(fdoc.data());

  double fx, cx, fy, cy;
  double k1, k2, k3, k4; // just fisheye, todo
  for (rapidxml::xml_node<> *param = doc.first_node("param")->first_node(); param; param = param->next_sibling()) {
		if (param->first_node() != NULL) {
			// std::cout << param->name()/* << " : " << param->value()*/ << std::endl;

      if(std::string(param->name()) == "cx") {
        cx = std::stod(param->value());
      }
      else if(std::string(param->name()) == "cy") {
        cy = std::stod(param->value());
      }
      else if(std::string(param->name()) == "fx") {
        fx = std::stod(param->value());
      }
      else if(std::string(param->name()) == "fy") {
        fy = std::stod(param->value());
      }
      else if(std::string(param->name()) == "image_height") {
        img_height = std::stoi(param->value());
      }
      else if(std::string(param->name()) == "image_width") {
        img_width = std::stoi(param->value());
      }
      if(std::string(param->name()) == "k1") {
        k1 = std::stod(param->value());
      }
      else if(std::string(param->name()) == "k2") {
        k2 = std::stod(param->value());
      }
      else if(std::string(param->name()) == "k3") {
        k3 = std::stod(param->value());
      }
      else if(std::string(param->name()) == "k4") {
        k4 = std::stod(param->value());
      }
      else if(std::string(param->name()) == "model_type") {
        model_type = std::string(param->value());
      }
      else if(std::string(param->name()) == "p1") {
        double p1 = std::stod(param->value());
      }
      else if(std::string(param->name()) == "p2") {
        double p2 = std::stod(param->value());
      }
		}
		else
			std::cout << "name: " << param->name() << "has no value" << std::endl;
	}

  camera_intrinsic = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
  camera_distort = (cv::Mat_<double>(4, 1) << k1, k2, k3, k4);
}

void readCamExFromYmlFile(const std::string& yml_file, Eigen::Matrix4d& cam_extrinsic) {
  cv::FileStorage fCamExSet(yml_file, cv::FileStorage::READ);
    if(!fCamExSet.isOpened()) {
      std::cerr << "Failed to open cams extrinsic file at " << yml_file  << std::endl;
      exit(-1);
    }

    cv::Mat fixed_ext;
    fCamExSet["Tdc"] >> fixed_ext;
    cam_extrinsic << static_cast<double>(fixed_ext.at<float>(0, 0)), static_cast<double>(fixed_ext.at<float>(0, 1)), 
                        static_cast<double>(fixed_ext.at<float>(0, 2)), static_cast<double>(fixed_ext.at<float>(0, 3)),
                        static_cast<double>(fixed_ext.at<float>(1, 0)), static_cast<double>(fixed_ext.at<float>(1, 1)),
                        static_cast<double>(fixed_ext.at<float>(1, 2)), static_cast<double>(fixed_ext.at<float>(1, 3)),
                        static_cast<double>(fixed_ext.at<float>(2, 0)), static_cast<double>(fixed_ext.at<float>(2, 1)), 
                        static_cast<double>(fixed_ext.at<float>(2, 2)), static_cast<double>(fixed_ext.at<float>(2, 3)),
                        0.0, 0.0, 0.0, 1.0;
}

void readExtrinsicFromYamlFile(const std::string& file, Eigen::Matrix4d& extrinsic) {
  cv::FileStorage settings(file, cv::FileStorage::READ);
  if (!settings.isOpened()) {
    std::cerr << "# ERROR: Failed to open settings file at: " << file << std::endl;
    return;
  }
  extrinsic = extrinsic.Identity();

  cv::FileNode n = settings["transform"];
  double qx = static_cast<double>(n["qx"]);
  double qy = static_cast<double>(n["qy"]);
  double qz = static_cast<double>(n["qz"]);
  double qw = static_cast<double>(n["qw"]);
  double tx = static_cast<double>(n["tx"]);
  double ty = static_cast<double>(n["ty"]);
  double tz = static_cast<double>(n["tz"]);

  Eigen::Quaterniond q = Eigen::Quaterniond(qw, qx, qy, qz);
  // std::cout << "qua: " << q.coeffs() << std::endl;

  Eigen::Matrix3d R = q.toRotationMatrix();
  Eigen::Vector3d t = Eigen::Vector3d(tx, ty, tz);
  
  extrinsic.block<3, 3>(0, 0) = R;
  extrinsic.block<3, 1>(0, 3) = t;
}

bool writeProtoToTextFile(std::string& file,
                          const google::protobuf::Message& proto) {
  int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);  // 0644 is the file permission
  if (fd == -1) {
    std::cerr << "ProtoIO: failed to open file(WR): " << file << std::endl;
    return false;
  }
  google::protobuf::io::FileOutputStream* output = new google::protobuf::io::FileOutputStream(fd);
  bool flag = google::protobuf::TextFormat::Print(proto, output);

  delete output;
  close(fd);

  return flag;
}

void writeExtrinsicToPbFile(const Eigen::Matrix4d& extrinsic, std::string& output_file) {
  tutorial::SensorExtrinsic extrinsic_pb;
  tutorial::Rotation* rot = extrinsic_pb.add_rotation();
  Eigen::Matrix3d rotation_matrix = extrinsic.topLeftCorner(3, 3);
  Eigen::Quaterniond qua = Eigen::Quaterniond(rotation_matrix);
  // std::cout << qua.coeffs() << std::endl;
  rot->set_x(static_cast<double>(qua.x()));
  rot->set_y(static_cast<double>(qua.y()));
  rot->set_z(static_cast<double>(qua.z()));
  rot->set_w(static_cast<double>(qua.w()));

  tutorial::Translation* trans = extrinsic_pb.add_translation();
  Eigen::Vector3d translation = extrinsic.topRightCorner(3, 1);
  trans->set_x(translation.x());
  trans->set_y(translation.y());
  trans->set_z(translation.z());
  
  tutorial::Rotation rot_test = extrinsic_pb.rotation()[0];
  // std::cout << rot_test.x() << std::endl;
  if(!writeProtoToTextFile(output_file, extrinsic_pb)) {
    return;
  }
}

void loadPcdFilePath(const std::string& pcd_folder, 
                     std::queue<std::string>& pcds_name_vec) {
  std::string pcds_path = pcd_folder + "/*.pcd";  
  glob_t pglob;
  if(glob(pcds_path.c_str(), GLOB_ERR, NULL, &pglob))  {
    std::cerr << "Glob lib can not match any path for pattern : " << pcds_path << std::endl;
    return ;
  }

  std::vector<std::string> paths;
  for (std::size_t iter = 0; iter < pglob.gl_pathc; iter++) {
    // pcds_name_vec.push(pglob.gl_pathv[iter]);
    paths.emplace_back(pglob.gl_pathv[iter]);
  }

  sort(paths.begin(), paths.end(), [&paths](std::string a, std::string b){
    return(std::stod(std::string(a.substr(a.rfind('/') + 1, a.rfind('.') - a.rfind('/') + 1))) 
    <= std::stod(std::string(b.substr(b.rfind('/') + 1, b.rfind('.') - b.rfind('/') + 1))));});
  
  for(auto path: paths) {
    // std::cout << path << std::endl;
    pcds_name_vec.push(path);
  }
}

void readStampPoseFromFile(const std::string& path, StampedPoseVectorPtr& stamp_pose_vec) {
  // std::queue<stamp_pose> stamp_pose_vec;
  std::ifstream file_if(path);
  if(!file_if) {
    std::cerr <<"error path: " << path << std::endl;
  }
  std::string line;
  while (std::getline(file_if, line)) {
    double timestamp = 0.;
    double tx = 0., ty = 0., tz = 0., qx = 0., qy = 0., qz = 0., qw = 1.;
    std::stringstream line_stream(line);
    std::string data[8];
    int i = 0;

    while (std::getline(line_stream, data[i++], ' ')) {
    }

    timestamp = atof(data[0].c_str());
    tx = atof(data[1].c_str());
    ty = atof(data[2].c_str());
    tz = atof(data[3].c_str());
    qx = atof(data[4].c_str());
    qy = atof(data[5].c_str());
    qz = atof(data[6].c_str());
    qw = atof(data[7].c_str());
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = (Eigen::Quaterniond(qw, qx, qy, qz)).toRotationMatrix();
    transform.block<3, 1>(0, 3) = Eigen::Vector3d(tx, ty, tz);
    stamp_pose_vec->emplace_back(StampedPose(timestamp, transform));
  }
}

void writeStampPoseToFile(const StampedPoseVectorPtr& stamp_pose_vec, std::string& path) {
  std::ofstream file_of(path); 
  file_of << std::fixed; 
  if(stamp_pose_vec->size() <= 0) {
    std::cerr << "empty pose file" << std::endl;
    return;
  }

  for(size_t i = 0; i < stamp_pose_vec->size(); ++i) {
    Eigen::Vector3d t(0, 0, 0);
    Eigen::Quaterniond q(1, 0, 0, 0);
    t = stamp_pose_vec->at(i).second.block<3, 1>(0, 3);
    q = Eigen::Quaterniond(stamp_pose_vec->at(i).second.block<3, 3>(0, 0));
    file_of << std::setprecision(18) << stamp_pose_vec->at(i).first << " "
            << t(0) << " " << t(1) << " " << t(2) << " "
            << q.x() << " " << q.y() << " "<< q.z() << " " << q.w() << std::endl;
  }
  std::cout << "write done" << std::endl;
  file_of.close();
}

bool read_timestamp_info(const std::string &input_img_stamp_file, std::vector<double> &img_stamps) {
    std::ifstream infile(input_img_stamp_file, std::ios::binary);
    if (infile.fail()) {
      return false;
    }
        

    unsigned int flag = 0;
    unsigned long long index = 0;
    unsigned long long time = 0;
    img_stamps.clear();
    while (!infile.eof()) {
        infile.read((char *) &flag, sizeof(flag));
        infile.read((char *) &index, sizeof(index));
        infile.read((char *) &time, sizeof(time));
        if (flag)
            img_stamps.push_back(double(time * 1.0e-6));         
    }
    infile.close();

    return true;
}


int getSyncTimestampIdex(const std::vector<double>& timestamp_vec, const double pcd_start_time, 
                        const StampedPoseVectorPtr& stamp_pose_vec) {
  double img_start_time = 0.0;

  double dr_start_still_time = 0.0, dr_end_still_time = 0.0;
  auto it = std::lower_bound(timestamp_vec.begin(), timestamp_vec.end(), pcd_start_time);
  if(it == timestamp_vec.begin()) {
    img_start_time = *it;
  }
  else if(it != timestamp_vec.end()) {
    auto prev = std::prev(it);
    img_start_time = (std::abs(pcd_start_time - *it) > std::abs(pcd_start_time - *prev)) ? *prev : *it;
  }
  else {
    std::cerr << "Cannot find sync image" << std::endl
              << "camera's time in " << timestamp_vec[0] << " and " 
              << timestamp_vec[timestamp_vec.size() - 1] << std::endl;
    exit(-1);
  }  

  if(std::abs(pcd_start_time - img_start_time) > 0.02) {
    std::vector<double> dr_stamp_vec;
    for(int pose_index = 0; pose_index < stamp_pose_vec->size(); ++pose_index) {
      dr_stamp_vec.emplace_back(stamp_pose_vec->at(pose_index).first);
    }

    if(pcd_start_time <= img_start_time) {
      auto dr_it = std::lower_bound(dr_stamp_vec.begin(), dr_stamp_vec.end(), pcd_start_time);
      if(dr_it == dr_stamp_vec.begin()) dr_start_still_time = *dr_it;
      else if(dr_it != dr_stamp_vec.end()) {
        auto dr_prev = std::prev(dr_it);
        dr_start_still_time = *dr_prev;
      }
      else {
        std::cerr << "dr stamp_pose is bad." << std::endl
                  << "pcd_stamp: " << pcd_start_time << std::endl
                  << "dr's time in " << dr_stamp_vec[0] << " and " 
                  << dr_stamp_vec[dr_stamp_vec.size() - 1] << std::endl;
        exit(-1);
      }

      dr_it = std::lower_bound(dr_stamp_vec.begin(), dr_stamp_vec.end(), img_start_time);
      if(dr_it != dr_stamp_vec.end()) dr_end_still_time = *dr_it;
      else {
        std::cerr << "dr stamp_pose is bad." << std::endl
                  << "img_stamp: " << img_start_time << std::endl
                  << "dr's time in " << dr_stamp_vec[0] << " and " 
                  << dr_stamp_vec[dr_stamp_vec.size() - 1] << std::endl;
        exit(-1);
      }
    }
    else {
      auto dr_it = std::lower_bound(dr_stamp_vec.begin(), dr_stamp_vec.end(), img_start_time);
      if(dr_it == dr_stamp_vec.begin()) dr_start_still_time = *dr_it;
      else if(dr_it != dr_stamp_vec.end()) {
        auto dr_prev = std::prev(dr_it);
        dr_start_still_time = *dr_prev;
      }
      else {
        std::cerr << "dr stamp_pose is bad." << std::endl
                  << "img_stamp: " << img_start_time << std::endl
                  << "dr's time in " << dr_stamp_vec[0] << " and " 
                  << dr_stamp_vec[dr_stamp_vec.size() - 1] << std::endl;
        exit(-1);
      }

      dr_it = std::lower_bound(dr_stamp_vec.begin(), dr_stamp_vec.end(), pcd_start_time);
      if(dr_it != dr_stamp_vec.end()) dr_end_still_time = *dr_it;
      else {
        std::cerr << "dr stamp_pose is bad." << std::endl;
        exit(-1);
      }
    }
    // std::cout << "dr_start_still_time: " << dr_start_still_time << std::endl
              // << "dr_end_still_time: " << dr_end_still_time << std::endl;

    int dr_cnt = 0;
    Eigen::Quaterniond last_q, curr_q;
    Eigen::Vector3d last_t, curr_t;
    while(dr_cnt < stamp_pose_vec->size()) {
      if(stamp_pose_vec->at(dr_cnt).first < dr_start_still_time) {
        ++dr_cnt;
        continue;
      }
      else if(stamp_pose_vec->at(dr_cnt).first == dr_start_still_time) {
        curr_q = Eigen::Quaterniond(stamp_pose_vec->at(dr_cnt).second.block<3, 3>(0, 0));
        curr_t = stamp_pose_vec->at(dr_cnt).second.block<3, 1>(0, 3);
        last_q = curr_q;
        last_t = curr_t;

      }

      curr_q = Eigen::Quaterniond(stamp_pose_vec->at(dr_cnt).second.block<3, 3>(0, 0));
      curr_t = stamp_pose_vec->at(dr_cnt).second.block<3, 1>(0, 3);
      if((curr_q.w() == last_q.w()) && (curr_q.x() == last_q.x()) && (curr_q.y() == last_q.y()) && (curr_q.z() == last_q.z())
          && (curr_t.x() == last_t.x()) && (curr_t.y() == last_t.y()) && (curr_t.z() == last_t.z())) {
        ++dr_cnt;
        last_q = curr_q;
        last_t = curr_t;
      }
      else {
        std::cerr << "The timestamps of the lidar and camera are not synchronized" << std::endl
                  << "vehicle starts moving, " << stamp_pose_vec->at(dr_cnt).first << std::endl;
        exit(-1);
      }

      if(stamp_pose_vec->at(dr_cnt).first == dr_end_still_time) 
        break;
    }
  }  

  for(size_t i = 0; i < timestamp_vec.size(); ++i) {
    if(img_start_time == timestamp_vec[i]) {
      return i;
    }
  }                   

}

} // namespace file_io


