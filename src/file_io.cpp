#include <fcntl.h>   // linux open() function
#include <unistd.h>  // linux close() function
#include <google/protobuf/io/coded_stream.h>           // CodedInputStream
#include <google/protobuf/io/zero_copy_stream.h>       // ZeroCopyInputStream,
#include <google/protobuf/io/zero_copy_stream_impl.h>  // FileInputStream, FileOutputStream
#include <google/protobuf/text_format.h>   
#include <google/protobuf/message.h>

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

void writeSensorExToPbFile(const Eigen::Matrix4d& extrinsic, std::string& output_file) {
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

} // namespace file_io


