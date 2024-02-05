#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <pcl/common/io.h>

#define SMALL_EPS 1e-10
#define HASH_P 116101
#define MAX_N 10000000019

#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

#define PLV(a) std::vector<Eigen::Matrix<double, a, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>

#define G_m_s2 9.81
#define DIMU 18
#define DIM 15
#define DNOI 12
#define NMATCH 5
#define DVEL 6



// int layer_limit = 5; // origin: 3
int MIN_PT = 30;
// double what = 0.95; // origin: 0.98

class VOXEL_LOC
{
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0): x(vx), y(vy), z(vz){}

  bool operator == (const VOXEL_LOC &other) const
  {
    return (x == other.x && y == other.y && z == other.z);
  }
};
namespace std
{
  template<>
  struct hash<VOXEL_LOC>
  {
    size_t operator() (const VOXEL_LOC &s) const
    {
      using std::size_t; using std::hash;
      // return (((hash<int64_t>()(s.z)*HASH_P)%MAX_N + hash<int64_t>()(s.y))*HASH_P)%MAX_N + hash<int64_t>()(s.x);
      long long index_x, index_y, index_z;
			double cub_len = 0.125;
			index_x = int(round(floor((s.x)/cub_len + SMALL_EPS)));
			index_y = int(round(floor((s.y)/cub_len + SMALL_EPS)));
			index_z = int(round(floor((s.z)/cub_len + SMALL_EPS)));
			return (((((index_z * HASH_P) % MAX_N + index_y) * HASH_P) % MAX_N) + index_x) % MAX_N;
    }
  };
}

Eigen::Matrix3d Exp(const Eigen::Vector3d &ang)
{
  double ang_norm = ang.norm();
  Eigen::Matrix3d Eye3 = Eigen::Matrix3d::Identity();
  if (ang_norm > 0.0000001)
  {
    Eigen::Vector3d r_axis = ang / ang_norm;
    Eigen::Matrix3d K;
    K << SKEW_SYM_MATRX(r_axis);
    /// Roderigous Tranformation
    return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  }
  else
  {
    return Eye3;
  }
}

Eigen::Vector3d Log(const Eigen::Matrix3d &R)
{
  double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Vector3d K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

struct IMUST
{
  double t;
  Eigen::Matrix3d R;
  Eigen::Vector3d p;
  Eigen::Vector3d v;
  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  Eigen::Vector3d g;
  
  IMUST()
  {
    setZero();
  }

  IMUST(double _t, const Eigen::Matrix3d& _R, const Eigen::Vector3d& _p, const Eigen::Vector3d& _v,
        const Eigen::Vector3d& _bg, const Eigen::Vector3d& _ba,
        const Eigen::Vector3d& _g = Eigen::Vector3d(0, 0, -G_m_s2)):
        t(_t), R(_R), p(_p), v(_v), bg(_bg), ba(_ba), g(_g){}

  IMUST &operator+=(const Eigen::Matrix<double, DIMU, 1> &ist)
  {
    this->R = this->R * Exp(ist.block<3, 1>(0, 0));
    this->p += ist.block<3, 1>(3, 0);
    this->v += ist.block<3, 1>(6, 0);
    this->bg += ist.block<3, 1>(9, 0);
    this->ba += ist.block<3, 1>(12, 0);
    this->g += ist.block<3, 1>(15, 0);
    return *this;
  }

  Eigen::Matrix<double, DIMU, 1> operator-(const IMUST &b) 
  {
    Eigen::Matrix<double, DIMU, 1> a;
    a.block<3, 1>(0, 0) = Log(b.R.transpose() * this->R);
    a.block<3, 1>(3, 0) = this->p - b.p;
    a.block<3, 1>(6, 0) = this->v - b.v;
    a.block<3, 1>(9, 0) = this->bg - b.bg;
    a.block<3, 1>(12, 0) = this->ba - b.ba;
    a.block<3, 1>(15, 0) = this->g - b.g;
    return a;
  }

  IMUST &operator=(const IMUST &b)
  {
    this->R = b.R;
    this->p = b.p;
    this->v = b.v;
    this->bg = b.bg;
    this->ba = b.ba;
    this->g = b.g;
    this->t = b.t;
    return *this;
  }

  void setZero()
  {
    t = 0; R.setIdentity();
    p.setZero(); v.setZero();
    bg.setZero(); ba.setZero();
    g << 0, 0, -G_m_s2;
  }
};

class VOX_FACTOR
{
public:
  Eigen::Matrix3d P;
  Eigen::Vector3d v;
  int N;

  #ifdef POINT_NOISE
  Eigen::Matrix<double, 6, 6> P_cov; Eigen::Matrix3d v_cov;
  #endif

  VOX_FACTOR()
  {
    P.setZero();
    v.setZero();
    N = 0;

    #ifdef POINT_NOISE
    P_cov.setZero(); v_cov.setZero();
    #endif
  }

  void clear()
  {
    P.setZero();
    v.setZero();
    N = 0;

    #ifdef POINT_NOISE
    P_cov.setZero(); v_cov.setZero();
    #endif
  }

  void push(const Eigen::Vector3d &vec)
  {
    N++;
    P += vec * vec.transpose();
    v += vec;

    #ifdef POINT_NOISE
    static double ang_error = 0.05*0.05 / 57.3 / 57.3;
    static double dis_error = 0.02 * 0.02;

    Eigen::Matrix<double, 6, 3> Bi;
    Bi << 2*vec(0), 0, 0,
          vec(1), vec(0), 0,
          vec(2), 0, vec(0),
          0, 2*vec(0), 0,
          0, vec(2), vec(1),
          0, 0, 2*vec(2);
    
    double d = vec.norm();
    Eigen::Vector3d w = vec / d;
    double co_ang = d * d * ang_error;
    Eigen::Matrix3d p_cov = (dis_error-co_ang)*w*w.transpose() + co_ang*I33;

    P_cov += Bi * p_cov * Bi.transpose();
    v_cov += p_cov;
    #endif
  }

  Eigen::Matrix3d cov()
  {
    Eigen::Vector3d center = v / N;
    return P/N - center*center.transpose();
  }

  VOX_FACTOR & operator+=(const VOX_FACTOR& sigv)
  {
    this->P += sigv.P;
    this->v += sigv.v;
    this->N += sigv.N;

    #ifdef POINT_NOISE
    this->P_cov += sigv.P_cov;
    this->v_cov += sigv.v_cov;
    #endif

    return *this;
  }

  void transform(const VOX_FACTOR &sigv, const IMUST &stat)
  {
    N = sigv.N;
    v = stat.R*sigv.v + N*stat.p;
    Eigen::Matrix3d rp = stat.R * sigv.v * stat.p.transpose();
    P = stat.R*sigv.P*stat.R.transpose() + rp + rp.transpose() + N*stat.p*stat.p.transpose();
  }
};

enum OT_STATE {UNKNOWN, MID_NODE, PLANE};
typedef struct Plane
{
  pcl::PointXYZINormal p_center;
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Matrix3d covariance;
  std::vector<Eigen::Vector3d> plane_points;
  double radius = 0;
  double min_eigen_value = 1;
  double d = 0;
  int points_size = 0;
  bool is_plane = false;
  bool is_init = false;
  int id;
  bool is_update = false;
} Plane;
class OCTO_TREE_NODE
{
public:
  OT_STATE octo_state;
  PLV(3) vec_orig;
  VOX_FACTOR sig_orig;

  OCTO_TREE_NODE* leaves[8];
  Plane* plane_ptr;
  double voxel_center[3];
  double quater_length;
  double eigen_thr;
  int layer;
  int layer_limit;
  double what;

  Eigen::Vector3d center, direct, value_vector;
  double eigen_ratio;

  OCTO_TREE_NODE(double _eigen_thr = 1.0/10, int _layer_limit = 3, double _what = 0.98): 
    eigen_thr(_eigen_thr), layer_limit(_layer_limit), what(_what)
  {
    octo_state = UNKNOWN;
    layer = 0;
    for(int i = 0; i < 8; i++)
      leaves[i] = nullptr;
  }

  virtual ~OCTO_TREE_NODE()
  {
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        delete leaves[i];
  }

  bool judge_eigen(int layer)
  {
    VOX_FACTOR covMat = sig_orig;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    center = covMat.v / covMat.N;
    direct = saes.eigenvectors().col(0);

    eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // [0] is the smallest

    if(eigen_ratio > eigen_thr) return 0;
    if(saes.eigenvalues()[0] / saes.eigenvalues()[1] > 0.1) return 0; // 排除线状点云
    
    double eva0 = saes.eigenvalues()[0];
    double sqr_eva0 = sqrt(eva0);
    Eigen::Vector3d center_turb = center + 5 * sqr_eva0 * direct;
    std::vector<VOX_FACTOR> covMats(8);

    for(Eigen::Vector3d ap: vec_orig)
    {
      int xyz[3] = {0, 0, 0};
      for(int k = 0; k < 3; k++)
        if(ap(k) > center_turb[k])
          xyz[k] = 1;

      Eigen::Vector3d pvec(ap(0), ap(1), ap(2));
      
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      covMats[leafnum].push(pvec);
    }
    
    int num_all = 0, num_qua = 0;
    for(int i = 0; i < 8; i++)
      if(covMats[i].N > MIN_PT)
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov());
        Eigen::Vector3d child_direct = saes.eigenvectors().col(0);

        if(fabs(child_direct.dot(direct)) > what)
          num_qua++;
        num_all++;
      }
    
    if(num_qua != num_all) return 0;
    return 1;
  }

  void cut_func()
  {
    PLV(3)& pvec_orig = vec_orig;
    uint a_size = pvec_orig.size();

    for(uint j = 0; j < a_size; j++)
    {
      int xyz[3] = {0, 0, 0};
      for(uint k = 0; k < 3; k++)
        if(pvec_orig[j][k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4*xyz[0] + 2*xyz[1] + xyz[2];
      if(leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OCTO_TREE_NODE(eigen_thr, layer_limit, what);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2*xyz[0]-1)*quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2*xyz[1]-1)*quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2*xyz[2]-1)*quater_length;
        leaves[leafnum]->quater_length = quater_length / 2.0;
        leaves[leafnum]->layer = layer + 1;
      }
      /*原始点云信息*/
      leaves[leafnum]->vec_orig.push_back(pvec_orig[j]);
      leaves[leafnum]->sig_orig.push(pvec_orig[j]);
    }
    PLV(3)().swap(pvec_orig);
  }

  void recut()
  {
    if(octo_state == UNKNOWN)
    {
      int point_size = sig_orig.N;
      
      if(point_size < MIN_PT)
      {
        octo_state = MID_NODE;
        PLV(3)().swap(vec_orig);
        return;
      }

      if(judge_eigen(layer))
      {
        octo_state = PLANE;

        plane_ptr = new Plane;
        VOX_FACTOR covMat = sig_orig;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
        value_vector = saes.eigenvalues();
        center = covMat.v / covMat.N;
        direct = saes.eigenvectors().col(0);

        plane_ptr->covariance = covMat.cov();
        plane_ptr->center = center;
        plane_ptr->normal = direct;
        plane_ptr->radius = sqrt(value_vector[2]);
        plane_ptr->min_eigen_value = value_vector[0];
        plane_ptr->d = -direct.dot(center);
        plane_ptr->p_center.x = center(0);
        plane_ptr->p_center.y = center(1);
        plane_ptr->p_center.z = center(2);
        plane_ptr->p_center.normal_x = direct(0);
        plane_ptr->p_center.normal_y = direct(1);
        plane_ptr->p_center.normal_z = direct(2);
        plane_ptr->points_size = point_size;
        plane_ptr->is_plane = true;
        for(auto pt: vec_orig)
          plane_ptr->plane_points.push_back(pt);

        return;
      }
      else
      {
        if(layer == layer_limit)
        {
          octo_state = MID_NODE;
          PLV(3)().swap(vec_orig);
          return;
        }
        cut_func();
      }
    }
    
    for(int i = 0; i < 8; i++)
      if(leaves[i] != nullptr)
        leaves[i]->recut();
  }

  void get_plane_list(std::vector<Plane*>& plane_list)
  {
    if(octo_state == PLANE)
      plane_list.push_back(plane_ptr);
    else
      if(layer <= layer_limit)
        for(int i = 0; i < 8; i++)
          if(leaves[i] != nullptr)
            leaves[i]->get_plane_list(plane_list);
  }

};

class OCTO_TREE_ROOT: public OCTO_TREE_NODE
{
public:
  OCTO_TREE_ROOT(double _eigen_thr, int _layer_limit, double _what): 
    OCTO_TREE_NODE(_eigen_thr, _layer_limit, _what){}
  PLV(3) all_points;
};

typedef struct Voxel {
  float size;
  Eigen::Vector3d voxel_origin;
  Eigen::Vector3d voxel_color;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
  Voxel(float _size) : size(_size) {
    voxel_origin << 0, 0, 0;
    cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
  };
} Voxel;

struct M_POINT {
  float xyz[3];
  float intensity;
  int count = 0;
};

// Similar with PCL voxelgrid filter
void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> feat_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      feat_map[position] = anp;
    }
  }
  plsize = feat_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}

typedef struct SinglePlane
{
  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZ p_center;
  Eigen::Vector3d normal;
  int index;
} SinglePlane;

template <class T> void calc(T matrix[4][5], Eigen::Vector3d &solution) {
  T base_D = matrix[1][1] * matrix[2][2] * matrix[3][3] +
             matrix[2][1] * matrix[3][2] * matrix[1][3] +
             matrix[3][1] * matrix[1][2] * matrix[2][3]; //计算行列式
  base_D = base_D - (matrix[1][3] * matrix[2][2] * matrix[3][1] +
                     matrix[1][1] * matrix[2][3] * matrix[3][2] +
                     matrix[1][2] * matrix[2][1] * matrix[3][3]);

  if (base_D != 0) {
    T x_D = matrix[1][4] * matrix[2][2] * matrix[3][3] +
            matrix[2][4] * matrix[3][2] * matrix[1][3] +
            matrix[3][4] * matrix[1][2] * matrix[2][3];
    x_D = x_D - (matrix[1][3] * matrix[2][2] * matrix[3][4] +
                 matrix[1][4] * matrix[2][3] * matrix[3][2] +
                 matrix[1][2] * matrix[2][4] * matrix[3][3]);
    T y_D = matrix[1][1] * matrix[2][4] * matrix[3][3] +
            matrix[2][1] * matrix[3][4] * matrix[1][3] +
            matrix[3][1] * matrix[1][4] * matrix[2][3];
    y_D = y_D - (matrix[1][3] * matrix[2][4] * matrix[3][1] +
                 matrix[1][1] * matrix[2][3] * matrix[3][4] +
                 matrix[1][4] * matrix[2][1] * matrix[3][3]);
    T z_D = matrix[1][1] * matrix[2][2] * matrix[3][4] +
            matrix[2][1] * matrix[3][2] * matrix[1][4] +
            matrix[3][1] * matrix[1][2] * matrix[2][4];
    z_D = z_D - (matrix[1][4] * matrix[2][2] * matrix[3][1] +
                 matrix[1][1] * matrix[2][4] * matrix[3][2] +
                 matrix[1][2] * matrix[2][1] * matrix[3][4]);

    T x = x_D / base_D;
    T y = y_D / base_D;
    T z = z_D / base_D;
    // cout << "[ x:" << x << "; y:" << y << "; z:" << z << " ]" << endl;
    solution[0] = x;
    solution[1] = y;
    solution[2] = z;
  } else {
    std::cout << "【无解】";
    solution[0] = 0;
    solution[1] = 0;
    solution[2] = 0;
    //        return DBL_MIN;
  }
}