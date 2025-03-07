#pragma once
#include <foundation/types.h>

using BooleanVec = std::vector<bool>; 

template <typename T>
class Refraction {
 public:

  static inline double const EPS_IOR = 0.001;
  static inline T const EPS_SZ = T(0.01);
  static inline T const EPS_ES = T(0.001);
  static inline T const EPS_IMG = T(0.001);

  Refraction(T ior = T(1.3));
  
  Vec3<T> Refract(const Vec3<T> &I, const Vec3<T> &N) const;
  
  bool CalcR(const Vec3<T> &E, const Vec3<T> &S, Vec3<T> &R) const;

  BooleanVec CalcRMany(const Vec3<T> &E, const MatX3<T> &S, MatX3<T> &R) const;
   
  bool CalcRefractionPlane(const Vec3<T> &E, const Vec3<T> &S, Eigen::Transform<T,3,Eigen::Affine> &Rt) const;

  bool SolveRy(T e, T sx, T sy, T &ry) const;
  bool SolveRyAnalytic(T e, T sx, T sy, T &ry) const;
  
  T ior;  
};

#include "./src/refraction.cc"

