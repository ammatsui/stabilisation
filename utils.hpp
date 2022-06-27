#ifndef UTILS_H   
#define UTILS_H


/*
utilities for transformation and trajectory calculations
*/

/* params for kalman filter */
// #define R1 0.5
// #define Q1 0.004


#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

const double R1 = 0.05; 
const double Q1 = 0.0005;
 
double diff_x = 0;
double diff_y = 0;
double diff_thetha = 0;
double diff_xtrans = 0;
double diff_ytrans = 0;
    
double err_x = 1;
double err_y = 1;
double err_thetha = 1;
double err_xtrans = 1;
double err_ytrans = 1;

double Q_x = Q1;
double Q_y = Q1;
double Q_thetha = Q1;
double Q_xtrans = Q1;
double Q_ytrans = Q1;

double R_x = R1;
double R_y = R1;
double R_thetha = R1;
double R_xtrans = R1;
double R_ytrans = R1;

double sum_x = 1;
double sum_y = 1;
double sum_thetha = 1;
double sum_xtrans = 1;
double sum_ytrans = 1;






struct TransformParam
{
  TransformParam(); 
  TransformParam(double _dx, double _dy, double _da);

  double dx;
  double dy;
  double da; 

  void getTransform(cv::Mat &T);

};

struct Trajectory
{
    Trajectory();
    Trajectory(double _x, double _y, double _a);
 
    double x;
    double y;
    double a; 
};

std::vector<Trajectory> cumsum(std::vector<TransformParam> &transforms);

std::vector <Trajectory> smooth(std::vector <Trajectory>& trajectory, int radius);

void smoothTransforms(std::vector<TransformParam>& transforms, std::vector<TransformParam>& transforms_smooth, 
                      std::vector<Trajectory>& trajectory, std::vector<Trajectory>& smoothed_trajectory);


void kalmanFilter(double& x , double& y , double& thetha , double& xtrans , double& ytrans);

cv::Mat kalmanPredict(cv::Mat& T, double& x , double& y , double& thetha , double& xtrans , double& ytrans);





TransformParam::TransformParam() 
{}

TransformParam::TransformParam(double _dx, double _dy, double _da) 
{
    dx = _dx;
    dy = _dy;
    da = _da;
}

void TransformParam::getTransform(cv::Mat &T)
{
    /* reconstruct thnew transformation matrix */
    T.at<double>(0,0) = cos(da);
    T.at<double>(0,1) = -sin(da);
    T.at<double>(1,0) = sin(da);
    T.at<double>(1,1) = cos(da);

    T.at<double>(0,2) = dx;
    T.at<double>(1,2) = dy;
}

Trajectory::Trajectory() 
{}

Trajectory::Trajectory(double _x, double _y, double _a) 
{
    x = _x;
    y = _y;
    a = _a;
}

std::vector<Trajectory> cumsum(std::vector<TransformParam> &transforms)
{
  std::vector <Trajectory> trajectory; 
  
  double a = 0;
  double x = 0;
  double y = 0;

  for(size_t i=0; i < transforms.size(); i++) 
  {
      x += transforms[i].dx;
      y += transforms[i].dy;
      a += transforms[i].da;

      trajectory.push_back(Trajectory(x,y,a));

  }

  return trajectory; 
}

std::vector <Trajectory> smooth(std::vector <Trajectory>& trajectory, int radius)
{
  std::vector <Trajectory> smoothed_trajectory; 
  for(size_t i=0; i < trajectory.size(); i++) {
      double sum_x = 0;
      double sum_y = 0;
      double sum_a = 0;
      int count = 0;

      for(int j=-radius; j <= radius; j++) {
          if(i+j >= 0 && i+j < trajectory.size()) {
              sum_x += trajectory[i+j].x;
              sum_y += trajectory[i+j].y;
              sum_a += trajectory[i+j].a;

              count++;
          }
      }

      double avg_a = sum_a / count;
      double avg_x = sum_x / count;
      double avg_y = sum_y / count;

      smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));
  }

  return smoothed_trajectory; 
}


void smoothTransforms(std::vector<TransformParam>& transforms, std::vector<TransformParam>& transforms_smooth, 
                      std::vector<Trajectory>& trajectory, std::vector<Trajectory>& smoothed_trajectory)
/* smoothes transforms to match new trajectory */
{
 std::vector<TransformParam> temp;
 for(size_t i=0; i < transforms.size(); i++)
  {
    /* difference between smoothed trjaectory and the original one */
    double diff_x = smoothed_trajectory[i].x - trajectory[i].x;
    double diff_y = smoothed_trajectory[i].y - trajectory[i].y;
    double diff_a = smoothed_trajectory[i].a - trajectory[i].a;

    /* the new transformation array */
    double dx = transforms[i].dx + diff_x;
    double dy = transforms[i].dy + diff_y;
    double da = transforms[i].da + diff_a;

    //transforms_smooth.push_back(TransformParam(dx, dy, da));
    temp.push_back(TransformParam(dx, dy, da));
  }
transforms_smooth.clear();
transforms_smooth = temp;
}


std::vector<double> smoothKalman(std::vector<double>& traj)
{
  std::vector <double> smoothed_trajectory; 
  double err_val = 0;
  smoothed_trajectory.push_back(traj[0]);
  for (int i = 1; i < traj.size(); i++)
  {
    double prev_val     = traj[i-1];
    double prev_err_val = err_val + Q1;

    double K = prev_err_val / (prev_err_val + R1);

    err_val = err_val * (1 - K);
    double val = prev_val + K *(traj[i] - prev_val);
    
    smoothed_trajectory.push_back(val);
  }

  return smoothed_trajectory;
}


double upd(double prev, double curr)
{ 
  double err_val = 0;
  
  double prev_val     = prev;
  double prev_err_val = err_val + Q1;

  double K = prev_err_val / (prev_err_val + R1);

  err_val = err_val * (1 - K);
  double val = prev_val + K *(curr - prev_val);
    
  return val;

}


std::vector<Trajectory> smoothKalman (std::vector<Trajectory>& traj)
{
  std::vector <Trajectory> smoothed_trajectory;
  double err_val = 0;
  smoothed_trajectory.push_back(traj[0]);
  for (int i = 1; i < traj.size(); i++)
  {
    double dx = upd(traj[i-1].x, traj[i].x);
    double dy = upd(traj[i-1].x, traj[i].x);
    double da = upd(traj[i-1].x, traj[i].x);
 
    smoothed_trajectory.push_back(Trajectory(dx, dy, da));
    
  }

  return smoothed_trajectory;
}


void kalmanFilter(double& x , double& y , double& thetha , double& xtrans , double& ytrans)
{
    double prev_x      = x;
    double prev_y      = y;
    double prev_thetha = thetha;
    double prev_xtrans = xtrans;
    double prev_ytrans = ytrans;

    double prev_err_x      = err_x      + Q_x;
    double prev_err_y      = err_y      + Q_y;
    double prev_err_thetha = err_thetha + Q_thetha;
    double prev_err_xtrans = err_xtrans + Q_xtrans;
    double prev_err_ytrans = err_ytrans + Q_ytrans;

    double K_x      = prev_err_x      / (prev_err_x      + R_x);
    double K_y      = prev_err_y      / (prev_err_y      + R_y);
    double K_thetha = prev_err_thetha / (prev_err_thetha + R_thetha);
    double K_xtrans = prev_err_xtrans / (prev_err_xtrans + R_xtrans);
    double K_ytrans = prev_err_ytrans / (prev_err_ytrans + R_ytrans);

    x      = prev_x      + K_x      * (sum_x      - prev_x);
    y      = prev_y      + K_y      * (sum_y      - prev_y);
    thetha = prev_thetha + K_thetha * (sum_thetha - prev_thetha);
    xtrans = prev_xtrans + K_xtrans * (sum_xtrans - prev_xtrans);
    ytrans = prev_ytrans + K_ytrans * (sum_ytrans - prev_ytrans);

    err_x      = ( 1 - K_x )      * prev_err_x;
    err_y      = ( 1 - K_y )      * prev_err_y;
    err_thetha = ( 1 - K_thetha ) * prev_err_thetha;
    err_xtrans = ( 1 - K_xtrans ) * prev_err_xtrans;
    err_ytrans = ( 1 - K_ytrans ) * prev_err_ytrans;
}


cv::Mat kalmanPredict(cv::Mat& T, double& x , double& y , double& thetha , double& xtrans , double& ytrans)
{
  /* translation */
  double dx = T.at<double>(0,2);
  double dy = T.at<double>(1,2);
    
  /* rotation angle */
  double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

  double ds_x = T.at<double>(0,0)/cos(da);
  double ds_y = T.at<double>(1,1)/cos(da);

  double sx = ds_x;
  double sy = ds_y;

  sum_x      += ds_x;
  sum_y      += ds_y;
  sum_thetha += da;
  sum_xtrans += dx;
  sum_ytrans += dy;
 
  diff_x      = x - sum_x;
  diff_y      = y - sum_y;
  diff_thetha = thetha - sum_thetha;
  diff_xtrans = xtrans - sum_xtrans;
  diff_ytrans = ytrans - sum_ytrans;
  
  ds_x = ds_x + diff_x;
  ds_y = ds_y + diff_y;
  da   = da   + diff_thetha;
  dx   = dx   + diff_xtrans;
  dy   = dy   + diff_ytrans;
  
  cv::Mat S(2,3,CV_64F);
  /* smooth matrix */
  S.at<double>(0,0) = sx * cos(da);
  S.at<double>(0,1) = sx * -sin(da);
  S.at<double>(1,0) = sy * sin(da);
  S.at<double>(1,1) = sy * cos(da);

  S.at<double>(0,2) = dx;
  S.at<double>(1,2) = dy;

  return S;
}


#endif