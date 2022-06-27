#include "utils.hpp"  // tranformations and trajctory operations, kalman filter
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <chrono> // for time measurment


/*
This video stablisation smooths the global trajectory using Kalman Filter (in real time)
1. Get previous to current frame transformation (dx, dy, da) for all frames
2. Accumulate the transformations to get the image trajectory
3. Smooth out the trajectory using Kalman filtering
4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
5. Apply the new transformation to the video
*/


using namespace std;
using namespace cv;


/* how often update features */
const int freq = 15;   // in frames //15
/* goodFeaturesToTrack params */
const int maxCorners = 200;
const double qualityLevel = 0.01;
const double minDistance = 20;
/* cropping out the black borders */
const int threshVal = 20;
const float borderThresh = 0.05f;
// /* canny edge detection */
 const int canny_low = 35;
 const int canny_high = 90;
//     //40
//     //120

/* for trajectory analysis */
ofstream forig_trajectory;
ofstream fsmooth_trajectory;

void save_traj(vector<Trajectory>& traj, ofstream& file)
{
  for (int i = 0; i < traj.size(); i++)
  {
    file << (i+1) << " " << traj[i].x << " " << traj[i].y << " " << traj[i].a << endl;
  }
}


void save_transform(Mat& T, vector<TransformParam>& transforms)
{
  /* translation */
  double dx = T.at<double>(0,2);
  double dy = T.at<double>(1,2);
    
  /* rotation angle */
  double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

  /* save transformation */ 
  transforms.push_back(TransformParam(dx, dy, da));
}


/* styling */
void cropBorder(Mat& frame_stabilised)
{
  Mat cropped, grey;
  cvtColor(frame_stabilised, grey, COLOR_BGR2GRAY);
  threshold(grey, cropped, threshVal, 255, THRESH_BINARY);
  morphologyEx(cropped, cropped, MORPH_CLOSE, 
              getStructuringElement(MORPH_RECT, Size(3, 3)),
              Point(-1, -1), 2, BORDER_CONSTANT, Scalar(0));
    

  /* find corners of the image */
  Point tl, br;
  for (int row = 0; row < cropped.rows; row++)
  {
    if (countNonZero(cropped.row(row)) > borderThresh * cropped.cols)
    {
      tl.y = row;
      break;
    }
  }
  for (int col = 0; col < cropped.cols; col++)
  {
    if (countNonZero(cropped.col(col)) > borderThresh * cropped.rows)
    {
      tl.x = col;
      break;
    }
  }
  for (int row = cropped.rows - 1; row >= 0; row--)
  {
    if (countNonZero(cropped.row(row)) > borderThresh * cropped.cols)
    {
      br.y = row;
      break;
    }
  }
  for (int col = cropped.cols - 1; col >= 0; col--)
  {
    if (countNonZero(cropped.col(col)) > borderThresh * cropped.rows)
    {
      br.x = col;
      break;
    }
  }
    
  Rect roi(tl, br);
  Mat temp;
  frame_stabilised(roi).copyTo(temp);
  frame_stabilised.setTo(Scalar(0, 0, 0));
  if (temp.rows > 0)
  {
    //resize(temp, temp, Size(temp.cols*0.95, temp.rows*0.95), 0.8, 0.8, INTER_AREA);
    temp.copyTo(frame_stabilised(Rect(0, 0, temp.cols, temp.rows)));
  }
}

/* removes black borders, makes them less wiggly */
void fixBorder(Mat& frame_stabilised)
{
  Mat T = getRotationMatrix2D(Point2f(frame_stabilised.cols/2, frame_stabilised.rows/2), 0, 1.04); 
  warpAffine(frame_stabilised, frame_stabilised, T, frame_stabilised.size()); 
  cropBorder(frame_stabilised);
}


void filterPoints(vector<Point2f>& curr_pts, vector<Point2f>& prev_pts, vector<uchar>& status)
{
  auto prev_it = prev_pts.begin(); 
  auto curr_it = curr_pts.begin(); 
  for(size_t k = 0; k < status.size(); k++) 
  {
    if(status[k]) 
    {
      prev_it++; 
      curr_it++; 
    }
    else 
    {
      prev_it = prev_pts.erase(prev_it);
      curr_it = curr_pts.erase(curr_it);
    }
  }
}


// void prepareFrame(Mat& grey)
// {
//    GaussianBlur(grey, grey, Size(21, 21), 0);
//    equalizeHist(grey, grey);
//    Canny(grey, grey, canny_low, canny_high);
//   // GaussianBlur(grey, grey, Size(21, 21), 0);
//   //cout << "fine\n";
// //  Sobel(grey, grey, -1, 0, 1);
  
// }

void stabilise(VideoCapture& cap, VideoWriter& out)
{
  Mat curr, curr_grey; 
  Mat prev, prev_grey;
  Mat frame_stabilised, frame_out;

  vector<Point2f> prev_pts;

  vector<TransformParam> transforms, smooth_transforms;

/* for kalman filter */
  double x      = 0 ;
  double y      = 0 ;
  double thetha = 0;
  double xtrans = 0;
  double ytrans = 0 ;
  

  cap.set(cv::CAP_PROP_POS_FRAMES, 0);

  /* time measuring */
  int fps_proc = 0;
  auto start = chrono::steady_clock::now();

  /* first frame */
  cap >> prev;
  /* first feature points */
  cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
 // prepareFrame(prev_grey);
  goodFeaturesToTrack(prev_grey, prev_pts, maxCorners, qualityLevel, minDistance);
  
  /* store the last successfull transformation */
  Mat last_T;

  int fcount = 0;
  int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT)); 
  int k = 0;
  
  for(int i = 1; i < n_frames-1; i++)
  {
    vector<Point2f> curr_pts;

    if (fcount > freq) 
    {
      goodFeaturesToTrack(prev_grey, prev_pts, maxCorners, qualityLevel, minDistance);
      fcount = 0;
    }
     
    /* next frame */
    bool success = cap.read(curr);
    if(!success) break; 
    
    cvtColor(curr, curr_grey, COLOR_BGR2GRAY);
   // prepareFrame(prev_grey);
  
   
    /* track the feature points */
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(prev_grey, curr_grey, prev_pts, curr_pts, status, err);
    /* leave only valid points */
    filterPoints(curr_pts, prev_pts, status);

    /* estimate the transformation matrix */
    Mat T = estimateRigidTransform(prev_pts, curr_pts, false); 

    /* in case there is no matrix found, use the last one available */
    if(T.data == NULL) last_T.copyTo(T);
    T.copyTo(last_T);

    /* translation */
    double dx = T.at<double>(0,2);
    double dy = T.at<double>(1,2);
    
    /* rotation angle */
    double da = atan2(T.at<double>(1,0), T.at<double>(0,0));

    transforms.push_back(TransformParam(dx, dy, da));

    Mat S;

    /* Kalman filter */
    /* pointless to use on the initial frame */
    if (k == 0)
    {
        k++;
        S = T;
    }
    else
    {
      kalmanFilter(x, y, thetha, xtrans, ytrans);
      S = kalmanPredict(T, x, y, thetha, xtrans, ytrans);
    }
    
   // Mat S = kalmanPredict(T, x, y, thetha, xtrans, ytrans);
    
    /* translation */
    double sx = S.at<double>(0,2);
    double sy = S.at<double>(1,2);
    
    /* rotation angle */
    double sa = atan2(S.at<double>(1,0), S.at<double>(0,0));

    smooth_transforms.push_back(TransformParam(sx, sy, sa));

    warpAffine(prev, frame_stabilised, S, curr.size());

    /* remove black borders */
    fixBorder(frame_stabilised); 
    //hconcat(prev, frame_stabilised, frame_out); 
    frame_stabilised.copyTo(frame_out);

    /* write the output */
    imshow("Output", frame_out);
    out.write(frame_out);
    waitKey(10);
  

    /* prepare for the next frame */
    curr_grey.copyTo(prev_grey);
    curr.copyTo(prev);
    prev_pts = curr_pts;

    /* check time */
    fps_proc++;
    fcount++;
    auto end = chrono::steady_clock::now();
    auto dur = chrono::duration_cast<chrono::seconds>(end - start).count();
    if (dur >= 1)
    {
      cout << "fps processed: " << fps_proc << endl;
      fps_proc = 0;
      start = chrono::steady_clock::now();
    }

  }
  /* get trajectory */
  vector<Trajectory> smooth_trajectory = cumsum(smooth_transforms);
  vector<Trajectory> trajectory        = cumsum(transforms);
  
  /* save it */
  save_traj(smooth_trajectory, fsmooth_trajectory);
  save_traj(trajectory, forig_trajectory);
  
}



int main(int argc, char **argv)
//int main()
{
  string path_in;
  string path_out;
//   /* output a single video or together with input for comparison */
//   bool compare = false;
  
  switch(argc)
  {
    case 1:
    {
      cout << "No input video." << endl;
      return 0;
    }
    case 2:
    {
      path_in = argv[1];
      break;
    }
    case 3:
    {
      path_out = argv[2];
      break;
    }
    default:
      break;
  }
  
  forig_trajectory.open(path_in + "_trajectory.txt");
  fsmooth_trajectory.open(path_out + "_trajectory.txt");
  
  /* input video file */
  VideoCapture cap(path_in);

    
  /* parameters from the original video */
  int w = int(cap.get(CAP_PROP_FRAME_WIDTH)); 
  int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));

  double fps = cap.get(cv::CAP_PROP_FPS);

  VideoWriter out(path_out, cv::VideoWriter::fourcc('M','J','P','G'), fps, Size(w, h));
  stabilise(cap, out);

  /* release video */
  cap.release();
  out.release();  
  
  destroyAllWindows();

  return 0;
}