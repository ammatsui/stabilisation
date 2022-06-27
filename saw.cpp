#include "utils.hpp"  // tranformations and trajctory operations
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <chrono> // for time measurment


/*
This video stablisation smooths the global trajectory using a sliding average window
1. Get previous to current frame transformation (dx, dy, da) for all frames
2. Accumulate the transformations to get the image trajectory
3. Smooth out the trajectory using an averaging window
4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
5. Apply the new transformation to the video
*/


using namespace std;
using namespace cv;


const int SMOOTHING_RADIUS = 50;  // in frames
/* how often update features */
const int freq = 15;   // in frames 
/* goodFeaturesToTrack params */
const int maxCorners = 200; 
const double qualityLevel = 0.01;
const double minDistance = 20;
/* cropping out the black borders */
const int threshVal = 20;
const float borderThresh = 0.05f;
// /* canny edge detection */
// const int canny_low = 35;
// const int canny_high = 90;
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
//   // GaussianBlur(grey, grey, Size(21, 21), 0);
//   // equalizeHist(grey, grey);
//   // Canny(grey, grey, canny_low, canny_high);
//   // GaussianBlur(grey, grey, Size(21, 21), 0);
// //  Sobel(grey, grey, -1, 0, 1);
// }

void storeTransforms(VideoCapture& cap, vector<TransformParam>& transforms)
{
  Mat curr, curr_grey; 
  Mat prev, prev_grey;
  
  vector<Point2f> prev_pts;

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

    /* save transformation */ 
    transforms.push_back(TransformParam(dx, dy, da));

    /* prepare for the next frame */
    curr_grey.copyTo(prev_grey);
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

}


void writeStable(VideoCapture& cap, VideoWriter& out, vector<TransformParam>& transforms_smooth, bool compare)
{
  cap.set(cv::CAP_PROP_POS_FRAMES, 0);
  int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT)); 
  Mat T(2,3,CV_64F);
  Mat frame, frame_stabilised, frame_out; 

  for( int i = 0; i < n_frames-1; i++) 
  { 
    bool success = cap.read(frame);
    if(!success) break;
    
    transforms_smooth[i].getTransform(T); 

    /* apply transformation to the frame */
    warpAffine(frame, frame_stabilised, T, frame.size());

    /* remove black borders */
    fixBorder(frame_stabilised); 
   
  
    /* comparison with the original video */
    if (compare)
      hconcat(frame, frame_stabilised, frame_out);
    else
    {
      frame_stabilised.copyTo(frame_out);
    }

    /* if the image is too big, resize it */
    if(frame_out.cols > 1920) 
    {
        resize(frame_out, frame_out, Size(frame_out.cols/2, frame_out.rows/2));
    }

    imshow("Output", frame_out);
    out.write(frame_out);
    waitKey(10);
  }

}


void stabilise(VideoCapture& cap, VideoWriter& out, bool compare)
{
  /* get transformations between frames from the original video */
  vector <TransformParam> transforms;  
  storeTransforms(cap, transforms);
  /* compute the trajectory using cumulative sum */
  vector <Trajectory> trajectory = cumsum(transforms);
  save_traj(trajectory, forig_trajectory);

  /* smooth the trajectory (average filter) */
  vector <Trajectory> smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS); 
  save_traj(smoothed_trajectory, fsmooth_trajectory);
  
  /* get smoothed transformations between frames for the new video */
  vector <TransformParam> transforms_smooth;
  smoothTransforms(transforms, transforms_smooth, trajectory, smoothed_trajectory);
  
  /* write stable video */
  writeStable(cap, out, transforms_smooth, compare);
  
}


int main(int argc, char **argv)
{
  string path_in;
  string path_out;
  /* output a single video or together with input for comparison */
  bool compare = false;
  
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
      path_out= argv[2];
      break;
    }
    case 4:
    {
      compare = argv[4];
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

  /* prepare for the output video */
  if (compare)
  {
    /* width is twice as big for comparison output */
    w = 2*w;
  }
  VideoWriter out(path_out, cv::VideoWriter::fourcc('M','J','P','G'), fps, Size(w, h));
  stabilise(cap, out, compare);

  /* release video */
  cap.release();
  out.release();  
  
  destroyAllWindows();

  return 0;
}