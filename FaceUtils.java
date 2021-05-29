import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Point;
import android.os.Environment;
import android.util.Log;

import com.dkzy.faceswap.FaceSwapApplication;
import com.dkzy.faceswap.models.Correspondens;
import com.tzutalin.dlib.Constants;
import com.tzutalin.dlib.FaceDet;
import com.tzutalin.dlib.VisionDetRet;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat6;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Subdiv2D;
import org.opencv.photo.Photo;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class FaceUtils {
    private String TAG = FaceUtils.class.getSimpleName();
    private FaceDet mFaceDet;
    private Context context = FaceSwapApplication.INSTANCE();
    private Paint mFaceLandmardkPaint;

    private FaceUtils() {
        initSetting();
    }

    private static class SingletonHolder {
        private static FaceUtils instance = new FaceUtils();
    }

    public static FaceUtils getInstance() {
        return SingletonHolder.instance;
    }


    //初始化人脸特征点检测器
    private void initSetting() {
        if (mFaceDet == null) {
            try {
                mFaceDet = new FaceDet(Constants.assetFilePath(context, "shape_predictor_68_face_landmarks.dat"));
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        mFaceLandmardkPaint = new Paint();
        mFaceLandmardkPaint.setColor(Color.GREEN);
        mFaceLandmardkPaint.setStrokeWidth(2);
        mFaceLandmardkPaint.setStyle(Paint.Style.STROKE);
    }

    public void FaceChange(Bitmap yourFaceBitmap, Bitmap originalBitmap, String filePath) {
        if (originalBitmap == null || yourFaceBitmap == null) return;
        List<Point> points1 = checkFace(originalBitmap);
        List<Point> points2 = checkFace(yourFaceBitmap);
        if (points1 == null || points1.size() == 0 || points2 == null || points2.size() == 0) {
            saveBitmap(originalBitmap, filePath);//没有检测出人脸的话，则使用原图，但是需要拷贝一份bitmap到目标路径
            return;
        }
        Mat mat1 = new Mat();
        Utils.bitmapToMat(originalBitmap, mat1, true);
        Mat mat2 = new Mat();
        Utils.bitmapToMat(yourFaceBitmap, mat2, true);
        Imgproc.cvtColor(mat1, mat1, Imgproc.COLOR_RGBA2BGR);
        Imgproc.cvtColor(mat2, mat2, Imgproc.COLOR_RGBA2BGR);//openCv使用的是bgr，因此需要转换颜色空间啊
        faceOff(points1, points2, mat1, mat2, filePath);//执行换脸操作
    }

    private List<Point> checkFace(Bitmap bitmap) {
        List<VisionDetRet> results;
        results = mFaceDet.detect(bitmap);
        Log.d(TAG, "checkFace| results : " + results);
        //人脸识别还没做研究,默认使用检测到的第一张人脸
        if (results != null) {
            for (final VisionDetRet ret : results) {
                //检索出来人脸关键点 68个
                float resizeRatio = 1.0f;
                Bitmap temp = bitmap.copy(bitmap.getConfig(), true);
                android.graphics.Rect bounds = new android.graphics.Rect();
                bounds.left = (int) (ret.getLeft() * resizeRatio);
                bounds.top = (int) (ret.getTop() * resizeRatio);
                bounds.right = (int) (ret.getRight() * resizeRatio);
                bounds.bottom = (int) (ret.getBottom() * resizeRatio);
                Canvas canvas = new Canvas(temp);
                canvas.drawRect(bounds, mFaceLandmardkPaint);
                // Draw landmark
                ArrayList<Point> landmarks = ret.getFaceLandmarks();
                Log.d(TAG, " 找到人脸特征点landmarks : " + landmarks.size());
                return landmarks;
            }
        }
        return null;
    }

    private void faceOff(List<Point> hull1s, List<Point> hull2s, Mat imgCV1, Mat imgCV2, String filePath) {
        org.opencv.core.Point[] points1 = pointToOpencvPoint(hull1s);
        org.opencv.core.Point[] points2 = pointToOpencvPoint(hull2s);
        Mat imgCV1Warped = imgCV2.clone();
        imgCV1.convertTo(imgCV1, CvType.CV_32F);
        imgCV1Warped.convertTo(imgCV1Warped, CvType.CV_32F);
        //寻找凸包点
        MatOfInt hullIndex = new MatOfInt();
        Imgproc.convexHull(new MatOfPoint(points2), hullIndex, true);
        int[] hullIndexArray = hullIndex.toArray();
        int hullIndexLen = hullIndexArray.length;
        //保存凸包点的容器
        List<org.opencv.core.Point> hull1 = new LinkedList<>();
        List<org.opencv.core.Point> hull2 = new LinkedList<>();
        // 保存组成凸包的关键点
        for (int i = 0; i < hullIndexLen; i++) {
            hull1.add(points1[hullIndexArray[i]]);
            hull2.add(points2[hullIndexArray[i]]);
        }

        Rect rect = new Rect(0, 0, imgCV1Warped.cols(), imgCV1Warped.rows());
        // delaunay triangulation 得劳内三角剖分和仿射变换
        List<Correspondens> delaunayTri = delaunay(hull2, rect);
        for (int i = 0; i < delaunayTri.size(); ++i) {
            List<org.opencv.core.Point> ts1 = new LinkedList<>();
            List<org.opencv.core.Point> ts2 = new LinkedList<>();
            Correspondens corpd = delaunayTri.get(i);
            for (int j = 0; j < 3; j++) {
                ts1.add(hull1.get(corpd.getIndex().get(j)));
                ts2.add(hull2.get(corpd.getIndex().get(j)));
            }
            dealTris(imgCV1, imgCV1Warped, list2MP(ts1), list2MP(ts2));
        }
        // 无缝融合
        List<org.opencv.core.Point> hull8U = new LinkedList<>();
        for (int i = 0; i < hull2.size(); ++i) {
            org.opencv.core.Point pt = new org.opencv.core.Point(hull2.get(i).x, hull2.get(i).y);
            hull8U.add(pt);
        }
        Mat mask = Mat.zeros(imgCV2.rows(), imgCV2.cols(), imgCV2.depth());
        Imgproc.fillConvexPoly(mask, list2MP(hull8U), new Scalar(255, 255, 255));
        Rect r = Imgproc.boundingRect(list2MP(hull2));
        double x = (r.tl().x + r.br().x) / 2;
        double y = (r.tl().y + r.br().y) / 2;
        org.opencv.core.Point center = new org.opencv.core.Point(x, y);
        Mat result = new Mat();
        imgCV1Warped.convertTo(imgCV1Warped, CvType.CV_8UC3);
        Photo.seamlessClone(imgCV1Warped, imgCV2, mask, center, result, Photo.NORMAL_CLONE);
        Imgcodecs.imwrite(filePath, result);
        Log.d(TAG, "done");
    }

    private org.opencv.core.Point[] pointToOpencvPoint(List<Point> points) {
        org.opencv.core.Point[] result = new org.opencv.core.Point[points.size()];
        org.opencv.core.Point temp;
        StringBuilder str = new StringBuilder();
        int index = 0;
        for (Point point : points) {
            temp = new org.opencv.core.Point(point.x, point.y);
            result[index] = temp;
            index++;
            str.append("[").append(point.x).append(point.y).append("]-");
        }
        return result;
    }

    private static List<Correspondens> delaunay(List<org.opencv.core.Point> hull, Rect rect) {
        Subdiv2D subdiv = new Subdiv2D(rect);
        for (int i = 0; i < hull.size(); i++) {
            subdiv.insert(hull.get(i));
        }
        MatOfFloat6 triangles = new MatOfFloat6();
        subdiv.getTriangleList(triangles);//分割后的三角形列表存在矩阵triangles中,(x1,y1,x2,y2,x3,y3)
        int cnt = triangles.rows();
        float[] buff = new float[cnt * 6];
        triangles.get(0, 0, buff);//将三角剖分结果存在一个一维的float数组中
        List<Correspondens> delaunayTri = new LinkedList<>();
        for (int i = 0; i < cnt; ++i) {
            List<org.opencv.core.Point> points = new LinkedList<>();//一个三角形有6个数据分别为三个点的横纵坐标
            points.add(new org.opencv.core.Point(buff[6 * i + 0], buff[6 * i + 1]));
            points.add(new org.opencv.core.Point(buff[6 * i + 2], buff[6 * i + 3]));
            points.add(new org.opencv.core.Point(buff[6 * i + 4], buff[6 * i + 5]));

            Correspondens ind = new Correspondens();
            //确认点都在rect中(即没有越界跑到矩形外面去)
            //这个方法比较笨，如果两个点的横坐标、纵坐标之间距离小于1则认为是同一个点。其实可以用java面向对象组合的方式，加个index进来。。后面优化吧
            if (rect.contains(points.get(0)) && rect.contains(points.get(1)) && rect.contains(points.get(2))) {
                int count = 0;
                for (int j = 0; j < 3; j++) {
                    for (int k = 0; k < hull.size(); k++) {
                        if (Math.abs(points.get(j).x - hull.get(k).x) < 1.0 && Math.abs(points.get(j).y - hull.get(k).y) < 1.0) {
                            ind.add(k);
                            count++;
                        }
                    }
                }
                if (count == 3)
                    delaunayTri.add(ind);
            }
        }
        return delaunayTri;
    }

    private Mat dealTris(Mat img1, Mat img2, MatOfPoint t1, MatOfPoint t2) {
        Rect r1 = Imgproc.boundingRect(t1);
        Rect r2 = Imgproc.boundingRect(t2);

        org.opencv.core.Point[] t1Points = t1.toArray();
        org.opencv.core.Point[] t2Points = t2.toArray();

        List<org.opencv.core.Point> t1Rect = new LinkedList<>();
        List<org.opencv.core.Point> t2Rect = new LinkedList<>();
        List<org.opencv.core.Point> t2RectInt = new LinkedList<>();

        for (int i = 0; i < 3; i++) {
            t1Rect.add(new org.opencv.core.Point(t1Points[i].x - r1.x, t1Points[i].y - r1.y));
            t2Rect.add(new org.opencv.core.Point(t2Points[i].x - r2.x, t2Points[i].y - r2.y));
            t2RectInt.add(new org.opencv.core.Point(t2Points[i].x - r2.x, t2Points[i].y - r2.y));
        }
        // mask 包含目标图片三个凸点的黑色矩形
        Mat mask = Mat.zeros(r2.height, r2.width, CvType.CV_32FC3);
        Imgproc.fillConvexPoly(mask, list2MP(t2RectInt), new Scalar(1.0, 1.0, 1.0), 16, 0);

        Mat img1Rect = new Mat();
        img1.submat(r1).copyTo(img1Rect);

        // img2Rect 原始图片适应mask大小并调整位置的图片
        Mat img2Rect = Mat.zeros(r2.height, r2.width, img1Rect.type());
        img2Rect = applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);
        Core.multiply(img2Rect, mask, img2Rect); // img2Rect在mask三个点之间的图片
        Mat dst = new Mat();
        Core.subtract(mask, new Scalar(1.0, 1.0, 1.0), dst);
        Core.multiply(img2.submat(r2), dst, img2.submat(r2));
        Core.absdiff(img2.submat(r2), img2Rect, img2.submat(r2));
        return img2;
    }

    private MatOfPoint list2MP(List<org.opencv.core.Point> points) {
        org.opencv.core.Point[] t = points.toArray(new org.opencv.core.Point[points.size()]);
        return new MatOfPoint(t);
    }

    private Mat applyAffineTransform(Mat warpImage, Mat src, List<org.opencv.core.Point> srcTri, List<org.opencv.core.Point> dstTri) {
        Mat warpMat = Imgproc.getAffineTransform(list2MP2(srcTri), list2MP2(dstTri));
        Imgproc.warpAffine(src, warpImage, warpMat, warpImage.size(), Imgproc.INTER_LINEAR);
        return warpImage;
    }

    private MatOfPoint2f list2MP2(List<org.opencv.core.Point> points) {
        org.opencv.core.Point[] t = points.toArray(new org.opencv.core.Point[points.size()]);
        return new MatOfPoint2f(t);
    }

    private void saveBitmap(Bitmap bitmap, String path) {
        String savePath;
        File filePic;
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            savePath = path;
        } else {
            Log.d(TAG, "saveBitmap failure : sdcard not mounted");
            return;
        }
        try {
            filePic = new File(savePath);
            if (!filePic.exists()) {
                filePic.getParentFile().mkdirs();
                filePic.createNewFile();
            }
            FileOutputStream fos = new FileOutputStream(filePic);
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, fos);
            fos.flush();
            fos.close();
        } catch (IOException e) {
            Log.d(TAG, "saveBitmap: " + e.getMessage());
            return;
        }
        Log.d(TAG, "saveBitmap success: " + filePic.getAbsolutePath());
    }


}
