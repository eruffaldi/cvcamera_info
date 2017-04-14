# Simple Manipulation of Camera Intrinsics
# Emanuele Ruffaldi 2016-2017
#
#Additional Material
#See: http://ksimek.github.io/2013/08/13/intrinsic/
#OpenCV: http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
#and http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
#http://ksimek.github.io/perspective_camera_toy.html
import numpy as np
import yaml
import cv2

# A yaml constructor is for loading from a yaml node.
# This is taken from: http://stackoverflow.com/a/15942429
def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat
yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)
 
# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
#http://www.morethantechnical.com/2016/03/02/opencv-python-yaml-persistance/
def opencv_matrix_representer(dumper, mat):
    mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)
yaml.add_representer(np.ndarray, opencv_matrix_representer)


def strip_malformed_directive(yaml_file):
    """
    Strip a malformed YAML directive from the top of a file.

    Returns the slurped (!) file.
    """
    lines = list(yaml_file)
    first_line = lines[0]
    if first_line.startswith('%') and ":" in first_line:
       return "\n".join(lines[1:])
    else:
       return "\n".join(lines)

# NOT USED
def convert_opencvmatrix_tag(yaml_events):
    """
    Convert an erroneous custom tag, !!opencv-matrix, to the correct 
    !opencv-matrix, in a stream of YAML events.
    """
    for event in yaml_events:
        if hasattr(event, "tag") and event.tag == u"tag:yaml.org,2002:opencv-matrix":
            event.tag = u"!opencv-matrix"
        yield event


class CameraInfo:
    # camera intrinsics structure: translation * scaling * shear 
    # distortion is invariant to the intrinsics
    def __init__(self,size,K=None,dist=None):
        if K is None:
            K = np.array([[1,0,size[0]/2],[0,1,size[1]],[0,0,1]])
        self.size = size
        self.K = K
        self.dist = dist
    def new_mirror(self,alongx,alongy):
        """Computes a new Camera Info in the case of mirroring"""
        Knew = K.clone()
        if alongx:
            Knew[0,2] = size[0]-Knew[0,2]
        if alongy:
            Knew[1,2] = size[1]-Knew[1,2]
        return CameraInfo(self.size,Knew,self.dist)
    def new_crop(self,topleft,size):
        Knew = np.array(np.dot([[1,0,-topleft[0]],[0,1,-topleft[1]],[0,0,1]],self.K))
        return CameraInfo(size,Knew,self.dist)
    def new_resize(self,newsize):
        sx = newsize[0]/float(self.size[0])
        sy = newsize[1]/float(self.size[1])
        return CameraInfo(newsize,np.dot(np.diag([sx,sy,1]),self.K),self.dist)                  
    def new_undistorted(self):
        """Removes Undistortion from CameraInfo"""
        return CameraInfo(self.size,self.K,None)                
    def new_makeOptimal(self,alpha,otherSize=None,centerPrincipalPoint=False):
        """computes the optimal transformation for the undistortion toward another or similar size. Wraps cv2.getOptimalNewCameraMatrix"""
        OK,validROI = cv2.getOptimalNewCameraMatrix(self.K,self.dist,alpha,otherSize,centerPrincipalPoint)
        if otherSize is None:
            otherSize = self.size
        return CameraInfo(otherSize,OK,None),validPixROI

    @staticmethod
    def fromyaml(ymlfile):
        if type(ymlfile) is str:
            ymlfile = open(ymlfile,"rb")

        #http://stackoverflow.com/questions/28058902/how-to-skip-lines-when-reading-a-yaml-file-in-python
        directive_processed = strip_malformed_directive(ymlfile)
        yaml_events = yaml.parse(directive_processed)
        #matrix_tag_converted = convert_opencvmatrix_tag(yaml_events)
        fixed_document = yaml.emit(yaml_events)

        q = yaml.load(fixed_document)
        print q

        if "rgb_intrinsics" in q:
            K = q["rgb_intrinsics"]
        else:
            raise Exception("rgb_intrinsics not found")
        if "rgb_distortion" in q:
            d = q["rgb_distortion"]
        else:
            raise Exception("rgb_distortion")
        if "image_height" in q:
            w = q["image_width"]
            h = q["image_height"]
        else:
            raise Exception("Size not found in YML")
        return CameraInfo((w,h),K,d)

    def toyaml(self,ymlfile,Kname="rgb_intrinsics",dname="rgb_distortion",sname=("image_width","image_height")):
        """To YAML file or as String"""
        q = dict()
        q[Kname] = self.K
        q[dname] = self.dist
        if type(sname) is tuple:
            q[sname[0]] = self.size[0]
            q[sname[1]] = self.size[1]
        else:
            q[sname] = self.size

        if ymlfile == "" or ymlfile is None:
            return yaml.dumps(q)
        elif type(ymlfile) is str:
            ymlfile = open(ymlfile,"wb")
            yaml.dump(q,ymlfile)

    def __repr__(self):
        return "CameraInfo(%d x %d,r %f,K\n\t%s,\n\tD %s)" % (self.size[0],self.size[1],self.ratio,self.K,self.dist)
    @property
    def ratio(self):
        return float(self.size[0])/self.size[1]
    def metrics(self,sensorsize_mm):
        r = cv2.calibrationMatrixValues(self.K,self.size,sensorsize_mm[0],sensorsize_mm[1])
        return dict(fovx_deg=r[0],fovy_deg=r[1],focalLength_mm=r[2],principalPoint_mm=r[3],aspectRatio=r[4])
    def undistort(self,src):
        """undistors the image using this camera info. For multiple images better use initUndistortRectifyMap and then remap"""
        # note: no check over src.shape and self.size
        return cv2.undistort(src,self.K,self.dist)
    def initUndistortRectifyMap(self,m1type,R=None,otherK=None):
        """calls OpenCV initUndistortRectifyMap using CameraInfo data. R is optional. otherK is used for stereo"""
        return cv2.initUndistortRectifyMap(self.K,self.dist,R,otherK is None and self.K or otherK,self.size,m1type)
if __name__ == '__main__':
    # Examples of Cameras
    K2_K = np.array([[1023.48095703125, 0.0, 959.43603515625],[0.0, 1023.1721801757812, 539.6748046875],[0.0, 0.0, 1.0]])
    K2_d = np.array([0.010865901596844196,0.011715753003954887,-0.0019090864807367325,0.0018720569787546992,-0.06741847097873688])

    a = CameraInfo((1920,1080),K2_K,K2_d)
    print a
    print "resize",a.new_resize((640,480))
    print "crop",a.new_crop((10,10),(640,480))

    import sys
    if len(sys.argv) > 1:
        q = CameraInfo.fromyaml(sys.argv[1])
        qn = q.new_resize((640,480))
        qn.toyaml(sys.argv[1]+"new.yaml")