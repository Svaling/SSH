# ------------------------------------------
# SSH: Single Stage Headless Face Detector
# Demo
# by Mahyar Najibi
# ------------------------------------------

from __future__ import print_function
from SSH.testcam import detect
from argparse import ArgumentParser
import os
from utils.get_config import cfg_from_file, cfg, cfg_print
import caffe
import cv2
import sys

windowName = "CameraCaffeDemo"
helpText = "'Esc' to Quit, 'H' to Toggle Help, 'F' to Toggle Fullscreen"


def parser():
    parser = ArgumentParser('SSH Demo!')
    parser.add_argument('--im', dest='im_path', help='Path to the image',
                        default='data/demo/demo.jpg', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='The GPU ide to be used',
                        default=0, type=int)
    parser.add_argument('--proto', dest='prototxt', help='SSH caffe test prototxt',
                        default='SSH/models/test_ssh.prototxt', type=str)
    parser.add_argument('--model', dest='model', help='SSH trained caffemodel',
                        default='data/SSH_models/SSH.caffemodel', type=str)
    parser.add_argument('--out_path', dest='out_path', help='Output path for saving the figure',
                        default='data/demo', type=str)
    parser.add_argument('--cfg', dest='cfg', help='Config file to overwrite the default configs',
                        default='SSH/configs/wider_pyramid.yml', type=str)
    parser.add_argument("--rtsp", dest="use_rtsp",
                        help="use IP CAM (remember to also set --uri)",
                        action="store_true")
    parser.add_argument("--uri", dest="rtsp_uri",
                        help="RTSP URI string, e.g. rtsp://192.168.1.64:554",
                        default=None, type=str)
    parser.add_argument("--latency", dest="rtsp_latency",
                        help="latency in ms for RTSP [200]",
                        default=200, type=int)
    parser.add_argument("--usb", dest="use_usb",
                        help="use USB webcam (remember to also set --vid)",
                        action="store_true")
    parser.add_argument("--vid", dest="video_dev",
                        help="video device # of USB webcam (/dev/video?) [1]",
                        default=1, type=int)
    parser.add_argument("--width", dest="image_width",
                        help="image width [640]",
                        default=640, type=int)
    parser.add_argument("--height", dest="image_height",
                        help="image width [480]",
                        default=480, type=int)
    parser.add_argument("--crop", dest="crop_center",
                        help="crop the square at center of image for Caffe inferencing [False]",
                        action="store_true")
    return parser.parse_args()


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ("rtspsrc location={} latency={} ! rtph264depay ! h264parse ! omxh264dec ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ("v4l2src device=/dev/video{} ! "
               "video/x-raw, width=(int){}, height=(int){}, format=(string)RGB ! "
               "videoconvert ! appsink").format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use Jetson onboard camera
    gst_str = ("nvcamerasrc ! "
               "video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, framerate=(fraction)30/1 ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(windowName, width, height):
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, width, height)
    cv2.moveWindow(windowName, 0, 0)
    cv2.setWindowTitle(windowName, "Dense scene crowd detection Demo for Jetson")


def read_cam_and_classify(windowName, cap, net, crop):
    showHelp = True
    showFullScreen = False
    font = cv2.FONT_HERSHEY_PLAIN
    while True:
        if cv2.getWindowProperty(windowName, 0) < 0:  # Check to see if the user closed the window
            # This will fail if the user closed the window; Nasties get printed to the console
            break
        ret_val, img = cap.read()

        if crop:
            height, width, channels = img.shape
            if height < width:
                img_crop = img[:, ((width - height) // 2):((width + height) // 2), :]
            else:
                img_crop = img[((height - width) // 2):((height + width) // 2), :, :]
        else:
            img_crop = img
        # Perform detection
        cls_dets, _, result_img = detect(net, img_crop, visualization_folder=args.out_path, visualize=True, pyramid=pyramid)

        if showHelp == True:
            cv2.putText(result_img, helpText, (11, 20), font, 1.0, (32, 32, 32), 4, cv2.LINE_AA)
            cv2.putText(result_img, helpText, (10, 20), font, 1.0, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.imshow(windowName, result_img)
        key = cv2.waitKey(10)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'):  # toggle help message
            showHelp = not showHelp
        elif key == ord('F') or key == ord('f'):  # toggle fullscreen
            showFullScreen = not showFullScreen
            if showFullScreen == True:
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


if __name__ == "__main__":

    # Parse arguments
    args = parser()

    # Load the external config
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    # Print config file
    cfg_print(cfg)

    # Loading the network
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(args.prototxt), 'Please provide a valid path for the prototxt!'
    assert os.path.isfile(args.model), 'Please provide a valid path for the caffemodel!'

    print('Loading the network...', end="")
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'SSH'
    print('Done!')
    print("OpenCV version: {}".format(cv2.__version__))

    # Read image
    # assert os.path.isfile(args.im_path), 'Please provide a path to an existing image!'
    pyramid = True if len(cfg.TEST.SCALES) > 1 else False

    # initialize camera
    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri, args.image_width, args.image_height, args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev, args.image_width, args.image_height)
    else:  # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width, args.image_height)

    if not cap.isOpened():
        sys.exit("Failed to open camera!")

    # start capturing live video and do inference
    open_window(windowName, args.image_width, args.image_height)

    read_cam_and_classify(windowName, cap, net, args.crop_center)

    cap.release()
    cv2.destroyAllWindows()
