import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import queue
import pandas as pd
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from trackNet.tracknet import trackNet
print(trackNet)

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

n_classes = 256
save_weights_path = "./WeightsTracknet/model.1"

data = []

# function filters contours and removes small and big ones and returns those which represent player
def filtercontours(contours):
    playercontours = list()
    for c in contours:
        rect = cv2.boundingRect(c)
        # if contour is too big or too small its not player
        if (rect[2] < 7 or rect[3] < 20) or (rect[2] > 60 or rect[3] > 100): continue
        playercontours.append(c)
    return playercontours


# function classifies contours to 2 lists based on color
def classifycontours(contours,frame, mask):
    classifiedObjects = {}
    ateamplayers = list()
    bteamplayers = list()

    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        crop_img = frame[y:y + h, x:x + w]
        # compute mean color with mask of background for better reults
        meanColor = cv2.mean(frame, mask)
        # comparison of Green color range is threshold for labels
        if meanColor[1] > 100:
            ateamplayers.append(c)
        else:
            bteamplayers.append(c)

    classifiedObjects['ateam'] = ateamplayers
    classifiedObjects['bteam'] = bteamplayers
    return classifiedObjects

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    # print(FLAGS.framework)

    nms_max_overlap = 1.0

    # # width and height in TrackNet
    TrackNetWidth, TrackNetHeight = 640, 360
    img, img1, img2 = None, None, None

    # # load TrackNet model
    modelFN = trackNet
    m = modelFN(n_classes, input_height=TrackNetHeight, input_width=TrackNetWidth)
    m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    m.load_weights(save_weights_path)

    # In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
    q = queue.deque()
    for i in range(0, 8):
        q.appendleft(None)
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
        length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0
    # while video is running
    while frame_num < 250:
        return_value, frame = vid.read()

        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num, frame_num/length*100)
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # detect the ball
        # img is the frame that TrackNet will predict the position
        # since we need to change the size and type of img, copy it to output_img
        output_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # resize it
        img = cv2.resize(output_img, (TrackNetWidth, TrackNetHeight))
        # input must be float type
        img = img.astype(np.float32)

        # since the odering of TrackNet  is 'channels_first', so we need to change the axis
        X = np.rollaxis(img, 2, 0)
        # prdict heatmap
        pr = m.predict(np.array([X]))[0]
        # print("pr: ", pr)

        # since TrackNet output is ( net_output_height*model_output_width , n_classes )
        # so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
        # .argmax( axis=2 ) => select the largest probability as class
        pr = pr.reshape((TrackNetHeight, TrackNetWidth, n_classes)).argmax(axis=2)

        # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
        pr = pr.astype(np.uint8)

        # reshape the image size as original input image
        heatmap = cv2.resize(pr, (width, height))

        # heatmap is converted into a binary image by threshold method.
        ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        # find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)
        print("circles: ", circles)

          # check if there have any tennis be detected
        if circles is not None:
            # if only one tennis be detected
            if len(circles) == 1:

                x = int(circles[0][0][0])
                y = int(circles[0][0][1])

                # push x,y to queue
                q.appendleft([x, y])
                # pop x,y from queue
                q.pop()

                data.append([frame_num, 'ball', '-', '-', x, y])      

            else:
                # push None to queue
                q.appendleft(None)
                # pop x,y from queue
                q.pop()

        else:
            # push None to queue
            q.appendleft(None)
            # pop x,y from queue
            q.pop()
        
        # draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
        for i in range(0, 8):
            if q[i] is not None:
                draw_x = q[i][0]
                draw_y = q[i][1]
                cv2.circle(frame, (draw_x, draw_y), 2, (255,255,0), thickness=1, lineType=8, shift=0)

        #update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # print(bbox)
            color_list=['red','blue','white']
            boundaries = [
                ([17, 15, 75], [50, 56, 200]), #red
                ([43, 31, 4], [250, 88, 50]), #blue
                ([187,169,112],[255,255,255]) #white
                ]
            ymin = int(bbox[1])
            ymax = int(bbox[3]) 
            xmin = int(bbox[0])
            xmax = int(bbox[2] )
            crop_img = frame[ymin:ymax, xmin:xmax]

            team = " "
            # print(crop_img, class_name)
            if crop_img.size and class_name == "person":
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
                # fgbg1 = cv2.bgsegm.createBackgroundSubtractorCNT(isParallel=True)

                # define range of green color in HSV
                lower_green = np.array([36, 0, 0])
                upper_green = np.array([86, 255, 255])
                # define range of orange color in HSV
                ORANGE_MIN = np.array([10, 0, 0], np.uint8)
                ORANGE_MAX = np.array([205, 255, 255], np.uint8)
                    # Convert BGR to HSV
                hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

                # Threshold the HSV image to get only green colors
                mask = cv2.inRange(hsv, lower_green, upper_green)
                # create mask for referees orange color
                refMask = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
                cv2.bitwise_not(refMask, refMask)

                refMask = cv2.erode(refMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations=1)
                refMask = cv2.dilate(refMask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

                # Bitwise-AND mask and original image
                cv2.bitwise_not(mask, mask)

                # fg mask from backg substraction
                fgmask = fgbg.apply(crop_img)

                # closing
                fgmask = cv2.erode(fgmask, kernel, iterations=1)
                fgmask = cv2.dilate(fgmask, kernel, iterations=1)

                # opening
                fgmask = cv2.dilate(fgmask, kernel, iterations=1)
                fgmask = cv2.erode(fgmask, kernel, iterations=1)


                clonedFrame = fgmask

                ret, thresh1 = cv2.threshold(clonedFrame, 100, 255, cv2.THRESH_BINARY)

                fgmask = cv2.bitwise_and(fgmask, mask)
                reffgMask= cv2.bitwise_and(fgmask, refMask)
                # create mask where are no referees
                notrefmask = cv2.bitwise_not(reffgMask)
                cv2.imwrite('ref_ext.jpg', reffgMask)
                cv2.imwrite('not_ref_ext.jpg', fgmask)
                # remove from mask referees
                fgmask = cv2.bitwise_and(fgmask, notrefmask)

                # find contours for referees and players
                contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                refcontours, hierarchy = cv2.findContours(reffgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mean = cv2.mean(crop_img, fgmask)
                print(mean)

                if mean[1] > 100:
                    team = " a"
                    print("team a")
                else:
                    team = " b"
                    print("team b")
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + team + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            """ Store player position in dataframe format """
            if track.track_id == 1 or track.track_id == 2:
                data.append([frame_num, class_name, team, track.track_id, (bbox[0] + bbox[2])/2, bbox[3]])      
                print(data)
       

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
            df_players_positions = pd.DataFrame(data, columns=["id", "class_name", "team", "x", "y"])
            df_players_positions.to_csv("tracking_players.csv")

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
