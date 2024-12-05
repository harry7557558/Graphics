#!/usr/bin/env python3

import cv2
import numpy as np
import os


quality = 95

def write_image(filename, image):
    if 0 <= quality <= 100:
        filename += '.jpg'
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(filename, image, encode_param)
    else:
        filename += '.png'
        cv2.imwrite(filename, image)
    return filename


class RosbagImageLoader:
    def __init__(self, bag_path, save_path, max_frames=100000, skip=1, required_topic=None):
        self.bag_path = bag_path

        os.makedirs(save_path, exist_ok=True)

        print("Loading rosbag (this may take a while on the first run)...")

        all_image_topics = set()
        frame_count = 0
        with __import__('rosbag').Bag(self.bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages():
                if required_topic is not None and topic != required_topic:
                    continue
                if msg._type in ['sensor_msgs/Image', 'sensor_msgs/CompressedImage']:
                    all_image_topics.add(topic)
                    if frame_count % skip != 0:
                        frame_count += 1
                        continue
                    image = self.get_image(msg)
                    filename = os.path.join(save_path, f"{frame_count:05d}")
                    filename = write_image(filename, image)
                    print(filename, topic)
                    frame_count += 1
                    if frame_count >= max_frames:
                        print("Maximum number of frames reached.")
                        break

        print(f"Extracted {frame_count//skip} frames from the rosbag.")
        if required_topic is None:
            print("Detected image topics:", all_image_topics)

    def get_image(self, msg):
        if msg._type == 'sensor_msgs/Image':
            cv_image = self._imgmsg_to_cv2(msg)
        else:  # CompressedImage
            cv_image = self._compressed_imgmsg_to_cv2(msg)
        return cv_image

    def _imgmsg_to_cv2(self, img_msg):
        
        dtype = np.dtype("uint8")
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),
                                  dtype=dtype, buffer=img_msg.data)
        
        if img_msg.encoding == "rgb8":
            image_opencv = cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB)
        elif img_msg.encoding != "bgr8":
            raise ValueError(f"Unsupported encoding: {img_msg.encoding}")
        
        return image_opencv

    def _compressed_imgmsg_to_cv2(self, compressed_img_msg):
        np_arr = np.frombuffer(compressed_img_msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def extract_video_frames(video_path, output_dir, max_frames, skip):

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video file.")
        return

    frame_count = 0

    while True:
        success, frame = video.read()
        if not success:
            break
        if frame_count % skip != 0:
            frame_count += 1
            continue
        filename = os.path.join(output_dir, f"{frame_count:05d}")
        filename = write_image(filename, frame)
        print(filename)
        frame_count += 1
        if frame_count >= max_frames:
            print("Maximum number of frames reached.")
            break

    video.release()

    print(f"Extracted {frame_count//skip} frames from the video.")


def extract_frames(filename, output_dir, max_frames=100000, skip=1, ros_topic=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if filename.split('.')[-1].lower() == 'bag':
        if ros_topic == "":
            ros_topic = None
        RosbagImageLoader(filename, output_dir, max_frames, skip, ros_topic)

    else:  # video
        extract_video_frames(filename, output_dir, max_frames, skip)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract frames from a rosbag or a video.")
    parser.add_argument("input_file", nargs=1, help="The input rosbag or video file.")
    parser.add_argument("--max_frames", "-n", default="100000", help="Maximum number of frames to output.")
    parser.add_argument("--skip", "-s", default="1", help="Take one frame every this number of frames.")
    parser.add_argument("--ros_topic", "-t", default="", help="rosbag topic. If not set, it will exact all detected frames.")
    parser.add_argument("--quality", "-q", default="95", help="Quality to save JPEG images. Save lossless PNG if this is not between 0 and 100.")
    args = parser.parse_args()

    quality = int(args.quality)
    filename = args.input_file[0]
    dirname = filename[:filename.rfind('.')]
    output_dir = os.path.join(dirname, 'images')
    extract_frames(
        filename, output_dir,
        int(args.max_frames), int(args.skip), args.ros_topic
    )

    open(os.path.join(dirname, 'command.bash'), 'w').write("""
colmap feature_extractor --database_path database.db --image_path ./images --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 0 --SiftExtraction.max_num_features 768  --SiftExtraction.num_threads 12
colmap vocab_tree_matcher --database_path database.db --SiftMatching.use_gpu 0 --VocabTreeMatching.vocab_tree_path ../../vocab_tree_flickr100K_words32K.bin

mkdir sparse
#glomap mapper --database_path database.db --output_path sparse

colmap mapper --database_path database.db --image_path ./images --output_path sparse
colmap bundle_adjuster --input_path sparse/0 --output_path sparse/0 --BundleAdjustment.refine_principal_point 1

ns-process-data images --data ./images --output-dir . --skip-image-processing --skip-colmap --colmap-model-path sparse/0/
""".strip())
