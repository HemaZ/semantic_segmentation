import rclpy
from rclpy.node import Node
import numpy as np
import imutils
import time
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('semantic_segmentation')
        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.publisher_ = self.create_publisher(Image, '/segmentation', 10)
        filepath = os.path.dirname(os.path.realpath(__file__))
        self.CLASSES = open(filepath +
                            "/enet-cityscapes/enet-classes.txt").read().strip().split("\n")
        self.COLORS = open(filepath +
                           "/enet-cityscapes/enet-colors.txt").read().strip().split("\n")
        self.COLORS = [np.array(c.split(",")).astype("int")
                       for c in self.COLORS]
        self.COLORS = np.array(self.COLORS, dtype="uint8")
        self.get_logger().info('Loading model')
        self.net = cv2.dnn.readNet(filepath+"/enet-cityscapes/enet-model.net")
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        self.get_logger().info('I got an image')
        image = self.bridge.imgmsg_to_cv2(msg)
        image = imutils.resize(image, width=500)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
                                     swapRB=True, crop=False)
        # perform a forward pass using the segmentation model
        self.net.setInput(blob)
        start = time.time()
        output = self.net.forward()
        end = time.time()

        # show the amount of time inference took
        print("[INFO] inference took {:.4f} seconds".format(end - start))

        # infer the total number of classes along with the spatial dimensions
        # of the mask image via the shape of the output array
        (numClasses, height, width) = output.shape[1:4]

        # our output class ID map will be num_classes x height x width in
        # size, so we take the argmax to find the class label with the
        # largest probability for each and every (x, y)-coordinate in the
        # image
        classMap = np.argmax(output[0], axis=0)

        # given the class ID map, we can map each of the class IDs to its
        # corresponding color
        mask = self.COLORS[classMap]

        # resize the mask and class map such that its dimensions match the
        # original size of the input image (we're not using the class map
        # here for anything else but this is how you would resize it just in
        # case you wanted to extract specific pixels/classes)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # perform a weighted combination of the input image with the mask to
        # form an output visualization
        output = ((0.4 * image) + (0.6 * mask)).astype("uint8")
        out_msg = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
        self.publisher_.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
