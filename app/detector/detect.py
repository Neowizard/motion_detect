import json
import traceback

import zmq
import numpy

from detector import motion_detection


def main():
    motion_detector = motion_detection.MotionDetector()
    context = zmq.Context()

    stream_socket = context.socket(zmq.PULL)
    stream_socket.connect("ipc:///tmp/detector")

    display_socket = context.socket(zmq.PUSH)
    display_socket.bind("ipc:///tmp/display")

    while True:
        try:
            print("Waiting for frames from ipc:///tmp/detector")
            header_bytes, frame_bytes = stream_socket.recv_multipart()
            header = json.loads(header_bytes.decode("utf-8"))
            print(f"Received frame pts={header.get('pts')} time={header.get('time')} shape={header.get('shape')}")

            shape = tuple(header["shape"])
            dtype = numpy.dtype(header["dtype"])

            frame_array = numpy.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

            print(f"Detecting ROIs in frame pts={header.get('pts')}")
            motion_rois = motion_detector.process_frame(frame_array)
            print(f"Detected {len(motion_rois)} ROIs in frame")

            rois_bytes = json.dumps(motion_rois).encode("utf-8")
            print("Sending frame + ROIs to ipc:///tmp/display")
            display_socket.send_multipart([header_bytes, frame_bytes, rois_bytes])

        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            break

    context.term()


if __name__ == "__main__":
    main()

