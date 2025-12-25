import json
import time

import cv2
import zmq
import numpy


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("ipc:///tmp/display")

    first_stream_time = None
    first_wall_time = None

    while True:
        print("Waiting for messages from ipc:///tmp/display")
        header_bytes, frame_bytes, rois_bytes = socket.recv_multipart()
        header = json.loads(header_bytes.decode("utf-8"))

        shape = tuple(header["shape"])
        dtype = numpy.dtype(header["dtype"])
        frame = numpy.frombuffer(frame_bytes, dtype=dtype).reshape(shape).copy()

        stream_time = header.get("time")
        if isinstance(stream_time, (int, float)):
            if first_stream_time is None:
                first_stream_time = float(stream_time)
                first_wall_time = time.monotonic()

            target_wall = first_wall_time + (float(stream_time) - first_stream_time)
            now = time.monotonic()
            sleep_s = target_wall - now
            if sleep_s > 0:
                time.sleep(sleep_s)

        rois = json.loads(rois_bytes.decode("utf-8"))

        print(f"Drawing {len(rois)} ROIs on frame")
        for roi in rois:
            x1, y1, x2, y2 = roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            roi_pixels = frame[y1:y2, x1:x2]
            roi_blurred = cv2.GaussianBlur(roi_pixels, (15, 15), 4)
            frame[y1:y2, x1:x2] = roi_blurred

        add_timestamp(frame, header)

        print("Displaying frame")
        cv2.imshow("Stream with ROIs", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    socket.close()
    context.term()


def add_timestamp(frame, header):
    t = header.get("time")
    time_label = f"t={float(t):.3f}s"
    cv2.putText(frame, time_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
