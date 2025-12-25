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
        draw_rois(frame, rois)

        add_timestamp(frame, header)

        print("Displaying frame")
        cv2.imshow("Stream with ROIs", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    socket.close()
    context.term()


def draw_rois(frame, rois):
    h_img, w_img = frame.shape[:2]
    for x, y, w, h in rois:
        x, y, w, h = int(x), int(y), int(w), int(h)
        x1 = max(0, min(x, w_img - 1))
        y1 = max(0, min(y, h_img - 1))
        x2 = max(0, min(x + w, w_img - 1))
        y2 = max(0, min(y + h, h_img - 1))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)



def add_timestamp(frame, header):
    t = header.get("time")
    time_label = f"t={float(t):.3f}s"
    cv2.putText(frame, time_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)


if __name__ == "__main__":
    main()
