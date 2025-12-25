import json
import sys
from typing import Iterator
import zmq
import av

def stream_mp4_from_filesystem(filepath: str, stream_index: int = 0) -> Iterator[av.VideoFrame]:
    with av.open(filepath, mode='r') as container:
        for frame in container.decode(video=stream_index):
            yield frame


# def stream_mp4_from_url(url: str) -> Iterator[av.VideoFrame]:
#     session = requests.Session()
#     response = session.get(url, stream=True, timeout=45)
#     response.raise_for_status()
#
#     container = av.open(response.raw, mode='r', format='mp4')
#
#     try:
#         for frame in container.decode(video=0):
#             yield frame
#
#     finally:
#         container.close()


def main(args: list[str]):
    mp4_path = args[1]
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("ipc:///tmp/detector")

    for i, video_frame in enumerate(stream_mp4_from_filesystem(mp4_path)):
        print(f"Frame {i} pts={video_frame.pts}, time={video_frame.time}")
        frame_data = video_frame.to_ndarray()

        header = {
            "shape": list(frame_data.shape),
            "dtype": str(frame_data.dtype),
            "pts": video_frame.pts,
            "time": video_frame.time,
            "format": "bgr24",
        }
        print(f"Sending frame {i} to ipc:///tmp/detector with header: {header}")

        socket.send_multipart([json.dumps(header).encode("utf-8"), frame_data.tobytes()])


    context.term()
    socket.close()


if __name__ == "__main__":
    main(sys.argv)
