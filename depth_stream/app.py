import edgeiq

from edgeiq.oak import Oak


def depth_stream():
    """
    This function shows how to retrieve the depth stream from the camera.
    """

    fps = edgeiq.FPS()
    with Oak('alwaysai/mobilenet_ssd_oak',
             capture_depth=True) as camera, edgeiq.Streamer() as streamer:

        fps.start()
        while True:
            text = ['FPS: {:2.2f}'.format(fps.compute_fps())]

            depth = camera.get_depth()
            if depth is not None:
                streamer.send_data(depth, text)
                fps.update()

            if streamer.check_exit():
                break

        print('fps = {}'.format(fps.compute_fps()))


if __name__ == '__main__':
    depth_stream()
