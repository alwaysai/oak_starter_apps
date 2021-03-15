import edgeiq


def object_detection():
    """
    Oak.get_model_result can return different results based on the purpose of the model running on the camera.

    This function shows how to work with object detection models.
    """
    fps = edgeiq.FPS()

    with edgeiq.Oak('alwaysai/mobilenet_ssd_oak') as camera, edgeiq.Streamer(
    ) as streamer:

        fps.start()
        while True:

            text = ['FPS: {:2.2f}'.format(fps.compute_fps())]

            frame = camera.get_frame()

            result = camera.get_model_result(confidence_level=.75)

            # Check for inferencing results. Oak.get_model_result is a non-blocking call and will return None when new data is not available.
            if result:
                frame = edgeiq.markup_image(frame, result.predictions)

                text.append("Objects:")

                for prediction in result.predictions:
                    text.append("{}: {:2.2f}%".format(
                        prediction.label, prediction.confidence * 100))

            streamer.send_data(frame, text)

            if streamer.check_exit():
                break

            fps.update()

        print('fps = {}'.format(fps.compute_fps()))


def pose_estimation():
    """
    Oak.get_model_result can return different results based on the purpose of the model running on the camera.

    This function shows how to work with pose estimation models.
    """
    fps = edgeiq.FPS()

    with edgeiq.Oak('alwaysai/human_pose_oak') as camera, edgeiq.Streamer(
    ) as streamer:

        fps.start()
        while True:

            text = ['FPS: {:2.2f}'.format(fps.compute_fps())]

            frame = camera.get_frame()

            result = camera.get_model_result()

            # Check for inferencing results. Oak.get_model_result is a non-blocking call and will return None when new data is not available.
            if result:
                frame = result.draw_poses(frame)

                text.append("Poses:")

                for ind, pose in enumerate(result.poses):
                    text.append("Person {}".format(ind))
                    text.append('-' * 10)
                    text.append("Key Points:")
                    for key_point in pose.key_points:
                        text.append(str(key_point))

            streamer.send_data(frame, text)

            if streamer.check_exit():
                break

            fps.update()

        print('fps = {}'.format(fps.compute_fps()))


if __name__ == '__main__':
    object_detection()
    # pose_estimation()
