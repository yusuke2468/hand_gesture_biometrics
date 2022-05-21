import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# mp4ファイルを手が映った場面ごとに分割し、それぞれmp4ファイルとして保存
def split_mp4_file(file_path = "video.mp4", previous_margin=0, back_margin=0, ignore_frame=10, video_id=1):
    frame_index = 0
    frame_index_list = []
    landmark_arrays = np.empty((0, 42))

    cap = cv2.VideoCapture(file_path)

    # MediaPipe Hands を使って手が認識できるフレームを抽出
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            
            image.flags.writeable = False
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                frame_index_list.append(frame_index)
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_array = np.array([[hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y]])

                    landmark_arrays = np.append(landmark_arrays, landmark_array, axis=0)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            frame_index+=1
    cap.release()
    cv2.destroyAllWindows()
    print(type(landmark_arrays))
    print(landmark_arrays.shape)
    print(landmark_arrays)


    # 動画を保存するための設定
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))             # カメラの縦幅を取得
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')         # 動画保存時のfourcc設定（mp4用）
    video = cv2.VideoWriter(str(video_id) + '.mp4', fourcc, fps, (w, h))

    # 一つ前のフレーム
    old_frame_index = -100

    # 手の認識に失敗するフレームがあるため、連続で失敗した時のみハンドジェスチャーが終了したと判断する
    # 手が見切れている場合は上手く認識できないため、前後に設定した数だけフレームを追加し、ハンドジェスチャーの途中で
    # 動画が切れないようにする
    for index in frame_index_list:
        if index - old_frame_index == 1:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            video.write(image)
        elif (index-old_frame_index >= 2) and (index-old_frame_index <= ignore_frame):
            for i in range(index-old_frame_index):
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break
                video.write(image)
            # print(index-old_frame_index)
        else:
            if old_frame_index != -100:
                for i in range(back_margin):
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        break
                    video.write(image)
                # print(index, "back_margin!!!")

            cap.set(cv2.CAP_PROP_POS_FRAMES, index-previous_margin)
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            video = cv2.VideoWriter(str(video_id) + '.mp4', fourcc, fps, (w, h))
            video.write(image)

            for i in range(previous_margin):
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    break
                video.write(image)
            # print(index, "previous_margin!!!")

            video_id+=1
        old_frame_index = index
    
    for i in range(back_margin):
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        video.write(image)
    # print(index, "back_margin!!!")

    video.release()
    cap.release()
    cv2.destroyAllWindows()
    print("done!")

    return video_id

if __name__ == '__main__':
    video_id = 1
    for i in range(10):
        video_id = split_mp4_file(file_path="./data/{}_full-1.mp4".format(i+1), previous_margin=10, back_margin=10, ignore_frame=20, video_id=video_id)
