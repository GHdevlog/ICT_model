import os
import cv2
import dlib

def process_single_image(image_path, crop_folder, class_id):
    try:
        # 이미지 읽기
        frame = cv2.imread(image_path)

        # dlib의 강아지 얼굴 탐지 모델 로드
        detector = dlib.cnn_face_detection_model_v1('preprocessing/dogHeadDetector.dat')

        # 강아지 얼굴 탐지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector(gray, 1)

        # 얼굴이 검출된 경우만 저장
        if detections:
            orig_height, orig_width = frame.shape[:2]

            for i, d in enumerate(detections):
                left, top, right, bottom, _ = (d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence)

                # 좌표가 이미지 범위를 벗어나지 않도록 조정
                left = max(0, left)
                top = max(0, top)
                right = min(orig_width, right)
                bottom = min(orig_height, bottom)

                # 박스에 맞게 이미지 크롭
                cropped_img = frame[top:bottom, left:right]

                # 크롭된 이미지가 유효한지 확인
                if cropped_img.size == 0:
                    print(f"Warning: Cropped image is empty. Skipping this detection.")
                    continue

                # 크롭된 이미지 리사이즈 (224x224)
                resized_cropped_img = cv2.resize(cropped_img, (224, 224))

                # 개별 클래스 폴더 생성
                class_folder = os.path.join(crop_folder, str(class_id))
                os.makedirs(class_folder, exist_ok=True)

                # 크롭된 이미지 저장 경로 설정
                crop_image_name = f'{os.path.basename(image_path)}'
                crop_image_path = os.path.join(class_folder, crop_image_name)

                # 크롭된 이미지 저장
                cv2.imwrite(crop_image_path, resized_cropped_img)
                print(f"Cropped and resized image saved as {crop_image_path}")

                return crop_image_path

    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)
    except RuntimeError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None

def extract_class_id_from_filename(filename):
    """확장자를 제거하고 파일명에서 숫자를 추출합니다."""
    basename = os.path.splitext(filename)[0]
    numbers = ''.join(filter(str.isdigit, basename))
    return int(numbers) if numbers else 0

# 전처리 폴더 경로 설정
crop_folder = 'cropped_image'
class_id = 11


for i in range(1,8):
    image_path = f'seonghyeon_dogs/img/dog11 ({i}).jpg'
    processed_image_path = process_single_image(image_path, crop_folder, class_id)
