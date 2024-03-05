from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import cv2
from io import BytesIO
import requests
import numpy as np
import base64
import io
import time
import re
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__)

# YOLOv8 모델 로드
car_model = YOLO(r'C:\last_project\best.pt')  # 첫 번째 모델 경로
plate_model = YOLO(r'C:\last_project\bunho\bunhobest.pt')  # 두 번째 모델 경로

# 차량 클래스 레이블
class_labels = {0: 'light_morning', 1: 'light_ray', 2: 'car_grandeur', 3: 'car_zeep', 4: 'truck_poter', 5: 'van_starrex', 6: 'buildcar'}
# 번호판 클래스 레이블 
class_labels2 = {0: 'plate'}

# Azure OCR 설정
azure_endpoint = ""
subscription_key = ""
computervision_client = ComputerVisionClient(azure_endpoint, CognitiveServicesCredentials(subscription_key))


# OpenAI API 키 설정
api_key = ""


@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image part'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected for uploading'}), 400

        # 이미지 읽기
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)

        # OpenCV에서 사용할 수 있도록 BGR로 변환
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 첫 번째 YOLO 모델로 차량 식별 및 바운딩 박스 그리기
        car_results = car_model(image_np)
        car_pred = []
        for car_result in car_results:
           for box in car_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
                # 클래스 인덱스 가져오기
                class_index = int(box.cls.item())  # 클래스 인덱스를 스칼라로 변환하여 가져옵니다
        
                # 클래스 레이블 가져오기
                class_label = class_labels[class_index]
        
                # 바운딩 박스 그리기
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 클래스 이름을 이미지에 출력
                cv2.putText(image_np, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
                car_pred.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_label})  # 클래스 정보를 추가

        # 첫 번째 모델 결과 이미지를 바이트로 변환
        _, car_image_bytes = cv2.imencode('.png', image_np)
        car_image_bytes = car_image_bytes.tobytes()

        # 차량 타입 식별
        car_types = []
        for car_result in car_results:
            for box in car_result.boxes:
                class_index = int(box.cls.item())
                class_label = class_labels[class_index]
                car_types.append(class_label)

        # 두 번째 YOLO 모델로 번호판 식별 및 바운딩 박스 그리기
        plate_results = plate_model(image_np)
        plate_pred = []
        for plate_result in plate_results:
           for box in plate_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
                # 클래스 인덱스 가져오기
                class_index = int(box.cls.item())  # 클래스 인덱스를 스칼라로 변환하여 가져옵니다
        
                # 클래스 레이블 가져오기
                class_label = class_labels2[class_index]
        
                # 바운딩 박스 그리기
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # 클래스 이름을 이미지에 출력
                cv2.putText(image_np, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
                plate_pred.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_label})  # 클래스 정보를 추가

        # 두 번째 모델 결과 이미지를 바이트로 변환
        _, plate_image_bytes = cv2.imencode('.png', image_np)
        plate_image_bytes = plate_image_bytes.tobytes()

        # 크롭된 번호판 이미지 생성
        cropped_plate_images = []
        for plate_result in plate_results:
            for box in plate_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # 이미지를 크롭합니다.
                cropped_img_np = image_np[y1:y2, x1:x2]
                # 크롭된 이미지를 리스트에 추가합니다.
                cropped_plate_images.append(cropped_img_np)
        _, cropped_plate_images_bytes = cv2.imencode('.png', cropped_img_np)
        cropped_plate_images_bytes = cropped_plate_images_bytes.tobytes()


        # OCR 처리 # 바이트로 변환해야 ocr이 된다 
        def get_text_from_image(image_bytes):
            try:
                img_stream = BytesIO(image_bytes)
                recognize_handw_results = computervision_client.read_in_stream(img_stream, raw=True)
                operation_location = recognize_handw_results.headers["Operation-Location"]
                operation_id = operation_location.split("/")[-1]

                # "GET" API를 호출하여 결과를 가져올 때까지 대기합니다
                while True:
                    get_handw_text_results = computervision_client.get_read_result(operation_id)
                    if get_handw_text_results.status not in ['notStarted', 'running']:
                        break
                    time.sleep(1)

                # 결과를 반환합니다
                text_results = []
                if get_handw_text_results.status == OperationStatusCodes.succeeded:
                    for text_result in get_handw_text_results.analyze_result.read_results:
                        for line in text_result.lines:
                            text_results.append(line.text)
                return text_results
            except Exception as e:
                print(f"Error: {e}")
                return "인식실패"

        # # 각각의 자른 이미지를 처리합니다
        # for cropped_image in cropped_plate_images:
        #     _, img_in_bytes = cv2.imencode('.png', cropped_image)
        #     extracted_texts = get_text_from_image(img_in_bytes.tobytes())
        #     print(extracted_texts)  # 추출된 텍스트를 출력하거나 다른 처리를 수행합니다

        def extract_first_number_sequence(text):
            match = re.search(r'\d+', text)
            if match:
                return int(match.group())
            return None

        def classify_vehicle(number):
            if number is None:
                return "분류불가"

            if len(str(number)) == 2:  # 앞에 번호가 2자리인 경우
                if 1 <= number <= 69:
                    return "승용차"
                elif 70 <= number <= 79:
                    return "승합차"
                elif 80 <= number <= 97:
                    return "화물차"
                elif 98 <= number <= 99:
                    return "특수차"
            elif len(str(number)) == 3:  # 앞에 번호가 3자리인 경우
                if 100 <= number <= 699:
                    return "승용차"
                elif 700 <= number <= 799:
                    return "승합차"
                elif 800 <= number <= 979:
                    return "화물차"
                elif 980 <= number <= 997:
                    return "특수차"
            return "분류불가"

        for cropped_image in cropped_plate_images:
            _, img_in_bytes = cv2.imencode('.png', cropped_image)
            extracted_texts = get_text_from_image(img_in_bytes.tobytes())

            for text in extracted_texts:
                    number = extract_first_number_sequence(text)
                    vehicle_type = classify_vehicle(number)
                    print(f"Text: {text}, Number: {number}, Vehicle Type: {vehicle_type}")

        # gpt 처리
        # 이미지를 BGR에서 RGB로 변환
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # NumPy 배열을 바이트 형태로 인코딩
        image_byte = cv2.imencode('.jpg', image_np)[1].tobytes()

        # 인코딩된 바이트 데이터를 Base64로 변환
        base64_image = base64.b64encode(image_byte).decode('utf-8')

        # OpenAI에 보낼 헤더와 페이로드 설정
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
    
        payload = {
          "model": "gpt-4-vision-preview",
          "messages": [
             {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "어떤 차량인지 설명하고 이 사진에 차량 말고 탄소가 적게 나오는 차량 추천과 이 사진의 차량에 대한 탄소중립 방안을 한줄로 간단하게 알려줘"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                         }
                     }
                 ]
             }
          ],
          "max_tokens": 1000
        }
    
        # OpenAI에 요청을 보내고 응답을 받음
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        content = response.json()['choices'][0]['message']['content']

        def limit_width(text, width=300):
            lines = text.split('\n')
            new_lines = []
            for line in lines:
                if len(line) <= width:
                    new_lines.append(line)
                else:
                    words = line.split(' ')
                    new_line = ''
                    for word in words:
                        if len(new_line + ' ' + word) <= width:
                            new_line += ' ' + word
                        else:
                            new_lines.append(new_line)
                            new_line = word
            new_text = '\n'.join(new_lines)
            return new_text

        print(limit_width(content))

        new_text = limit_width(content)


        # 처리 결과를 응답 데이터로 구성
        response_data = {
        'car_image': base64.b64encode(car_image_bytes).decode('utf-8'),  # 첫 번째 모델로 처리된 차량 이미지
        'plate_image': base64.b64encode(plate_image_bytes).decode('utf-8'),  # 두 번째 모델로 처리된 번호판 이미지
        'enlarged_plate_images': base64.b64encode(cropped_plate_images_bytes).decode('utf-8'),  # 크롭 번호판
        'vehicle_type': car_types,  # 식별된 차량 타입들
        'ocr_text': extracted_texts,  # OCR로 추출된 텍스트
        'ocr_vehicle_type': vehicle_type,
        'g_print': new_text
          }

        # 처리 결과를 반환
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)