import cv2
import numpy as np


def preprocess_image(image):
    # Optimize edilmiş kontrast ve parlaklık ayarları
    alpha = 1.2
    beta = 10
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Optimize edilmiş gürültü azaltma
    denoised = cv2.fastNlMeansDenoisingColored(adjusted, None, 5, 5, 7, 21)

    return denoised


def create_roi_mask(image):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Genişletilmiş ROI bölgesi
    roi_points = np.array([
        [(0, height), (0, height // 3),
         (width, height // 3), (width, height)]], dtype=np.int32)

    cv2.fillPoly(mask, roi_points, 255)
    return mask


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    return intersection / float(box1_area + box2_area - intersection)


def merge_overlapping_boxes(boxes, threshold=0.2):
    if not boxes:
        return []

    merged_boxes = []
    while boxes:
        base_box = boxes.pop(0)
        boxes_to_merge = []

        i = 0
        while i < len(boxes):
            if compute_iou(base_box, boxes[i]) > threshold:
                boxes_to_merge.append(boxes.pop(i))
            else:
                i += 1

        for box in boxes_to_merge:
            x = min(base_box[0], box[0])
            y = min(base_box[1], box[1])
            w = max(base_box[0] + base_box[2], box[0] + box[2]) - x
            h = max(base_box[1] + base_box[3], box[1] + box[3]) - y
            base_box = (x, y, w, h)

        merged_boxes.append(base_box)

    return merged_boxes


def verify_vehicle(contour):
    # Optimize edilmiş doğrulama kriterleri
    solidity_threshold = 0.5
    extent_threshold = 0.4
    min_width = 30
    min_height = 25
    aspect_ratio_range = (0.5, 4.0)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)

    if hull_area == 0:
        return False

    solidity = float(contour_area) / hull_area

    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h

    if rect_area == 0:
        return False

    extent = float(contour_area) / rect_area
    aspect_ratio = float(w) / h

    return (solidity > solidity_threshold and
            extent > extent_threshold and
            w > min_width and
            h > min_height and
            aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1])


def detect_vehicles(image_path):
    # Görüntüyü oku
    image = cv2.imread(image_path)
    if image is None:
        print("Görüntü okunamadı!")
        return None

    # Görüntüyü kopyala
    output = image.copy()

    # ROI maskesi oluştur
    roi_mask = create_roi_mask(image)

    # Genişletilmiş ölçek aralığı
    scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    all_boxes = []

    for scale in scales:
        # Görüntüyü yeniden boyutlandır
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        scaled_image = cv2.resize(image, dim)
        scaled_mask = cv2.resize(roi_mask, dim)

        # Ön işleme
        processed = preprocess_image(scaled_image)

        # Gri seviyeye dönüştürme
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # ROI uygula
        gray = cv2.bitwise_and(gray, gray, mask=scaled_mask)

        # Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptif eşikleme
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Kenar tespiti - parametreler optimize edildi
        edges = cv2.Canny(blur, 30, 150)

        # Morfolojik işlemler
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Kontur tespiti
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Ölçek için kutuları ayarla
        scale_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Düşürülmüş minimum alan eşiği
                if verify_vehicle(contour):
                    x, y, w, h = cv2.boundingRect(contour)
                    # Kutuları orijinal boyuta çevir
                    x = int(x / scale)
                    y = int(y / scale)
                    w = int(w / scale)
                    h = int(h / scale)
                    scale_boxes.append((x, y, w, h))

        all_boxes.extend(scale_boxes)

    # Kesişen kutuları birleştir
    final_boxes = merge_overlapping_boxes(all_boxes)

    # Tespit edilen araç sayısı
    vehicle_count = len(final_boxes)

    # Kutuları çiz
    for box in final_boxes:
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tespit edilen araç sayısını görüntüye ekle
    cv2.putText(output, f'Tespit Edilen Arac: {vehicle_count}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return output


def main():
    # Görüntü yolu
    image_path = '/Users/erolatik/Desktop/arac1.jpeg'  # Kendi görüntünüzün yolunu belirtin

    # Araç tespiti yap
    result = detect_vehicles(image_path)

    if result is not None:
        # Sonucu göster
        cv2.imshow('Arac Tespiti', result)

        # Sonucu kaydet
        cv2.imwrite('sonuc.jpg', result)

        # Pencereyi kapatmak için bir tuşa basılmasını bekle
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
