import cv2
import numpy as np


def preprocess_image(image):
    # Kontrast iyileştirme
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Gürültü azaltma
    denoised = cv2.fastNlMeansDenoisingColored(enhanced)

    return denoised


def verify_vehicle(contour):
    # Şekil analizi
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    solidity = float(contour_area) / hull_area

    # Dikdörtgensellik kontrolü
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h
    extent = float(contour_area) / rect_area

    return solidity > 0.8 and extent > 0.6


def detect_vehicles(image_path):
    # Görüntüyü oku
    image = cv2.imread(image_path)
    if image is None:
        print("Görüntü okunamadı!")
        return None

    # Görüntüyü kopyala (orijinali korumak için)
    output = image.copy()

    # Ön işleme
    processed = preprocess_image(image)

    # Gri seviyeye dönüştürme
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur uygula
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kenar tespiti
    edges = cv2.Canny(blur, 50, 150)

    # Morfolojik işlemler
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Kontur tespiti
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tespit edilen araç sayısı
    vehicle_count = 0

    # Araç tespiti
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:  # Minimum alan kontrolü
            if verify_vehicle(contour):
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                # Araç oranı kontrolü
                if 1.0 <= aspect_ratio <= 3.0:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    vehicle_count += 1

    # Tespit edilen araç sayısını görüntüye ekle
    cv2.putText(output, f'Tespit Edilen Arac: {vehicle_count}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return output


def main():
    # Görüntü yolu
    image_path = 'arac.jpeg'  # Kendi görüntünüzün yolunu belirtin

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
