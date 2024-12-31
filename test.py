import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model

# Device configuration
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# Image transformations
test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def get_model(num_classes):
    model = create_model('tf_efficientnetv2_b0', pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)

def predict(model, image, classes):
    model.eval()
    with torch.no_grad():
        image = test_transforms(image).unsqueeze(0).to(DEVICE)
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_index = predicted.item()
        predicted_class = classes[class_index]
        return predicted_class

def main():
    # Kamera veya video dosyasını kullanmak için bir video yakalama nesnesi oluşturun
    cap = cv2.VideoCapture(0)  # Kamera için 0, video dosyası için dosya yolunu kullanın

    # Meyve ve sebze sınıf etiketlerini belirtin
    classes = ['Banana', 'Apple', 'Pear', 'Grapes', 'Orange', 'Kiwi', 'Watermelon', 'Pomegranate', 'Pineapple', 'Mango',
               'Cucumber', 'Carrot', 'Capsicum', 'Onion', 'Potato', 'Lemon', 'Tomato', 'Radish', 'Beetroot', 'Cabbage',
               'Lettuce', 'Spinach', 'Soybean', 'Cauliflower', 'Bell Pepper', 'Chilli Pepper', 'Turnip', 'Corn',
               'Sweetcorn', 'Sweet Potato', 'Paprika', 'Jalapeño', 'Ginger', 'Garlic', 'Peas', 'Eggplant']

    # Modeli yükle
    num_classes = len(classes)  # Sınıf sayısını al
    model = get_model(num_classes)  # Modeli doğru sınıf sayısıyla oluştur
    model.load_state_dict(torch.load('best_model.pth'))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Görüntüyü modele uygun hale getirin
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tahmin yapın
        predicted_class = predict(model, image, classes)

        # Sonucu ekranda gösterin
        cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Fruit and Vegetable Recognition', frame)

        # 'q' tuşuna basıldığında döngüden çıkın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()