from commons import get_model, transform_image

model = get_model()


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = model.forward(tensor)
    except Exception:
        return 0, 'error'
    _, y_hat = outputs.max(1)
    if str(y_hat.item()) == '1':
        return "Malignant"
    return "Benign"
