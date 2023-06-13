
import os

folder_path = r'D:\fypmodels\b\H_images\\'  # Replace with the actual path to your folder

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Update with the image file extensions you have
        new_filename = f"h_{filename}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)









# numeric_values = re.findall(r'\d+(\.\d+)?', text)

# print(numeric_values)





# prediction_groups = pipeline.recognize(images)

# predicted_image = prediction_groups[1]
# for text, box in predicted_image:
#     print(text)



































# from ultralytics import YOLO
# import numpy as np
# from PIL import Image
# # import requests
# from io import BytesIO
# import cv2
# import os







# def count_lines(file_path):
#     try:
#         files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
#         if len(files) == 0:
#             print("No .txt files found in the directory.")
#         else:
#             for file_name in files:
#                 with open(os.path.join(file_path, file_name), 'r') as file:
#                     lines = file.readlines()
#                     num_lines = len(lines)
#                     return num_lines
#                     # print("Number of lines in", file_name, ":", num_lines)
#     except FileNotFoundError:
#         print("Directory not found.")
#     except IOError:
#         print("Error reading the file.")

# # Prompt the user to enter the directory path
# # file_path = input("Enter the path to the directory containing .txt files: ")

# # count_lines(file_path)


# # Example usage
# file_path = 'D:/fypmodels/copy_mitwebb/runs/detect/predict/labels/'  # Replace with the actual path to your file
# line_count = count_lines(file_path)
# print("Number of lines in the file:", line_count)
# # boxes = output['boxes']
 
# from ultralytics import YOLO

# model = YOLO('best.pt')
# output=model.predict(
#    source='image1.jpg',
#    conf=0.25,
#     save=True,save_txt=True
# )







# # Count the number of boxes
# num_boxes = len(boxes)

# # Print the number of boxes
# print(num_boxes)



# import re
# import subprocess

# # Run model.predict() and capture the output
# result = subprocess.run(['python', '-c', 'from your_script import model; output = model.predict(source="2.jpg", conf=0.25, save=True); print(output)'], capture_output=True, text=True)

# # Get the captured output
# output_str = result.stdout

# # Extract the desired value using regular expressions
# pattern = r'(\d+)\smitotic-cells'
# match = re.search(pattern, output_str)

# if match:
#     temp = match.group(1)
#     print(temp)  # Output: 2
# else:
#     print("No match found.")





# output = model.predict(
#     source='2.jpg',
#     conf=0.25,
#     save=True
# )
# print(str(output))
# pattern = r'(\d+)\smitotic-cells'
# match = re.search(pattern, str(output))

# if match:
#     temp = match.group(1)
#     print(temp)  # Output: 2
# else:
#     print("No match found.")

# num_objects = len(model.xyxy[0])
# print(f"Detected {num_objects} objects.")

# output_str = str(results)


# print(output_str)


# output_str = "image 1/1 D:\fypmodels\copy_mitwebb\imgg1.jpg: 928x1024 2 mitotic-cells, 1350.3ms Speed: 28.0ms preprocess, 1350.3ms inference, 10.0ms postprocess per image at shape (1, 3, 1024, 1024) Results saved to runs\detect\predict4 <ultralytics.yolo.engine.model.YOLO object at 0x000001B3D3467D60>"
# print(output_str)

# output_list = output_str.split()
# print(output_list)

# if "(no detections)" in output_str:
#     mitotic_cells = "(no detections)"
# else:
#     if len(output_list) >= 8:
#         mitotic_cells = output_list[5] + " " + output_list[6]
#     else:
#         mitotic_cells = "Error: Insufficient output"

# print(mitotic_cells)


# img = Image.open(myImageFile)
#         print(type(img))
        
#         model_path = "latest_model.pt"
#         # image_path = "A06_00Ab.png"

#         device = torch.device("cpu")
#         model = torch.load(model_path, map_location=device)
#         input_image = img

#         # Apply the necessary transformations to the input image
#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.ToTensor(),
#         ])

#         input_tensor = transform(input_image).unsqueeze(0)

#         # Run the model to get the segmentation mask
#         with torch.no_grad():
#             output = model(input_tensor)

#         # Convert the logits tensor to a probability tensor
#         logits = output.logits
#         probs = torch.softmax(logits, dim=1)

#         # Get the predicted class labels
#         pred_labels = torch.argmax(probs, dim=1)

#         # Map the predicted class labels to RGB colors
#         palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
#         colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
#         colors = (colors % 255).numpy().astype("uint8")
#         rgb_labels = np.zeros((pred_labels.shape[1], pred_labels.shape[2], 3), dtype=np.uint8)
#         for i in range(21):
#             rgb_labels[pred_labels[0] == i] = colors[i]

#         # Save the segmented image
#         segmented_image = Image.fromarray(rgb_labels)
#         segmented_image.save("segmented_image.png")
#         img=segmented_image

#         img = img.resize((250, 250))
    
    