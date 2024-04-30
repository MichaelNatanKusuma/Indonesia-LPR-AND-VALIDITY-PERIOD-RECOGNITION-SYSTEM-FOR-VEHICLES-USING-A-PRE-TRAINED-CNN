# INSTALL PACKAGES
1.  !pip install ultralytics
2.  from IPython import display
3.  display.clear_output()
4.  !yolo mode=checks
5.  !pip install easyocr

#IMPORT LIBRARY
1.  from ultralytics import YOLO
2.  import IPython
3.  from IPython.display import display, Image
4.  from google.colab import drive
5.  import zipfile
6.  import os
7.  import cv2
8.  import easyocr
9.  from datetime import datetime
10.import re
11.from google.colab.patches import cv2_imshow

#GPU CHECK
12.!nvidia-smi drive.mount('/content/drive')

#IMPORTING FILE FROM GDRIVE
13.# Path to the zip file on Google Drive
14.zip_file_path = '/content/drive/My Drive/Colab Notebooks/indonesian lpr final project.zip'
15. 
16.# Destination folder for extracting the contents
17.extract_folder_path = '/content/'
18. 
19.# Extract the contents of the zip file
20.with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
21.zip_ref.extractall(extract_folder_path)

#TRAINING
22.!yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=5

#TESTING
23.!yolo task=detect mode=predict  model=runs/detect/train/weights/best.pt source=/content/putih

#PLATE READING
24.# Load a model
25.model = YOLO('uns/detect/train/weights/best.pt')  # pretrained YOLOv8n model
26. 
27.# Specify the folder containing images
28.folder_path = '/content/putih'  # Change this to your folder name
29. 
30.# Get a list of image files in the folder
31.image_files = [file for file in os.listdir(folder_path) if file.endswith('.png') or file.endswith('.jpg')]
32. 
33.# Process each image in the folder
34.for image_file in image_files:
35.	# Initialize flags before processing each image
36.	text_detected = False
37.	date_detected = False
38. 
39.	# Construct the full path to the image
40.	image_path = os.path.join(folder_path, image_file)
41.	image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extracting the image name without the extension
42. 
43.	# Run batched inference on the current image
44.	results = model([image_path])
45. 
46.	# Process results list
47.	for result in results:
48.    	# Extract bounding box coordinates for each detection in the current image
49.    	for detection in result.boxes[0]:
50.        	xyxy_coordinates = detection.xyxy.cpu().numpy()
51.        	output_path = f'{image_name}.txt'
52. 
53.        	# Save xyxy coordinates to the text file
54.        	with open(output_path, 'w') as file:
55.            	for coordinates in xyxy_coordinates.tolist():
56.                	file.write(' '.join(map(str, coordinates)) + '\n')
57. 
58.        	# Open the image
59.        	image = cv2.imread(image_path)
60. 
61.        	# Extract coordinates
62.        	x1, y1, x2, y2 = map(int, coordinates)
63. 
64.        	# Crop the region using coordinates
65.        	cropped_region = image[y1:y2, x1:x2]
66. 
67.        	# Display the cropped region (for visualization purposes)
68.        	cv2_imshow(cropped_region)
69. 
70.        	# Initialize EasyOCR reader
71.        	reader = easyocr.Reader(['en'])
72. 
73.        	# Read text from the cropped region
74.        	results_ocr = reader.readtext(cropped_region)
75. 
76.        	# Print and handle text detection
77.        	for result_ocr in results_ocr:
78.            	detected_text = result_ocr[1]
79.            	print(f"Detected text: {detected_text}")
80.            	text_detected = True
81. 
82.            	# Check if the detected text contains a date in the format mm.yy
83.            	date_pattern = r"\b\d{2}\.\d{2}\b"
84.            	date_match = re.search(date_pattern, detected_text)
85. 
86.            	if date_match:
87.                	try:
88.                    	# Convert the matched string to a datetime object
89.                    	date_str = date_match.group()
90.                    	date_obj = datetime.strptime(date_str, "%m.%y")
91. 
92.                    	# Extract the month and year
93.                        month = date_obj.month
94.                    	year = date_obj.year
95. 
96.                    	print(f"Detected date: {month}/{year}")
97. 
98.                    	# Check if the detected date is still valid
99.                    	current_date = datetime.now()
100.                            if date_obj > current_date:
101.                                print("Detected date is still valid.")
102.                            else:
103.                                print("Detected date has expired.")
104.     
105.                            date_detected = True
106.     
107.                        except ValueError as e:
108.                            # Handle the case where the date string does not match the format
109.                            print(f"Error processing date: {e}")
110.     
111.                # Check if no text or date is detected
112.                if not text_detected:
113.                    print("No text detected.")
114.                if not date_detected:
115.                    print("No date detected.")

#PLATE READING WITH SHARP FILTER
116.    import numpy as np
117.     
118.    # Load a model
119.    model = YOLO('/content/best(1).pt')  # pretrained YOLOv8n model
120.     
121.    # Specify the folder containing images
122.    folder_path = '/content/putih'  # Change this to your folder name
123.     
124.    # Get a list of image files in the folder
125.    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png') or file.endswith('.jpg')]
126.     
127.    # Process each image in the folder
128.    for image_file in image_files:
129.        # Initialize flags before processing each image
130.        text_detected = False
131.        date_detected = False
132.     
133.        # Construct the full path to the image
134.        image_path = os.path.join(folder_path, image_file)
135.        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extracting the image name without the extension
136.     
137.        # Open the image
138.        original_image = cv2.imread(image_path)
139.     
140.        # Apply sharpening filter
141.        sharpening_filter = np.array([[-1, -1, -1],
142.                                      [-1, 9, -1],
143.                                      [-1, -1, -1]])
144.        sharpened_image = cv2.filter2D(original_image, -1, sharpening_filter)
145.     
146.        # Run batched inference on the sharpened image
147.        results = model([sharpened_image])
148.     
149.        # Process results list
150.        for result in results:
151.            # Extract bounding box coordinates for each detection in the current image
152.            for detection in result.boxes[0]:
153.                xyxy_coordinates = detection.xyxy.cpu().numpy()
154.                output_path = f'{image_name}.txt'
155.     
156.                # Save xyxy coordinates to the text file
157.                with open(output_path, 'w') as file:
158.                    for coordinates in xyxy_coordinates.tolist():
159.                        file.write(' '.join(map(str, coordinates)) + '\n')
160.     
161.                # Display the original and sharpened images
162.                cv2_imshow(original_image)
163.                cv2_imshow(sharpened_image)
164.     
165.                # Initialize EasyOCR reader
166.                reader = easyocr.Reader(['en'])
167.     
168.                # Read text from the cropped region
169.                results_ocr = reader.readtext(sharpened_image)
170.     
171.                # Print and handle text detection
172.                for result_ocr in results_ocr:
173.                    detected_text = result_ocr[1]
174.                    print(f"Detected text: {detected_text}")
175.                    text_detected = True
176.     
177.                    # Check if the detected text contains a date in the format mm.yy
178.                    date_pattern = r"\b\d{2}\.\d{2}\b"
179.                    date_match = re.search(date_pattern, detected_text)
180.     
181.                    if date_match:
182.                        try:
183.                            # Convert the matched string to a datetime object
184.                            date_str = date_match.group()
185.                            date_obj = datetime.strptime(date_str, "%m.%y")
186.     
187.                            # Extract the month and year
188.                            month = date_obj.month
189.                            year = date_obj.year
190.     
191.                            print(f"Detected date: {month}/{year}")
192.     
193.                            # Check if the detected date is still valid
194.                            current_date = datetime.now()
195.                            if date_obj > current_date:
196.                                print("Detected date is still valid.")
197.                            else:
198.                                print("Detected date has expired.")
199.     
200.                            date_detected = True
201.     
202.                        except ValueError as e:
203.                            # Handle the case where the date string does not match the format
204.                            print(f"Error processing date: {e}")
205.     
206.                # Check if no text or date is detected
207.                if not text_detected:
208.                    print("No text detected.")
209.                if not date_detected:
210.                    print("No date detected.")

#PLATE READING WITH SHARP + GAUSSIAN BLUR FILTER
211.    import numpy as np
212.     
213.    # Load a model
214.    model = YOLO('/content/best(1).pt')  # pretrained YOLOv8n model
215.     
216.    # Specify the folder containing images
217.    folder_path = '/content/putih'  # Change this to your folder name
218.     
219.    # Get a list of image files in the folder
220.    image_files = [file for file in os.listdir(folder_path) if file.endswith('.png') or file.endswith('.jpg')]
221.     
222.    # Process each image in the folder
223.    for image_file in image_files:
224.        # Initialize flags before processing each image
225.        text_detected = False
226.        date_detected = False
227.     
228.        # Construct the full path to the image
229.        image_path = os.path.join(folder_path, image_file)
230.        image_name = os.path.splitext(os.path.basename(image_path))[0]  # Extracting the image name without the extension
231.     
232.        # Open the image
233.        original_image = cv2.imread(image_path)
234.     
235.        # Apply sharpening filter
236.        sharpening_filter = np.array([[-1, -1, -1],
237.                                      [-1, 9, -1],
238.                                      [-1, -1, -1]])
239.        sharpened_image = cv2.filter2D(original_image, -1, sharpening_filter)
240.     
241.        # Apply Gaussian blur to the sharpened image
242.        blurred_image = cv2.GaussianBlur(sharpened_image, (3, 3), 0)
243.     
244.        # Run batched inference on the blurred image
245.        results = model([blurred_image])
246.     
247.        # Process results list
248.        for result in results:
249.            # Extract bounding box coordinates for each detection in the current image
250.            for detection in result.boxes[0]:
251.                xyxy_coordinates = detection.xyxy.cpu().numpy()
252.                output_path = f'{image_name}.txt'
253.     
254.                # Save xyxy coordinates to the text file
255.                with open(output_path, 'w') as file:
256.                    for coordinates in xyxy_coordinates.tolist():
257.                        file.write(' '.join(map(str, coordinates)) + '\n')
258.     
259.                # Display the original, sharpened, blurred, and final images
260.                cv2_imshow(original_image)
261.                cv2_imshow(sharpened_image)
262.                cv2_imshow(blurred_image)
263.     
264.                # Initialize EasyOCR reader
265.                reader = easyocr.Reader(['en'])
266.     
267.                # Read text from the blurred image
268.                results_ocr = reader.readtext(blurred_image)
269.     
270.                # Print and handle text detection
271.                for result_ocr in results_ocr:
272.                    detected_text = result_ocr[1]
273.                    print(f"Detected text: {detected_text}")
274.                    text_detected = True
275.     
276.                    # Check if the detected text contains a date in the format mm.yy
277.                    date_pattern = r"\b\d{2}\.\d{2}\b"
278.                    date_match = re.search(date_pattern, detected_text)
279.     
280.                    if date_match:
281.                        try:
282.                            # Convert the matched string to a datetime object
283.                            date_str = date_match.group()
284.                            date_obj = datetime.strptime(date_str, "%m.%y")
285.     
286.                            # Extract the month and year
287.                            month = date_obj.month
288.                            year = date_obj.year
289.     
290.                            print(f"Detected date: {month}/{year}")
291.     
292.                            # Check if the detected date is still valid
293.                            current_date = datetime.now()
294.                            if date_obj > current_date:
295.                                print("Detected date is still valid.")
296.                            else:
297.                                print("Detected date has expired.")
298.     
299.                            date_detected = True
300.     
301.                        except ValueError as e:
302.                            # Handle the case where the date string does not match the format
303.                            print(f"Error processing date: {e}")
304.     
305.                # Check if no text or date is detected
306.                if not text_detected:
307.                    print("No text detected.")
308.                if not date_detected:
309.                    print("No date detected.")
