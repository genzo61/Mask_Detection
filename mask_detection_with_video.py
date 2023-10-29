import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    # frame = cv2.flip()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB = True, crop = False)

    labels = ["good","bad"]

    colors = ["0,255,0","255,0,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(15,1))

    model = cv2.dnn.readNetFromDarknet("yolov3_mask.cfg","yolov3_mask_last.weights")
    layers = model.getLayerNames()
    output_layer = [layers[layer-1] for layer in model.getUnconnectedOutLayers()]

    model.setInput(frame_blob)

    detection_layers = model.forward(output_layer)

########### non maximum  operation 1 #############
    id_list = []
    boxes_list = []
    confidance_list = []

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidance = scores[predicted_id]

            if confidance > 0.30:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_X = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))

            ########### non maximum operation 2 ############

                id_list.append(predicted_id)
                confidance_list.append(float(confidance))
                boxes_list.append([start_X,start_y,int(box_width),int(box_height)])

########### non maximum operation 3 #############
    max_ids = cv2.dnn.NMSBoxes(boxes_list,confidance_list, 0.5, 0.4)

    for i in max_ids:
        max_class_id = i

        box = boxes_list[max_class_id]

        start_X = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]


        predicted_id = id_list[max_class_id]
        label = labels[predicted_id]
        confidance = confidance_list[max_class_id]


        end_X = start_X + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each  in box_color]


        label = "{}: {:.2f}%".format(label, confidance * 100)
        print("predicted object {}".format(label))


        cv2.rectangle(frame,(start_X,start_y),(end_X,end_y),box_color,1)
        cv2.putText(frame,label,(start_X,start_y-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, box_color, 1)

    cv2.imshow("detection window",frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
cap.release()    
cv2.destroyAllWindows()