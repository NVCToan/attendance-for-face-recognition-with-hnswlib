import tkinter as tk
import cv2
import os
import dlib
import imutils
import face_recognition
import hnswlib
import time
import numpy as np
import matplotlib.pyplot as plt
from tkinter import font as tkfont
from tkinter import messagebox,PhotoImage
from imutils import face_utils,paths
from imutils.face_utils import FaceAligner
from constant import IMAGE_SIZE, MAX_NUMBER_OF_IMAGES,DIM, NUM_ELEMENTS, IMAGE_SIZE, EF_CONSTRUCTION, DIMENTIONAL, M, FX, FY
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from collections import Counter
from tkinter import *
from tkmacosx import Button
from PIL import ImageTk, Image
# from gender_prediction import emotion,ageAndgender
mssvs = set()

class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global mssvs
        with open("mssvslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                mssvs.add(i)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Face Recognizer")
        self.resizable(False, False)
        self.geometry("500x300")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_mssv = None
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
            frame = self.frames[page_name]
            frame.tkraise()

    def on_closing(self):

        if messagebox.askokcancel("Quit", "Are you sure?"):
            global mssvs
            f =  open("mssvslist.txt", "a+")
            for i in mssvs:
                    f.write(i+" ")
            self.destroy()


class StartPage(tk.Frame):

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            self.controller = controller
            load = Image.open("homepagepic.png")
            load = load.resize((250, 250), Image.ANTIALIAS)

            render = PhotoImage(file='homepagepic.png')
            img = tk.Label(self, image=render)
            img.image = render
            img.grid(row=0, column=1, rowspan=5, sticky="nsew")
            
            #    print(mssvs)
            label = tk.Label(self, text="   Class Attendance   ", width=15, height=1, font=self.controller.title_font,fg="#263942")
            label.grid(row=0, sticky="ew")
            button1 = tk.Button(self, text="Add a User", width=15, height=1, fg="#ffffff", bg="#263942", command=lambda: self.controller.show_frame("PageOne"))
            trainbutton = tk.Button(self, text="Train The Model", width=15, height=1, fg="#ffffff", bg="#263942", command=self.trainmodel)
            button2 = tk.Button(self, text="ATTENDANCE", width=15, height=1, fg="#ffffff", bg="green", command=self.openwebcam)
            button3 =tk.Button(self, text="QUIT", width=15, height=1, fg="#263942", bg="red", command=self.on_closing)
            button1.grid(row=2, column=0, ipady=3, ipadx=2)
            button2.grid(row=1, column=0, ipady=3, ipadx=2)
            # button2.place(relx=0.6,rely=0.5,anchor=CENTER)
            trainbutton.grid(row=3, column=0, ipady=3, ipadx=2)
            button3.grid(row=4, column=0, ipady=3, ipadx=2)

        def openwebcam(self):
            output_mssv = []
            known_face_mssvs = []
            video_capture = cv2.VideoCapture(0)
            p = hnswlib.Index(space='l2', dim=DIM)  # the space can be changed - keeps the data, alters the distance function.
            p.load_index("images.bin", max_elements = NUM_ELEMENTS)
            imagePaths = list(paths.list_images('images'))

            for i, imagePath in enumerate(imagePaths):
                mssv = imagePath.split(os.path.sep)[-2]
                known_face_mssvs.append(mssv)
            
            def append_mssvs(frame): 
                frame = cv2.flip(frame, 1)
                
                small_frame = cv2.resize(frame, (0, 0), fx=FX, fy=FY)
                rgb_small_frame = small_frame[:, :, ::-1]

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_mssvs = []
                
                for face_encoding in face_encodings:
                    labels, distances = p.knn_query(np.expand_dims(face_encoding, axis = 0), k = 1)
                    known_face_encoding = p.get_items([labels])
                    mssv = "unknown"
                    if distances < 0.13:
                        mssv = known_face_mssvs[labels[0][0]]
                    face_mssvs.append(mssv)
                    
                #Draw rectangle in faces 
                
                for (top, right, bottom, left), mssv in zip(face_locations, face_mssvs):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    if (mssv=="unknown"):
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, mssv, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    #Device frame
            #        left = np.zeros((frame.shape[0], 400, frame.shape[2]), dtype=frame.dtype)
            #        cv2.putText(left, "Some information", (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            #        cv2.putText(left, "More information", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            #        cv2.putText(left, "Some more information", (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            #        img = np.hstack((left,frame))

                    # right = np.zeros((frame.shape[0], 400, frame.shape[2]), dtype=frame.dtype)
                    # cv2.putText(left, "Some information", (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    # cv2.putText(left, "More information", (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    # cv2.putText(left, "Some more information", (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    # img = np.hstack((left,frame))
                    #Device frame
                    #resize frame
                    scale_percent = 65 # percent of original size
                    width = int(frame.shape[1] * scale_percent / 100)
                    height = int(frame.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    # dim = (900, 530)
                    imResize = cv2.resize(frame,dsize = dim ,fx=-100,fy=-100, interpolation = cv2.INTER_AREA )
                    #resize frame  
                    cv2.imshow('Video', imResize)
                    
                return face_mssvs


            while True:
                ret, frame = video_capture.read()
                #Test
                mssvs =  append_mssvs(frame)
                if(cv2.waitKey(2) == 27 ):
                    break
            #    print(mssvs)
            # cap.release()
            cv2.destroyAllWindows()
        def trainmodel(self):
            p = hnswlib.Index(space = 'l2', dim = DIM) 
            p.init_index(max_elements = NUM_ELEMENTS, ef_construction = EF_CONSTRUCTION, M = M)
            imagePaths = list(paths.list_images('images'))
            detector = MTCNN()

            def check_image_path(imagePath):
                img = face_recognition.load_image_file(imagePath)
                try:
                    #Check image
                    coodirnate = detector.detect_faces(img)[0]['box']
                except IndexError:
                    coodirnate = []
                return coodirnate, img

            def image_encoding(coodirnate, img):
                x, y, w, h = [v for v in coodirnate]
                x2, y2 = x + w, y + h
                face = img[y:y+h, x:x+w]
                img_emb = face_recognition.face_encodings(face)
                return img_emb

            for i, imagePath in tqdm(enumerate(imagePaths)):
                coodirnate, img = check_image_path(imagePath)
                if img.shape == (IMAGE_SIZE, IMAGE_SIZE, DIMENTIONAL):
                    img_emb  = face_recognition.face_encodings(img)
                    if len(img_emb) == 0:
                        pass
                    else:
                        p.add_items(np.expand_dims(img_emb[0], axis = 0), i)
                else:    
                    if len(coodirnate) == 0:
                        pass
                    else:
                        img_emb = image_encoding(tuple(coodirnate), img)
                        if len(img_emb) == 0:
                            pass
                        else:
                            p.add_items(np.expand_dims(img_emb[0], axis = 0), i)

            index_path='images.bin'

            print("Saving index to '%s'" % index_path)
            print("Completed!!")
            p.save_index("images.bin")
            labelcomplete = tk.Label(self, text="Training completed!!", font=self.controller.title_font,fg="green")
            labelcomplete.grid(row=5,column=1)

            del p
        def on_closing(self):
            if messagebox.askokcancel("Quit", "Are you sure?"):
                global mssvs
                with open("mssvslist.txt", "w") as f:
                    for i in mssvs:
                        f.write(i + " ")
                self.controller.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text="MSSV", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, pady=10, padx=5)
        tk.Label(self, text="Tên", fg="#263942", font='Helvetica 12 bold').grid(row=1, column=0, pady=10, padx=5)
        tk.Label(self, text="Lớp", fg="#263942", font='Helvetica 12 bold').grid(row=2, column=0, pady=10, padx=5)

        self.user_mssv = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_ten = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_lop = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')

        self.user_mssv.grid(row=0, column=1, pady=10, padx=10)
        self.user_ten.grid(row=1, column=1, pady=10, padx=10)
        self.user_lop.grid(row=2, column=1, pady=10, padx=10)
              
        self.buttonnext = tk.Button(self, text="Next", fg="#ffffff", bg="#263942", command=self.getdata)
        self.buttoncancel = tk.Button(self, text="Cancel", bg="#ffffff", fg="#263942", command=lambda: controller.show_frame("StartPage"))
        self.buttonnext.grid(row=3, column=0, pady=10, ipadx=5, ipady=4)
        self.buttoncancel.grid(row=3, column=1, pady=10, ipadx=5, ipady=4)
        
        
    def getdata(self):
        global mssvs
        if self.user_mssv.get() == "None":
            messagebox.showerror("Error", "MSSV cannot be 'None'")
            return
        elif self.user_mssv.get() in mssvs:
            messagebox.showerror("Error", "MSSV already exists!")
            return
        elif len(self.user_mssv.get()) == 0:
            messagebox.showerror("Error", "MSSV cannot be empty!")
            return
        mssv = self.user_mssv.get()
        mssvs.add(mssv)
        self.controller.active_mssv = mssv
        # self.controller.frames["PageTwo"].refresh_mssvs()
        # self.controller.show_frame("PageThree")
        #source hnsw
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_aligner = FaceAligner(shape_predictor, desiredFaceWidth=IMAGE_SIZE)
        video_capture = cv2.VideoCapture(0)
        # mssv = input("Enter mssv of person:")

        path = 'images'
        directory = os.path.join(path, mssv)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok = 'True')

        number_of_images = 0
        

        while number_of_images < MAX_NUMBER_OF_IMAGES:
            ret, frame = video_capture.read()

            # frame = cv2.flip(frame, 1)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            faces = detector(frame_gray)
            if len(faces) == 1:
                face = faces[0]
                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                face_img = frame_gray[y-50:y + h+100, x-50:x + w+100]
                face_aligned = face_aligner.align(frame, frame_gray, face)
                cv2.imwrite(os.path.join(directory, str(mssv+str(number_of_images)+'.jpg')), face_aligned)
                number_of_images += 1
                
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), cv2.FILLED)

            cv2.putText(frame, str(str(number_of_images)+ " images captured"), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

            #resize frame           
            scale_percent = 65 # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            # dim = (900, 530)
            imResize = cv2.resize(frame,dsize = dim ,fx=-100,fy=-100, interpolation = cv2.INTER_AREA )
            #resize frame  
            cv2.imshow('Video', imResize)

            if(cv2.waitKey(2) == 27 ):
                break
                        
        video_capture.release()
        cv2.destroyAllWindows()



app = MainUI()

#Chen hinh anh

app.iconphoto(False, tk.PhotoImage(file='icon.ico'))
app.mainloop()

