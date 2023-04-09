import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time

class Video(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fps=int(self.cap.get(cv2.CAP_PROP_FPS))
        self.start_time = time.time() 
        # cv2.namedWindow("tracking")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.peaks=0
        self.elapsed_time=0
        self.breath_rate=0
        self.prev_frame=0
        self.gsums=[]
        self.idf=0
        self.m_diff = 0
        self.bbox=[0, 0, 10, 10]
        
    def __del__(self):
        self.cap.release()
    def get_frame(self):
        x1 = 0.4
        x2 = 0.6
        y1 = 0.1
        y2 = 0.25
        face_cascade=self.face_cascade

        def getChestROI(img):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            chest = face_cascade.detectMultiScale(gray, 1.2, 5)

            if len(chest) > 0:
                img = cv2.rectangle(img, (chest[0][0] + int(x1*chest[0][2]), chest[0][1] + int(y1*chest[0][3])),
                                    (chest[0][0] + int(x2*chest[0][2]), chest[0][1] + int(y2*chest[0][3])), (255, 0, 0), 2)

                return [chest[0][0] + int(x1*chest[0][2]),  chest[0][1] + int(y1*chest[0][3]),
                        chest[0][0] + int(x2*chest[0][2]), chest[0][1] + int(y2*chest[0][3])]
            else:
                return [0, 0, 0, 0]

        def getColorSum(frame, color_id):
            return frame[:, :, color_id].sum()

        def getColorAverage(frame, color_id):
            return frame[:, :, color_id].sum() * 1.0 / (frame.shape[0] * frame.shape[1])

        # idf = self.idf
        rsums = []
        # gsums = self.gsums
        bsums = []
        # breath_rate=self.breath_rate
        bbox = self.bbox
        m_diff = self.m_diff
        cap=self.cap
        # cv2.namedWindow("tracking")
        fps=self.fps
        step = int(1000 / fps)
        start_time = self.start_time
        elapsed_time = self.elapsed_time
        prev_frame=self.prev_frame
        # while elapsed_time<=30:
        # ans=[]
        while True:
            ret, frame = cap.read()
            self.elapsed_time = time.time() - start_time
            # ans=[]
            # cv2.putText(frame, f"No of Breaths in 30 seconds {breath_rate}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            # if(elapsed_time>12):
            #     break
            if (self.elapsed_time>31):
                peaks, _ = find_peaks(self.gsums, height=0)
                breath_rate=self.breath_rate
                self.breath_rate = len(peaks)/2
                # print(breath_rate)
                cv2.putText(frame, f"No of Breaths after 30 seconds {int((self.breath_rate)/3)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                # break
                _,jpg=cv2.imencode('.jpg',frame)
                return jpg.tobytes()
            previous_frame = frame
            if(self.elapsed_time<31):
                if self.idf == 0:
                    droi = getChestROI(frame)
                    if droi[3] > 0:
                        self.bbox = droi
                        
                if self.idf > 0:
                    df = cv2.absdiff(frame, previous_frame)
                    m_diff = 1.0 * df.sum() / (df.shape[0] * df.shape[1])
                    if m_diff > 15:
                        droi = getChestROI(frame)
                        if droi[3] > 0:
                            self.bbox = droi
                            
                roi = frame[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
                # print(roi)
                frame = cv2.rectangle(frame, (self.bbox[0], self.bbox[1]), (self.bbox[2], self.bbox[3]), (255, 0, 0), 2)
                self.green = getColorAverage(roi, 1)  # 2nd channel for Green color
                if self.idf > 50:
                    # print("ENTER")
                    self.gsums.append(self.green)
                # print(self.gsums)
                self.idf += 1
                previous_frame = frame
                
                cv2.putText(frame, f"Time {int(self.elapsed_time)}", (50,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)    
                prev_frame=frame
                # ans=gsums
                # peaks, _ = find_peaks(self.gsums, height=0)
                # # breath_rate=self.breath_rate
                # self.breath_rate = (len(peaks))/2
                # print(breath_rate)
                _,jpg=cv2.imencode('.jpg',frame)
                return jpg.tobytes()
            # break
        # peaks=self.peaks
        # peaks, _ = find_peaks(gsums, height=0)
        # breath_rate=self.breath_rate
        # breath_rate = (len(peaks))/2
# break
        
            # if elapsed_time==30:
            #     break
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # cap.release()
        # cv2.destroyAllWindows()

        # Applying peak detection to detect breath rate
        # peaks, _ = find_peaks(gsums, height=0)
        # breath_rate = (len(peaks))/2
        # print("Breath rate: {} breaths till now".format(int(breath_rate)))
        

